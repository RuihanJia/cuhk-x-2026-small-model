"""Regenerate training-only inputs from the authorized raw dataset, without fitting.

Outputs are local data artifacts and must stay out of Git. Existing inputs are
never overwritten. The default recipe targets the frozen selected solutions.
"""
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import itertools
import json
import os
import sys
import time

DEFAULT_RUNTIME = Path(__file__).resolve().parents[1] / 'solutions/dual_mirror/code/runtime'
sys.path.insert(0,str(DEFAULT_RUNTIME))
from atomic_io import atomic as verified_atomic

USERS = set(range(1, 10)) | set(range(16, 25))
EXTENSIONS = {'.png', '.jpg', '.jpeg'}

def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def atomic(path, value):
    verified_atomic(path,value)

def images(path):
    return sorted(p.name for p in Path(path).iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS) if Path(path).is_dir() else []

def records(root, modality):
    base = root / 'Training/data/HAR/data' / modality
    assert base.is_dir(), f'MISSING_TRAINING_MODALITY:{base}'
    result = []
    for activity in base.iterdir():
        if not activity.is_dir():
            continue
        label = int(activity.name.split('_', 1)[0])
        assert 0 <= label < 40
        for subject in activity.iterdir():
            if not subject.is_dir() or not subject.name.startswith('user'):
                continue
            user = int(subject.name[4:])
            assert user in USERS, f'UNEXPECTED_TRAINING_USER:{user}'
            for trial in subject.iterdir():
                if not trial.is_dir():
                    continue
                files = images(trial)
                if files:
                    result.append(dict(clip_id=f'{activity.name}/{user}/{trial.name}',
                                       label=label, user=user, directory=trial.relative_to(root).as_posix(), files=files))
    return sorted(result, key=lambda r: (r['label'], r['user'], r['clip_id'].rsplit('/', 1)[1]))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--runtime-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'solutions/dual_mirror/code/runtime')
    p.add_argument('--pose-state', type=Path)
    p.add_argument('--stage', choices=['indexes', 'sensors', 'conflicts', 'pose', 'all'], default='all')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--max-minutes', type=float, default=30)
    p.add_argument('--reference-inputs', type=Path, help='Optional historical POSE_ADAPTER_INPUTS.json; comparison only')
    a = p.parse_args()
    root, out, runtime = a.data_dir.resolve(), a.output.resolve(), a.runtime_dir.resolve()
    assert root.name != 'Testing' and (root / 'Training').is_dir()
    assert not (out / (a.stage.upper() + '_DONE.json')).exists(), 'COMPLETED_STAGE_EXISTS_USE_NEW_OUTPUT'
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    deadline = started + 60 * a.max_minutes
    def progress(stage, completed, total):
        assert time.time() < deadline, 'PREPROCESSING_WALL_LIMIT'
        assert not (out / 'STOP_REQUESTED').exists(), 'USER_STOP_REQUESTED'
        value = dict(stage=stage, completed=completed, total=total, seconds=time.time()-started)
        atomic(out / 'PROGRESS.json', value)
        print(json.dumps(value), flush=True)
    spec = dict(operation='training_input_regeneration', training_fits=0, model_updates=0,
                allowed_users=sorted(USERS), read_scope='Training only', Test_read=False,
                max_minutes=a.max_minutes, workers=a.workers, script_sha256=digest(__file__),
                stage=a.stage, data_dir=str(root), started_unix=started)
    atomic(out / f'{a.stage.upper()}_PROTOCOL.json', spec)
    ir = records(root, 'IR')
    depth = {r['clip_id']: r for r in records(root, 'Depth_Color')}
    def stream(m,s):
        inventory=sorted((name,(root/s['directory']/name).stat().st_size)for name in s['files'])
        sample=s['files'][len(s['files'])//2]
        return dict(modality=m,directory=s['directory'],files=s['files'],
                    inventory_sha256=hashlib.sha256(json.dumps(inventory,sort_keys=True,separators=(',',':')).encode()).hexdigest().upper(),
                    sample_name=sample,sample_sha256=digest(root/s['directory']/sample).upper())
    rows = [dict(clip_id=r['clip_id'], label=r['label'], user=r['user'], streams=[
        stream(m,s) for m,s in [('Depth_Color',depth[r['clip_id']]),('IR',r)]])
        for r in ir if r['clip_id'] in depth]
    assert len(rows) == 2931, f'FROZEN_VIDEO_COHORT_DRIFT:{len(rows)}'
    assert {r['user'] for r in rows} == USERS
    if not (out / 'frame_index.json').exists():
        atomic(out / 'frame_index.json', rows)
    else:
        assert read(out / 'frame_index.json') == rows, 'FRAME_INDEX_DRIFT'
    reference = read(a.reference_inputs) if a.reference_inputs else None
    audit = dict(video_records=len(rows), users=sorted(USERS), Test_read=False, training_fits=0)
    if reference:
        reference_rows = {r['clip_id']:r for r in reference['rows']}
        assert set(reference_rows) == {r['clip_id'] for r in rows}
        audit['reference_IR_rows_equal'] = all(dict(r, streams=[r['streams'][1]]) == reference_rows[r['clip_id']] for r in rows)
        assert audit['reference_IR_rows_equal'], 'IR_INDEX_REFERENCE_MISMATCH'
    if a.stage in ('indexes', 'all'):
        thermal = records(root, 'Thermal')
        assert len(thermal) == 2891, f'FROZEN_THERMAL_COHORT_DRIFT:{len(thermal)}'
        atomic(out / 'thermal_index.json', thermal)
        audit['thermal_records'] = len(thermal)
        progress('INDEXES_COMPLETE', len(rows), len(rows))
    sys.path.insert(0, str(runtime))
    if a.stage in ('sensors', 'all'):
        import numpy as np
        from raw_sensor import imu_feature
        from device_inputs import device_feature
        def one(row):
            activity, _, trial = row['clip_id'].split('/')
            directory = root / 'Training/data/HAR/data/IMU' / activity / f"user{row['user']}" / trial
            arm, valid = imu_feature(directory)
            devices, masks, fallback = device_feature(directory, arm, valid)
            return arm, valid, devices, masks, fallback
        arms, arm_valid, devices, masks, fallbacks = [], [], [], [], []
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            for i, value in enumerate(pool.map(one, rows), 1):
                arm, valid, dev, mask, fallback = value
                arms.append(arm); arm_valid.append(valid); devices.append(dev); masks.append(mask); fallbacks.append(fallback)
                if i % 100 == 0:
                    progress('NATIVE_IMU_REGENERATION', i, len(rows))
        ids = np.asarray([r['clip_id'] for r in rows]); users = np.asarray([r['user'] for r in rows]); targets = np.asarray([r['label'] for r in rows])
        arm = np.stack(arms); valid = np.asarray(arm_valid); dev = np.stack(devices); mask = np.stack(masks)
        assert np.isfinite(arm).all() and np.isfinite(dev).all()
        assert np.array_equal(dev[:, :2].reshape(-1, 6, 256), arm)
        assert int(valid.sum()) == 2543, f'ARM_COHORT_DRIFT:{valid.sum()}'
        np.savez_compressed(out / 'arm_acceleration_t256.npz', x=arm, valid=valid, clip_ids=ids, users=users, targets=targets)
        np.savez_compressed(out / 'device_acceleration_t256.npz', x=dev, valid=mask, clip_ids=ids, users=users, targets=targets)
        ready = dict(status='REGENERATED_FROM_RAW_TRAINING', path='bundle/data/device_acceleration_t256.npz',
                     sha256=digest(out/'device_acceleration_t256.npz').upper(), arm_numeric_parity_clips=int(valid.sum()),
                     valid_devices=mask.sum(0).tolist(), fallback_count=sum(fallbacks), Test_read=False)
        atomic(out / 'DEVICE_CACHE_READY.json', ready)
        audit['sensors'] = ready
        progress('SENSORS_COMPLETE', len(rows), len(rows))
    if a.stage in ('conflicts', 'all'):
        # Same-user, different-label matching frame names, then actual byte hashes.
        inverted = defaultdict(list)
        for i, r in enumerate(ir):
            for name in r['files']:
                if name.startswith('IR_') and name.endswith('.png'):
                    inverted[(r['user'], name)].append(i)
        pairs = defaultdict(list)
        for (_, name), indices in inverted.items():
            for i, j in itertools.combinations(indices, 2):
                if ir[i]['label'] != ir[j]['label']:
                    pairs[i,j].append(name)
        fractions, hashes, pair_audit = {}, {}, []
        def hashed(i, name):
            key = i, name
            if key not in hashes:
                hashes[key] = digest(root / ir[i]['directory'] / name)
            return hashes[key]
        for n, ((i,j), names) in enumerate(sorted(pairs.items()), 1):
            equal = sum(hashed(i,name) == hashed(j,name) for name in names)
            if equal >= 3:
                for k in (i,j):
                    cid = ir[k]['clip_id']; fractions[cid] = max(fractions.get(cid,0.), equal/len(ir[k]['files']))
            pair_audit.append(dict(a=ir[i]['clip_id'],b=ir[j]['clip_id'],shared_names=len(names),identical_bytes=equal))
            if n % 25 == 0:
                progress('TRAINING_OVERLAP_BYTE_CHECK', n, len(pairs))
        weights = {r['clip_id']:max(.25,1-.75*fractions.get(r['clip_id'],0.)) for r in rows}
        atomic(out/'conflict_weights.json', weights)
        atomic(out/'CONFLICT_AUDIT.json', dict(pairs=pair_audit, raw_IR_clips=len(ir), hashed_frames=len(hashes), Test_read=False))
        audit['downweighted_clips'] = sum(v < 1 for v in weights.values())
        if reference:
            audit['reference_conflict_weights_equal'] = weights == reference['conflict_weights']
            assert audit['reference_conflict_weights_equal'], 'CONFLICT_WEIGHT_REFERENCE_MISMATCH'
        progress('CONFLICT_WEIGHTS_COMPLETE', len(pairs), len(pairs))
    if a.stage in ('pose', 'all'):
        assert a.pose_state and a.pose_state.is_file(), 'POSE_STATE_REQUIRED'
        import numpy as np
        import torch
        from PIL import Image
        os.environ['YOLO_CONFIG_DIR'] = str(out/'pose_settings')
        os.environ['YOLO_AUTOINSTALL'] = 'false'
        os.environ['PYTEST_CURRENT_TEST'] = 'pose_geometry_audit'
        sys.path.insert(0,str(runtime/'pose_site'))
        from pose_detector import load_detector
        from pose_roi import box_from_detections
        from activity_inputs import box_from_thumbnails
        import cv2
        torch.set_num_threads(2); cv2.setNumThreads(1)
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        assert digest(a.pose_state).lower() == 'bc709bb30a90c8f230b0cb6288cecaf852cab29a2578752e1a43004a9ad9bdb4'
        detector = load_detector(a.pose_state, runtime/'yolo11n-pose.yaml')
        regions, raw_regions = {}, {}
        cache = out/'POSE_PARTIAL.json'
        if cache.exists():
            previous=read(cache);regions=previous['regions'];raw_regions=previous['raw_regions']
        def fallback_region(stream):
            files=stream['files'];frames=[];native=None
            for i in sorted(set(min(len(files)-1,int((k+.5)*len(files)/16)) for k in range(16))):
                with Image.open(root/stream['directory']/files[i]) as im:
                    im=im.convert('RGB');native=im.size;scale=120/min(im.size)
                    frames.append(np.asarray(im.resize(tuple(round(v*scale)for v in im.size),Image.Resampling.BILINEAR),dtype=np.uint8))
            return box_from_thumbnails(np.stack(frames),native)['box']
        with torch.inference_mode():
            for i,row in enumerate(rows,1):
                cid=row['clip_id']
                if cid in regions:
                    continue
                stream=row['streams'][1];frames=[];frame_hashes=[]
                for index in sorted(set(min(len(stream['files'])-1,int((k+.5)*len(stream['files'])/8)) for k in range(8))):
                    file=root/stream['directory']/stream['files'][index]
                    with Image.open(file) as im:frame=np.asarray(im.convert('RGB'))
                    frames.append(np.round(np.sqrt(frame.astype(np.float32)/255)*255).astype(np.uint8)[:,:,::-1].copy())
                    frame_hashes.append(dict(index=index,sha256=digest(file)))
                h,w=frames[0].shape[:2]
                preds=detector.predict(frames,imgsz=640,conf=.25,iou=.7,max_det=1,device=device,half=False,verbose=False,save=False)
                detections=[dict(boxes=v.boxes.xyxy.tolist(),confidence=v.boxes.conf.tolist(),keypoints=v.keypoints.data.tolist())for v in preds]
                box,reason=box_from_detections(detections,w,h)
                raw_regions[cid]=dict(box=box,reason=reason,native_size=[w,h],sampled_frame_hashes=frame_hashes,detections=detections)
                if box is None:
                    box=fallback_region(row['streams'][0]) or fallback_region(stream)
                    reason='EXISTING_DEPTH_OR_IR_FALLBACK'
                regions[cid]=[dict(box=box,native_size=[w,h],reason=reason)]
                if i%25==0:
                    atomic(cache,dict(regions=regions,raw_regions=raw_regions));progress('TRAINING_POSE_GEOMETRY',i,len(rows))
        atomic(out/'POSE_REGIONS.json',dict(regions=raw_regions))
        weights=read(out/'conflict_weights.json')
        payload=dict(rows=[dict(r,streams=[r['streams'][1]])for r in rows],regions=regions,conflict_weights=weights,
                     geometry_sha256=digest(out/'POSE_REGIONS.json'),input_kind='POSE_HEAD_ARMS_WITH_EXISTING_DEPTH_OR_IR_FALLBACK')
        atomic(out/'POSE_ADAPTER_INPUTS.json',payload)
        if reference:
            mismatch=[cid for cid in regions if regions[cid][0]['box'] != reference['regions'][cid][0]['box']]
            audit['reference_pose_box_mismatches']=mismatch
            audit['pose_boxes_compared']=len(regions)
        audit['pose_device']=device
        progress('POSE_COMPLETE',len(rows),len(rows))
    audit.update(status='COMPLETE',seconds=time.time()-started,stage=a.stage,
                 generated_files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p))for p in out.iterdir() if p.is_file() and p.suffix in ('.npz','.json') and p.name not in ('POSE_PARTIAL.json','PROGRESS.json')})
    atomic(out/(a.stage.upper()+'_DONE.json'),audit)
    print(json.dumps(audit,ensure_ascii=True),flush=True)

if __name__ == '__main__':
    main()
