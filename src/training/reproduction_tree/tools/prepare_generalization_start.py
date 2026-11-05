"""Freeze the authorized first two queues; no training or external writes in this preparer."""
from pathlib import Path
import ast, hashlib, json, shutil, time
from datetime import datetime

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/generalization_training_20260910_v1';PY=ROOT.parent/'CUHK-X/.venv/Scripts/python.exe'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest().upper()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    assert not BASE.exists(),'NEW_RUN_ALREADY_EXISTS'
    state=read(ROOT/'PROJECT_STATE.json');allusers=list(range(1,10))+list(range(16,25));folds={'0':[2,16,20],'3':[5,7,8,21]};now=datetime.now().astimezone().isoformat(timespec='seconds');start=time.time();assetroot=ROOT/state['assets']['training_cache_and_video_starting_bundle'];source=ROOT/'runs/s4_tail_average_20260910_v1/bundle'
    for host in ['cloud','local']:
        root=BASE/host;bundle=root/'bundle';bundle.mkdir(parents=True)
        for folder in ['research_runtime','dual_host_runtime']:shutil.copytree(source/folder,bundle/folder,ignore=shutil.ignore_patterns('__pycache__'))
        # Same validated atomic implementation on Windows and Linux.
        common=bundle/'research_runtime/common.py';common.write_text(common.read_text(encoding='utf-8')+'\nfrom dual_host_runtime.io import atomic,save_torch\n',encoding='utf-8')
        shutil.copy2(ROOT/'tools/generalization_native_queue.py',bundle/'native_queue.py')
        for name in (['generalization_cloud_worker.py'] if host=='cloud' else ['generalization_imu_worker.py']):shutil.copy2(ROOT/'tools'/name,bundle/name)
        (bundle/'data').mkdir();(bundle/'assets').mkdir()
    cloud=BASE/'cloud';b=cloud/'bundle'
    for f in ['frame_index.json','measured_sensors.npz']:shutil.copy2(assetroot/'data'/f,b/'data'/f)
    shutil.copy2(assetroot/'assets/mvit_k400.pth',b/'assets/mvit_k400.pth')
    code={p.relative_to(b).as_posix():sha(p) for p in b.rglob('*.py')};jobs={};order=[];groups={}
    for fold,held in folds.items():
        available=[u for u in allusers if u not in held];groups[fold]=[available[i::3] for i in range(3)]
        assert set(sum(groups[fold],[]))==set(available) and len(sum(groups[fold],[]))==len(available)
    for k in [0,1,2,'outer']:
        for fold,held in folds.items():
            pred=held if k=='outer' else groups[fold][k];train=[u for u in allusers if u not in held and u not in (pred if k!='outer' else [])]
            for kind,prefix in [('video','V'),('sensor','S')]:
                name=f'{prefix}_F{fold}_{k}';jobs[name]=dict(kind=kind,train_users=train,predict_users=pred,source='mvit_public',epochs=20 if kind=='video' else 60,seed=2026091003,arm='individual',fold=int(fold),calibration=False,micro_batch=4,accumulation=2,effective_batch=8,loader_workers=4,non_blocking=True,backbone_lr=.0001,head_lr=.001);order.append(name)
    # Long video jobs launch first on both halves; light matched sensors follow without assistant intervention.
    order=[k for k in order if k.startswith('V')]+[k for k in order if k.startswith('S')]
    protocol=dict(id='GENERALIZATION-USER-CROSSFIT-VIDEO-SENSOR-20260910-V1',status='FROZEN_BEFORE_TRAINING',frozen_at_local=now,authorization='User explicitly requested both hosts start training at2026-09-10 20:51:56 +08',hypothesis='User-excluded inner predictions permit training reliable fusion without inner encoder label leakage; fresh matching outer models isolate recipe mismatch from calibration benefit.',training_users=allusers,folds=folds,inner_user_groups=groups,code_sha256=code,assets={'mvit_public':dict(path='assets/mvit_k400.pth',sha256=sha(b/'assets/mvit_k400.pth'),train_users=[],admission='evidence/pretrained/mvit_v2_s_kinetics400_20260908_admission.json')},data_sha256={f:sha(b/'data'/f) for f in ['frame_index.json','measured_sensors.npz']},input=dict(frames=16,resize=256,crop=224,normalization_mean=[.45]*3,normalization_std=[.225]*3,train_sampling='shared segment jitter and0.5horizontalflip; center crop',prediction='fixed five-view cross: center,low/high long-axis crops atphase.5;center crops atphases.25/.75; FP32 logits averaged equally; per-modality exported'),jobs=jobs,remote_data='/home/featurize/cuhkx_dual_host_20260910_v1/dataset/Small-Model-Track',budget=dict(max_wall_seconds=21600,deadline_unix=min(start+21600,state['current_authorization']['cloud_access_deadline_unix']),concurrency=2),fusion_preregistration=dict(outer_folds=[0,3],strategies=['matched fixed video.8_sensor.2','train-only scalar-temperature convex video/sensor/Thermal','train-only reliability gate with entropy,top2margin,Depth-IR disagreement and availability'],calibration_objective='inverse-user-count weighted NLL; no heldouter fitting',temperature_bounds=[.25,8],gate_l2=.01,optimizer='CPU FP64 LBFGS max500iterations, one initialization, no sweep',missing_thermal='mask and renormalize weights; report coverage',outer_metrics='fixed final predictions only; effect size,error flips,per-user and uncertainty; no arbitrary count gate',reference='Existing R1 five-view805/1094 is a separate historical recipe reference, not the matched control'),limitations=['F0/F3 repeatedly used development; no independent Private-score improvement claim','New seed and microbatch differ from historical R1; matching new outer video/sensor models required','F2/F4 users may appear only on permitted training/calibration sides of F0/F3, not as separately scored outer folds'],Test_access=False,all18=False,external_data=False,kaggle_upload=False)
    write(cloud/'protocol.json',protocol);ph=sha(cloud/'protocol.json');write(cloud/'QUEUE.json',dict(deadline_unix=protocol['budget']['deadline_unix'],concurrency=2,jobs=[dict(id=k,worker='generalization_cloud_worker.py',protocol='protocol.json',protocol_sha256=ph) for k in order]))
    local=BASE/'local';b=local/'bundle';arm=ROOT/'archive/20260907_handoff/project/runs/_native_imu_20260903/arm_acceleration_t256.npz';shutil.copy2(arm,b/'data'/arm.name)
    code={p.relative_to(b).as_posix():sha(p) for p in b.rglob('*.py')};jobs={}
    for arm_name in 'ABC':
        for fold,held in folds.items():jobs[f'IMU_{arm_name}_F{fold}']=dict(arm=arm_name,fold=int(fold),train_users=[u for u in allusers if u not in held],predict_users=held)
    lp=dict(id='GENERALIZATION-PER-DEVICE-IMU-20260910-V1',status='FROZEN_BEFORE_TRAINING',frozen_at_local=now,hypothesis='Independent rotation already helped arm IMU; shared per-device temporal encoding may exploit the extra measured devices without channel-appending confounds.',code_sha256=code,folds=folds,jobs=jobs,data=dict(arm_sha256=sha(arm),manifest_sha256=sha(ROOT/'data/manifests/train_manifest.jsonl'),cohort='Same2543valid native arm clips for all three arms',slots=['WTLA','WTRA','WTC','WTLL','WTRL'],extra_devices='Original arm tensors remain byte-identical; extras interpolated onto arm-overlap time grid; per-device missing mask. Source builder and concrete cache bound before C starts.'),training=dict(seed=2026091003,epochs=60,batch=64,optimizer='AdamW',lr=.001,weight_decay=.0001,scheduler='cosine60_eta1e-5',label_smoothing=.1,clip_norm=1.,precision='FP32',orientation='Bernoulli.5 per clip;independent HaarSO3 perdevice,constant over time;raw rotation before train-only per-channel standardization; same A/B/C schedule'),controls=dict(A='historical ArmIMU topology, two-arm acceleration plus orientation',B='shared3to64to128to128 temporal encoder; five fixed slots; extras zero;645to128to40 head',C='exact B architecture and seed; valid extra measured acceleration included',capacity='A and B differ in capacity; B and C exactly same parameter count and initial state'),budget=dict(max_wall_seconds=10800,deadline_unix=start+10800,concurrency=1),metrics='Fixed final60epoch serialized FP16 model reconstructed in FP32. No validation during fit. Report paired correctness flips,macroF1,eachuser,uncertainty; no hard improvement-count gate.',limitations=['New60epoch seed-matched batch is not an exact reproduction of older100epoch IMU result','Per-device availability is observed input quality, not identity features; coverage and missingness reported'],Test_access=False,F2_F4_outer=False,all18=False,external_data=False)
    write(local/'protocol.json',lp);lh=sha(local/'protocol.json');write(local/'QUEUE.json',dict(deadline_unix=start+10800,concurrency=1,jobs=[dict(id=k,worker='generalization_imu_worker.py',protocol='protocol.json',protocol_sha256=lh,requires=['DEVICE_CACHE_READY.json'] if v['arm']=='C' else []) for k,v in jobs.items()]))
    for p in list((BASE/'cloud/bundle').rglob('*.py'))+list((BASE/'local/bundle').rglob('*.py')):ast.parse(p.read_text(encoding='utf-8'))
    facts=dict(event='E-GENERALIZATION-DUAL-HOST-REGISTERED-20260910-V1',at_local=now,status='FROZEN_NOT_YET_LAUNCHED',cloud_protocol=str((cloud/'protocol.json').relative_to(ROOT)).replace('\\','/'),cloud_sha256=ph,local_protocol=str((local/'protocol.json').relative_to(ROOT)).replace('\\','/'),local_sha256=lh,cloud_video_fits=8,cloud_sensor_fits=8,local_imu_fits=6,cloud_thermal='Preparatory artifact/queue extension pending before its own training; total phase6h cap preserved',gpu_training_started=False)
    with (ROOT/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(facts,ensure_ascii=False)+'\n')
    write(BASE/'REGISTRATION.json',facts);state.update(updated_at_local=now,phase='DUAL_HOST_GENERALIZATION_FROZEN_AWAITING_PROCESS_LAUNCH',next_frozen_candidate=facts);state['successor_preparation'].update(status='PROTOCOLS_FROZEN_QUEUES_READY',run_root=BASE.relative_to(ROOT).as_posix());write(ROOT/'PROJECT_STATE.json',state);print(json.dumps(facts,ensure_ascii=False))

if __name__=='__main__':main()
