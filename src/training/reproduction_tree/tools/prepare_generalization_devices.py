"""Add real device acceleration without altering any existing arm sample."""
from pathlib import Path
from collections import defaultdict,Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import csv,hashlib,json,os,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/generalization_training_20260910_v1/local';SLOTS=['WTLA','WTRA','WTC','WTLL','WTRL'];FIELDS=['加速度X(g)','加速度Y(g)','加速度Z(g)']
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest().upper()
def write(p,v):
    temp=p.with_name(p.name+'.partial');temp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(temp,p)
def prepare_one(arg):
    clip,directory=arg;signals=defaultdict(list);sources={};issues=Counter()
    for name in ['up(LA+RA+C).csv','down(LL+RL).csv']:
        p=Path(directory)/name
        if not p.is_file():issues['missing_file']+=1;continue
        data=p.read_bytes();sources[name]=hashlib.sha256(data).hexdigest().upper();rows=csv.DictReader(data.decode('utf-8-sig').splitlines())
        if not all(f in rows.fieldnames for f in ['时间','设备名称']+FIELDS):raise ValueError('SCHEMA:'+clip)
        for row in rows:
            device=row['设备名称'].split('(')[0].strip()
            if device not in SLOTS:continue
            try:
                stamp=datetime.fromisoformat(row['时间']).timestamp();values=[float(row[f]) for f in FIELDS]
                if not np.isfinite(values).all():raise ValueError('NONFINITE')
                signals[device].append((stamp,values))
            except (ValueError,TypeError):issues['invalid_rows']+=1
    processed={}
    for device,values in signals.items():
        groups=defaultdict(list)
        for t,v in values:groups[t].append(v)
        times=np.array(sorted(groups));data=np.array([np.mean(groups[t],axis=0) for t in times])
        if len(times)>=8:processed[device]=(times,data)
    assert 'WTLA' in processed and 'WTRA' in processed,'ARM_VALIDITY_DRIFT:'+clip
    begin=max(processed[s][0][0] for s in SLOTS[:2]);end=min(processed[s][0][-1] for s in SLOTS[:2]);assert 0<end-begin<=600
    grid=np.linspace(begin,end,256);values=np.zeros((5,3,256),np.float32);valid=np.zeros(5,bool)
    for i,s in enumerate(SLOTS):
        if s not in processed:issues['device_lt8_timestamps']+=1;continue
        times,data=processed[s]
        if i>=2:
            coverage=max(0,min(end,times[-1])-max(begin,times[0]))/(end-begin)
            interior=times[(times>=begin)&(times<=end)]
            if coverage<.9 or len(interior)<8 or np.diff(interior).max(initial=0)>max(1.,.2*(end-begin)) or (data.std(0)==0).any():issues['extra_quality_rejected']+=1;continue
        values[i]=np.stack([np.interp(grid,times,data[:,axis]) for axis in range(3)]).astype(np.float32);valid[i]=True
    return clip,values,valid,dict(sources),dict(issues)
def main():
    start=time.time();protocol=dict(status='FROZEN_BEFORE_CACHE_AND_C_TRAINING',builder_sha256=sha(__file__),manifest_sha256=sha(ROOT/'data/manifests/train_manifest.jsonl'),arm_sha256=sha(OUT/'bundle/data/arm_acceleration_t256.npz'),slots=SLOTS,policy='Keep all original arm tensors exactly. Extras:>=8unique finite timestamps,>=90percent arm-overlap coverage,>=8timestamps inside overlap,maxgap<=max(1second,.2overlap duration),nonconstant axes. Linear interpolation on the original arm time grid. Missing extras zero with explicit per-device mask. No normalizer fitting in cache.',Test_access=False,started_unix=start)
    assert not (OUT/'DEVICE_CACHE_PROTOCOL.json').exists();write(OUT/'DEVICE_CACHE_PROTOCOL.json',protocol)
    z=dict(np.load(OUT/'bundle/data/arm_acceleration_t256.npz',allow_pickle=False));rows={r['clip_id']:r for r in [json.loads(line) for line in (ROOT/'data/manifests/train_manifest.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip()]};valid_idx=np.flatnonzero(z['valid']);index={c:i for i,c in enumerate(z['clip_ids'])};x=np.zeros((len(index),5,3,256),np.float32);x[:,:2]=z['x'].reshape(-1,2,3,256);valid=np.zeros((len(index),5),bool);valid[:,:2]=z['valid'][:,None];sources={};issues=Counter();parity=0
    args=[(str(z['clip_ids'][i]),rows[str(z['clip_ids'][i])]['modalities']['IMU']) for i in valid_idx]
    with ProcessPoolExecutor(max_workers=6) as pool:
        for n,(clip,value,mask,hashes,errors) in enumerate(pool.map(prepare_one,args,chunksize=8),1):
            i=index[clip];assert np.array_equal(value[:2].reshape(6,256),z['x'][i]),'ARM_RESAMPLING_PARITY:'+clip;parity+=1;x[i,2:]=value[2:];valid[i,2:]=mask[2:];sources[clip]=hashes;issues.update(errors)
            if n%64==0:write(OUT/'DEVICE_PREPARATION_STATUS.json',dict(status='PREPARING',completed=n,total=len(args),elapsed_seconds=time.time()-start))
    assert np.array_equal(x[:,:2].reshape(-1,6,256),z['x']) and np.array_equal(valid[:,:2],np.repeat(z['valid'][:,None],2,1));path=OUT/'bundle/data/device_acceleration_t256.npz';np.savez_compressed(path,x=x,valid=valid,clip_ids=z['clip_ids'],users=z['users'],targets=z['targets'])
    write(OUT/'DEVICE_SOURCE_HASHES.json',sources);r=dict(status='PASS',path=path.relative_to(OUT).as_posix(),sha256=sha(path),bytes=path.stat().st_size,protocol_sha256=sha(OUT/'DEVICE_CACHE_PROTOCOL.json'),source_hashes_sha256=sha(OUT/'DEVICE_SOURCE_HASHES.json'),arm_numeric_parity_clips=parity,all2931arm_tensor_bytes_unchanged=True,valid_devices=dict(zip(SLOTS,valid.sum(0).tolist())),issues=dict(issues),elapsed_seconds=time.time()-start,at_unix=time.time());write(OUT/'DEVICE_CACHE_READY.json',r);print(json.dumps(r,ensure_ascii=False))
if __name__=='__main__':main()
