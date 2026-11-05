from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime
import csv,hashlib
import numpy as np
SLOTS=['WTLA','WTRA','WTC','WTLL','WTRL']
FIELDS=['加速度X(g)','加速度Y(g)','加速度Z(g)']

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

def device_feature(directory,arm=None,arm_valid=None):
    from raw_sensor import imu_feature
    if arm is None:arm,arm_valid=imu_feature(directory)
    values=np.zeros((5,3,256),np.float32);valid=np.zeros(5,bool)
    if not arm_valid:return values,valid,False
    values[:2]=arm.reshape(2,3,256);valid[:2]=True
    try:
        _,new,mask,_,_=prepare_one(('inference',str(directory)))
        assert np.array_equal(new[:2],values[:2])
        return new,mask,False
    except (AssertionError,ValueError,TypeError,UnicodeDecodeError):
        # Preserve verified arm readings if an extra-device file cannot be parsed.
        return values,valid,True
