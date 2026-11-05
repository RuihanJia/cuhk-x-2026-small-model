"""Exact numeric sensor features used by the accepted training caches."""
import csv,json,math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np

def normalize_person(person):
 points=np.asarray(person.get('keypoints',[]),dtype=np.float32);scores=np.asarray(person.get('keypoint_scores',[]),dtype=np.float32)
 if points.shape!=(17,3) or scores.shape!=(17,):return -1.,np.zeros((17,3),np.float32),np.zeros(17,np.float32)
 mask=(np.isfinite(points).all(axis=1)&np.isfinite(scores)&(scores>0)).astype(np.float32)
 if not mask.any():return -1.,np.zeros((17,3),np.float32),mask
 work=np.where(mask[:,None]>0,points,0.);center=work.sum(axis=0)/mask.sum();centered=np.where(mask[:,None]>0,points-center,0.);spread=float(np.sqrt(np.square(centered).sum(axis=1).sum()/mask.sum()))
 return spread,(centered/max(spread,1e-6)).astype(np.float32),mask

def skeleton_feature(directory):
 files=sorted((Path(directory)/'predictions').glob('*.json'),key=lambda p:p.name)
 if not files:raise ValueError('MISSING_SKELETON_FRAMES')
 pos=[];mask=[]
 for k in range(64):
  p=files[min(len(files)-1,int((k+.5)*len(files)/64))];people=json.loads(p.read_text(encoding='utf-8-sig'))
  if not isinstance(people,list):raise ValueError('INVALID_SKELETON_SCHEMA')
  candidates=[normalize_person(x) for x in people]
  if candidates:
   _,points,valid=max(candidates,key=lambda x:x[0]);scalar=math.log1p(min(len(people),3))/math.log(4)
  else:points=np.zeros((17,3),np.float32);valid=np.zeros(17,np.float32);scalar=0.
  frame=np.concatenate((points.reshape(-1),valid,np.asarray([scalar],np.float32)));pos.append(frame[:51]);mask.append(frame[51:])
 position=np.stack(pos);motion=np.zeros_like(position);motion[1:]=position[1:]-position[:-1];result=np.concatenate((position,motion,np.stack(mask)),axis=1)
 assert result.shape==(64,120) and np.isfinite(result).all();return result

def imu_feature(directory):
 slots=['WTLA','WTRA'];signals=defaultdict(list);p=Path(directory)/'up(LA+RA+C).csv';output=np.zeros((6,256),np.float32)
 if not p.is_file():return output,False
 try:
  rows=csv.DictReader(p.read_bytes().decode('utf-8-sig').splitlines());required=['时间','设备名称','加速度X(g)','加速度Y(g)','加速度Z(g)']
  if not all(k in rows.fieldnames for k in required):return output,False
  for row in rows:
   device=row['设备名称'].split('(')[0].strip()
   if device not in slots:continue
   try:
    stamp=datetime.fromisoformat(row['时间']).timestamp();values=[float(row[k]) for k in required[2:]]
    if not np.isfinite(values).all():continue
    signals[device].append((stamp,values))
   except (ValueError,TypeError):continue
 except (UnicodeDecodeError,ValueError,TypeError):return output,False
 processed={}
 for device,values in signals.items():
  groups=defaultdict(list)
  for stamp,v in values:groups[stamp].append(v)
  times=np.array(sorted(groups));data=np.array([np.mean(groups[t],axis=0) for t in times])
  if len(times)>=8:processed[device]=(times,data)
 if len(processed)!=2:return output,False
 begin=max(x[0][0] for x in processed.values());end=min(x[0][-1] for x in processed.values())
 if not 0<end-begin<=600:return output,False
 grid=np.linspace(begin,end,256)
 for k,device in enumerate(slots):
  times,values=processed[device];output[k*3:k*3+3]=np.stack([np.interp(grid,times,values[:,axis]) for axis in range(3)]).astype(np.float32)
 return output,True
