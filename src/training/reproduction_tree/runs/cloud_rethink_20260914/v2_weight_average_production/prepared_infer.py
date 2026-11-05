"""Reproducible T/I/VideoMAE-Base inference with one shared pose locator."""
from pathlib import Path
import argparse,json,os,sys,time
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'pose_site'))
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'runtime/pose_settings');os.environ['YOLO_AUTOINSTALL']='false';os.environ['PYTEST_CURRENT_TEST']='pose_geometry_audit'
import numpy as np
import torch
from torch.utils.data import DataLoader
import base_runtime as b
import video_io as io
from detail_inputs import detector_frames,detector_result,local_five
from temporal_inputs import local_halves
from pose_detector import load_detector
from mae_model import VideoMAEBase
from mae_codec import unpack
import classwise_fusion as cw
import fusion_math as fm
from atomic_io import atomic as atomic_with_retry
io.atomic_json=atomic_with_retry

def sha_matches(path,expected):return io.sha256(path).lower()==str(expected).lower()

def constructor(cp,s):
 assert cp['protocol_sha256']==s['checkpoints']['B']['protocol_sha256']
 assert cp['train_users']==s['training_users'] and cp['held_users']==[] and len(cp['quantized_names'])==50
 for key,value in s['checkpoint_provenance'].items():assert cp[key]==value,key
 model=VideoMAEBase(classes=40,fast=True);model.load_state_dict(unpack(cp),strict=True)
 assert sum(p.numel()for p in model.parameters())==86257960
 return model.eval().requires_grad_(False)

class BaseViews:
 def __init__(self,samples,regions):self.samples=samples;self.regions=regions
 def __len__(self):return len(self.samples)
 def __getitem__(self,i):
  v,sub,bad=local_five(self.samples[i].sample_dir/'IR',self.regions[i]['box'])
  half,hs,hb=local_halves(self.samples[i].sample_dir/'IR',self.regions[i]['box']);sub+=hs;bad+=hb
  v=torch.cat([v,half],0) if v is not None and half is not None else None
  return dict(index=i,mae=v if v is not None else torch.empty(0),sub=sub,bad=bad)

@torch.inference_mode()
def predict(model,item,device,micro=4,kind='Base'):
 x=item['mae']
 if not x.numel():return np.zeros(40,np.float32),False
 assert x.shape==(7,3,16,224,224)
 x=x if kind=='Base' else x[:5];dup=torch.equal(x[0],x[1]) and torch.equal(x[0],x[2]);sel=[0,3,4,5,6] if kind=='Base' else [0,3,4];selected=x[sel] if dup else x;values=[]
 for i in range(0,len(selected),micro):
  z=model(selected[i:i+micro].to(device,non_blocking=True));assert z.dtype==torch.float32 and torch.isfinite(z).all();values.append(z.cpu().numpy())
 z=np.concatenate(values)
 if dup:z=z[[0,0,0,1,2,3,4] if kind=='Base' else [0,0,0,1,2]]
 return ((.5*z[:5].mean(0)+.25*z[5]+.25*z[6]) if kind=='Base' else z.mean(0)).astype(np.float32),True

def fusion(logits,available,cal):
 if 'delta' in cal:
  z=cw.predict(logits,available,cal);delta=np.asarray(cal['delta']);delta=delta-delta.mean(0,keepdims=True);scores=np.broadcast_to(np.asarray(cal['global_bias'])[None,:,None]+delta[None],logits.shape).copy()
 else:
  model=fm.Calibrator(cal['description']);model.load_state_dict({k:torch.tensor(v,dtype=torch.float64) for k,v in cal['state'].items()},strict=True)
  with torch.no_grad():z=model(dict(logits=torch.tensor(logits,dtype=torch.float64),mask=torch.tensor(available))).numpy().astype(np.float32)
  scores=np.broadcast_to(np.asarray(cal['state']['bias'])[None,:,None],logits.shape).copy()
 scores[~available]=-np.inf;scores-=np.max(scores,1,keepdims=True);weights=np.exp(scores);weights/=weights.sum(1,keepdims=True)
 assert np.isfinite(z).all() and not weights[~available].any()
 return z,weights.astype(np.float32)

def infer(data_dir,output,guard=None,workers=4,micro=4,sensor_cache=None,strict_original_template=False):
 b.setup();output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);s=json.loads((ROOT/'release.json').read_text());device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');start=time.time();cp={};size=0
 def progress(phase,n,total):
  if guard:guard.check(force=True,phase=phase,records=n,total=total)
  value=dict(phase=phase,records=n,total=total,at_unix=time.time());io.atomic_json(Path(str(output)+'.status.json'),value);print(json.dumps(value),flush=True)
 for k,v in s['checkpoints'].items():
  p=ROOT/v['path'];assert sha_matches(p,v['sha256']);size+=p.stat().st_size
  if k not in ('POSE','PAIR'):
   cp[k]=torch.load(p,map_location='cpu',weights_only=True);assert cp[k]['protocol_sha256']==v['protocol_sha256'] and sorted(cp[k]['train_users'])==s['training_users']
 calpath=ROOT/'calibrator.json';assert sha_matches(calpath,s['calibrator_sha256']);cal=json.loads(calpath.read_text());size+=calpath.stat().st_size;assert size==s['serialized_weight_bytes']<=100000000
 samples,discovery=io.discover_samples(Path(data_dir));n=len(samples);assert n>0
 if strict_original_template:
  assert n==405;b.verify_paths([x.output_path for x in samples])
 ids=np.asarray([x.sample_dir.name for x in samples]);recoveries=dict(video_substitutions=0,video_invalid_files=0,thermal_substitutions=0,thermal_invalid_files=0,pose_substitutions=0,pose_invalid_files=0)
 if sensor_cache:
  assert strict_original_template;assert sha_matches(sensor_cache,s['optional_sensor_cache_sha256']);cache=dict(np.load(sensor_cache,allow_pickle=False));assert np.array_equal(cache['clip_ids'],ids) and np.array_equal(cache['output_paths'],np.asarray([x.output_path for x in samples]))
  tz=cache['branch_logits'][:,2].copy();imu=cache['branch_logits'][:,3].copy();tvalid=cache['available'][:,2].copy();ivalid=cache['available'][:,3].copy();progress('REUSED_EXACT_FROZEN_TI',n,n)
 else:
  devices=[];masks=[];ivalid=[];fallback=0
  for sample in samples:
   arm,valid=b.imu_feature(sample.sample_dir/'IMU');x,mask,rec=b.device_feature(sample.sample_dir/'IMU',arm,valid);devices.append(x);masks.append(mask);ivalid.append(valid);fallback+=int(rec)
  data=dict(devices=np.stack(devices),device_valid=np.stack(masks));imu=b.imu_predictions(data,cp['I'],device);recoveries['device_parse_fallback']=fallback
  thermal=b.thermal_constructor().to(device);thermal.load_state_dict(cp['T']['state_dict'],strict=True);thermal.eval().requires_grad_(False);tz=[np.zeros(40,np.float32) for _ in samples];tvalid=[];pending=[];indices=[]
  with torch.inference_mode():
   for i,sample in enumerate(samples):
    x,sub,bad=b.thermal_input(sample.sample_dir/'Thermal',cp['T']['mean'],cp['T']['std']);recoveries['thermal_substitutions']+=sub;recoveries['thermal_invalid_files']+=bad;tvalid.append(x is not None)
    if x is not None:pending.append(x);indices.append(i)
    if len(pending)==2 or (i+1==len(samples) and pending):
     z=thermal(torch.stack(pending).to(device)).float().cpu().numpy()
     for index,value in zip(indices,z):tz[index]=value
     pending=[];indices=[]
    if (i+1)%50==0:progress('THERMAL_FP32',i+1,n)
  del thermal;torch.cuda.empty_cache()
 cache=s.get('optional_geometry_cache');regions=None
 if strict_original_template and cache and (ROOT/cache['path']).exists():
  assert sha_matches(ROOT/cache['path'],cache['sha256']);value=json.loads((ROOT/cache['path']).read_text());assert value['clip_ids']==ids.tolist() and not value['labels_used'] and value['model_sha256'].lower()==s['checkpoints']['POSE']['sha256'].lower();regions=value['regions'];progress('REUSED_LABEL_FREE_GEOMETRY',n,n)
 else:
  import cv2
  from ultralytics.utils import torch_utils
  torch_utils.NUM_THREADS=2;torch.set_num_threads(2);cv2.setNumThreads(1);detector=load_detector(ROOT/s['checkpoints']['POSE']['path'],ROOT/'yolo11n-pose.yaml');regions=[]
  with ThreadPoolExecutor(max_workers=workers) as pool:
   for i,(frames,sub,bad) in enumerate(pool.map(detector_frames,samples)):
    pred=detector.predict(frames,imgsz=640,conf=.25,iou=.7,max_det=1,device=str(device),half=False,verbose=False,save=False) if frames else [];regions.append(detector_result(pred,frames));recoveries['pose_substitutions']+=sub;recoveries['pose_invalid_files']+=bad
    if (i+1)%50==0:progress('LABEL_FREE_PERSON_GEOMETRY',i+1,n)
  del detector;torch.cuda.empty_cache()
 models={'Base':constructor(cp['B'],s).to(device)};values={kind:[]for kind in models};available={kind:[]for kind in models};loader=DataLoader(BaseViews(samples,regions),batch_size=None,num_workers=workers,pin_memory=True,prefetch_factor=1)
 for i,item in enumerate(loader):
  assert item['index']==i
  for kind,model in models.items():
   z,valid=predict(model,item,device,micro,kind);values[kind].append(z);available[kind].append(valid)
  recoveries['video_substitutions']+=item['sub'];recoveries['video_invalid_files']+=item['bad']
  if (i+1)%25==0:progress('PAIR_BASE6_SMALL8_FP32',i+1,n)
 del models,model;torch.cuda.empty_cache();logits=np.stack([tz,imu,values['Base']],1).astype(np.float32);mask=np.stack([tvalid,ivalid,available['Base']],1).astype(bool);fused,weights=fusion(logits,mask,cal);assert fused.shape==(n,40)

 io.write_submission(output,[(sample.output_path,int(z.argmax())) for sample,z in zip(samples,fused)]);np.savez_compressed(str(output)+'.logits.npz',clip_ids=ids,output_paths=np.asarray([x.output_path for x in samples]),branch_logits=logits,available=mask,fused=fused,weights=weights)
 audit=dict(status='WEIGHT_AVERAGED_BASE_TI_CSV_CREATED',rows=n,classes=40,csv_sha256=io.sha256(output),all_serialized_inference_weight_bytes=size,calibrator_sha256=s['calibrator_sha256'],checkpoints=s['checkpoints'],availability={k:int(mask[:,i].sum()) for i,k in enumerate(['T','I','Base'])},recoveries=recoveries,elapsed_seconds=time.time()-start,Test_labels_read=False,uploaded=False,LLM_in_inference=False,used_exact_TI_cache=bool(sensor_cache));io.atomic_json(Path(str(output)+'.audit.json'),audit);progress('COMPLETE',n,n);return audit
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--micro',type=int,default=4);p.add_argument('--sensor-cache');p.add_argument('--strict-original-template',action='store_true');a=p.parse_args();print(json.dumps(infer(**vars(a))))
