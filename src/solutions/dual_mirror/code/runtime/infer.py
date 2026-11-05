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
from dual_model import PairVideo
from loader import load_state
import classwise_fusion as cw
import fusion_math as fm
from atomic_io import atomic as atomic_with_retry
io.atomic_json=atomic_with_retry

def sha_matches(path,expected):return io.sha256(path).lower()==str(expected).lower()

def constructor(container,kind,s):
 state,origin=load_state(container,kind)
 assert origin['train_users']==s['training_users'] and origin['held_users']==[] and origin['epochs']==16
 assert origin==s['pair_provenance'], 'PAIR_HEADER_PROVENANCE_MISMATCH'
 model=PairVideo(kind);model.load_state_dict(state,strict=True);return model.eval().requires_grad_(False)

class BaseViews:
 def __init__(self,samples,regions):self.samples=samples;self.regions=regions
 def __len__(self):return len(self.samples)
 def __getitem__(self,i):
  v,sub,bad=local_five(self.samples[i].sample_dir/'IR',self.regions[i]['box'])
  half,hs,hb=local_halves(self.samples[i].sample_dir/'IR',self.regions[i]['box']);sub+=hs;bad+=hb
  v=torch.cat([v,half],0) if v is not None and half is not None else None
  return dict(index=i,mae=v if v is not None else torch.empty(0),sub=sub,bad=bad)

from mirror_predict import predict,predict_mirror_only

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

def infer(data_dir,output,guard=None,workers=4,micro=4,sensor_cache=None,strict_original_template=False,original_video_cache=None):
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
 ids=np.asarray([x.sample_dir.name for x in samples]);original_video=None;mirrored_values={'Base':[],'Small':[]}
 if original_video_cache:
  assert strict_original_template and sha_matches(original_video_cache,s['optional_original_video_cache_sha256'])
  assert s['original_video_cache_pair_sha256'].lower()==s['checkpoints']['PAIR']['sha256'].lower()
  original_video=dict(np.load(original_video_cache,allow_pickle=False));assert np.array_equal(original_video['clip_ids'],ids)and np.array_equal(original_video['output_paths'],np.asarray([x.output_path for x in samples]))
  assert original_video['branch_logits'].shape==(n,4,40)and original_video['branch_logits'].dtype==np.float32 and np.isfinite(original_video['branch_logits']).all()and original_video['available'].shape==(n,4)and original_video['available'].dtype==bool
 recoveries=dict(video_substitutions=0,video_invalid_files=0,thermal_substitutions=0,thermal_invalid_files=0,pose_substitutions=0,pose_invalid_files=0)
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
 models={kind:constructor(ROOT/s['checkpoints']['PAIR']['path'],kind,s).to(device) for kind in ('Base','Small')};values={kind:[]for kind in models};available={kind:[]for kind in models};loader=DataLoader(BaseViews(samples,regions),batch_size=None,num_workers=workers,pin_memory=True,prefetch_factor=1)
 for i,item in enumerate(loader):
  assert item['index']==i
  for kind,model in models.items():
   if original_video is None:z,valid=predict(model,item,device,micro,kind)
   else:
    mirror,valid=predict_mirror_only(model,item,device,micro,kind);mirrored_values[kind].append(mirror);branch=2 if kind=='Base'else 3;assert valid==bool(original_video['available'][i,branch]);z=(.5*original_video['branch_logits'][i,branch]+.5*mirror).astype(np.float32)
   values[kind].append(z);available[kind].append(valid)
  recoveries['video_substitutions']+=item['sub'];recoveries['video_invalid_files']+=item['bad']
  if (i+1)%25==0:progress('PAIR_BASE6_SMALL8_FP32',i+1,n)
 del models,model;torch.cuda.empty_cache();logits=np.stack([tz,imu,values['Base'],values['Small']],1).astype(np.float32);mask=np.stack([tvalid,ivalid,available['Base'],available['Small']],1).astype(bool);fused,weights=fusion(logits,mask,cal);assert fused.shape==(n,40)

 io.write_submission(output,[(sample.output_path,int(z.argmax())) for sample,z in zip(samples,fused)]);np.savez_compressed(str(output)+'.logits.npz',clip_ids=ids,output_paths=np.asarray([x.output_path for x in samples]),branch_logits=logits,available=mask,fused=fused,weights=weights,**(dict(original_video_branch_logits=original_video['branch_logits'],mirrored_video_logits=np.stack([mirrored_values['Base'],mirrored_values['Small']],1).astype(np.float32)) if original_video is not None else {}))
 audit=dict(status='BASE6_SMALL8_TI_CLASSWISE_CSV_CREATED',rows=n,classes=40,csv_sha256=io.sha256(output),all_serialized_inference_weight_bytes=size,calibrator_sha256=s['calibrator_sha256'],checkpoints=s['checkpoints'],availability={k:int(mask[:,i].sum()) for i,k in enumerate(['T','I','Base','Small'])},recoveries=recoveries,elapsed_seconds=time.time()-start,Test_labels_read=False,uploaded=False,LLM_in_inference=False,used_exact_TI_cache=bool(sensor_cache),used_exact_original_video_cache=bool(original_video_cache));io.atomic_json(Path(str(output)+'.audit.json'),audit);progress('COMPLETE',n,n);return audit
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--micro',type=int,default=4);p.add_argument('--sensor-cache');p.add_argument('--original-video-cache');p.add_argument('--strict-original-template',action='store_true');a=p.parse_args();print(json.dumps(infer(**vars(a))))
