"""Matched four-epoch continuation, actual INT8 replay, no Test data."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,sys,time,traceback
E=Path(__file__).resolve().parent;C=E.parent;W=C/'videomaev2_base';O=C.parent/'error_resolution_20260913'
sys.path[:0]=[str(W),str(O),str(O/'bundle')]
spec=importlib.util.spec_from_file_location('v2_window_parent_runtime',W/'worker.py');base=importlib.util.module_from_spec(spec);sys.modules[spec.name]=base;spec.loader.exec_module(base)
sys.path.insert(0,str(E))
from data import WindowClips,SevenViews
from codec import pack,unpack
from dual_host_runtime.io import atomic,read,save_torch,retry
from dual_host_runtime.health import Guard,identity,owned_tree
import numpy as np
import torch
from torch.utils.data import DataLoader
OriginalModel=base.VideoMAEBase
PARENT=None
ARM=None

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(p,h):assert sha(p)==h,str(p);return Path(p)

def warm_model(*args,**kwargs):
 model=OriginalModel(*args,**kwargs);cp=torch.load(PARENT,map_location='cpu',weights_only=True);model.load_state_dict(cp['model'],strict=True)
 return model

class WarmClips(WindowClips):
 def __init__(self,*args,arm='RAW_CONTROL',**kwargs):super().__init__(*args,arm=ARM,**kwargs)

def evaluate(job,s,ph,out,inp,guard):
 frozen=read(out/'FROZEN.json');cp=torch.load(check(out/'model_fp16.pt',frozen['checkpoint_sha256']),map_location='cpu',weights_only=True)
 packed=pack(cp);save_torch(out/'model_int8.pt',packed);loaded=torch.load(out/'model_int8.pt',map_location='cpu',weights_only=True)
 model=OriginalModel(classes=40,fast=True,seed=s['seed']).cuda();model.load_state_dict(unpack(loaded),strict=True);model.eval()
 held=[r for r in inp['rows'] if r['user'] in job['held_users']];ds=SevenViews(held,s['data_root'],inp['regions'])
 loader=DataLoader(ds,batch_size=1,num_workers=s['workers'],pin_memory=True,prefetch_factor=1)
 ids=[];ys=[];us=[];values=[];indices=[];unique=0
 with torch.inference_mode():
  for b in loader:
   guard.check(phase='WINDOW_INT8_HELD',job=job['id'],records=len(ids),total=len(held));x=b['x'].flatten(0,1)
   duplicate=torch.equal(x[0],x[1]) and torch.equal(x[0],x[2]);selected=x[[0,3,4,5,6]] if duplicate else x;z=[]
   for k in range(0,len(selected),s['eval_micro']):
    v=model(selected[k:k+s['eval_micro']].cuda(non_blocking=True));assert v.dtype==torch.float32 and torch.isfinite(v).all();z.append(v.cpu().numpy())
   z=np.concatenate(z);unique+=len(z)
   if duplicate:z=z[[0,0,0,1,2,3,4]]
   values.append(z[None]);indices.append(b['frame_indices'].numpy());ids+=b['clip_id'];ys+=b['y'].tolist();us+=b['user'].tolist()
 values=np.concatenate(values).astype(np.float32);assert values.shape==(len(held),7,40)
 blended=.5*values[:,:5].mean(1)+.25*values[:,5]+.25*values[:,6]
 temp=out/'predictions.partial.npz';np.savez_compressed(temp,clip_ids=np.array(ids),targets=np.array(ys,dtype=np.int64),users=np.array(us,dtype=np.int64),videomae_logits=blended.astype(np.float32),five_view_logits=values[:,:5],temporal_logits=values[:,5:],seven_view_logits=values,frame_indices=np.concatenate(indices));retry(lambda:os.replace(temp,out/'predictions.npz'))
 atomic(out/'HELD_DONE.json',dict(status='COMPLETE_INT8_SEVEN_VIEWS',protocol_sha256=ph,checkpoint_sha256=sha(out/'model_int8.pt'),checkpoint_bytes=(out/'model_int8.pt').stat().st_size,records=len(ids),unique_forwarded_views=unique,no_fit=True,no_Test=True,at_unix=time.time()))
 del model,cp,packed,loaded,loader;torch.cuda.empty_cache()

def main(protocol,host):
 global PARENT,ARM
 s=read(protocol);ph=sha(protocol);assert s['status']=='FROZEN_FOR_EXECUTION' and (host=='local')==(os.name=='nt');h=s['hosts'][host];out=Path(h['output_directory']);assert out.resolve().is_relative_to(E)
 for p,v in s['code_sha256'].items():check(C/p,v)
 for p,v in h['input_bindings'].items():check(p,v)
 bd=read(out/'BUDGET.json');ct=read(out/'CONTROLLER.json');me=identity(os.getpid());assert bd['protocol_sha256']==ct['protocol_sha256']==ph and any(v['pid']==me['pid'] and abs(v['create_time']-me['create_time'])<.01 for v in owned_tree(ct['worker']))
 guard=Guard(out,bd['deadline_unix'],out/'TELEMETRY.json');guard.check(force=True,phase='WINDOW_PREFLIGHT')
 torch.set_num_threads(2 if host=='local' else 1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert ('4070' if host=='local' else '5090') in torch.cuda.get_device_name(0)
 inp=read(h['pose_inputs']);outer=read(h['original_protocol']);inner=read(h['inner_protocol']);runtime=dict(s);runtime.update(h)
 base.VideoMAEBase=warm_model;base.EpochClips=WarmClips;jobs=[j for j in s['jobs'] if j['host']==host];metadata=[]
 for job in jobs:
  par=next(j for j in (inner if job['kind']=='source_inner' else outer)['jobs'] if j['id']==job['split_parent'])
  for key in ['train_users','held_users','fold']+(['outer_excluded_users','inner'] if job['kind']=='source_inner' else []):assert job[key]==par[key]
  PARENT=check(job['parent_checkpoint'],job['parent_checkpoint_sha256']);ARM=job['arm'];cp=torch.load(PARENT,map_location='cpu',weights_only=True)
  assert cp['protocol_sha256']==job['parent_protocol_sha256'] and cp['epochs']==16
  for key in ['train_users','held_users']:assert cp[key]==job[key]
  subset=dict(inp);subset['rows']=[r for r in inp['rows'] if r['user'] not in job.get('outer_excluded_users',[])];assert {r['user'] for r in subset['rows']}==set(job['train_users'])|set(job['held_users']);del cp
  dest=out/job['id'];dest.mkdir(exist_ok=True);atomic(dest/'PARENT_BINDING.json',dict(protocol_sha256=ph,parent_checkpoint_sha256=job['parent_checkpoint_sha256'],parent_protocol_sha256=job['parent_protocol_sha256'],arm=ARM,outer_excluded_users=job.get('outer_excluded_users',[]),at_unix=time.time()))
  fg=Guard(dest,min(bd['deadline_unix'],time.time()+h['job_wall_cap_seconds']),out/'TELEMETRY.json');base.fit(job,runtime,ph,dest,subset,fg)
  metadata.append(dict(id=job['id'],arm=ARM,checkpoint_sha256=sha(dest/'model_fp16.pt')))
 atomic(out/'ALL_FITS_FROZEN.json',dict(protocol_sha256=ph,jobs=metadata,held_evaluation_started=False,at_unix=time.time()));files={'ALL_FITS_FROZEN.json':sha(out/'ALL_FITS_FROZEN.json')}
 for job in jobs:
  subset=dict(inp);subset['rows']=[r for r in inp['rows'] if r['user'] not in job.get('outer_excluded_users',[])];dest=out/job['id'];evaluate(job,runtime,ph,dest,subset,guard)
  for name in ['PARENT_BINDING.json','TRAIN_DONE.json','model_fp16.pt','model_int8.pt','INITIAL.json','SPLIT.json','HISTORY.json','FIRST_UPDATE.json','FROZEN.json','predictions.npz','HELD_DONE.json']:files[job['id']+'/'+name]=sha(dest/name)
 guard.check(force=True,phase='WINDOW_COMPLETE');atomic(out/'DONE.json',dict(status='COMPLETE_WINDOW_MATCHED4',protocol_sha256=ph,artifacts=files,no_Test=True,at_unix=time.time()))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--host',required=True);a=p.parse_args()
 try:main(a.protocol,a.host)
 except BaseException as exc:
  s=read(a.protocol);atomic(Path(s['hosts'][a.host]['output_directory'])/'ERROR.json',dict(protocol_sha256=sha(a.protocol),error=str(exc),traceback=traceback.format_exc(),at_unix=time.time()));raise
