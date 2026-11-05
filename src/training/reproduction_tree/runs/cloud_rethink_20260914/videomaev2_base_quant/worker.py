"""Fixed storage conversion and FP32 five-view replay; zero optimizer updates."""
from pathlib import Path
import argparse,hashlib,os,sys,time,traceback
E=Path(__file__).resolve().parent;C=E.parent;W=C/'videomaev2_base';Q=E;O=C.parent/'error_resolution_20260913'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
sys.path[:0]=[str(Q),str(W),str(C/'aux_public_regression_20260914/dynamic_interaction/photometric_candidate'),str(O),str(O/'bundle')]
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import VideoMAEBase
from codec import pack,unpack,FORMAT
from photometric_data_numpy import PhotometricFiveViews
if os.name=='nt':
 sys.path.insert(0,str(E));import resilient_io;resilient_io.install()
from dual_host_runtime.io import read,atomic,save_torch,retry
from dual_host_runtime.health import Guard,identity,owned_tree

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(p,h):assert sha(p)==h,str(p);return Path(p)

def main(protocol,host):
 assert host in ('cloud','local') and (os.name=='nt')==(host=='local');s=read(protocol);ph=sha(protocol);h=s['hosts'][host];out=Path(h['output_directory']);assert out.is_relative_to(E)
 assert s['status']=='FROZEN_FOR_EXECUTION' and s['optimizer_updates']==0 and s['storage_format']==FORMAT and len(s['jobs'])==8
 for n,v in s['code_sha256'].items():check(C/n,v)
 for n,v in h['input_bindings'].items():
  check(n,v)
 bd=read(out/'BUDGET.json');ct=read(out/'CONTROLLER.json');actual=identity(os.getpid());assert bd['protocol_sha256']==ct['protocol_sha256']==ph
 assert any(v['pid']==actual['pid'] and abs(v['create_time']-actual['create_time'])<.01 for v in owned_tree(ct['worker']))
 guard=Guard(out,bd['deadline_unix'],out/'TELEMETRY.json');guard.check(force=True,phase='QUANT_REPLAY_PREFLIGHT')
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert ('5090' if host=='cloud' else '4070') in torch.cuda.get_device_name(0)
 inp=read(check(h['pose_inputs'],s['pose_inputs_sha256']));assert len(inp['rows'])==2931
 files={}
 for job in [j for j in s['jobs'] if j['host']==host]:
  p=out/job['id'];p.mkdir(exist_ok=True)
  parent=Path(job['parent_directory']);td=read(parent/'TRAIN_DONE.json');fr=read(parent/'FROZEN.json')
  assert td['protocol_sha256']==fr['protocol_sha256']==job['parent_protocol_sha256'] and td['status']=='COMPLETE_TRAIN_ONLY'
  for n,v in td['artifacts'].items():check(parent/n,v)
  job=dict(job,checkpoint_sha256=td['artifacts']['model_fp16.pt'])
  assert fr['checkpoint_sha256']==job['checkpoint_sha256']
  atomic(p/'EXACT_PARENT_BINDING.json',dict(parent_directory=str(parent),parent_protocol_sha256=job['parent_protocol_sha256'],checkpoint_sha256=job['checkpoint_sha256'],train_done_sha256=sha(parent/'TRAIN_DONE.json'),at_unix=time.time()))
  cp=torch.load(check(parent/'model_fp16.pt',job['checkpoint_sha256']),map_location='cpu',weights_only=True)
  assert cp['protocol_sha256']==job['parent_protocol_sha256'] and cp['epochs']==16 and cp['train_users']==job['train_users'] and cp['held_users']==job['held_users']
  assert not set(cp['train_users'])&set(cp['held_users'])
  assert not (set(cp['train_users'])|set(cp['held_users']))&set(job['outer_excluded_users'])
  quant=pack(cp);save_torch(p/'model_int8.pt',quant);q=torch.load(p/'model_int8.pt',map_location='cpu',weights_only=True)
  assert (p/'model_int8.pt').stat().st_size<87000000
  state=unpack(q);reference=unpack(quant);assert all(torch.equal(state[k],reference[k]) for k in state);del reference
  model=VideoMAEBase(classes=40,fast=True,seed=s['seed']).cuda();model.load_state_dict(state,strict=True);model.eval()
  held=[r for r in inp['rows'] if r['user'] in job['held_users']];assert len(held)==job['records']
  ds=PhotometricFiveViews(held,h['data_root'],inp['regions'],arm='RAW_CONTROL')
  loader=DataLoader(ds,batch_size=1,num_workers=h['workers'],pin_memory=True,prefetch_factor=1)
  ids=[];ys=[];us=[];values=[];unique_views=0
  with torch.inference_mode():
   for b in loader:
    guard.check(phase='QUANT_REPLAY_FP32_FIVE_VIEW',job=job['id'],records=len(ids),total=len(held));x=b['x'].flatten(0,1)
    duplicate=torch.equal(x[0],x[1]) and torch.equal(x[0],x[2]);selected=x[[0,3,4]] if duplicate else x;z=[]
    for k in range(0,len(selected),h['eval_micro']):
     v=model(selected[k:k+h['eval_micro']].cuda(non_blocking=True));assert v.dtype==torch.float32 and torch.isfinite(v).all();z.append(v.cpu().numpy())
    z=np.concatenate(z);unique_views+=len(z)
    if duplicate:z=z[[0,0,0,1,2]]
    values.append(z[None]);ids+=b['clip_id'];ys+=b['y'].tolist();us+=b['user'].tolist()
  values=np.concatenate(values).astype(np.float32);assert values.shape==(len(held),5,40)
  temporary=p/'predictions.partial.npz';np.savez_compressed(temporary,clip_ids=np.array(ids),targets=np.array(ys,dtype=np.int64),users=np.array(us,dtype=np.int64),videomae_logits=values.mean(1),five_view_logits=values);retry(lambda:os.replace(temporary,p/'predictions.npz'))
  atomic(p/'DONE.json',dict(status='COMPLETE_FIXED_INT8_FP32_FIVE_VIEW_REPLAY',protocol_sha256=ph,parent_protocol_sha256=job['parent_protocol_sha256'],parent_checkpoint_sha256=job['checkpoint_sha256'],checkpoint_sha256=sha(p/'model_int8.pt'),checkpoint_bytes=(p/'model_int8.pt').stat().st_size,storage_format=FORMAT,quantized_matrices=len(q['quantized_names']),records=len(ids),train_users=job['train_users'],held_users=job['held_users'],outer_excluded_users=job['outer_excluded_users'],unique_forwarded_views=unique_views,predictions_sha256=sha(p/'predictions.npz'),no_optimizer=True,no_Test=True,at_unix=time.time()))
  for n in ['model_int8.pt','predictions.npz','DONE.json','EXACT_PARENT_BINDING.json']:files[job['id']+'/'+n]=sha(p/n)
  del model,loader,ds,cp,q,quant,state;torch.cuda.empty_cache()
 atomic(out/'DONE.json',dict(status='COMPLETE_VIDEOMAE_QUANT_REPLAY',protocol_sha256=ph,jobs=sum(j['host']==host for j in s['jobs']),host=host,optimizer_updates=0,artifacts=files,no_Test=True,at_unix=time.time()))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--host',required=True);a=p.parse_args()
 try:main(a.protocol,a.host)
 except BaseException as exc:
  s=read(a.protocol);atomic(Path(s['hosts'][a.host]['output_directory'])/'ERROR.json',dict(protocol_sha256=sha(a.protocol),error=str(exc),traceback=traceback.format_exc(),at_unix=time.time()));raise
