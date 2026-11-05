"""One fixed endpoint average. Source-held INT8 readout only, zero optimizer updates."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time,traceback
E=Path(__file__).resolve().parent;C=E.parent;O=C.parent/'error_resolution_20260913'
P=C/'aux_public_regression_20260914/dynamic_interaction/photometric_candidate'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
sys.path[:0]=[str(E),str(C/'videomaev2_base_quant'),str(C/'videomaev2_base'),str(P),str(C/'v2_window_resume'),str(O),str(O/'bundle')]
import numpy as np,torch
from torch.utils.data import DataLoader
from model import VideoMAEBase
from codec import pack,unpack,FORMAT
from data import SevenViews
if os.name=='nt':
 import resilient_io;resilient_io.install()
from dual_host_runtime.io import read,atomic,retry,save_torch
from dual_host_runtime.health import Guard,identity,owned_tree

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def check(p,h):assert h and sha(p)==h,str(p);return Path(p)

def average_state(a,b):
 assert a.keys()==b.keys();out={};changed=0
 for k,x in a.items():
  y=b[k];assert x.shape==y.shape and x.dtype==y.dtype==torch.float16
  assert torch.isfinite(x).all() and torch.isfinite(y).all()
  changed+=int(not torch.equal(x,y));out[k]=((x.float()+y.float())*.5).half()
  assert torch.isfinite(out[k]).all()
 assert changed>0
 return out,changed

def parent(job,kind):
 v=job[kind];root=Path(v['directory'])
 fr=read(check(root/'FROZEN.json',v['frozen_sha256']));sp=read(check(root/'SPLIT.json',v['split_sha256']))
 td=read(check(root/'TRAIN_DONE.json',v['train_done_sha256']))
 assert fr['protocol_sha256']==sp['protocol_sha256']==td['protocol_sha256']==v['protocol_sha256']
 assert td['artifacts']['model_fp16.pt']==fr['checkpoint_sha256']==v['checkpoint_sha256']
 assert sp['held_labels_used_in_fit'] is False and fr['epochs']==(16 if kind=='raw' else 4)
 for k in ('train_users','held_users'):assert fr[k]==sp[k]==job[k]
 if kind=='control':
  pb=read(check(root/'PARENT_BINDING.json',v['parent_binding_sha256']))
  assert pb['parent_checkpoint_sha256']==job['raw']['checkpoint_sha256'] and pb['parent_protocol_sha256']==job['raw']['protocol_sha256'] and pb['arm']=='CONTROL'
 cp=torch.load(check(root/'model_fp16.pt',v['checkpoint_sha256']),map_location='cpu',weights_only=True)
 assert cp['protocol_sha256']==v['protocol_sha256'] and cp['epochs']==fr['epochs']
 for k in ('train_users','held_users'):assert cp[k]==job[k]
 return cp,sp

def main(protocol,host):
 s=read(protocol);ph=sha(protocol);h=s['hosts'][host];out=Path(h['output_directory']).resolve()
 assert s['status']=='FROZEN_FOR_EXECUTION' and (os.name=='nt')==(host=='local') and out.is_relative_to(E)
 assert s['optimizer_updates']==0 and s['alpha']==.5 and len(s['jobs'])==6 and s['storage_format']==FORMAT
 for n,v in s['code_sha256'].items():check(C/n,v)
 for n,v in h['input_bindings'].items():check(n,v)
 bd=read(out/'BUDGET.json');ct=read(out/'CONTROLLER.json');me=identity(os.getpid())
 assert bd['protocol_sha256']==ct['protocol_sha256']==ph and bd['deadline_unix']<=s['deadline_unix']
 assert bd['deadline_unix']<=bd['first_started_unix']+h['wall_cap_seconds']
 assert any(v['pid']==me['pid'] and abs(v['create_time']-me['create_time'])<.01 for v in owned_tree(ct['worker']))
 guard=Guard(out,bd['deadline_unix'],out/'TELEMETRY.json');guard.check(force=True,phase='AVERAGE_PREFLIGHT')
 torch.set_num_threads(2 if host=='local' else 1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert ('4070' if host=='local' else '5090') in torch.cuda.get_device_name(0)
 inp=read(check(h['pose_inputs'],s['pose_inputs_sha256']));assert len(inp['rows'])==2931
 jobs=[j for j in s['jobs'] if j['host']==host];assert len(jobs)==(1 if host=='local'else 5);files={}
 for j in jobs:
  start=time.time();p=out/j['id'];p.mkdir(exist_ok=True);assert not(p/'DONE.json').exists()
  assert j['scope'] in ('source','outer') and set(j['train_users']).isdisjoint(j['held_users']) and (set(j['train_users'])|set(j['held_users'])).isdisjoint(j['outer_excluded_users'])
  a,sa=parent(j,'raw');b,sb=parent(j,'control')
  assert sa['train_ids']==sb['train_ids'] and sa['held_ids']==sb['held_ids']
  averaged,changed=average_state(a['model'],b['model']);del a,b
  packed=pack(dict(model=averaged,protocol_sha256=ph,train_users=j['train_users'],held_users=j['held_users'],seed=s['seed'],alpha=.5,raw_sha256=j['raw']['checkpoint_sha256'],control_sha256=j['control']['checkpoint_sha256']))
  cp=p/'model_int8.pt';save_torch(cp,packed);assert cp.stat().st_size<=s['checkpoint_cap_bytes'];expected=unpack(packed)
  loaded=unpack(torch.load(cp,map_location='cpu',weights_only=True));assert all(torch.equal(expected[k],loaded[k])for k in expected)
  atomic(p/'AVERAGE_FROZEN.json',dict(protocol_sha256=ph,alpha=.5,arithmetic='(raw_fp16.float()+control_fp16.float())*0.5 then FP16 then original INT8 codec',checkpoint_sha256=sha(cp),checkpoint_bytes=cp.stat().st_size,raw=j['raw'],control=j['control'],changed_source_tensors=changed,tensors=len(averaged),parameters=sum(v.numel()for v in averaged.values()),quantized_matrices=len(packed['quantized_names']),FP32_reload_exact=True,optimizer_updates=0,held_readout_started=False,at_unix=time.time()))
  del averaged,packed,expected
  held=[r for r in inp['rows'] if r['user'] in j['held_users']];assert len(held)==j['records'] and {r['clip_id']for r in held}==set(sa['held_ids'])
  assert all(r['streams'][0]['directory'].startswith('Training/')for r in held)
  model=VideoMAEBase(classes=40,fast=True,seed=s['seed']).cuda();model.load_state_dict(loaded,strict=True);model.eval();del loaded
  assert not any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm)for m in model.modules())
  ds=SevenViews(held,h['data_root'],inp['regions']);loader=DataLoader(ds,batch_size=1,num_workers=h['workers'],pin_memory=True,prefetch_factor=1)
  ids=[];ys=[];us=[];zs=[];idx=[];unique=0
  with torch.inference_mode():
   for batch in loader:
    guard.check(phase='AVERAGE_INT8_SOURCE_HELD',job=j['id'],records=len(ids),total=len(held))
    x=batch['x'].flatten(0,1);duplicate=torch.equal(x[0],x[1]) and torch.equal(x[0],x[2]);sel=x[[0,3,4,5,6]]if duplicate else x;z=[]
    for k in range(0,len(sel),h['eval_micro']):
     v=model(sel[k:k+h['eval_micro']].cuda(non_blocking=True));assert v.dtype==torch.float32 and torch.isfinite(v).all();z.append(v.cpu().numpy())
    z=np.concatenate(z);unique+=len(z)
    if duplicate:z=z[[0,0,0,1,2,3,4]]
    zs.append(z[None]);idx.append(batch['frame_indices'].numpy());ids+=batch['clip_id'];ys+=batch['y'].tolist();us+=batch['user'].tolist()
  z=np.concatenate(zs).astype(np.float32);assert z.shape==(j['records'],7,40) and ids==[r['clip_id']for r in held]
  value=(.5*z[:,:5].mean(1)+.25*z[:,5]+.25*z[:,6]).astype(np.float32)
  tmp=p/'predictions.partial.npz';np.savez_compressed(tmp,clip_ids=np.array(ids),targets=np.array(ys,np.int64),users=np.array(us,np.int64),videomae_logits=value,seven_view_logits=z,frame_indices=np.concatenate(idx));retry(lambda:os.replace(tmp,p/'predictions.npz'))
  atomic(p/'DONE.json',dict(status='COMPLETE_FIXED_AVERAGE_SOURCE_READOUT',protocol_sha256=ph,job_id=j['id'],records=len(ids),unique_forwarded_views=unique,optimizer_updates=0,no_Test=True,no_outer=False,seconds=time.time()-start,artifacts={n:sha(p/n)for n in ['AVERAGE_FROZEN.json','model_int8.pt','predictions.npz']},at_unix=time.time()))
  for n in ['AVERAGE_FROZEN.json','model_int8.pt','predictions.npz','DONE.json']:files[j['id']+'/'+n]=sha(p/n)
  del model,loader,ds;torch.cuda.empty_cache()
 atomic(out/'DONE.json',dict(status='COMPLETE_WEIGHT_AVERAGE_COMPLETION',protocol_sha256=ph,host=host,optimizer_updates=0,no_Test=True,no_outer=False,artifacts=files,at_unix=time.time()))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--host',choices=['cloud','local'],required=True);a=p.parse_args()
 try:main(a.protocol,a.host)
 except BaseException as e:
  s=read(a.protocol);atomic(Path(s['hosts'][a.host]['output_directory'])/'ERROR.json',dict(protocol_sha256=sha(a.protocol),error=str(e),traceback=traceback.format_exc(),at_unix=time.time()));raise
