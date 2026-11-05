"""Two fixed temporal half windows of existing INT8 models; zero updates."""
from pathlib import Path
import argparse,hashlib,os,sys,time,traceback
E=Path(__file__).resolve().parent;C=E.parent;W=C/'videomaev2_base';Q=C/'videomaev2_base_quant';O=C.parent/'error_resolution_20260913'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
sys.path[:0]=[str(E),str(Q),str(W),str(O),str(O/'bundle')]
import numpy as np,torch
from torch.utils.data import DataLoader
from model import VideoMAEBase
from codec import unpack,FORMAT
from data import TemporalHalves
if os.name=='nt':
 import resilient_io;resilient_io.install()
from dual_host_runtime.io import read,atomic,retry
from dual_host_runtime.health import Guard,identity,owned_tree
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(protocol,host):
 s=read(protocol);ph=sha(protocol);h=s['hosts'][host];out=Path(h['output_directory']);assert out.is_relative_to(E) and (os.name=='nt')==(host=='local')
 assert s['status']=='FROZEN_FOR_EXECUTION' and s['optimizer_updates']==0 and len(s['jobs'])==8
 for n,v in s['code_sha256'].items():assert sha(C/n)==v,n
 for n,v in h['input_bindings'].items():assert sha(n)==v,n
 bd=read(out/'BUDGET.json');ct=read(out/'CONTROLLER.json');me=identity(os.getpid());assert bd['protocol_sha256']==ct['protocol_sha256']==ph
 assert any(v['pid']==me['pid'] and abs(v['create_time']-me['create_time'])<.01 for v in owned_tree(ct['worker']))
 guard=Guard(out,bd['deadline_unix'],out/'TELEMETRY.json');guard.check(force=True,phase='TEMPORAL_HALVES_PREFLIGHT')
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;assert ('5090' if host=='cloud' else '4070') in torch.cuda.get_device_name(0)
 inp=read(h['pose_inputs']);assert sha(h['pose_inputs'])==s['pose_inputs_sha256'] and len(inp['rows'])==2931;files={}
 for j in [j for j in s['jobs'] if j['host']==host]:
  p=out/j['id'];p.mkdir(exist_ok=True);qroot=Path(j['quant_directory']);done=read(qroot/'DONE.json');assert sha(qroot/'DONE.json')==j['quant_done_sha256']
  assert done['protocol_sha256']==s['quant_protocol_sha256'] and done['checkpoint_sha256']==j['checkpoint_sha256'] and done['quantized_matrices']==50
  assert sha(qroot/'model_int8.pt')==j['checkpoint_sha256'];cp=torch.load(qroot/'model_int8.pt',map_location='cpu',weights_only=True)
  assert cp['train_users']==j['train_users'] and cp['held_users']==j['held_users'] and not set(cp['train_users'])&set(cp['held_users']) and cp['epochs']==16
  assert not (set(cp['train_users'])|set(cp['held_users']))&set(j['outer_excluded_users'])
  model=VideoMAEBase(classes=40,fast=True,seed=s['seed']).cuda();model.load_state_dict(unpack(cp),strict=True);model.eval()
  assert sha(qroot/'predictions.npz')==j['old_predictions_sha256']
  with np.load(qroot/'predictions.npz',allow_pickle=False) as ar:old={k:ar[k] for k in ar.files}
  held=[r for r in inp['rows'] if r['user'] in j['held_users']];assert len(held)==j['records'] and old['clip_ids'].tolist()==[r['clip_id'] for r in held]
  ds=TemporalHalves(held,h['data_root'],inp['regions']);loader=DataLoader(ds,batch_size=2,num_workers=h['workers'],pin_memory=True,prefetch_factor=1)
  ids=[];ys=[];us=[];values=[];indices=[];start=time.time()
  with torch.inference_mode():
   for b in loader:
    guard.check(phase='TEMPORAL_HALVES_FP32',job=j['id'],records=len(ids),total=len(held));n=len(b['clip_id']);x=b['x'].flatten(0,1);assert len(x)<=4;z=model(x.cuda(non_blocking=True));assert z.dtype==torch.float32 and torch.isfinite(z).all()
    values.append(z.cpu().numpy().reshape(n,2,40));indices.append(b['frame_indices'].numpy());ids+=b['clip_id'];ys+=b['y'].tolist();us+=b['user'].tolist()
  values=np.concatenate(values).astype(np.float32);assert values.shape==(len(held),2,40) and ids==old['clip_ids'].tolist() and np.array_equal(ys,old['targets']) and np.array_equal(us,old['users'])
  combined=.5*old['videomae_logits']+.25*values[:,0]+.25*values[:,1]
  tmp=p/'predictions.partial.npz';np.savez_compressed(tmp,clip_ids=np.array(ids),targets=np.array(ys,dtype=np.int64),users=np.array(us,dtype=np.int64),videomae_logits=combined,temporal_logits=values,frame_indices=np.concatenate(indices));retry(lambda:os.replace(tmp,p/'predictions.npz'))
  d=dict(status='COMPLETE_FIXED_TEMPORAL_HALF_REPLAY',protocol_sha256=ph,checkpoint_sha256=j['checkpoint_sha256'],parent_quant_protocol_sha256=s['quant_protocol_sha256'],records=len(ids),train_users=j['train_users'],held_users=j['held_users'],outer_excluded_users=j['outer_excluded_users'],predictions_sha256=sha(p/'predictions.npz'),old_predictions_sha256=j['old_predictions_sha256'],optimizer_updates=0,model_bytes_unchanged=True,at_unix=time.time(),seconds=time.time()-start)
  atomic(p/'DONE.json',d)
  for n in ['predictions.npz','DONE.json']:files[j['id']+'/'+n]=sha(p/n)
  del model,loader,ds,cp,old;torch.cuda.empty_cache()
 atomic(out/'DONE.json',dict(status='COMPLETE_TEMPORAL_HALVES',protocol_sha256=ph,artifacts=files,optimizer_updates=0,no_Test=True,at_unix=time.time()))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--host',required=True);a=p.parse_args()
 try:main(a.protocol,a.host)
 except BaseException as e:
  s=read(a.protocol);atomic(Path(s['hosts'][a.host]['output_directory'])/'ERROR.json',dict(error=str(e),traceback=traceback.format_exc(),protocol_sha256=sha(a.protocol),at_unix=time.time()));raise
