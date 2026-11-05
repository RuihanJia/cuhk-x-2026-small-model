"""Only new context3 FP32 readout of exact old INT8 endpoints; no fitting."""
from pathlib import Path
import argparse,hashlib,os,sys,time,traceback
E=Path(__file__).resolve().parent;C=E.parent;W=C/'videomaev2_base';Q=C/'videomaev2_base_quant';O=C.parent/'error_resolution_20260913'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
sys.path[:0]=[str(E),str(Q),str(W),str(O),str(O/'bundle')]
import numpy as np,torch
from torch.utils.data import DataLoader
from model import VideoMAEBase
from codec import pack,unpack,FORMAT
from data import ContextThreeViews,PHASES,METHOD
from quant_identity import validate
if os.name=='nt':
 import resilient_io;resilient_io.install()
from dual_host_runtime.io import read,atomic,retry
from dual_host_runtime.health import Guard,identity,owned_tree
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def checked(p,h):
 assert h and sha(p)==h,str(p)
 return Path(p)

def load_exact_quant(job,s,guard):
 receipt=read(checked(job['identity_receipt'],job['identity_receipt_sha256']))
 assert receipt['original_quant_checkpoint_sha256']==job['original_quant_checkpoint_sha256']
 assert receipt['parent_fp16_sha256']==job['parent_fp16_sha256']
 parent=Path(job['parent_directory'])
 td=read(checked(parent/'TRAIN_DONE.json',job['parent_train_done_sha256']))
 fr=read(checked(parent/'FROZEN.json',job['parent_frozen_sha256']))
 split=read(checked(parent/'SPLIT.json',job['parent_split_sha256']))
 assert td['status']=='COMPLETE_TRAIN_ONLY'
 assert td['protocol_sha256']==fr['protocol_sha256']==split['protocol_sha256']==job['parent_protocol_sha256']
 assert td['artifacts']['model_fp16.pt']==fr['checkpoint_sha256']==job['parent_fp16_sha256']
 assert fr['epochs']==16 and split['held_labels_used_in_fit'] is False
 for key in ('train_users','held_users'):
  assert fr[key]==split[key]==job[key]
 assert set(split['train_ids']).isdisjoint(split['held_ids'])
 if job['quant_mode']=='EXISTING_INT8':
  quant=torch.load(checked(job['quant_checkpoint'],job['original_quant_checkpoint_sha256']),map_location='cpu',weights_only=True)
 elif job['quant_mode']=='REPACK_FROZEN_FP16':
  original=torch.load(checked(parent/'model_fp16.pt',job['parent_fp16_sha256']),map_location='cpu',weights_only=True)
  guard.check(force=True,phase='CONTEXT_CPU_FIXED_REPACK',job=job['id'])
  quant=pack(original);del original
 else:raise AssertionError('UNKNOWN_QUANT_MODE')
 validate(quant,receipt)
 assert quant['protocol_sha256']==job['parent_protocol_sha256'] and quant['epochs']==16
 assert quant['train_users']==job['train_users'] and quant['held_users']==job['held_users']
 assert set(job['train_users']).isdisjoint(job['held_users'])
 assert (set(job['train_users'])|set(job['held_users'])).isdisjoint(job['outer_excluded_users'])
 return quant,split,receipt

def main(protocol,host):
 s=read(protocol);ph=sha(protocol);h=s['hosts'][host];out=Path(h['output_directory']).resolve()
 assert out.is_relative_to(E) and (os.name=='nt')==(host=='local')
 assert s['status']=='FROZEN_FOR_EXECUTION' and s['optimizer_updates']==0 and len(s['jobs'])==8
 assert s['factor']==1.5 and s['phases']==list(PHASES) and s['crop_method']==METHOD and s['storage_format']==FORMAT
 assert sum(j['host']=='cloud' for j in s['jobs'])==6 and sum(j['host']=='local' for j in s['jobs'])==2
 for n,v in s['code_sha256'].items():checked(C/n,v)
 for n,v in h['input_bindings'].items():checked(n,v)
 bd=read(out/'BUDGET.json');ct=read(out/'CONTROLLER.json');me=identity(os.getpid())
 assert bd['protocol_sha256']==ct['protocol_sha256']==ph
 assert bd['deadline_unix']<=s['deadline_unix'] and bd['deadline_unix']<=bd['first_started_unix']+h['wall_cap_seconds']
 assert any(v['pid']==me['pid'] and abs(v['create_time']-me['create_time'])<.01 for v in owned_tree(ct['worker']))
 guard=Guard(out,bd['deadline_unix'],out/'TELEMETRY.json');guard.check(force=True,phase='CONTEXT_PREFLIGHT')
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert ('5090' if host=='cloud' else '4070') in torch.cuda.get_device_name(0)
 inp=read(checked(h['pose_inputs'],s['pose_inputs_sha256']));assert len(inp['rows'])==2931
 files={}
 for j in [j for j in s['jobs'] if j['host']==host]:
  p=out/j['id'];p.mkdir(exist_ok=True);assert not(p/'DONE.json').exists();start=time.time()
  guard.check(force=True,phase='CONTEXT_JOB_START',job=j['id'])
  quant,split,receipt=load_exact_quant(j,s,guard)
  atomic(p/'QUANT_IDENTITY.json',dict(protocol_sha256=ph,job_id=j['id'],mode=j['quant_mode'],parent_fp16_sha256=j['parent_fp16_sha256'],original_quant_checkpoint_sha256=j['original_quant_checkpoint_sha256'],receipt_sha256=j['identity_receipt_sha256'],all_quantized_tensors_exact=True,checkpoint_metadata_exact=True,source_model_protocol_sha256=j['parent_protocol_sha256'],no_float_parent_inference=True))
  held=[r for r in inp['rows'] if r['user'] in j['held_users']]
  assert len(held)==j['records'] and {r['clip_id'] for r in held}==set(split['held_ids'])
  assert set(r['user'] for r in held)==set(j['held_users'])
  assert all(r['streams'][0]['directory'].startswith('Training/') for r in held)
  model=VideoMAEBase(classes=40,fast=True,seed=s['seed']).cuda();state=unpack(quant);model.load_state_dict(state,strict=True);model.eval()
  del state,quant
  ds=ContextThreeViews(held,h['data_root'],inp['regions']);loader=DataLoader(ds,batch_size=1,num_workers=h['workers'],pin_memory=True,prefetch_factor=1)
  ids=[];ys=[];us=[];values=[];indices=[];boxes=[];fallback=[]
  with torch.inference_mode():
   for b in loader:
    guard.check(phase='CONTEXT_FP32_NEW_THREE_ONLY',job=j['id'],records=len(ids),total=len(held))
    x=b['x'].flatten(0,1);assert x.shape[0]==3;zs=[]
    for k in range(0,3,h['eval_micro']):
     z=model(x[k:k+h['eval_micro']].cuda(non_blocking=True));assert z.dtype==torch.float32 and torch.isfinite(z).all();zs.append(z.cpu().numpy())
    values.append(np.concatenate(zs)[None]);indices.append(b['frame_indices'].numpy());boxes.append(b['context_box'].numpy());fallback.append(b['fallback'].numpy())
    ids+=b['clip_id'];ys+=b['y'].tolist();us+=b['user'].tolist()
  values=np.concatenate(values).astype(np.float32);assert values.shape==(j['records'],3,40)
  assert ids==[r['clip_id'] for r in held]
  tmp=p/'predictions.partial.npz'
  np.savez_compressed(tmp,clip_ids=np.array(ids),targets=np.array(ys,np.int64),users=np.array(us,np.int64),context_logits=values,context_mean=values.mean(1),frame_indices=np.concatenate(indices),context_boxes=np.concatenate(boxes),fallback=np.concatenate(fallback))
  retry(lambda:os.replace(tmp,p/'predictions.npz'))
  d=dict(status='COMPLETE_CONTEXT3_READOUT',protocol_sha256=ph,job_id=j['id'],fold=j['fold'],scope=j['scope'],records=len(ids),train_users=j['train_users'],held_users=j['held_users'],outer_excluded_users=j['outer_excluded_users'],parent_protocol_sha256=j['parent_protocol_sha256'],parent_fp16_sha256=j['parent_fp16_sha256'],original_quant_checkpoint_sha256=j['original_quant_checkpoint_sha256'],identity_sha256=sha(p/'QUANT_IDENTITY.json'),predictions_sha256=sha(p/'predictions.npz'),original_seven_views_recomputed=False,optimizer_updates=0,no_Test=True,no_held_metrics_computed=True,seconds=time.time()-start,at_unix=time.time())
  atomic(p/'DONE.json',d)
  for name in ('predictions.npz','DONE.json','QUANT_IDENTITY.json'):files[j['id']+'/'+name]=sha(p/name)
  del model,loader,ds;torch.cuda.empty_cache()
 atomic(out/'DONE.json',dict(status='COMPLETE_CONTEXT_READOUT',protocol_sha256=ph,host=host,jobs=sum(j['host']==host for j in s['jobs']),artifacts=files,optimizer_updates=0,no_Test=True,no_held_metrics_computed=True,at_unix=time.time()))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--host',choices=['local','cloud'],required=True);a=p.parse_args()
 try:main(a.protocol,a.host)
 except BaseException as e:
  s=read(a.protocol);atomic(Path(s['hosts'][a.host]['output_directory'])/'ERROR.json',dict(error=str(e),traceback=traceback.format_exc(),protocol_sha256=sha(a.protocol),at_unix=time.time()));raise
