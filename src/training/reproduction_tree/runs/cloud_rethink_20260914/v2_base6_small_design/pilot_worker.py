"""One F3 Base6 seven-view and Small50INT8 five-view qualification; no fit."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,sys,time,traceback
E=Path(__file__).resolve().parent;C=E.parent;O=C.parent/'error_resolution_20260913'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
sys.path[:0]=[str(E),str(O),str(O/'bundle')]
import torch,numpy as np
from torch.utils.data import DataLoader
from pose_view_data import PoseFiveViews
from temporal_inputs import TemporalHalves
from loader import load_state
from dual_host_runtime.io import read,atomic,retry
from dual_host_runtime.health import Guard,identity,owned_tree
import resilient_io;resilient_io.install()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(p,h):assert sha(p)==h,str(p);return Path(p)
def arrays(p):
 with np.load(p,allow_pickle=False)as a:return {k:a[k] for k in a.files}
def align(a,b):
 d={str(c):i for i,c in enumerate(a['clip_ids'])};assert len(d)==len(a['clip_ids']);ix=np.array([d[str(c)] for c in b['clip_ids']]);r={k:v[ix] for k,v in a.items()}
 for k in ('clip_ids','targets','users'):assert np.array_equal(r[k],b[k])
 return r
def pair(y,u,a,b):
 ok=a.argmax(1)==y;new=b.argmax(1)==y
 return dict(n=len(y),old_correct=int(ok.sum()),new_correct=int(new.sum()),fix=int((~ok&new).sum()),harm=int((ok&~new).sum()),prediction_changes=int((a.argmax(1)!=b.argmax(1)).sum()),maxabs=float(np.max(abs(a-b))),per_user={str(int(k)):dict(n=int((u==k).sum()),fix=int((~ok&new&(u==k)).sum()),harm=int((ok&~new&(u==k)).sum())) for k in np.unique(u)})
def module(path,name):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m

def replay(model,ds,views,guard,phase):
 loader=DataLoader(ds,batch_size=1,num_workers=2,pin_memory=True,prefetch_factor=1);values=[];ids=[];ys=[];us=[]
 with torch.inference_mode():
  for b in loader:
   guard.check(phase=phase,records=len(ids),total=len(ds));x=b['x'].flatten(0,1)
   duplicate=views==5 and torch.equal(x[0],x[1]) and torch.equal(x[0],x[2]);x=x[[0,3,4]] if duplicate else x
   z=[]
   for k in range(0,len(x),3):
    v=model(x[k:k+3].cuda(non_blocking=True));assert v.dtype==torch.float32 and torch.isfinite(v).all();z.append(v.cpu().numpy())
   z=np.concatenate(z)
   if duplicate:z=z[[0,0,0,1,2]]
   assert z.shape==(views,40);values.append(z[None]);ids+=b['clip_id'];ys+=b['y'].tolist();us+=b['user'].tolist()
 return dict(clip_ids=np.array(ids),targets=np.array(ys,np.int64),users=np.array(us,np.int64),values=np.concatenate(values).astype(np.float32))

def main(protocol,host):
 s=read(protocol);ph=sha(protocol);assert s['status']=='FROZEN_FOR_EXECUTION' and host=='local' and os.name=='nt' and s['optimizer_updates']==0
 out=Path(s['hosts']['local']['output_directory']);bd=read(out/'BUDGET.json');ct=read(out/'CONTROLLER.json');me=identity(os.getpid())
 assert bd['protocol_sha256']==ct['protocol_sha256']==ph
 assert any(v['pid']==me['pid'] and abs(v['create_time']-me['create_time'])<.01 for v in owned_tree(ct['worker']))
 guard=Guard(out,bd['deadline_unix'],out/'TELEMETRY.json');guard.check(force=True,phase='PAIR6S8_PREFLIGHT')
 for p,h in s['bindings'].items():check(p,h)
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 inp=read(s['pose_inputs']);rows=[r for r in inp['rows'] if r['user'] in s['held_users']]
 assert len(rows)==582 and set(s['train_users']).isdisjoint(s['held_users'])
 assert all(r['streams'][0]['directory'].startswith('Training/') for r in rows)
 values={};metrics={};files={};start=time.time()
 for label in ('Base','Small'):
  guard.check(force=True,phase='PAIR6S8_LOAD_'+label)
  state,origin=load_state(s['container'],label)
  assert origin['train_users']==s['train_users'] and origin['held_users']==s['held_users'] and origin['epochs']==16
  config=s['models'][label];m=module(config['model_code'],'pilot_'+label+'_model');cls=getattr(m,config['constructor'])
  model=cls(classes=40,fast=True,seed=s['seed']);model.load_state_dict(state,strict=True);del state;model=model.cuda();model.eval()
  five=replay(model,PoseFiveViews(rows,s['data_root'],inp['regions']),5,guard,label+'_FIVE_FP32');old=align(arrays(config['old_five_predictions']),five)
  z=five['values'].mean(1);metrics[label+'_five']=pair(five['targets'],five['users'],old['videomae_logits'],z)
  v={k:five[k] for k in ('clip_ids','targets','users')};v.update(five_view_logits=five['values'],five_logits=z,old_five_logits=old['videomae_logits'])
  if label=='Base':
   half=replay(model,TemporalHalves(rows,s['data_root'],inp['regions']),2,guard,'BASE6_HALVES_FP32')
   for k in ('clip_ids','targets','users'):assert np.array_equal(half[k],five[k])
   seven=.5*z+.25*half['values'][:,0]+.25*half['values'][:,1];old7=align(arrays(config['old_seven_predictions']),five)
   v.update(temporal_logits=half['values'],seven_logits=seven,old_seven_logits=old7['videomae_logits']);metrics['Base_seven']=pair(five['targets'],five['users'],old7['videomae_logits'],seven)
  tmp=out/(label+'.partial.npz');np.savez_compressed(tmp,**v);retry(lambda:os.replace(tmp,out/(label+'.npz')));files[label+'.npz']=sha(out/(label+'.npz'));values[label]=v
  del model,m,cls;torch.cuda.empty_cache()
 base=values['Base'];small=values['Small']
 for k in ('clip_ids','targets','users'):assert np.array_equal(base[k],small[k])
 # Fixed marginal old3 calibrator replay can be audited later. No new four-route fit here.
 result=dict(status='COMPLETE_BASE6_SMALL8_F3_QUALIFICATION',protocol_sha256=ph,metrics=metrics,container_sha256=s['bindings'][s['container']],optimizer_updates=0,calibrators_fitted=0,no_Test=True,no_CSV=True,model_bytes_with_calibrator_reserve=s['model_bytes_with_calibrator_reserve'],seconds=time.time()-start,limits='F3 adaptive precision qualification only; source recalibration required before multi-model promotion.')
 atomic(out/'RESULT.json',result);files['RESULT.json']=sha(out/'RESULT.json');atomic(out/'DONE.json',dict(status='COMPLETE_BASE6_SMALL8_PILOT',protocol_sha256=ph,artifacts=files,optimizer_updates=0,at_unix=time.time()));print(json.dumps(metrics))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--host',required=True);a=p.parse_args()
 try:main(a.protocol,a.host)
 except BaseException as e:
  s=read(a.protocol);atomic(Path(s['hosts'][a.host]['output_directory'])/'ERROR.json',dict(protocol_sha256=sha(a.protocol),error=str(e),traceback=traceback.format_exc(),at_unix=time.time()));raise
