"""Only mirror forwards of frozen original Base6/Small8 pairs; no fit or update."""
from pathlib import Path
import os,sys,argparse,json,hashlib,time,traceback
E=Path(__file__).resolve().parent;C=E.parent;O=C.parent/'error_resolution_20260913'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
sys.path[:0]=[str(E),str(O/'bundle')]
import torch,numpy as np
from torch.utils.data import DataLoader
from pose_view_data import PoseFiveViews
from temporal_inputs import TemporalHalves
from loader import load_state
from dual_model import PairVideo
from mirror_math import horizontal_mirror,aggregate,mix_original
from dual_host_runtime.io import read,atomic,retry
from dual_host_runtime.health import Guard,identity,owned_tree
if os.name=='nt':
 import resilient_io;resilient_io.install()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(p,h):assert sha(p)==h,str(p);return Path(p)
def arrays(p):
 with np.load(p,allow_pickle=False)as a:return {k:a[k]for k in a.files}
def save(p,v):
 t=p.with_suffix('.partial.npz');np.savez_compressed(t,**v);retry(lambda:os.replace(t,p))
class Views:
 def __init__(self,rows,root,regions):self.rows=rows;self.five=PoseFiveViews(rows,root,regions);self.half=TemporalHalves(rows,root,regions)
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  a=self.five[i];b=self.half[i];n=len(self.rows[i]['streams'][0]['files'])
  ix=[[min(n-1,int((k+v)*n/16))for k in range(16)]for v in [.5,.5,.5,.25,.75]]
  return dict(a,x=torch.cat([a['x'],b['x']]),frame_indices=torch.tensor(ix+b['frame_indices'].tolist()))
def replay(model,ds,kind,h,guard,job):
 loader=DataLoader(ds,batch_size=1,num_workers=h['workers'],pin_memory=True,prefetch_factor=1);out=[];ids=[];ys=[];us=[];ix=[]
 with torch.inference_mode():
  for b in loader:
   guard.check(phase='MIRROR_'+kind,job=job,records=len(ids),total=len(ds));x=b['x'][0];x=x if kind=='Base'else x[:5];x=horizontal_mirror(x)
   duplicate=torch.equal(x[0],x[1])and torch.equal(x[0],x[2]);selected=x[[0,3,4,5,6]if kind=='Base'else[0,3,4]]if duplicate else x;parts=[]
   for k in range(0,len(selected),h['eval_micro']):
    z=model(selected[k:k+h['eval_micro']].cuda(non_blocking=True));assert z.dtype==torch.float32 and torch.isfinite(z).all();parts.append(z.cpu().numpy())
   z=np.concatenate(parts)
   if duplicate:z=z[[0,0,0,1,2,3,4]if kind=='Base'else[0,0,0,1,2]]
   assert z.shape==((7,40)if kind=='Base'else(5,40));out.append(z);ids+=b['clip_id'];ys+=b['y'].tolist();us+=b['user'].tolist();ix.append(b['frame_indices'].numpy())
 return dict(clip_ids=np.array(ids),targets=np.array(ys,np.int64),users=np.array(us,np.int64),values=np.stack(out),frame_indices=np.concatenate(ix))
def main(protocol,host):
 s=read(protocol);ph=sha(protocol);h=s['hosts'][host];out=Path(h['output_directory']);assert s['optimizer_updates']==0 and len(s['jobs'])==8
 assert (os.name=='nt')==(host=='local')
 for n,v in s['code_sha256'].items():check(C/n,v)
 for n,v in h['input_bindings'].items():check(n,v)
 ct=read(out/'CONTROLLER.json');bd=read(out/'BUDGET.json');me=identity(os.getpid());assert ct['protocol_sha256']==bd['protocol_sha256']==ph
 assert any(v['pid']==me['pid']and abs(v['create_time']-me['create_time'])<.01 for v in owned_tree(ct['worker']))
 guard=Guard(out,bd['deadline_unix'],out/'TELEMETRY.json');torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 inp=read(h['pose_inputs']);assert len(inp['rows'])==2931;files={}
 for j in [j for j in s['jobs']if j['host']==host]:
  start=time.time();p=out/j['id'];p.mkdir(exist_ok=True);assert not(p/'DONE.json').exists();cp=Path(j['container']['path'])
  while not cp.is_file():guard.check(force=True,phase='WAIT_ORIGINAL_PAIR_TRANSFER',job=j['id']);time.sleep(2)
  check(cp,j['container']['sha256']);rows=[r for r in inp['rows']if r['user']in j['held_users']];assert len(rows)==j['records']
  assert set(j['train_users']).isdisjoint(set(j['held_users'])|set(j['outer_excluded_users']));assert all(r['streams'][0]['directory'].startswith('Training/')for r in rows)
  ds=Views(rows,h['data_root'],inp['regions']);values={};arts={}
  atomic(p/'BACKBONE_FROZEN.json',dict(protocol_sha256=ph,container=j['container'],parents=j['parents'],train_users=j['train_users'],held_users=j['held_users'],optimizer_updates=0,weight_values_changed=False,at_unix=time.time()))
  for kind in ['Base','Small']:
   state,origin=load_state(cp,kind);assert origin['train_users']==j['train_users']and origin['held_users']==j['held_users']and origin['epochs']==16
   model=PairVideo(kind);model.load_state_dict(state,strict=True);assert all(torch.equal(v,state[k])for k,v in model.state_dict().items());del state;model.cuda().eval()
   a=replay(model,ds,kind,h,guard,j['id']);v=j['original'][kind];old=arrays(check(v['path'],v['sha256']));lookup={str(x):i for i,x in enumerate(old['clip_ids'])};idx=np.array([lookup[str(x)]for x in a['clip_ids']]);assert len(lookup)==len(old['clip_ids'])
   for k in ['clip_ids','targets','users']:assert np.array_equal(old[k][idx],a[k])
   original=old['seven_logits'if kind=='Base'else'five_logits'][idx];mirrored=aggregate(a['values'],kind);mixed=mix_original(original,mirrored)
   values[kind]=dict(a,original=original,mirrored=mirrored,mixed=mixed)
   q={k:a[k]for k in ['clip_ids','targets','users','frame_indices']};q.update({('seven_view_logits'if kind=='Base'else'five_view_logits'):a['values']},original_logits=original,mirror_logits=mirrored,mixed_logits=mixed)
   save(p/(kind+'.npz'),q);arts[kind+'.npz']=sha(p/(kind+'.npz'));del model;torch.cuda.empty_cache()
  a=values['Base'];b=values['Small']
  for k in ['clip_ids','targets','users']:assert np.array_equal(a[k],b[k])
  q={k:a[k]for k in ['clip_ids','targets','users','frame_indices']};q.update(base6=a['mixed'],small8=b['mixed'],base6_mirror=a['mirrored'],small8_mirror=b['mirrored'],base6_original=a['original'],small8_original=b['original'])
  save(p/'predictions.npz',q);arts.update({'predictions.npz':sha(p/'predictions.npz'),'BACKBONE_FROZEN.json':sha(p/'BACKBONE_FROZEN.json')})
  atomic(p/'DONE.json',dict(status='COMPLETE_MIRROR_DUAL_JOB_READOUT',protocol_sha256=ph,job_id=j['id'],container_sha256=j['container']['sha256'],records=j['records'],optimizer_updates=0,seconds=time.time()-start,artifacts=arts,at_unix=time.time()))
  for n in list(arts)+['DONE.json']:files[j['id']+'/'+n]=sha(p/n)
 atomic(out/'DONE.json',dict(status='COMPLETE_MIRROR_DUAL_READOUT',protocol_sha256=ph,host=host,optimizer_updates=0,artifacts=files,at_unix=time.time()))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--host',required=True);a=p.parse_args()
 try:main(a.protocol,a.host)
 except BaseException as e:
  s=read(a.protocol);atomic(Path(s['hosts'][a.host]['output_directory'])/'ERROR.json',dict(error=str(e),traceback=traceback.format_exc(),protocol_sha256=sha(a.protocol),at_unix=time.time()));raise
