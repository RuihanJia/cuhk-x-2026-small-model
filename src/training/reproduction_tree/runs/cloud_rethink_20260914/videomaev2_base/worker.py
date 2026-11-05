"""One preregistered VideoMAE-Small source fit and fixed FP32 held evaluation."""
from pathlib import Path
import argparse,hashlib,json,math,os,sys,time,traceback,multiprocessing
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
E=Path(__file__).resolve().parent;C=E.parent;O=C.parent/'error_resolution_20260913';P=C/'aux_public_regression_20260914/dynamic_interaction/photometric_candidate'
sys.path[:0]=[str(E),str(P),str(O),str(O/'bundle')]
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from model import VideoMAEBase
from photometric_data_numpy import PhotometricClips,PhotometricFiveViews
import resilient_io;resilient_io.install()
from dual_host_runtime.io import atomic,read,save_torch,retry
from dual_host_runtime.health import Guard,identity,owned_tree

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def checked(p,h):assert sha(p)==h,str(p);return Path(p)
def state(model,half=False):return {k:v.detach().cpu().to(torch.float16 if half else torch.float32).clone() for k,v in model.state_dict().items()}
def digest(values):
 h=hashlib.sha256()
 for k,v in values.items():h.update(k.encode());h.update(v.contiguous().numpy().tobytes())
 return h.hexdigest()

class EpochClips(PhotometricClips):
 def __init__(self,*args,epoch_value,**kwargs):super().__init__(*args,**kwargs);self.epoch_value=epoch_value
 def __getitem__(self,i):self.epoch=self.epoch_value.value;return super().__getitem__(i)

def groups(model,s):
 result={}
 for name,p in model.named_parameters():
  if name.startswith('video.head.'):scale=s['head_lr']/s['backbone_lr']
  elif name.startswith('video.blocks.'):scale=s['layer_decay']**(11-int(name.split('.')[2]))
  elif name.startswith('video.fc_norm.'):scale=1.
  else:scale=s['layer_decay']**12
  decay=0. if p.ndim==1 or name.endswith('.bias') else s['weight_decay']
  key=(scale,decay)
  if key not in result:result[key]=dict(params=[],lr=s['backbone_lr']*scale,lr_scale=scale,weight_decay=decay)
  result[key]['params'].append(p)
 return list(result.values())

def fit(job,s,ph,out,inp,guard):
 train=[r for r in inp['rows'] if r['user'] in job['train_users']];held=[r for r in inp['rows'] if r['user'] in job['held_users']]
 assert not set(job['train_users'])&set(job['held_users']) and len(train)+len(held)==len(inp['rows'])
 assert set(r['user'] for r in train)==set(job['train_users'])
 atomic(out/'SPLIT.json',dict(protocol_sha256=ph,fold=job['fold'],train_users=job['train_users'],held_users=job['held_users'],train_ids=[r['clip_id'] for r in train],held_ids=[r['clip_id'] for r in held],held_labels_used_in_fit=False))
 model=VideoMAEBase(classes=40,fast=True,seed=s['seed']).cuda();initial=state(model);initial_digest=digest(initial)
 assert sum(p.numel() for p in model.parameters())==s['trainable_parameters']
 atomic(out/'INITIAL.json',dict(protocol_sha256=ph,pretrained_sha256=s['pretrained_sha256'],initial_digest=initial_digest,all_parameters_trainable=True,at_unix=time.time()))
 params=list(model.parameters());opt=torch.optim.AdamW(groups(model,s),lr=s['backbone_lr']);scaler=torch.amp.GradScaler('cuda',init_scale=128)
 micro=s['micro_clips'];accum=s['effective_batch']//micro;per_epoch=math.ceil(len(train)/s['effective_batch']);total=s['epochs']*per_epoch;warm=s['warmup_epochs']*per_epoch
 updates=0;hist=[];first=1
 if(out/'resume.pt').exists():
  saved=torch.load(out/'resume.pt',map_location='cpu',weights_only=True);assert saved['protocol_sha256']==ph and saved['initial_digest']==initial_digest
  model.load_state_dict(saved['model'],strict=True);opt.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler']);updates=saved['updates'];hist=saved['history'];first=saved['epoch']+1
  torch.set_rng_state(saved['rng']);torch.cuda.set_rng_state_all(saved['cuda_rng']);del saved
 shared=multiprocessing.get_context('spawn' if os.name=='nt' else 'fork').Value('i',first)
 ds=EpochClips(train,s['data_root'],s['seed'],inp['regions'],training=True,epoch_value=shared,arm='RAW_CONTROL')
 generator=torch.Generator();loader=DataLoader(ds,batch_size=micro,shuffle=True,generator=generator,num_workers=s['workers'],pin_memory=True,prefetch_factor=1,persistent_workers=True)
 quality=inp['conflict_weights']
 for epoch in range(first,s['epochs']+1):
  guard.check(force=True,phase='VIDEOMAE_EPOCH_START',epoch=epoch);shared.value=epoch;generator.manual_seed(s['seed']+epoch);model.train();opt.zero_grad(set_to_none=True)
  start=time.time();previous=time.perf_counter();wait=0.;seen=0;sum_loss=0.;sampling=hashlib.sha256()
  for bi,b in enumerate(loader):
   wait+=time.perf_counter()-previous;n=len(b['y']);x=b['x'].cuda(non_blocking=True);y=b['y'].cuda(non_blocking=True);q=torch.tensor([quality[k] for k in b['clip_id']],device='cuda')
   assert torch.isfinite(q).all() and (q>0).all()
   for i,cid in enumerate(b['clip_id']):sampling.update(json.dumps([cid,b['frame_indices'][i].tolist(),bool(b['flip'][i])],separators=(',',':')).encode())
   if bi%accum==0:
    factor=(updates+1)/warm if updates<warm else s['min_lr_ratio']+(1-s['min_lr_ratio'])*.5*(1+math.cos(math.pi*(updates-warm)/max(1,total-warm)))
    for g in opt.param_groups:g['lr']=s['backbone_lr']*g['lr_scale']*factor
   with torch.autocast('cuda',dtype=torch.float16):z=model(x);loss=(q*F.cross_entropy(z,y,label_smoothing=s['label_smoothing'],reduction='none')).mean()
   assert torch.isfinite(loss),'NONFINITE_LOSS';window=min(s['effective_batch'],len(train)-(bi//accum)*s['effective_batch'])
   scaler.scale(loss*n/window).backward();seen+=n;sum_loss+=float(loss.detach())*n
   if bi%accum==accum-1 or bi+1==len(loader):
    scale=scaler.get_scale();scaler.unscale_(opt);norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);scaler.step(opt);scaler.update();assert scaler.get_scale()>=scale,'AMP_SKIP';opt.zero_grad(set_to_none=True);updates+=1
    if updates==1:atomic(out/'FIRST_UPDATE.json',dict(protocol_sha256=ph,epoch=epoch,loss=float(loss.detach()),gradient_norm=float(norm),AMP_scale=scaler.get_scale(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),updates=updates,at_unix=time.time()))
   guard.check(phase='VIDEOMAE_SOURCE_TRAIN',epoch=epoch,epochs=s['epochs'],records=seen,total=len(train),updates=updates,job=job['id']);previous=time.perf_counter()
  assert seen==len(train) and updates==epoch*per_epoch
  hist.append(dict(epoch=epoch,records=seen,updates=updates,loss=sum_loss/seen,seconds=time.time()-start,input_wait_seconds=wait,sampling_sha256=sampling.hexdigest()))
  atomic(out/'HISTORY.json',hist)
  save_torch(out/'resume.pt',dict(protocol_sha256=ph,initial_digest=initial_digest,model=state(model),optimizer=opt.state_dict(),scaler=scaler.state_dict(),history=hist,updates=updates,epoch=epoch,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
  print(json.dumps(dict(job=job['id'],**hist[-1])),flush=True)
 assert updates==total and len(hist)==s['epochs'];trained=state(model);changed=[k for k in trained if not torch.equal(trained[k],initial[k])];assert 'video.patch_embed.proj.weight' in changed and 'video.head.weight' in changed
 cp=out/'model_fp16.pt';save_torch(cp,dict(model=state(model,True),protocol_sha256=ph,pretrained_sha256=s['pretrained_sha256'],train_users=job['train_users'],held_users=job['held_users'],epochs=s['epochs'],seed=s['seed']))
 assert cp.stat().st_size<=s['checkpoint_cap_bytes'];saved=torch.load(cp,map_location='cpu',weights_only=True);model.load_state_dict(saved['model'],strict=True)
 assert all(v.dtype==torch.float16 and torch.equal(v.float(),model.state_dict()[k].cpu()) for k,v in saved['model'].items())
 atomic(out/'FROZEN.json',dict(protocol_sha256=ph,checkpoint_sha256=sha(cp),checkpoint_bytes=cp.stat().st_size,epochs=s['epochs'],updates=updates,initial_digest=initial_digest,pretrained_sha256=s['pretrained_sha256'],train_users=job['train_users'],held_users=job['held_users'],changed_parameters=changed,FP16_reload_exact=True,held_evaluation_started=False,at_unix=time.time()))
 names=['model_fp16.pt','INITIAL.json','SPLIT.json','HISTORY.json','FIRST_UPDATE.json','FROZEN.json'];atomic(out/'TRAIN_DONE.json',dict(status='COMPLETE_TRAIN_ONLY',protocol_sha256=ph,held_evaluation_started=False,artifacts={n:sha(out/n) for n in names},at_unix=time.time()))
 del model,opt,scaler,params,initial,trained,saved,loader,ds;torch.cuda.empty_cache()

def evaluate(job,s,ph,out,inp,guard):
 fr=read(out/'FROZEN.json');assert fr['protocol_sha256']==ph and fr['epochs']==s['epochs'];cp=torch.load(checked(out/'model_fp16.pt',fr['checkpoint_sha256']),map_location='cpu',weights_only=True)
 model=VideoMAEBase(classes=40,fast=True,seed=s['seed']).cuda();model.load_state_dict(cp['model'],strict=True);model.eval()
 held=[r for r in inp['rows'] if r['user'] in job['held_users']];ds=PhotometricFiveViews(held,s['data_root'],inp['regions'],arm='RAW_CONTROL')
 loader=DataLoader(ds,batch_size=1,num_workers=s['workers'],pin_memory=True,prefetch_factor=1)
 ids=[];ys=[];us=[];values=[];unique_views=0
 with torch.inference_mode():
  for b in loader:
   guard.check(phase='VIDEOMAE_HELD_FP32',records=len(ids),total=len(held),job=job['id']);x=b['x'].flatten(0,1)
   duplicate=torch.equal(x[0],x[1]) and torch.equal(x[0],x[2]);selected=x[[0,3,4]] if duplicate else x;z=[]
   for k in range(0,len(selected),s['eval_micro']):
    v=model(selected[k:k+s['eval_micro']].cuda(non_blocking=True));assert v.dtype==torch.float32 and torch.isfinite(v).all();z.append(v.cpu().numpy())
   z=np.concatenate(z);unique_views+=len(z)
   if duplicate:z=z[[0,0,0,1,2]]
   values.append(z[None]);ids+=b['clip_id'];ys+=b['y'].tolist();us+=b['user'].tolist()
 values=np.concatenate(values).astype(np.float32);assert values.shape==(len(held),5,40)
 dest=out/'predictions.npz';temp=out/'predictions.partial.npz';np.savez_compressed(temp,clip_ids=np.array(ids),targets=np.array(ys,dtype=np.int64),users=np.array(us,dtype=np.int64),videomae_logits=values.mean(1),five_view_logits=values);retry(lambda:os.replace(temp,dest))
 atomic(out/'HELD_DONE.json',dict(status='COMPLETE_FP32_FIXED_FIVE_VIEW',protocol_sha256=ph,checkpoint_sha256=fr['checkpoint_sha256'],records=len(ids),unique_forwarded_views=unique_views,duplicate_views_reused_only_after_tensor_equality=True,no_fit=True,no_Test=True,at_unix=time.time()))
 del model,cp,loader;torch.cuda.empty_cache()

def main(protocol,host):
 pp=Path(protocol).resolve();s=read(pp);ph=sha(pp);assert pp.parent==E and s['status']=='FROZEN_FOR_EXECUTION' and (os.name=='nt')==(host=='local')
 h=s['hosts'][host];out=Path(h['output_directory']);assert out.resolve().is_relative_to(E)
 for rel,hsh in s['code_sha256'].items():checked(C/rel,hsh)
 for name,hsh in h['input_bindings'].items():checked(name,hsh)
 checked(E/'vit_b_k710_dl_from_giant.pth',s['pretrained_sha256']);checked(E/'ADMISSION.json',s['admission_sha256'])
 bd=read(out/'BUDGET.json');ct=read(out/'CONTROLLER.json');assert bd['protocol_sha256']==ct['protocol_sha256']==ph
 me=identity(os.getpid());assert any(v['pid']==me['pid'] and abs(v['create_time']-me['create_time'])<.01 for v in owned_tree(ct['worker']))
 guard=Guard(out,bd['deadline_unix'],out/'TELEMETRY.json');guard.check(force=True,phase='V2_BASE_PREFLIGHT')
 torch.set_num_threads(2 if host=='local' else 1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert ('4070' if host=='local' else '5090') in torch.cuda.get_device_name(0)
 inp=read(checked(h['pose_inputs'],s['pose_inputs_sha256']));outer=read(checked(h['original_protocol'],s['original_protocol_sha256']));inner=read(checked(h['inner_protocol'],s['inner_protocol_sha256']));assert len(inp['rows'])==2931
 runtime=dict(s);runtime.update(h);jobs=[j for j in s['jobs'] if j['host']==host];metadata=[]
 for job in jobs:
  source=job['kind']=='source_inner';parent=next(j for j in (inner if source else outer)['jobs'] if j['id']==job['split_parent'])
  for key in ['train_users','held_users','fold']+(['inner','outer_excluded_users'] if source else []):assert parent[key]==job[key]
  subset=dict(inp);excluded=job.get('outer_excluded_users',[]);subset['rows']=[r for r in inp['rows'] if r['user'] not in excluded]
  assert not set(job['train_users'])&(set(job['held_users'])|set(excluded)) and not set(job['held_users'])&set(excluded)
  assert {r['user'] for r in subset['rows']}==set(job['train_users'])|set(job['held_users'])
  path=out/job['id'];path.mkdir(exist_ok=True);bp=path/'JOB_BUDGET.json'
  if not bp.exists():atomic(bp,dict(protocol_sha256=ph,first_started_unix=time.time(),deadline_unix=min(bd['deadline_unix'],time.time()+h['job_wall_cap_seconds'])))
  jb=read(bp);assert jb['protocol_sha256']==ph;fg=Guard(path,jb['deadline_unix'],out/'TELEMETRY.json')
  fit(job,runtime,ph,path,subset,fg)
  metadata.append(dict(job=job['id'],kind=job['kind'],checkpoint_sha256=sha(path/'model_fp16.pt')))
 atomic(out/'ALL_FITS_FROZEN.json',dict(protocol_sha256=ph,jobs=metadata,held_evaluation_started=False,at_unix=time.time()))
 files={'ALL_FITS_FROZEN.json':sha(out/'ALL_FITS_FROZEN.json')}
 for job in jobs:
  subset=dict(inp);subset['rows']=[r for r in inp['rows'] if r['user'] not in job.get('outer_excluded_users',[])];path=out/job['id'];evaluate(job,runtime,ph,path,subset,guard)
  for n in ['TRAIN_DONE.json','model_fp16.pt','INITIAL.json','SPLIT.json','HISTORY.json','FIRST_UPDATE.json','FROZEN.json','predictions.npz','HELD_DONE.json']:files[job['id']+'/'+n]=sha(path/n)
 guard.check(force=True,phase='V2_BASE_COMPLETE');atomic(out/'DONE.json',dict(status='COMPLETE_V2_BASE_MATCHED_STAGE',protocol_sha256=ph,host=host,artifacts=files,no_Test=True,at_unix=time.time()))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--host',choices=['local','cloud'],required=True);a=p.parse_args()
 try:main(a.protocol,a.host)
 except BaseException as exc:
  s=read(a.protocol);atomic(Path(s['hosts'][a.host]['output_directory'])/'ERROR.json',dict(protocol_sha256=sha(a.protocol),error=str(exc),traceback=traceback.format_exc(),at_unix=time.time()));raise
