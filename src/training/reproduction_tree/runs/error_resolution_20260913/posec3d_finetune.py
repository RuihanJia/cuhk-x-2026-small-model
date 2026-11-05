"""Small pretrained temporal-pose network; source-user-only adaptation."""
from pathlib import Path
import sys,json,time,hashlib,traceback,math,argparse
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'bundle'))
from dual_host_runtime.io import atomic,save_torch
from dual_host_runtime.health import Guard
from posec3d_model import PoseC3D
from posec3d_data import sample_indices,compact_and_resize,heatmaps,SWAP
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def training_coords(kps,size,seed):
 rng=np.random.RandomState(seed);z=compact_and_resize(kps[sample_indices(len(kps),rng)],size);valid=z[...,2]>0
 cropw=croph=64;left=top=0
 for _ in range(10):
  area=rng.uniform(.56,1.)*64*64;ratio=np.exp(rng.uniform(np.log(.75),np.log(4/3)));w=int(round(np.sqrt(area*ratio)));h=int(round(np.sqrt(area/ratio)))
  if 0<w<=64 and 0<h<=64:cropw,croph=w,h;left=int(rng.randint(65-w));top=int(rng.randint(65-h));break
 z[...,0][valid]=(z[...,0][valid]-left)*56/cropw;z[...,1][valid]=(z[...,1][valid]-top)*56/croph
 if rng.random()<.5:z=z[:,SWAP].copy();z[...,0][z[...,2]>0]=56-z[...,0][z[...,2]>0]
 return z

def main(host):
 p=R/'POSEC3D_FINETUNE_PROTOCOL.json';s=json.loads(p.read_text());ph=sha(p);outroot=R/'posec3d_finetune_output'/host;outroot.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2 if host=='local' else 1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 for n,h in s['code_sha256'].items():assert sha(R/n)==h,n
 source=R/'assets/posec3d/state_fp16.pt';assert sha(source)==s['weights_sha256'];root=R/'pose_dense_output';assert sha(root/'INDEX.json')==s['dense_index_sha256'];records=json.loads((root/'INDEX.json').read_text())['records'];meta={r['clip_id']:r for r in json.loads((R/'IR_ADAPTER_INPUTS.json').read_text())['rows']};samples=[]
 for rec in records:
  path=root/rec['file'];assert sha(path)==rec['sha256'];kps=np.load(path,allow_pickle=False)['keypoints'];av=int(((kps[...,2]>0).sum(1)>=6).sum())>=2;r=meta[rec['clip_id']];samples.append(dict(clip_id=rec['clip_id'],user=r['user'],label=r['label'],kps=kps,size=rec['native_size'],available=av))
 setting=s['hosts'][host];micro=setting['micro_clips'];accum=16//micro
 for job in s['jobs']:
  if job['host']!=host:continue
  out=outroot/job['id'];out.mkdir(exist_ok=True)
  if(out/'DONE.json').exists():continue
  start=time.time();budget=out/'JOB_BUDGET.json'
  if not budget.exists():atomic(budget,dict(protocol_sha256=ph,first_started_unix=start,deadline_unix=min(s['deadline_unix'],start+s['job_wall_cap_seconds'])))
  bd=json.loads(budget.read_text());assert bd['protocol_sha256']==ph;guard=Guard(out,bd['deadline_unix'],outroot/'TELEMETRY.json');tr=[r for r in samples if r['user'] in job['train_users'] and r['available']];va=[r for r in samples if r['user'] in job['held_users']];assert not set(job['train_users'])&set(job['held_users']) and not(set(job['train_users'])|set(job['held_users']))&set(job.get('outer_excluded_users',[]))
  atomic(out/'SPLIT.json',dict(train_users=job['train_users'],held_users=job['held_users'],outer_excluded_users=job.get('outer_excluded_users',[]),train_ids=[r['clip_id'] for r in tr],held_ids=[r['clip_id'] for r in va],unavailable_excluded_from_fit=True));counts={u:sum(r['user']==u for r in tr) for u in job['train_users']};torch.manual_seed(s['seed']);m=PoseC3D(40);state=torch.load(source,map_location='cpu',weights_only=True)['state_dict'];m.backbone.load_state_dict({k.removeprefix('backbone.'):v for k,v in state.items() if k.startswith('backbone.')},strict=True)
  rng=np.random.RandomState(s['seed'])
  with torch.no_grad():m.cls_head.fc_cls.weight.copy_(torch.from_numpy(rng.normal(0,.01,(40,512)).astype(np.float32)));m.cls_head.fc_cls.bias.zero_()
  m.cuda();opt=torch.optim.AdamW([dict(params=m.backbone.parameters(),lr=1e-4,base_lr=1e-4),dict(params=m.cls_head.parameters(),lr=1e-3,base_lr=1e-3)],weight_decay=.01);scaler=torch.amp.GradScaler('cuda',init_scale=128);history=[];updates=0;first=1;total=s['epochs']*math.ceil(len(tr)/16)
  if(out/'resume.pt').exists():
   q=torch.load(out/'resume.pt',map_location='cpu',weights_only=True);assert q['protocol_sha256']==ph;m.load_state_dict(q['model']);opt.load_state_dict(q['optimizer']);scaler.load_state_dict(q['scaler']);history=q['history'];updates=q['updates'];first=q['epoch']+1;torch.set_rng_state(q['rng']);torch.cuda.set_rng_state_all(q['cuda_rng'])
  for epoch in range(first,s['epochs']+1):
   m.train()
   for module in m.modules():
    if isinstance(module,nn.BatchNorm3d):module.eval()
   order=np.random.RandomState(s['seed']+epoch).permutation(len(tr));seen=0;loss_sum=0;t0=time.time();opt.zero_grad(set_to_none=True)
   for bi,begin in enumerate(range(0,len(tr),micro)):
    rr=[tr[i] for i in order[begin:begin+micro]];k=[]
    for row in rr:
     seed=int.from_bytes(hashlib.sha256(f"{s['seed']}:{epoch}:{row['clip_id']}".encode()).digest()[:4],'big');k.append(training_coords(row['kps'],row['size'],seed))
    x=heatmaps(torch.from_numpy(np.stack(k)).cuda(),size=56);y=torch.tensor([r['label'] for r in rr],device='cuda');w=torch.tensor([len(tr)/(len(counts)*counts[r['user']]) for r in rr],device='cuda')
    if bi%accum==0:
     for g in opt.param_groups:g['lr']=g['base_lr']*(.05+.95*.5*(1+math.cos(math.pi*updates/total)))
    with torch.autocast('cuda',dtype=torch.float16):z=m(x);loss=(F.cross_entropy(z,y,label_smoothing=.1,reduction='none')*w).mean()
    assert torch.isfinite(loss);window=min(16,len(tr)-(bi//accum)*16);scaler.scale(loss*len(rr)/window).backward();seen+=len(rr);loss_sum+=float(loss.detach())*len(rr)
    if bi%accum==accum-1 or begin+micro>=len(tr):
     old=scaler.get_scale();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);scaler.step(opt);scaler.update();assert scaler.get_scale()>=old,'AMP_SKIP';opt.zero_grad(set_to_none=True);updates+=1
    guard.check(phase='TEMPORAL_POSE_FINETUNE',job=job['id'],epoch=epoch,epochs=s['epochs'],records=seen,total=len(tr),updates=updates)
   history.append(dict(epoch=epoch,loss=loss_sum/seen,records=seen,seconds=time.time()-t0,updates=updates));atomic(out/'HISTORY.json',history);save_torch(out/'resume.pt',dict(protocol_sha256=ph,model=m.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),history=history,epoch=epoch,updates=updates,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()));print(json.dumps(dict(job=job['id'],**history[-1])),flush=True)
  save_torch(out/'model_fp16.pt',dict(state_dict={k:(v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu()) for k,v in m.state_dict().items()},train_users=job['train_users'],protocol_sha256=ph,source_sha256=s['weights_sha256']));q=torch.load(out/'model_fp16.pt',map_location='cpu',weights_only=True);m.load_state_dict(q['state_dict']);m.eval().requires_grad_(False);values=[]
  with torch.inference_mode():
   for begin in range(0,len(va),setting['eval_batch']):
    rr=va[begin:begin+setting['eval_batch']];k=np.stack([compact_and_resize(r['kps'][sample_indices(len(r['kps']))],r['size']) for r in rr]);hm=heatmaps(torch.from_numpy(k).cuda());z=.5*(m(hm)+m(hm[:,SWAP].flip(-1)));values.append(z.float().cpu().numpy());guard.check(phase='SERIALIZED_TEMPORAL_POSE_HELD',job=job['id'],records=begin+len(rr),total=len(va))
  logits=np.concatenate(values);targets=np.array([r['label'] for r in va]);available=np.array([r['available'] for r in va]);np.savez_compressed(out/'predictions.npz',clip_ids=np.array([r['clip_id'] for r in va]),targets=targets,users=np.array([r['user'] for r in va]),logits=logits,available=available);d=dict(status='COMPLETE',job=job['id'],protocol_sha256=ph,checkpoint_sha256=sha(out/'model_fp16.pt'),checkpoint_bytes=(out/'model_fp16.pt').stat().st_size,predictions_sha256=sha(out/'predictions.npz'),epochs=s['epochs'],n=len(va),available=int(available.sum()),seconds=time.time()-start,source_sha256=s['weights_sha256'],no_Test=True)
  if job['kind']=='outer':d['correct_on_available']=int(((logits.argmax(1)==targets)&available).sum())
  atomic(out/'DONE.json',d);print(json.dumps(d),flush=True);del m,opt,scaler;torch.cuda.empty_cache()
 atomic(outroot/'QUEUE_DONE.json',dict(status='COMPLETE',at_unix=time.time(),protocol_sha256=ph))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--host',required=True,choices=['local','cloud']);args=a.parse_args()
 try:main(args.host)
 except BaseException as e:
  p=R/'posec3d_finetune_output'/args.host;p.mkdir(parents=True,exist_ok=True);atomic(p/'ERROR.json',dict(error=str(e),traceback=traceback.format_exc(),at_unix=time.time()));raise
