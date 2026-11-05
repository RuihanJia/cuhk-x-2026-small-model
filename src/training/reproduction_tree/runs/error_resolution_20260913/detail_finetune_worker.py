"""Finite last-four-block adaptation of the new public visual backbone."""
from pathlib import Path
import os,sys,json,time,hashlib,random,math,traceback
os.environ['XFORMERS_DISABLED']='1'
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from PIL import Image,ImageOps
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'bundle'));sys.path.insert(0,str(R/'assets/dino/source'))
from dual_host_runtime.io import atomic,save_torch
from dual_host_runtime.health import Guard
from detail_model import DetailHead
from detail_worker import Frames
from dinov2.hub.backbones import dinov2_vits14
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def unpack(cp):return {k:(v['q'].float()*v['scale'].float() if 'q' in v else v['value'].float()) for k,v in cp['state_dict'].items()}
def pack(model):
 state={}
 for k,v in model.state_dict().items():
  v=v.detach().float().cpu()
  if k.endswith('weight') and v.ndim>=2:
   shape=[v.shape[0]]+[1]*(v.ndim-1);scale=v.abs().flatten(1).amax(1).clamp_min(1e-8).div(127).half().reshape(shape);q=(v/scale.float()).round().clamp(-127,127).to(torch.int8);state[k]=dict(q=q,scale=scale)
  else:state[k]=dict(value=v.half())
 return state

class TrainFrames(Dataset):
 def __init__(self,rows,data,seed,epoch):self.rows,self.data,self.seed,self.epoch=rows,Path(data),seed,epoch
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  row=self.rows[i];rng=random.Random(int.from_bytes(hashlib.sha256(f'{self.seed}:{self.epoch}:{row["clip_id"]}'.encode()).digest()[:8],'big'));phase=[rng.random() for _ in range(16)];flip=rng.random()<.5;gamma=rng.uniform(.75,1.35);brightness=rng.uniform(.85,1.15);views=[];available=[]
  for j,s in enumerate(row['views']):
   files=s['files'];available.append(bool(files));decoded={};frames=[]
   if not files:views.append(np.zeros((16,3,224,224),np.uint8));continue
   directory=self.data/s['directory'];assert 'Training' in directory.parts and 'Test' not in directory.parts
   for k in range(16):
    index=min(len(files)-1,int((k+phase[k])*len(files)/16))
    if index not in decoded:
     with Image.open(directory/files[index]) as im:
      im=im.convert('RGB')
      if s['box'] is not None:
       x,y,w,h=s['box'];im=im.crop((x,y,x+w,y+h))
      im=ImageOps.pad(im,(224,224),method=Image.Resampling.BICUBIC,color=(0,0,0))
      a=np.asarray(im,dtype=np.uint8)
      if j<2:a=np.clip((a.astype(np.float32)/255.)**gamma*brightness*255,0,255).astype(np.uint8)
      if flip:a=a[:,::-1]
      decoded[index]=a.transpose(2,0,1).copy()
    frames.append(decoded[index])
   views.append(np.stack(frames))
  return np.stack(views),row['label'],np.array(available,bool),max(.25,1-.75*row['conflict_fraction'])

def features(model,x,available,micro,mean,std):
 # Available missing Thermal is always represented by zeros and masked at the head.
 b=x.shape[0];valid=available.unsqueeze(-1).expand(-1,-1,16).flatten();flat=x.flatten(0,2);allfeat=[]
 for st in range(0,len(flat),micro):
  xx=flat[st:st+micro].cuda(non_blocking=True).float().div_(255);xx=(xx-mean)/std
  with torch.autocast('cuda',dtype=torch.float16):
   f=model.forward_features(xx);patch=f['x_norm_patchtokens'];pool=F.adaptive_avg_pool2d(patch.transpose(1,2).reshape(-1,384,16,16),(2,2)).flatten(2).transpose(1,2);z=torch.cat([f['x_norm_clstoken'].unsqueeze(1),pool],1)
  allfeat.append(z)
 out=torch.cat(allfeat).reshape(b,4,16,5,384)
 return out*available[:,:,None,None,None]

def main():
 spec=json.loads((R/'DETAIL_FINETUNE_PROTOCOL.json').read_text(encoding='utf-8'));checksum=sha(R/'DETAIL_FINETUNE_PROTOCOL.json')
 for n,h in spec['code_sha256'].items():assert sha(R/n)==h,n
 for n,h in spec['asset_sha256'].items():assert sha(R/n)==h,n
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 rows=json.loads((R/'DETAIL_INPUTS.json').read_text(encoding='utf-8'));outroot=R/'detail_finetune_output';outroot.mkdir(exist_ok=True)
 mean=torch.tensor([.485,.456,.406],device='cuda').reshape(1,3,1,1);std=torch.tensor([.229,.224,.225],device='cuda').reshape(1,3,1,1)
 for job in spec['jobs']:
  out=outroot/job['id'];out.mkdir(exist_ok=True)
  if (out/'DONE.json').exists():continue
  start=time.time();budget=out/'JOB_BUDGET.json'
  if not budget.exists():atomic(budget,dict(protocol_sha256=checksum,deadline_unix=min(spec['deadline_unix'],start+spec['job_wall_cap_seconds']),first_started_unix=start))
  bd=json.loads(budget.read_text());assert bd['protocol_sha256']==checksum;guard=Guard(out,bd['deadline_unix'],outroot/'TELEMETRY.json')
  torch.manual_seed(spec['seed']);np.random.seed(spec['seed']);random.seed(spec['seed'])
  model=dinov2_vits14(pretrained=False);model.load_state_dict(unpack(torch.load(R/'assets/dino/dino_int8.pt',map_location='cpu',weights_only=True)));model.requires_grad_(False)
  for block in list(model.blocks)[-4:]:block.requires_grad_(True)
  model.norm.requires_grad_(True);model.cuda().train()
  cp=torch.load(R/job['head_source'],map_location='cpu',weights_only=True);assert cp['train_users']==job['train_users'];head=DetailHead(True);head.load_state_dict(cp['state_dict']);head.cuda().train();del cp
  tr=[r for r in rows if r['user'] in job['train_users']];va=[r for r in rows if r['user'] in job['held_users']];assert len(tr)+len(va)==len(rows) and not set(job['train_users'])&set(job['held_users'])
  atomic(out/'SPLIT.json',dict(train_users=job['train_users'],held_users=job['held_users'],train_ids=[r['clip_id'] for r in tr],held_ids=[r['clip_id'] for r in va],outer_labels_used_for_fit=False))
  params=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW([dict(params=params,lr=1e-5),dict(params=head.parameters(),lr=1e-4)],weight_decay=.01);scaler=torch.amp.GradScaler('cuda',init_scale=128);hist=[];updates=0;first=1;total=spec['epochs']*math.ceil(len(tr)/4)
  if (out/'resume.pt').exists():
   saved=torch.load(out/'resume.pt',map_location='cpu',weights_only=True);assert saved['protocol_sha256']==checksum;model.load_state_dict(saved['backbone']);head.load_state_dict(saved['head']);opt.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler']);hist=saved['history'];updates=saved['updates'];first=saved['epoch']+1;torch.set_rng_state(saved['rng']);torch.cuda.set_rng_state_all(saved['cuda_rng']);del saved
  for epoch in range(first,spec['epochs']+1):
   ds=TrainFrames(tr,spec['data_root'],spec['seed'],epoch);gen=torch.Generator().manual_seed(spec['seed']+epoch);loader=DataLoader(ds,batch_size=1,shuffle=True,generator=gen,num_workers=4,pin_memory=True,prefetch_factor=2)
   seen=0;loss_sum=0;began=time.time();opt.zero_grad(set_to_none=True)
   for bi,(x,y,av,q) in enumerate(loader):
    y=y.cuda(non_blocking=True);av=av.cuda(non_blocking=True);q=q.cuda(non_blocking=True).float()
    if bi%4==0:
     factor=.5*(1+math.cos(math.pi*updates/max(1,total)))
     for group,lr in zip(opt.param_groups,[1e-5,1e-4]):group['lr']=1e-6+(lr-1e-6)*factor
    feats=features(model,x,av,spec['image_micro_batch'],mean,std);z,v,g=head(feats.float(),av,drop_modalities=True);loss=F.cross_entropy(z,y,label_smoothing=.1,reduction='none');aux=F.cross_entropy(v.flatten(0,1),y.repeat_interleave(4),label_smoothing=.1,reduction='none').reshape(-1,4);aw=av.float();aw[:,:3]*=q[:,None];aux=(aux*aw).sum(1)/aw.sum(1).clamp_min(1);objective=(loss+.25*aux).mean();assert torch.isfinite(objective);window=min(4,len(tr)-(bi//4)*4);scaler.scale(objective/window).backward();loss_sum+=float(objective.detach());seen+=1
    if bi%4==3 or bi+1==len(loader):
     old=scaler.get_scale();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(params+list(head.parameters()),1.,error_if_nonfinite=True);scaler.step(opt);scaler.update();assert scaler.get_scale()>=old,'AMP_SKIP';opt.zero_grad(set_to_none=True);updates+=1
    guard.check(phase='DINO_LAST4_ADAPT',job=job['id'],epoch=epoch,epochs=spec['epochs'],records=seen,total=len(tr),updates=updates)
   hist.append(dict(epoch=epoch,loss=loss_sum/seen,records=seen,seconds=time.time()-began,updates=updates));atomic(out/'HISTORY.json',hist);save_torch(out/'resume.pt',dict(protocol_sha256=checksum,backbone=model.state_dict(),head=head.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),history=hist,epoch=epoch,updates=updates,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()));print(json.dumps(dict(job=job['id'],**hist[-1])),flush=True)
  bp=out/'dino_int8.pt';hp=out/'head_fp16.pt';save_torch(bp,dict(state_dict=pack(model),protocol_sha256=checksum,train_users=job['train_users'],source_sha256=spec['dino_source_sha256']));save_torch(hp,dict(state_dict={k:v.detach().cpu().half() for k,v in head.state_dict().items()},ordered=True,protocol_sha256=checksum,train_users=job['train_users']))
  assert 76193153+bp.stat().st_size+hp.stat().st_size+10000<100000000
  model.load_state_dict(unpack(torch.load(bp,map_location='cpu',weights_only=True)));head.load_state_dict(torch.load(hp,map_location='cpu',weights_only=True)['state_dict']);model.eval().requires_grad_(False);head.eval().requires_grad_(False)
  loader=DataLoader(Frames(va,[0,1,2,3],spec['data_root']),batch_size=1,num_workers=2,pin_memory=True,prefetch_factor=1);values=[];ids=[];ys=[];users=[]
  with torch.inference_mode():
   for indices,x,av in loader:
    av=av.cuda(non_blocking=True);f=features(model,x,av,spec['image_micro_batch'],mean,std);z,_,_=head(f.float(),av);values.append(z.cpu().numpy());row=va[int(indices[0])];ids.append(row['clip_id']);ys.append(row['label']);users.append(row['user']);guard.check(phase='SERIALIZED_DINO_FINAL_HELD',job=job['id'],records=len(ids),total=len(va))
  values=np.concatenate(values);np.savez_compressed(out/'predictions.npz',clip_ids=np.array(ids),targets=np.array(ys),users=np.array(users),logits=values)
  done=dict(status='COMPLETE',job=job['id'],fold=job['fold'],epochs=spec['epochs'],protocol_sha256=checksum,backbone_sha256=sha(bp),head_sha256=sha(hp),predictions_sha256=sha(out/'predictions.npz'),serialized_added_bytes=bp.stat().st_size+hp.stat().st_size,seconds=time.time()-start,n=len(ids),correct=int((values.argmax(1)==ys).sum()),Test_read=False)
  atomic(out/'DONE.json',done);print(json.dumps(done),flush=True);del model,head,opt,scaler;torch.cuda.empty_cache()
 atomic(outroot/'QUEUE_DONE.json',dict(status='COMPLETE',protocol_sha256=checksum,at_unix=time.time()))
if __name__=='__main__':
 try:main()
 except BaseException as e:
  atomic(R/'detail_finetune_output/ERROR.json',dict(error=str(e),at_unix=time.time(),traceback=traceback.format_exc()));raise
