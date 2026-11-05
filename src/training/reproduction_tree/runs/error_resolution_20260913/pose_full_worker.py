"""Full18 IR-pose adapter, supervised CE only, same eight-epoch developmentrecipe."""
"""One fullTraining IR-pose adapter on frozen R6full18."""
from pathlib import Path
import sys,time,json,hashlib,math,random,argparse,traceback
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.models.video import mvit_v2_s
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'bundle'))
from dual_host_runtime.io import atomic,save_torch
from dual_host_runtime.health import Guard
from pose_view_data import PoseClips as ActivityClips,PoseFiveViews as ActivityFiveViews
from activity_token_readout import ActivityTokenModel
from thermal_adapter_model import ThermalAdapter
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(host):
 spec=json.loads((R/'POSE_FULL_PROTOCOL.json').read_text(encoding='utf-8'));checksum=sha(R/'POSE_FULL_PROTOCOL.json')
 for n,h in spec['code_sha256'].items():assert sha(R/n)==h,n
 for n,h in spec['asset_sha256'].items():assert sha(R/n)==h,n
 torch.set_num_threads(2 if host=='local' else 1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 inp=json.loads((R/'POSE_ADAPTER_INPUTS.json').read_text(encoding='utf-8'));rows=inp['rows'];regions=inp['regions'];quality=inp['conflict_weights'];setting=spec['hosts'][host];outroot=R/'pose_full_output'/host;outroot.mkdir(parents=True,exist_ok=True)
 for job in spec['jobs']:
  if job['host']!=host:continue
  out=outroot/job['id'];out.mkdir(exist_ok=True)
  if(out/'DONE.json').exists():continue
  start=time.time();budget=out/'JOB_BUDGET.json'
  if not budget.exists():atomic(budget,dict(protocol_sha256=checksum,first_started_unix=start,deadline_unix=min(spec['deadline_unix'],start+spec['job_wall_cap_seconds'])))
  bd=json.loads(budget.read_text());assert bd['protocol_sha256']==checksum;guard=Guard(out,bd['deadline_unix'],outroot/'TELEMETRY.json')
  source=Path(job['source'][host]);assert sha(source)==job['source_sha256'];cp=torch.load(source,map_location='cpu',weights_only=True);assert sorted(cp['train_users'])==job['train_users'];assert not set(cp['train_users'])&set(job['held_users'])
  torch.manual_seed(spec['seed']);np.random.seed(spec['seed']);random.seed(spec['seed'])
  v=mvit_v2_s(weights=None);v.head=nn.Linear(768,40);base=ActivityTokenModel(v,'TOKEN_DIFF');base.load_state_dict(cp['state_dict']);del cp
  m=ThermalAdapter(base,spec['rank']).cuda();frozen_hash=m.frozen_digest();params=[p for p in m.parameters() if p.requires_grad];assert sum(p.numel() for p in params)<600000
  opt=torch.optim.AdamW(params,lr=spec['lr'],weight_decay=.01);scaler=torch.amp.GradScaler('cuda',init_scale=128);hist=[];updates=0;first=1
  tr=[r for r in rows if r['user'] in job['train_users']];va=[r for r in rows if r['user'] in job['held_users']];assert len(tr)+len(va)==len(rows)
  atomic(out/'SPLIT.json',dict(train_users=job['train_users'],held_users=job['held_users'],train_ids=[r['clip_id'] for r in tr],held_ids=[r['clip_id'] for r in va],only_IR_local_view_adapted=True,outer_labels_used=False))
  micro=setting['micro_clips'];accum=8//micro;total=spec['epochs']*math.ceil(len(tr)/8)
  if(out/'resume.pt').exists():
   q=torch.load(out/'resume.pt',map_location='cpu',weights_only=True);assert q['protocol_sha256']==checksum;m.load_private(q['private']);opt.load_state_dict(q['optimizer']);scaler.load_state_dict(q['scaler']);hist=q['history'];updates=q['updates'];first=q['epoch']+1;torch.set_rng_state(q['rng']);torch.cuda.set_rng_state_all(q['cuda_rng']);del q
  for epoch in range(first,spec['epochs']+1):
   m.train();ds=ActivityClips(tr,setting['data_root'],spec['seed'],regions,True,True,epoch);gen=torch.Generator().manual_seed(spec['seed']+epoch)
   loader=DataLoader(ds,batch_size=micro,shuffle=True,generator=gen,num_workers=setting['workers'],pin_memory=True,prefetch_factor=1);seen=0;loss_sum=0;began=time.time();wait=0;previous=time.perf_counter();opt.zero_grad(set_to_none=True)
   for bi,b in enumerate(loader):
    wait+=time.perf_counter()-previous;n=len(b['y']);x=b['x'].cuda(non_blocking=True);y=b['y'].cuda(non_blocking=True);q=torch.tensor([quality[k] for k in b['clip_id']],device='cuda')
    if bi%accum==0:
     lr=1e-5+(spec['lr']-1e-5)*.5*(1+math.cos(math.pi*updates/total))
     for g in opt.param_groups:g['lr']=lr
    with torch.autocast('cuda',dtype=torch.float16):
     z=m(x,True);ce=F.cross_entropy(z,y,label_smoothing=.1,reduction='none');loss=(q*ce).mean()
    assert torch.isfinite(loss);window=min(8,len(tr)-(bi//accum)*8);scaler.scale(loss*n/window).backward();seen+=n;loss_sum+=float(loss.detach())*n
    if bi%accum==accum-1 or bi+1==len(loader):
     old=scaler.get_scale();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);scaler.step(opt);scaler.update();assert scaler.get_scale()>=old,'AMP_SKIP';opt.zero_grad(set_to_none=True);updates+=1
    guard.check(phase='POSE_PRIVATE_CE_TRAIN',job=job['id'],epoch=epoch,epochs=spec['epochs'],records=seen,total=len(tr),updates=updates);previous=time.perf_counter()
   assert seen==len(tr);hist.append(dict(epoch=epoch,loss=loss_sum/seen,records=seen,seconds=time.time()-began,input_wait_seconds=wait,updates=updates));atomic(out/'HISTORY.json',hist)
   save_torch(out/'resume.pt',dict(protocol_sha256=checksum,private={k:p.detach().cpu() for k,p in m.named_parameters() if p.requires_grad},optimizer=opt.state_dict(),scaler=scaler.state_dict(),history=hist,updates=updates,epoch=epoch,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
   print(json.dumps(dict(job=job['id'],**hist[-1])),flush=True)
  assert m.frozen_digest()==frozen_hash,'BASE_CHANGED'
  cpfile=out/'adapter_fp16.pt';save_torch(cpfile,dict(private_state=m.private_state(),source_sha256=job['source_sha256'],train_users=job['train_users'],protocol_sha256=checksum,rank=spec['rank']))
  assert cpfile.stat().st_size<1500000;m.load_private(torch.load(cpfile,map_location='cpu',weights_only=True)['private_state']);m.eval().requires_grad_(False)
  done=dict(status='COMPLETE_FULL18_POSE_ADAPTER',job=job['id'],protocol_sha256=checksum,source_sha256=job['source_sha256'],frozen_base_hash_before=frozen_hash,frozen_base_unchanged=True,checkpoint_sha256=sha(cpfile),checkpoint_bytes=cpfile.stat().st_size,epochs=spec['epochs'],training_records=len(tr),seconds=time.time()-start,no_Test=True)
  atomic(out/'DONE.json',done);print(json.dumps(done),flush=True)
 atomic(outroot/'QUEUE_DONE.json',dict(status='COMPLETE',protocol_sha256=checksum,at_unix=time.time()))
if __name__=='__main__':
 try:main('cloud')
 except BaseException as e:
  out=R/'pose_full_output/cloud';out.mkdir(parents=True,exist_ok=True);atomic(out/'ERROR.json',dict(error=str(e),traceback=traceback.format_exc(),at_unix=time.time()));raise
