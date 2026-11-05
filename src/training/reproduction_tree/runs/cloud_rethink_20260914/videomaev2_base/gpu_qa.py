"""Bounded local source-only gradient-equivalence and microbatch capacity check."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time
E=Path(__file__).resolve().parent;C=E.parent;O=C.parent/'error_resolution_20260913'
sys.path[:0]=[str(E),str(O),str(O/'bundle')]
import resilient_io;resilient_io.install()
from dual_host_runtime.io import atomic,read
from dual_host_runtime.health import Guard,identity,owned_tree
from pose_view_data import PoseClips
from model import VideoMAEBase
import torch
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(protocol,host):
 s=read(protocol);ph=sha(protocol);assert host=='local' and os.name=='nt';out=Path(s['hosts'][host]['output_directory']);bd=read(out/'BUDGET.json');ct=read(out/'CONTROLLER.json');me=identity(os.getpid());assert ct['protocol_sha256']==bd['protocol_sha256']==ph and any(v['pid']==me['pid'] and abs(v['create_time']-me['create_time'])<.01 for v in owned_tree(ct['worker']))
 for p,h in s['bindings'].items():assert sha(C.parents[1]/p)==h
 guard=Guard(out,bd['deadline_unix'],out/'TELEMETRY.json');guard.check(force=True,phase='BASE_LOCAL_BATCH_QA');torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 inp=read(O/'POSE_ADAPTER_INPUTS.json');row=next(x for x in inp['rows'] if x['clip_id']==s['source_clip']);assert row['user']==4
 x=PoseClips([row],s['data_root'],20260915,inp['regions'])[0]['x'][None].cuda();model=VideoMAEBase(classes=40,fast=True).cuda().train();params=list(model.parameters());optimizer_resident=torch.empty(2*sum(p.numel() for p in params),device='cuda',dtype=torch.float32);start=time.time();ref=None;checks=[]
 for micro in [4,8]:
  try:
   model.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();begin=time.time();losses=[]
   for _ in range(16//micro):
    guard.check(force=True,phase='BASE_MICROBATCH_GRADIENT_QA',micro=micro)
    with torch.autocast('cuda',dtype=torch.float16):z=model(x.repeat(micro,1,1,1,1));loss=torch.nn.functional.cross_entropy(z,torch.full((micro,),row['label'],device='cuda',dtype=torch.long),label_smoothing=.1)
    (loss*128.*micro/16.).backward();losses.append(float(loss.detach()))
   torch.cuda.synchronize();seconds=time.time()-begin;peak=torch.cuda.max_memory_allocated();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
   grad=torch.cat([p.grad.detach().cpu().flatten()/128. for p in params]);r=dict(micro=micro,effective_batch=16,seconds=seconds,peak_allocated_bytes=peak,finite_all_gradients=True,mean_loss=sum(losses)/len(losses))
   if ref is None:ref=grad
   else:
    r['gradient_cosine']=float(torch.dot(ref,grad)/(ref.norm()*grad.norm()));r['gradient_relative_l2']=float((ref-grad).norm()/ref.norm());assert r['gradient_cosine']>.999 and r['gradient_relative_l2']<.02
   checks.append(r);print(json.dumps(r),flush=True)
  except torch.cuda.OutOfMemoryError:
   assert micro==8;model.zero_grad(set_to_none=True);torch.cuda.empty_cache();checks.append(dict(micro=8,status='OOM_NO_OPTIMIZER_UPDATES'));break
 selected=8 if len(checks)==2 and checks[1].get('finite_all_gradients') and checks[1]['peak_allocated_bytes']<s['memory_margin_limit_bytes'] and checks[1]['seconds']<checks[0]['seconds'] else 4
 r=dict(status='PASS_BASE_LOCAL_BATCH_QA',protocol_sha256=ph,checks=checks,selected_micro=selected,optimizer_updates=0,simulated_adam_moment_bytes=optimizer_resident.numel()*4,no_Test=True,source_clip=row['clip_id'],scope='Repeatedsingle-sourceclip effective16forward/backward only; notfulltraining speedup. Existing90/85thermalpolicy unchanged.',seconds=time.time()-start,at_unix=time.time());atomic(out/'RESULT.json',r);atomic(out/'DONE.json',dict(status='PASS_BASE_LOCAL_BATCH_QA',protocol_sha256=ph,artifacts={'RESULT.json':sha(out/'RESULT.json')},at_unix=time.time()))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--host',required=True);a=p.parse_args();main(**vars(a))
