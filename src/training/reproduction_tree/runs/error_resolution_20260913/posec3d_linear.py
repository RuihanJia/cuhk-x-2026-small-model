"""User-isolated linear transfer from the admitted external pose representation."""
from pathlib import Path
import json,time,hashlib,sys,traceback
import numpy as np
import torch
from torch.nn import functional as F
R=Path(__file__).resolve().parent;ROOT=R.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 torch.set_num_threads(2);s=json.loads((R/'POSEC3D_LINEAR_PROTOCOL.json').read_text());ph=sha(R/'POSEC3D_LINEAR_PROTOCOL.json');assert sha(__file__)==s['worker_sha256'];out=R/'posec3d_linear_output';out.mkdir(exist_ok=True);d=json.loads((R/'posec3d_feature_output/DONE.json').read_text());assert d['protocol_sha256']==s['feature_protocol_sha256'];p=R/'posec3d_feature_output/features.npz';assert sha(p)==d['features_sha256'];a=dict(np.load(p,allow_pickle=False));x=a['features'].mean(1).astype(np.float64);available=a['available'].astype(bool);start=time.time();results=[]
 for job in s['jobs']:
  dst=out/job['id'];dst.mkdir(exist_ok=True)
  if(dst/'DONE.json').exists():results.append(json.loads((dst/'DONE.json').read_text()));continue
  assert time.time()<min(s['deadline_unix'],start+s['wall_cap_seconds']);tr=np.isin(a['users'],job['train_users'])&available;va=np.isin(a['users'],job['held_users']);assert tr.sum()>0 and va.sum()>0 and not set(job['train_users'])&set(job['held_users']);assert not(set(job['train_users'])|set(job['held_users']))&set(job.get('outer_excluded_users',[]));t0=time.time()
  mean=x[tr].mean(0);scale=x[tr].std(0).clip(1e-4,None);tx=torch.from_numpy((x[tr]-mean)/scale);ty=torch.from_numpy(a['targets'][tr]).long();u=a['users'][tr];weights=torch.from_numpy(np.array([1/sum(u==v) for v in u],np.float64));weights/=weights.sum();w=torch.zeros((40,512),dtype=torch.float64,requires_grad=True);b=torch.zeros(40,dtype=torch.float64,requires_grad=True);opt=torch.optim.LBFGS([w,b],lr=1,max_iter=s['LBFGS_max_iter'],line_search_fn='strong_wolfe',tolerance_grad=1e-8,tolerance_change=1e-10);calls=0
  def closure():
   nonlocal calls
   assert time.time()<min(s['deadline_unix'],start+s['wall_cap_seconds']);opt.zero_grad();z=F.linear(tx,w,b);loss=(weights*F.cross_entropy(z,ty,reduction='none')).sum()+.5/tr.sum()*w.square().sum();assert torch.isfinite(loss);loss.backward();calls+=1;return loss
  loss=float(opt.step(closure));model=dict(weight=w.detach().float(),bias=b.detach().float(),mean=torch.from_numpy(mean).float(),scale=torch.from_numpy(scale).float(),train_users=job['train_users'],held_users=job['held_users'],outer_excluded_users=job.get('outer_excluded_users',[]),protocol_sha256=ph,source_features_sha256=d['features_sha256']);torch.save(model,dst/'linear.pt');write(dst/'FROZEN.json',dict(protocol_sha256=ph,model_sha256=sha(dst/'linear.pt'),at_unix=time.time(),held_labels_used_for_fit=False,source_users=job['train_users']))
  model=torch.load(dst/'linear.pt',map_location='cpu',weights_only=True)
  with torch.inference_mode():logits=F.linear((torch.from_numpy(x[va]).float()-model['mean'])/model['scale'],model['weight'],model['bias']).numpy()
  np.savez_compressed(dst/'predictions.npz',clip_ids=a['clip_ids'][va],users=a['users'][va],targets=a['targets'][va],available=available[va],logits=logits);done=dict(status='COMPLETE',job=job['id'],protocol_sha256=ph,checkpoint_sha256=sha(dst/'linear.pt'),checkpoint_bytes=(dst/'linear.pt').stat().st_size,predictions_sha256=sha(dst/'predictions.npz'),n=int(va.sum()),available=int(available[va].sum()),train_users=job['train_users'],held_users=job['held_users'],outer_excluded_users=job.get('outer_excluded_users',[]),closure_calls=calls,seconds=time.time()-t0)
  if job['kind']=='outer':done['correct_on_available']=int(((logits.argmax(1)==a['targets'][va])&available[va]).sum())
  write(dst/'DONE.json',done);results.append(done);print(json.dumps(done),flush=True)
 write(out/'QUEUE_DONE.json',dict(status='COMPLETE',protocol_sha256=ph,results=results,at_unix=time.time(),seconds=time.time()-start))
 with(ROOT/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf-8') as fp:fp.write(json.dumps(dict(event='POSEC3D_LINEAR_EIGHT_SOURCE_FITS_COMPLETE',at_unix=time.time(),protocol_sha256=ph,artifact=str((out/'QUEUE_DONE.json').relative_to(ROOT))))+'\n')
if __name__=='__main__':
 try:main()
 except BaseException as e:
  (R/'posec3d_linear_output').mkdir(exist_ok=True);write(R/'posec3d_linear_output/ERROR.json',dict(error=str(e),traceback=traceback.format_exc(),at_unix=time.time()));raise
