"""Source-excluded frozen crop inference for the six existing R6 inner sources."""
from pathlib import Path
import json,time,sys,hashlib,traceback
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.video import mvit_v2_s
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'bundle'))
from activity_token_readout import ActivityTokenModel
from dual_host_runtime.health import Guard
from dual_host_runtime.io import atomic
from crop_probe import Views
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=R/'POSE_INNER_PROTOCOL.json';s=json.loads(p.read_text());ph=sha(p);out=R/'pose_inner_output';out.mkdir(exist_ok=True)
 for n,h in {**s['code_sha256'],**s['asset_sha256']}.items():assert sha(R/n)==h
 guard=Guard(out,min(time.time()+s['wall_cap_seconds'],s['deadline_unix']),out/'TELEMETRY.json');torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 inputs=json.loads((R/'IR_ADAPTER_INPUTS.json').read_text());rows={r['clip_id']:r for r in inputs['rows']};roi=json.loads((R/'pose_regions_gpu/POSE_REGIONS.json').read_text())['regions'];all_results=[]
 for job in s['jobs']:
  d=out/job['id'];d.mkdir(exist_ok=True)
  if (d/'DONE.json').exists():continue
  start=time.time();cpfile=Path(job['source']);assert sha(cpfile)==job['source_sha256'];npfile=Path(job['prediction_source']);assert sha(npfile)==job['prediction_sha256'];base=dict(np.load(npfile,allow_pickle=False));cp=torch.load(cpfile,map_location='cpu',weights_only=True)
  assert sorted(cp['train_users'])==job['train_users'];assert not set(cp['train_users'])&set(job['held_users']);assert not (set(cp['train_users'])|set(job['held_users']))&set(job['outer_excluded_users'])
  v=mvit_v2_s(weights=None);v.head=nn.Linear(768,40);m=ActivityTokenModel(v,'TOKEN_DIFF');m.load_state_dict(cp['state_dict']);del cp;m.eval().requires_grad_(False).cuda()
  original=base['phase_spatial_modality_logits'];ids=base['clip_ids'];phase=original.copy();assert original.shape==(len(ids),5,2,2,40);rr=[]
  for k in ids:
   row=rows[str(k)];q=roi[str(k)];stream=row['streams'][0];assert row['user'] in job['held_users']
   rr.append(dict(clip_id=str(k),IR_directory=str(Path(s['data_root'])/stream['directory']),depth_box=q['box'],frames=len(stream['files']),native_size=q['native_size']))
  with torch.inference_mode():
   for i,x in DataLoader(Views(rr),batch_size=None,num_workers=4,prefetch_factor=1,pin_memory=True):
    if x.numel():
     # Reuse exactly the original probe preprocessing. Discard gamma views.
     z=m(x[::2].cuda(non_blocking=True)).float().cpu().numpy();phase[i,:,1,1]=z[[0,0,0,1,2]]
    guard.check(phase='POSE_SOURCE_INFERENCE',job=job['id'],records=i+1,total=len(rr))
  assert np.array_equal(phase[:,:,0],original[:,:,0]);assert np.array_equal(phase[:,:,1,0],original[:,:,1,0])
  logits=.5*(phase[:,:,0].mean((1,2))+phase[:,:,1].mean((1,2)))
  np.savez_compressed(d/'predictions.npz',clip_ids=ids,targets=base['targets'],users=base['users'],phase_spatial_modality_logits=phase,logits=logits)
  # Targets are preserved for source calibration. No closed-fold score reported.
  done=dict(status='COMPLETE',job=job['id'],protocol_sha256=ph,source_sha256=job['source_sha256'],predictions_sha256=sha(d/'predictions.npz'),n=len(ids),seconds=time.time()-start,training_fits=0,no_Test=True)
  atomic(d/'DONE.json',done);all_results.append(done);print(json.dumps(done),flush=True);del m;torch.cuda.empty_cache()
 atomic(out/'QUEUE_DONE.json',dict(status='COMPLETE',protocol_sha256=ph,at_unix=time.time(),jobs=all_results))
if __name__=='__main__':
 try:main()
 except BaseException as e:atomic(R/'pose_inner_output'/'ERROR.json',dict(error=str(e),traceback=traceback.format_exc(),at_unix=time.time()));raise
