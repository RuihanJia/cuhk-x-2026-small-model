"""Single frozen-model intervention: automatic pose-directed IR local crop."""
from pathlib import Path
import json,time,sys,hashlib,traceback
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.video import mvit_v2_s
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'bundle'))
from crop_probe import Views
from activity_token_readout import ActivityTokenModel
from dual_host_runtime.health import Guard
from dual_host_runtime.io import atomic
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 s=json.loads((R/'POSE_CROP_PROTOCOL.json').read_text());ph=sha(R/'POSE_CROP_PROTOCOL.json');assert s['worker_sha256']==sha(__file__);out=R/'pose_crop_output';out.mkdir(exist_ok=True);done=json.loads((R/'pose_regions_gpu/DONE.json').read_text());geo=R/'pose_regions_gpu/POSE_REGIONS.json';assert done['regions_sha256']==sha(geo) and done['protocol_sha256']==s['geometry_protocol_sha256'];regions=json.loads(geo.read_text())['regions'];rows={r['clip_id']:r for r in json.loads((R/'IR_ADAPTER_INPUTS.json').read_text())['rows']};guard=Guard(out,min(s['deadline_unix'],time.time()+s['wall_cap_seconds']),out/'TELEMETRY.json');torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;results=[]
 for job in s['jobs']:
  f=job['fold'];p=out/f'F{f}';p.mkdir(exist_ok=True)
  if(p/'DONE.json').exists():continue
  start=time.time();source=Path(job['source']);assert sha(source)==job['source_sha256'];cp=torch.load(source,map_location='cpu',weights_only=True);assert cp['train_users']==job['train_users'] and not set(cp['train_users'])&set(job['held_users']);v=mvit_v2_s(weights=None);v.head=nn.Linear(768,40);m=ActivityTokenModel(v,'TOKEN_DIFF');m.load_state_dict(cp['state_dict']);del cp;m.eval().requires_grad_(False).cuda()
  src=R/f'crop_probe_output/F{f}_INTERVENTIONS.npz';assert sha(src)==job['phase_source_sha256'];a=dict(np.load(src,allow_pickle=False));original=a['depth_box_phase'];phase=original.copy();rr=[]
  for k in a['clip_ids']:
   row=rows[str(k)];g=regions[str(k)];st=row['streams'][0];assert row['user'] in job['held_users'];rr.append(dict(IR_directory=str(Path(s['data_root'])/st['directory']),frames=len(st['files']),native_size=g['native_size'],depth_box=g['box']))
  with torch.inference_mode():
   for i,x in DataLoader(Views(rr),batch_size=None,num_workers=8,prefetch_factor=1,pin_memory=True):
    if x.numel():phase[i,:,1,1]=m(x[::2].cuda(non_blocking=True)).float().cpu().numpy()[[0,0,0,1,2]]
    guard.check(phase='POSE_CROP_ZERO_FIT',fold=f,records=i+1,total=len(rr))
  assert np.array_equal(phase[:,:,0],original[:,:,0]) and np.array_equal(phase[:,:,1,0],original[:,:,1,0]);logits=.5*(phase[:,:,0].mean((1,2))+phase[:,:,1].mean((1,2)));np.savez_compressed(p/'predictions.npz',clip_ids=a['clip_ids'],targets=a['targets'],users=a['users'],phase_spatial_modality_logits=phase,logits=logits);d=dict(status='COMPLETE',protocol_sha256=ph,source_sha256=job['source_sha256'],geometry_sha256=sha(geo),predictions_sha256=sha(p/'predictions.npz'),n=len(rr),correct=int((logits.argmax(1)==a['targets']).sum()),seconds=time.time()-start,training_fits=0,no_Test=True,unchanged_other3routes=True);atomic(p/'DONE.json',d);results.append(d);print(json.dumps(d),flush=True);del m;torch.cuda.empty_cache()
 atomic(out/'QUEUE_DONE.json',dict(status='COMPLETE',results=results,at_unix=time.time(),protocol_sha256=ph))
if __name__=='__main__':
 try:main()
 except BaseException as e:atomic(R/'pose_crop_output/ERROR.json',dict(error=str(e),traceback=traceback.format_exc(),at_unix=time.time()));raise
