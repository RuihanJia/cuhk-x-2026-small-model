"""Frozen external temporal pose features; never fit CUHK labels here."""
from pathlib import Path
import sys,json,time,hashlib,traceback
import numpy as np
import torch
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'bundle'))
from dual_host_runtime.io import atomic
from dual_host_runtime.health import Guard
from posec3d_model import PoseC3D
from posec3d_data import compact_and_resize,sample_indices,heatmaps,SWAP
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 s=json.loads((R/'POSEC3D_FEATURE_PROTOCOL.json').read_text());ph=sha(R/'POSEC3D_FEATURE_PROTOCOL.json');out=R/'posec3d_feature_output';out.mkdir(exist_ok=True);start=time.time();guard=Guard(out,min(start+s['wall_cap_seconds'],s['deadline_unix']),out/'TELEMETRY.json')
 for n,h in s['code_sha256'].items():assert sha(R/n)==h,n
 root=R/'pose_dense_output';assert sha(root/'INDEX.json')==s['dense_index_sha256'];records=json.loads((root/'INDEX.json').read_text())['records'];assert sha(R/'assets/posec3d/state_fp16.pt')==s['weights_sha256'];torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 rows={r['clip_id']:r for r in json.loads((R/'IR_ADAPTER_INPUTS.json').read_text())['rows']};model=PoseC3D();model.load_state_dict(torch.load(R/'assets/posec3d/state_fp16.pt',map_location='cpu',weights_only=True)['state_dict'],strict=True);model.eval().requires_grad_(False).cuda();coords=[];ids=[];available=[]
 for rec in records:
  p=root/rec['file'];assert sha(p)==rec['sha256'];k=np.load(p,allow_pickle=False)['keypoints'];assert k.shape==(rec['frames'],17,3) and np.isfinite(k).all();valid=(k[...,2]>0).sum(-1)>=6;available.append(int(valid.sum())>=2);coords.append(compact_and_resize(k[sample_indices(len(k))],rec['native_size']));ids.append(rec['clip_id'])
 coords=np.stack(coords);np.savez_compressed(out/'coordinate_inputs.npz',clip_ids=np.array(ids),coords=coords,available=np.array(available));features=[]
 with torch.inference_mode():
  for i in range(0,len(ids),s['batch_size']):
   k=torch.from_numpy(coords[i:i+s['batch_size']]).cuda();hm=heatmaps(k);z=model.features(hm).float().cpu().numpy();zz=model.features(hm[:,SWAP].flip(-1)).float().cpu().numpy();features.append(np.stack([z,zz],1));guard.check(phase='EXTERNAL_POSEC3D_FROZEN_FEATURES',records=min(i+s['batch_size'],len(ids)),total=len(ids))
   if i%(s['batch_size']*25)==0:print(json.dumps(dict(records=i+s['batch_size'],seconds=time.time()-start)),flush=True)
 features=np.concatenate(features);assert features.shape==(len(ids),2,512) and np.isfinite(features).all();np.savez_compressed(out/'features.npz',clip_ids=np.array(ids),features=features,available=np.array(available),targets=np.array([rows[k]['label'] for k in ids]),users=np.array([rows[k]['user'] for k in ids]));atomic(out/'DONE.json',dict(status='COMPLETE',protocol_sha256=ph,features_sha256=sha(out/'features.npz'),coordinate_inputs_sha256=sha(out/'coordinate_inputs.npz'),n=len(ids),available=int(sum(available)),seconds=time.time()-start,action_fits=0,action_labels_used_for_features=False,no_Test=True));print(json.dumps(dict(status='COMPLETE',seconds=time.time()-start)),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:atomic(R/'posec3d_feature_output/ERROR.json',dict(error=str(e),traceback=traceback.format_exc(),at_unix=time.time()));raise
