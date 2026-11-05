"""Compare new standalone preprocessing and model graph with frozen held outputs."""
from pathlib import Path
import sys,json,time,hashlib
import numpy as np
import torch
from torch import nn
from torchvision.models.video import mvit_v2_s
R=Path(__file__).resolve().parent;ROOT=R.parents[1];D=ROOT/'deliverables/pose_dual_context_candidate_20260914';sys.path.insert(0,str(D));sys.path.insert(1,str(R/'bundle'))
from activity_token_readout import ActivityTokenModel
from thermal_adapter_model import ThermalAdapter
from detail_inputs import ContextViews,predict_views,local_five
from video_io import Sample
sys.path.insert(2,str(R))
from pose_view_data import PoseFiveViews
from dual_host_runtime.io import atomic
from dual_host_runtime.health import Guard

def main():
 out=R/'release_qa_output';out.mkdir(exist_ok=True);guard=Guard(out,1789323192.0883653,out/'TELEMETRY.json');torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;root=Path(r'D:\Datasets\CUHK-X_Small_Model_Track\Small-Model-Track');inp=json.loads((R/'POSE_ADAPTER_INPUTS.json').read_text());rows={q['clip_id']:q for q in inp['rows']};geo=json.loads((R/'pose_regions_gpu/POSE_REGIONS.json').read_text())['regions'];result=[]
 for f in [0,3]:
  a=dict(np.load(R/f'crop_probe_output/F{f}_INTERVENTIONS.npz',allow_pickle=False));z=dict(np.load(R/'pose_crop_output'/f'F{f}'/'predictions.npz',allow_pickle=False));pr=dict(np.load(R/'pose_adapter_output/cloud'/f'POSE_PRIVATE_F{f}'/'predictions.npz',allow_pickle=False));zi={k:i for i,k in enumerate(z['clip_ids'])};pi={k:i for i,k in enumerate(pr['clip_ids'])};ids=list(a['clip_ids']);chosen=[ids[0],next(k for k in ids if geo[k]['box'] is None)] if any(geo[k]['box'] is None for k in ids) else ids[:2];chosen+= [next(k for k in ids if inp['regions'][k][0]['box'] is not None)]
  cp=torch.load(ROOT/f'runs/activity_token_stopgrad_20260913/cloud_mirror/output/C_TOKEN_STOPGRAD_F{f}_AUX025/model_fp16.pt',map_location='cpu',weights_only=True);v=mvit_v2_s(weights=None);v.head=nn.Linear(768,40);base=ActivityTokenModel(v,'TOKEN_DIFF');base.load_state_dict(cp['state_dict']);model=ThermalAdapter(base,8);p=torch.load(R/'pose_adapter_output/cloud'/f'POSE_PRIVATE_F{f}'/'adapter_fp16.pt',map_location='cpu',weights_only=True);model.load_private(p['private_state']);model.cuda().eval().requires_grad_(False)
  for k in dict.fromkeys(chosen):
   row=rows[k]
   class TrainingPath:
    def __truediv__(self,name):return root/row['streams'][0]['directory'].replace('/IR/','/'+name+'/')
   sample=Sample(k,TrainingPath());assert(sample.sample_dir/'IR').is_dir()
   ds=ContextViews([sample],[geo[k]]);item=ds[0];oldpose=PoseFiveViews([row],str(root),inp['regions'])[0]['x'];pixel=float((oldpose-item['pose']).abs().max());assert pixel==0,(k,pixel);q=predict_views(model,item,'cuda',4);i=ids.index(k);phase=a['depth_box_phase'][i];reference=.5*(phase[:,0].mean((0,1))+phase[:,1].mean((0,1)));errs=dict(video=float(np.abs(q['video']-reference).max()),pose_zero=float(np.abs(q['pose_zero']-z['phase_spatial_modality_logits'][zi[k],:,1,1].mean(0)).max()),pose_private=float(np.abs(q['pose_private']-pr['phase_spatial_modality_logits'][pi[k],:,1,1].mean(0)).max()));assert max(errs.values())<.0003,(k,errs);result.append(dict(fold=f,clip_id=k,pixel_max_abs_difference=pixel,logit_max_abs_difference=errs,pose_fallback=geo[k]['box'] is None));guard.check(force=True,phase='RAW_DEVELOPMENT_PARITY',cases=len(result))
  del model;torch.cuda.empty_cache()
 atomic(out/'DONE.json',dict(status='PASS',at_unix=time.time(),cases=result,Test_read=False,training_fits=0));print(json.dumps(result))
if __name__=='__main__':main()
