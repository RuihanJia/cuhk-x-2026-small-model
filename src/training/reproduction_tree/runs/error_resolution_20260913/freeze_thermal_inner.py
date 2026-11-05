"""Freeze six source-excluded private Thermal fits and identical CPU initial states."""
from pathlib import Path
import json,time,hashlib,sys,copy
import numpy as np
import torch
from torch import nn
from torchvision.models.video import mvit_v2_s
R=Path(__file__).resolve().parent;ROOT=R.parents[1];sys.path.insert(0,str(R/'bundle'))
from activity_token_readout import ActivityTokenModel
from thermal_adapter_model import ThermalAdapter
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
 torch.set_num_threads(2);old=read(R/'THERMAL_ADAPTER_PROTOCOL.json');template=read(R/'VIDEO_SUCCESSOR_TEMPLATE.json');p=R/'THERMAL_INNER_PROTOCOL.json';assert not p.exists();jobs=[];initroot=R/'thermal_initial';initroot.mkdir(exist_ok=True)
 for j in template['jobs']:
  if j['fold']==18:continue
  job={k:j[k] for k in ['fold','inner','train_users','held_users','outer_excluded_users','source_sha256','source_protocol_sha256']};job['id']=f"THERMAL_F{j['fold']}_INNER{j['inner']}";job['host']='local' if j['fold']==3 and j['inner'] in [1,2] else 'cloud';job['source']={'local':j['local_source'],'cloud':j['source']}
  assert not (set(job['train_users'])|set(job['held_users']))&set(job['outer_excluded_users']);assert not set(job['train_users'])&set(job['held_users']);assert sha(j['local_source'])==job['source_sha256']
  cp=torch.load(j['local_source'],map_location='cpu',weights_only=True);assert cp['train_users']==job['train_users'];torch.manual_seed(old['seed']);v=mvit_v2_s(weights=None);v.head=nn.Linear(768,40);base=ActivityTokenModel(v,'TOKEN_DIFF');base.load_state_dict(cp['state_dict']);m=ThermalAdapter(base,old['rank'])
  # FP32 avoids a host-dependent random initialization and avoids rounding A at init.
  state={k:v.detach().cpu().clone() for k,v in m.named_parameters() if v.requires_grad};assert sum(v.numel() for v in state.values())==207954
  ip=initroot/(job['id']+'.pt');torch.save(dict(private_state=state,source_sha256=job['source_sha256'],seed=old['seed']),ip);job['initial_private_path']=str(ip.relative_to(R)).replace('\\','/');job['initial_private_sha256']=sha(ip);jobs.append(job);del m,base,v,cp,state
 s=copy.deepcopy(old);s.update(id='THERMAL_PRIVATE_SOURCE_CALIBRATION_SIX_FITS_V1',registered_at_unix=time.time(),post_observation_design=True,jobs=jobs,source_evidence=dict(private_F3_result_sha256=sha(R/'private_analysis/RESULT.json'),interpretation='First outer Thermal standalone improves27 but fixed fused regresses6; changed confidence and complementary replacements require source-only calibration, not a held-label weight scan.'),hypothesis='A0.43MB shared-video private Thermal route learns useful new information; test calibrated fifth branch while retaining oldThermal, plus a direct replacement control, with every source checkpoint excluding both outer and inner prediction users.',job_wall_cap_seconds=2700,estimated_total_minutes_range=dict(cloud=[25,45],local=[35,55]),init='Exactly hash-bound CPU FP32 private tensors, shared across hosts; same batch8, seed, augmentation, optimizer, epochs and evaluation.',fit_scope='Six inner fits only. Full18 requires subsequent concrete decision; not launched by this protocol.',calibration_plan=dict(frozen_before_inner_outputs=True,strategies=['CROP_VSTI_CONVEX','CROP_VSTI_WITH_PRIVATE_T_FIFTH_CONVEX','CROP_VS_PRIVATE_T_I_CONVEX'],source_only=True,max_iterations=500,outer_folds=[0,3],outer_labels_used=False,closed_F2_F4_not_reopened=True,no_class_or_user_route=True,no_weight_grid=True),code_sha256={**{k:v for k,v in old['code_sha256'].items() if k.startswith('bundle/') or k=='thermal_adapter_model.py'},**{k:sha(R/k) for k in ['thermal_inner_worker.py','thermal_inner_local_controller.py','thermal_inner_cloud_controller.py']}},asset_sha256={'THERMAL_INPUTS.json':sha(R/'THERMAL_INPUTS.json'),**{j['initial_private_path']:j['initial_private_sha256'] for j in jobs}})
 s['hosts']['cloud']=dict(data_root='/home/featurize/cuhkx_motion_transfer_20260911_v1/error_resolution_20260913/thermal_dataset',micro_clips=8,workers=8,eval_videos=8)
 p.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 with(ROOT/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf-8') as fp:fp.write(json.dumps(dict(event='REGISTERED',experiment=s['id'],protocol=str(p.relative_to(ROOT)),protocol_sha256=sha(p),at_unix=time.time(),fits=6,hosts={h:[j['id'] for j in jobs if j['host']==h] for h in ['cloud','local']},no_Test=True))+'\n')
 print(json.dumps(dict(protocol_sha256=sha(p),jobs=[(j['id'],j['host']) for j in jobs],CPU_init_only=True)))
if __name__=='__main__':main()
