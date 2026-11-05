"""CPU-only binding preparation; --freeze only after the actual GPU protocol exists."""
from pathlib import Path
import argparse,hashlib,json,time
E=Path(__file__).resolve().parent;C=E.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def bound(p):p=Path(p).resolve();return dict(path=str(p),sha256=sha(p))
def main(freeze=False):
 now=time.time();old=read(C/'v2_mean6_small8_source/fusion_protocol.json');previous=read(C/'v2_temporal_residual_source_completion/fusion/protocol.json');stage=E.parent/'protocol.json';proto=read(stage) if stage.exists() else None
 fs=dict(status='FROZEN_FOR_EXECUTION' if freeze else 'PREPARED_NOT_EXECUTABLE',id='FIXED_HALF_MIRROR_RTN6_SMALL8_ORIGINAL_FOUR_BRANCH_SOURCE_FUSION',registered_at_unix=now if freeze else None,deadline_unix=min(now+3600,proto['deadline_unix']) if freeze else None,wall_cap_seconds=3600,cpu_fit_cap_seconds=240,source_authorization_sha256=old['source_authorization_sha256'],source_recipe=dict(branches=['T','I','half_original_plus_half_mirrored_Base6_seven','half_original_plus_half_mirrored_Small8_five'],blend_domain='FP32_logits',blend_weight=.5,global_recipe='Original convex_TI_BASE_SMALL4 FP64 user-balanced LBFGS max_iter500',classwise='Original classwise4.py 160 centered deviations, fixed penalty .1, global temperatures frozen, FP64 LBFGS max_iter500',primary='classwise4',four_calibrators_before_new_outer_arrays=True,no_scan=True),outer_users=old['outer_users'],origins=[],inputs={},references=previous['references'],code_bindings={})
 fs['expected_containers']={j['id']:j['container']['sha256'] for j in old['small_readouts'] if 'sha256' in j['container']}
 pilot=read(C/'v2_base6_small_design/protocol.json');fs['expected_containers']['V2_BASE_F3']=pilot['bindings'][pilot['container']]
 fs['inputs']['original_F3_pilot_protocol']=bound(C/'v2_base6_small_design/protocol.json')
 for k in ['source_oof','outer_scores_0','outer_scores_3']:fs['inputs'][k]=old['inputs'][k]
 fs['inputs']['raw_source']=bound(C/'v2_base6_small8_source_completion/output/local/SOURCE_FITTED.npz');assert fs['inputs']['raw_source']['sha256']=='0380c8f5a388ca4edfbc892ba35ae9d8e029a2d0f9632bb8fbe9adf7c81987fc'
 fs['inputs']['parent_inventory']=old['inputs']['Small_inventory'];fs['inputs']['raw_fusion_protocol']=old['inputs']['Small_source_protocol'];fs['inputs']['mirror_pixel_QA']=bound(E.parent/'CPU_QA.json');fs['inputs']['source_authorization']=bound(C/'AUTHORIZATION_DUAL_RESOURCE_20260914.json')
 assert fs['inputs']['source_authorization']['sha256']==fs['source_authorization_sha256']
 for host in ['cloud','local']:
  fs['origins'].append(dict(protocol_path=str(stage.resolve()),protocol_sha256=sha(stage) if proto else None,host=host,output_root=str((E.parent/'cloud' if host=='cloud' else E.parent/'output/local').resolve()),success_status='COMPLETE_MIRROR_DUAL_READOUT'))
 for p in [E/'calibrate.py',E/'independent_audit.py',E/'run.py',E/'prepare.py',C/'frozen_fusion_math.py',C/'v2_base6_small8_source_completion/classwise4.py',C.parent/'error_resolution_20260913/bundle/dual_host_runtime/io.py']:fs['code_bindings'][str(p.resolve())]=sha(p)
 for v in list(fs['inputs'].values())+list(fs['references'].values()):assert sha(v['path'])==v['sha256'],v['path']
 if freeze:
  assert proto and proto['status']=='FROZEN_FOR_EXECUTION' and len(proto['jobs'])==8 and time.time()<fs['deadline_unix']
  assert {j['scope'] for j in proto['jobs']}=={'source','outer'}
 p=E/('protocol.json' if freeze else 'protocol_candidate.json')
 with p.open('x',encoding='utf8') as f:json.dump(fs,f,indent=2);f.write('\n')
 print(json.dumps(dict(path=str(p),sha256=sha(p),status=fs['status'])))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');main(**vars(p.parse_args()))
