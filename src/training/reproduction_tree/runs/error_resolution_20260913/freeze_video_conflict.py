import json,time,hashlib,shutil
from pathlib import Path
R=Path(__file__).resolve().parent;REPO=R.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (R/'VIDEO_CONFLICT_PROTOCOL.json').exists()
needed=['activity_view_data.py','activity_token_readout.py','motion_video_runtime/__init__.py','motion_video_runtime/data.py','research_runtime/common.py']
for n in needed:
 p=R/'bundle'/n;p.parent.mkdir(parents=True,exist_ok=True);source=REPO/'runs/activity_token_stopgrad_20260913/bundle'/n
 if p.exists():assert sha(p)==sha(source),n
 else:shutil.copyfile(source,p)
rows=json.loads((REPO/'runs/activity_token_stage_20260912/local_extended/data/frame_index.json').read_text(encoding='utf-8'));roi=json.loads((REPO/'runs/activity_token_stage_20260912/local_extended/data/activity_roi_v1.json').read_text(encoding='utf-8'))['regions'];detail=json.loads((R/'DETAIL_INPUTS.json').read_text(encoding='utf-8'))
for r in rows:
 q=roi[r['clip_id']]
 if q[0]['box'] is not None:
  assert q[0]['native_size']==q[1]['native_size'];q[1]['box']=q[0]['box'];q[1]['reason']='DEPTH_GUIDED_IR_FROM_LABEL_FREE_SYNCHRONIZED_FRAMES'
weights={r['clip_id']:max(.25,1-.75*r['conflict_fraction']) for r in detail}
(R/'VIDEO_INPUTS.json').write_text(json.dumps(dict(rows=rows,regions=roi,conflict_weights=weights),ensure_ascii=False)+'\n',encoding='utf-8')
prior=json.loads((R/'CROP_PROBE_PROTOCOL.json').read_text());jobs=[]
for fold in [0,3]:
 for weighted in [False,True]:
  b=prior['folds'][str(fold)];jobs.append(dict(id=f'VIDEO_F{fold}_'+('CONFLICT' if weighted else 'CONTROL'),fold=fold,weighted=weighted,source=b['checkpoint'],source_sha256=b['checkpoint_sha256'].lower(),source_protocol_sha256=b['source_protocol_sha256'],train_users=b['train_users'],held_users=b['held_users']))
code=['video_conflict_worker.py','video_conflict_controller.py',*[f'bundle/{n}' for n in needed],'bundle/dual_host_runtime/__init__.py','bundle/dual_host_runtime/io.py','bundle/dual_host_runtime/health.py']
auth=json.loads((R/'AUTHORIZATION.json').read_text(encoding='utf-8'))
spec=dict(id='R6_DEPTH_GUIDED_CONFLICT_WEIGHTING_V1',registered_at_unix=time.time(),hypothesis='Automatic same-user cross-class frame overlap creates conflicting visual gradients. Reduce their CE contribution while preserving labels, and adapt training to the independently positive Depth-guided IR crop.',post_observation_design=True,evidence=['audit/OBSERVATIONS.jsonl','audit/FULL_FRAME_OBSERVATIONS.jsonl','audit/FULL_CONTAINMENT_DEPTH_PROOF.json','crop_probe_output/FROZEN_FUSION_RESULT.json','detail_output/RESULT.json'],jobs=jobs,epochs=4,eval_epochs=[4],seed=20260913,backbone_lr=5e-6,head_lr=5e-5,min_lr=1e-6,weight_decay=.05,auxiliary_backbone_stopgrad=True,auxiliary_ce_weight=.25,label_smoothing=.1,effective_clip_batch=8,micro_clip_batch=4,video_forward_batch=16,control='Both arms identical source/crop/split/schedule; only per-clip CE weight changes. Incumbent and zero-fit crop results remain additional references.',weight_formula='max(.25,1-.75*max_same_user_cross_class_identical_IR_fraction), overlaps<3frames ignored; weights multiply combined and auxiliary losses without normalizing by weight sum',fusion='Frozen original R6 calibrator, no new fusion weight fitting or Public search',deadline_unix=auth['execution_deadline_unix'],job_wall_cap_seconds=2700,estimated_total_minutes=65,host='cloud',data_root='/home/featurize/cuhkx_dual_host_20260910_v1/dataset/Small-Model-Track',code_sha256={n:sha(R/n) for n in code},asset_sha256={'VIDEO_INPUTS.json':sha(R/'VIDEO_INPUTS.json')},no_Test=True,no_LLM_chain=True,model_bytes_cap=100000000,confirmation='F0/F3 development only; F2/F4 not opened for selection',resume='Epoch model+optimizer+scaler+RNG restored under same protocol and original per-job/total deadline')
(R/'VIDEO_CONFLICT_PROTOCOL.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (REPO/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(event='REGISTERED',at_unix=time.time(),experiment=spec['id'],protocol='runs/error_resolution_20260913/VIDEO_CONFLICT_PROTOCOL.json',protocol_sha256=sha(R/'VIDEO_CONFLICT_PROTOCOL.json'),jobs=4,epochs=4,estimated_minutes=65,deadline_unix=spec['deadline_unix']),ensure_ascii=False)+'\n')
print(json.dumps(dict(protocol_sha256=sha(R/'VIDEO_CONFLICT_PROTOCOL.json'),jobs=[x['id'] for x in jobs],downweighted_records=sum(x<1 for x in weights.values()))))
