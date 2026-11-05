"""Pre-register finite dual-host detail experiments after the complete per-case audit."""
from pathlib import Path
import json,hashlib,time,shutil,sys
R=Path(__file__).resolve().parent;REPO=R.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (R/'DETAIL_PROTOCOL.json').exists()
review={json.loads(x)['case_id'] for x in (R/'audit/OBSERVATIONS.jsonl').read_text(encoding='utf-8').splitlines()};assert len(review)==285
index=REPO/'runs/activity_token_stage_20260912/local_extended/data/frame_index.json';roi=REPO/'runs/activity_token_stage_20260912/local_extended/data/activity_roi_v1.json'
rows=json.loads(index.read_text(encoding='utf-8'));regions=json.loads(roi.read_text(encoding='utf-8'))['regions'];DATA=Path('D:/Datasets/CUHK-X_Small_Model_Track/Small-Model-Track')
overlap=json.loads((R/'audit/ALL_TRAINING_TEMPORAL_OVERLAP.json').read_text(encoding='utf-8'))
fractions={};conflicts={}
for p in overlap['cross_class_overlaps']:
 if p['identical_IR_bytes']<3:continue
 for a,b in [(p['a'],p['b']),(p['b'],p['a'])]:
  cid=a['activity']+'/'+a['user'][4:]+'/'+a['clip'];other=b['activity']+'/'+b['user'][4:]+'/'+b['clip'];assert a['user']==b['user']
  value=p['identical_IR_bytes']/a['frames'];fractions[cid]=max(fractions.get(cid,0),value);conflicts.setdefault(cid,[]).append(dict(other_id=other,fraction=value,user=int(a['user'][4:])))
inputs=[]
for r in rows:
 d,ir=r['streams'];assert d['modality']=='Depth_Color' and ir['modality']=='IR'
 thermal='Training/data/HAR/data/Thermal/'+r['clip_id'].split('/')[0]+'/user'+str(r['user'])+'/'+r['clip_id'].split('/')[-1]
 fs=sorted(p.name for p in (DATA/thermal).glob('*') if p.suffix.lower() in ['.png','.jpg'])
 views=[dict(directory=ir['directory'],files=ir['files'],box=None),dict(directory=ir['directory'],files=ir['files'],box=regions[r['clip_id']][0]['box'] or regions[r['clip_id']][1]['box']),dict(directory=d['directory'],files=d['files'],box=None),dict(directory=thermal,files=fs,box=None)]
 inputs.append(dict(clip_id=r['clip_id'],label=r['label'],user=r['user'],views=views,conflict_fraction=fractions.get(r['clip_id'],0),conflicts=conflicts.get(r['clip_id'],[])))
(R/'DETAIL_INPUTS.json').write_text(json.dumps(inputs,ensure_ascii=False)+'\n',encoding='utf-8')
ad=json.loads((R/'assets/dino/ADMISSION.json').read_text());assert ad['remaining_head_budget']>800000
code=['detail_model.py','detail_worker.py','detail_controller.py','bundle/dual_host_runtime/__init__.py','bundle/dual_host_runtime/io.py','bundle/dual_host_runtime/health.py']
assets=['DETAIL_INPUTS.json','assets/dino/dino_int8.pt','assets/dino/ADMISSION.json']+[x.relative_to(R).as_posix() for x in (R/'assets/dino/source').rglob('*') if x.is_file() and x.suffix=='.py']
auth=json.loads((R/'AUTHORIZATION.json').read_text(encoding='utf-8'))
source=json.loads((R/'CROP_PROBE_PROTOCOL.json').read_text(encoding='utf-8'))
spec=dict(schema=1,registered_at_unix=time.time(),id='DINO_DETAIL_TEMPORAL_MULTIMODAL_V1',hypotheses=[
 'A public non-language image encoder can supply object/contact cues missing from the current video expert.',
 'Signed change and local temporal convolution distinguish process direction and short events from time-invariant pooling.',
 'Reducing only duplicate-conflict IR/Depth auxiliary supervision lets intact Thermal supervision contribute, without relabeling.'
 ],observed_development_scope='F0/F3 heavily exposed development; not unbiased unseen-user or Public proof',audit_scope='285 distinct union error/harm montages and full-frame targeted review; diagnostic observations NEVER become training labels, features or per-case crops',
 deadline_unix=auth['execution_deadline_unix'],cache_wall_cap_seconds=3600,head_wall_cap_seconds=1800,training_fits=6,head_arms=['STATIC','TEMPORAL','TEMPORAL_CONFLICT'],head_training=dict(epochs=40,eval_epochs=[40],batch=64,seed=20260913,optimizer='AdamW',lr=.001,min_lr=.00005,weight_decay=.01,dropout=.3,label_smoothing=.1,auxiliary_loss_weight=.25,modality_dropout=.15,temporal_frames=16,selection='fixed final epoch, no held-dependent early stopping'),
 folds={k:dict(train_users=v['train_users'],held_users=v['held_users']) for k,v in source['folds'].items()},
 hosts={'cloud':dict(cache_parts=[0,1,2],cache_clip_batch=2,image_batch=96,cache_workers=6,fold=0,data_root='/home/featurize/cuhkx_dual_host_20260910_v1/dataset/Small-Model-Track'),'local':dict(cache_parts=[3],cache_clip_batch=2,image_batch=32,cache_workers=2,fold=3,data_root=str(DATA))},
 sources=dict(frame_index_sha256=sha(index),roi_sha256=sha(roi),overlap_sha256=sha(R/'audit/ALL_TRAINING_TEMPORAL_OVERLAP.json'),observations_sha256=sha(R/'audit/OBSERVATIONS.jsonl'),full_frame_observations_sha256=sha(R/'audit/FULL_FRAME_OBSERVATIONS.jsonl')),
 feature_definition='Shared quantized DINOv2-S/14; 16 center-segment frames, aspect-preserving pad to224; CLS+2x2 mean patch cells; ImageNet normalization. Views IRfull, IRDepthguidedROI, Depthfull, Thermalfull. No temporal alignment assumed for Thermal.',
 fusion_development_probe=dict(recipe='0.8*softmax(frozen_R6_logits)+0.2*softmax(new_head_logits)',weight=.2,fit_parameters=0,no_Public_search=True,report_separately=True),
 conflict_handling='Automatic same-user Training frame-byte overlap of at least3frames; q=max(.25,1-.75*maximum_shared_fraction). Third arm weights IR/Depth auxiliary CE by q, retains original fused CE and original labels. Held evaluation unchanged.',
 dino_int8_sha256=ad['int8_sha256'],serialized_byte_cap=100000000,head_serialized_cap=800000,expected_package_max_bytes=ad['existing_R6_bytes']+ad['int8_bytes']+800000,
 no_Test_access=True,no_LLM_chain=True,no_automatic_submission=True,code_sha256={n:sha(R/n) for n in code},asset_sha256={n:sha(R/n) for n in assets})
assert set(spec['folds'])=={'0','3'}
(R/'DETAIL_PROTOCOL.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (REPO/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(event='REGISTERED',at_unix=time.time(),experiment=spec['id'],protocol=str((R/'DETAIL_PROTOCOL.json').relative_to(REPO)),protocol_sha256=sha(R/'DETAIL_PROTOCOL.json'),fits=6,epochs=40,hosts=['local','cloud'],deadline_unix=spec['deadline_unix'],evidence_before_registration=spec['sources']),ensure_ascii=False)+'\n')
p=REPO/'PROJECT_STATE.json';state=json.loads(p.read_text(encoding='utf-8'));state['phase']='ERROR_DRIVEN_DETAIL_ARCHITECTURE_VALIDATION';state['next_frozen_candidate']=dict(protocol=str((R/'DETAIL_PROTOCOL.json').relative_to(REPO)),sha256=sha(R/'DETAIL_PROTOCOL.json'));state['current_research']['error_resolution_20260913'].update(status='DETAIL_PROTOCOL_FROZEN',protocol_frozen=True,visually_reviewed_cases=285,full_frame_review_cases=20,training_started=False,detail_protocol=str((R/'DETAIL_PROTOCOL.json').relative_to(REPO)));state['updated_at_unix']=time.time();tmp=p.with_suffix('.detail.tmp');tmp.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(p)
print(json.dumps(dict(protocol_sha256=sha(R/'DETAIL_PROTOCOL.json'),records=len(inputs),thermal_available=sum(bool(r['views'][3]['files']) for r in inputs),conflicting_records=sum(r['conflict_fraction']>0 for r in inputs),expected_package_max_bytes=spec['expected_package_max_bytes'])))
