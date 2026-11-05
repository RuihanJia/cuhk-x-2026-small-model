from pathlib import Path,PurePosixPath
import csv,json,time,shutil,hashlib,ast
R=Path(__file__).resolve().parent;ROOT=R.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert not (R/'CROP_PROBE_PROTOCOL.json').exists()
bundle=R/'bundle';bundle.mkdir(exist_ok=True)
for n in ['activity_token_readout.py','activity_inputs.py','video_io.py']:
    shutil.copy2(ROOT/'deliverables/activity_token_stopgrad_candidate_20260913'/n,bundle/n)
(bundle/'dual_host_runtime').mkdir(exist_ok=True)
for n in ['__init__.py','health.py','io.py']:
    shutil.copy2(ROOT/'runs/joint_relation_20260913/throughput_probe/bundle/dual_host_runtime'/n,bundle/'dual_host_runtime'/n)
for p in [R/'crop_probe.py',R/'crop_controller.py',*bundle.rglob('*.py')]:ast.parse(p.read_text(encoding='utf-8'))
gp=R/'audit/ALL_1094_INPUT_GEOMETRY.json';geometry=json.loads(gp.read_text(encoding='utf-8'))
g={x['clip_id']:x for x in geometry['cases']}
cp=ROOT/'runs/joint_relation_20260913/analysis/ALL_1094_COMPARISONS.csv'
with cp.open(encoding='utf-8-sig',newline='') as f:comparisons=list(csv.DictReader(f))
rows=[];DATA=PurePosixPath('/home/featurize/cuhkx_dual_host_20260910_v1/dataset/Small-Model-Track')
for c in comparisons:
    a=g[c['clip_id']];ir=a['IR'];d=a['Depth_Color']
    assert ir['native_size']==d['native_size'] and ir['frames']==d['frames']
    assert not ir['decode_failures'] and not d['decode_failures']
    assert ir['timestamp']==d['timestamp']
    row=dict(clip_id=c['clip_id'],fold=int(c['fold']),user=int(c['user']),target=int(c['true_class']),depth_box=d['region']['box'],native_size=ir['native_size'],frames=ir['frames'])
    for m in ['IR','Depth_Color']:
        p=Path(c[m+'_directory']);parts=p.parts[p.parts.index('Training'):]
        row[m+'_directory']=str(DATA.joinpath(*parts))
    rows.append(row)
write(R/'CROP_INPUTS.json',rows)
folds={}
for f in [0,3]:
    source=ROOT/f'runs/activity_token_stopgrad_20260913/cloud_mirror/output/C_TOKEN_STOPGRAD_F{f}_AUX025'
    done=json.loads((source/'DONE.json').read_text(encoding='utf-8'))
    dst=R/f'base_F{f}.npz';shutil.copy2(source/'predictions.npz',dst)
    folds[str(f)]=dict(checkpoint=f'/home/featurize/cuhkx_motion_transfer_20260911_v1/activity_token_stopgrad_20260913/cloud/output/C_TOKEN_STOPGRAD_F{f}_AUX025/model_fp16.pt',checkpoint_sha256=sha(source/'model_fp16.pt'),train_users=done['spec']['train_users'],held_users=done['spec']['held_users'],source_protocol_sha256=done['protocol_sha256'],source_done_sha256=sha(source/'DONE.json'),prediction_sha256=sha(dst))
auth=json.loads((R/'AUTHORIZATION.json').read_text(encoding='utf-8'))
spec=dict(id='ZERO_FIT_DEPTH_GUIDED_IR_CROP_GAMMA_V1',registered_at_unix=time.time(),hypothesis='Mislocalized or absent IR activity crops hide the actual actor and hand/object evidence. Depth-guided IR local crops, with optional fixed sqrt contrast, can expose whether this preprocessing contributes to held-user errors.',arms=['original R6 phase logits','replace IR local view by existing Depth activity box when valid','same Depth-guided IR local view with fixed sqrt intensity'],fixed_global_Depth_and_fusion=True,training_fits=0,LLM_labels_or_features=False,manual_per_case_crops=False,Test_read=False,Public_selection=False,scope='All 1094 previously exposed F0/F3 development rows, paired controls retained. Adaptive mechanistic inference probe; negative zero-fit distribution-shift results cannot exclude retraining.',eval_points='All 1094; fixed video aggregation, then inherited frozen calibrator; fixes/harms and per-user/class reported without choosing weights or thresholds.',wall_cap_seconds=3600,deadline_unix=auth['host_deadline_unix']['cloud'],cloud_temperature_stop=88,resource=dict(host='existing cloud 5090',workers=4,prefetch=1,micro_batch=6,precision='FP32',TF32=False,unique_phases=[.5,.25,.75],phase_replication=[0,0,0,1,2]),source_geometry_sha256=sha(gp),source_comparisons_sha256=sha(cp),alignment=dict(native_sizes_match=1094,frame_counts_match=1094,IR_Depth_timestamp_extents_match=1094,scope='Timestamp metadata and dimensions verified for all rows; paired PNG timestamp equality verified separately before launch.'),inputs_sha256=sha(R/'CROP_INPUTS.json'),folds=folds,code_sha256=sha(R/'crop_probe.py'),controller_sha256=sha(R/'crop_controller.py'),bundle_hashes={str(p.relative_to(R)).replace('\\','/'):sha(p) for p in bundle.rglob('*.py')})
write(R/'CROP_PROBE_PROTOCOL.json',spec)
with (ROOT/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf-8') as out:out.write(json.dumps(dict(event='REGISTERED_ZERO_FIT_CROP_DIAGNOSTIC',at_unix=time.time(),protocol=str((R/'CROP_PROBE_PROTOCOL.json').relative_to(ROOT)),protocol_sha256=sha(R/'CROP_PROBE_PROTOCOL.json'),n=1094,training_fits=0,wall_cap_seconds=3600),ensure_ascii=False)+'\n')
print(json.dumps(dict(status='REGISTERED',n=len(rows),depth_box_available=sum(r['depth_box'] is not None for r in rows),protocol_sha256=sha(R/'CROP_PROBE_PROTOCOL.json'))))
