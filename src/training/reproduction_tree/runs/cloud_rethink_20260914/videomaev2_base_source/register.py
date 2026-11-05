"""Five fixed source fits following the completed Base V2 paired development audit."""
from pathlib import Path
import ast,copy,hashlib,json,time
E=Path(__file__).resolve().parent;C=E.parent;R=C.parents[1];W=C/'videomaev2_base';O=C.parent/'error_resolution_20260913';RC='/home/featurize/cuhkx_motion_transfer_20260911_v1/cloud_rethink_20260914'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert not(E/'protocol.json').exists();d=read(W/'review/DONE.json')
for n,h in d['artifacts'].items():assert sha(W/'review'/n)==h
result=read(W/'review/RESULT.json');assert result['status']=='COMPLETE_V2_BASE_TWO_OUTER_AUDIT' and all(result['source_sampling_exact'].values())
parent=read(W/'protocol.json');s=read(C/'videomaev2_source/protocol.json');now=time.time()
s.update(id='VIDEOMAEV2_BASE_REMAINING_FIVE_SOURCE16_CLOUD',registered_at_unix=now,deadline_unix=now+2820,wall_cap_seconds=1800,job_wall_cap_seconds=660,video_protocol_sha256=sha(W/'protocol.json'),hypothesis='Completed matched Base V2 standalone918 versus K400 Base888 and V2 Small882; source sampling exact. Raw Base V2 also18 net above current900 whole model. Complete remaining five original user-isolated source folds to determine actual TI/POSE calibrated complementarity under100MB. Existing local sourceF3inner2 remains same recipe, not repeated. Public0.83582 retained and not used to tune parameters.',outer_audit_sha256=sha(W/'review/RESULT.json'),optimizer_and_sampling_recipe='Reuse admitted V2Base exact16 recipe and existing deterministic RAW_CONTROL frames/ROIs/conflict weights; fresh public weights in every fit. No warm-start from outer or from an earlier source fold.',jobs=[],hosts={},source_sampling_assertion='Base V2 fit already supports source-subset cardinality; no code rewriting at runtime.',resource_note='Five cloud jobs, maximum45minutes including queue; each660seconds training cap, host/globalbound also covers held evaluation and transport. No new instance or old budget reset. Local previously registered sourceF3inner2 continues independently.')
original=read(O/'POSE_PRIVATE_INNER_PROTOCOL.json')
for par in original['jobs']:
 if par['fold'] not in [0,3] or (par['fold']==3 and par['inner']==2):continue
 j={k:par[k] for k in ['fold','inner','train_users','held_users','outer_excluded_users']};j.update(id='V2_BASE_SOURCE_F%d_INNER%d'%(par['fold'],par['inner']),host='cloud',split_parent=par['id']);s['jobs'].append(j)
assert len(s['jobs'])==5
h=copy.deepcopy(parent['hosts']['cloud']);h.update(output_directory=RC+'/'+E.name+'/output/cloud',wall_cap_seconds=2700,job_wall_cap_seconds=660,original_protocol=RC.replace('/cloud_rethink_20260914','')+'/error_resolution_20260913/POSE_PRIVATE_INNER_PROTOCOL.json',predecessor={'directory':parent['hosts']['cloud']['output_directory'],'protocol_sha256':sha(W/'protocol.json'),'success_only':True});h['input_bindings'][RC+'/videomaev2_base/protocol.json']=sha(W/'protocol.json');s['hosts']={'cloud':h}
s['code_sha256']=copy.deepcopy(parent['code_sha256'])
for n in ['worker.py','controller.py']:
 p=E/n;ast.parse(p.read_text(encoding='utf-8'));s['code_sha256'][p.relative_to(C).as_posix()]=sha(p)
s['worker_sha256']=sha(E/'worker.py');s['controller_sha256']=sha(E/'controller.py');s['existing_local_source']=dict(protocol_sha256=sha(W/'protocol.json'),job='V2_BASE_SOURCE_F3_INNER2',output_directory=str(W/'output/local/V2_BASE_SOURCE_F3_INNER2'))
s['original_protocol_sha256']=sha(O/'POSE_PRIVATE_INNER_PROTOCOL.json');s['no_new_outer_held_evaluation']=True;s['production_not_registered']=True
write(E/'protocol.json',s)
files=[dict(name=n,bytes=(E/n).stat().st_size,sha256=sha(E/n)) for n in ['protocol.json','worker.py','controller.py']]
write(E/'TRANSFER_MANIFEST.json',dict(status='READY',files=files,total_bytes=sum(v['bytes'] for v in files),remote_directory=RC+'/'+E.name,no_raw_data=True,no_credentials=True,existing_public_weight_not_reuploaded=True,at_unix=now))
access=C/'post_aux_cloud_access.py';text=access.read_text(encoding='utf-8');old="'videomaev2_base/protocol.json']";assert old in text;text=text.replace(old,"'videomaev2_base/protocol.json','videomaev2_base_source/protocol.json']");access.write_text(text,encoding='utf-8')
record=dict(event='V2_BASE_FIVE_SOURCE_REGISTERED_AFTER_PAIRED_AUDIT',at_unix=now,protocol=str((E/'protocol.json').relative_to(R)),protocol_sha256=sha(E/'protocol.json'),outer_audit_sha256=sha(W/'review/RESULT.json'),deadline_unix=s['deadline_unix'],no_public_fitting=True)
with(R/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
state=read(R/'PROJECT_STATE.json');state['phase']='V2_BASE918_OUTER_COMPLETE_FIVE_SOURCE_REGISTERED';state['next_frozen_candidate']=str((E/'protocol.json').relative_to(R));state['current_authorization']['active_stage_protocols']=[str((W/'protocol.json').relative_to(R)),str((E/'protocol.json').relative_to(R))];state['current_authorization']['active_stage_deadline_unix']=s['deadline_unix'];state['latest_v2_base_audit']=dict(result=str((W/'review/RESULT.json').relative_to(R)),result_sha256=sha(W/'review/RESULT.json'),matched_Base_before=888,BaseV2_after=918,fix=74,harm=44,diagnostic917_not_production=True);write(R/'PROJECT_STATE.json',state)
print(json.dumps(dict(protocol_sha256=sha(E/'protocol.json'),deadline_unix=s['deadline_unix'],jobs=[j['id'] for j in s['jobs']],bytes=sum(v['bytes'] for v in files),users_vs_current_whole={u:{k:v[k] for k in ['before','after','fix','harm','net']} for u,v in result['private_vs_current_whole']['per_user'].items()})))
