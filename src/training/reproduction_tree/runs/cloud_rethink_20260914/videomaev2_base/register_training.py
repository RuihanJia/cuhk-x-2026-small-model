"""Freeze one Base V2 pair and one provisional source fit before any update."""
from pathlib import Path
import ast,copy,hashlib,json,time
E=Path(__file__).resolve().parent;C=E.parent;R=C.parents[1];O=C.parent/'error_resolution_20260913';B='/home/featurize/cuhkx_motion_transfer_20260911_v1';RC=B+'/'+C.name
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert not(E/'protocol.json').exists()
q=read(E/'QA_RESULT.json');assert q['status']=='PASS' and q['parameters_unchanged'] and q['fresh40head_exact_original_Base'] and sha(E/'model.py')==q['model_sha256'] and sha(E/'vit_b_k710_dl_from_giant.pth')==q['public_weight_sha256']
ad=read(C/'videomaev2_small/ADMISSION.json');ad.update(status='ADMITTED_PUBLIC_VIDEOMAEV2_BASE_RESEARCH',at_unix=time.time(),model='VideoMAEv2 Base K710 distilled from a vision-only giant',pretrained_weight_sha256=q['public_weight_sha256'],public_weight_bytes=173574417,parameters=q['parameters'],FP16_measured_bytes=q['FP16_serialized_bytes'],implementation_sha256=q['model_sha256'],implementation_correction='All 162 tensors mapped strictly. V2 author versus retained runtime maxabs 0; SDPA maxabs 5.25e-6; fresh 40-head exactly matches original Base.',use='Noncommercial competition research. Public vision-only teacher; no language model. Fresh 40-class head uses original Training labels.',pretraining_source='Official VideoMAEv2 vision-only giant distillation on K710, public Base checkpoint; no local teacher labels.',allowed_stage='Two source-isolated outer development fits plus one provisional source-inner fit, fixed 16 epochs. No promotion implied.',final_ensemble_admission='Previously measured same Base shape after fixed 50-tensor INT8 plus T/I/POSE: 99,427,234 bytes before calibrator. Must verify newly trained actual serialized total and quantized held outputs before promotion.',qa_result_sha256=sha(E/'QA_RESULT.json'),qa_type='CPU source-only 710-class author comparison and fresh40-head all-parameter finite gradient; zero optimizer updates.')
assert sha(R/ad['rule_evidence'])==ad['rule_evidence_sha256']
write(E/'ADMISSION.json',ad)
w=(C/'videomaev2_small/worker.py').read_text(encoding='utf-8').replace('VideoMAESmall','VideoMAEBase').replace('len(train)+len(held)==2931',"len(train)+len(held)==len(inp['rows'])")
w=w.replace('from dual_host_runtime.io import atomic,read,save_torch,retry','import resilient_io;resilient_io.install()\nfrom dual_host_runtime.io import atomic,read,save_torch,retry')
w=w[:w.index('def main(protocol,host):')]+'''def main(protocol,host):
 pp=Path(protocol).resolve();s=read(pp);ph=sha(pp);assert pp.parent==E and s['status']=='FROZEN_FOR_EXECUTION' and (os.name=='nt')==(host=='local')
 h=s['hosts'][host];out=Path(h['output_directory']);assert out.resolve().is_relative_to(E)
 for rel,hsh in s['code_sha256'].items():checked(C/rel,hsh)
 for name,hsh in h['input_bindings'].items():checked(name,hsh)
 checked(E/'vit_b_k710_dl_from_giant.pth',s['pretrained_sha256']);checked(E/'ADMISSION.json',s['admission_sha256'])
 bd=read(out/'BUDGET.json');ct=read(out/'CONTROLLER.json');assert bd['protocol_sha256']==ct['protocol_sha256']==ph
 me=identity(os.getpid());assert any(v['pid']==me['pid'] and abs(v['create_time']-me['create_time'])<.01 for v in owned_tree(ct['worker']))
 guard=Guard(out,bd['deadline_unix'],out/'TELEMETRY.json');guard.check(force=True,phase='V2_BASE_PREFLIGHT')
 torch.set_num_threads(2 if host=='local' else 1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert ('4070' if host=='local' else '5090') in torch.cuda.get_device_name(0)
 inp=read(checked(h['pose_inputs'],s['pose_inputs_sha256']));outer=read(checked(h['original_protocol'],s['original_protocol_sha256']));inner=read(checked(h['inner_protocol'],s['inner_protocol_sha256']));assert len(inp['rows'])==2931
 runtime=dict(s);runtime.update(h);jobs=[j for j in s['jobs'] if j['host']==host];metadata=[]
 for job in jobs:
  source=job['kind']=='source_inner';parent=next(j for j in (inner if source else outer)['jobs'] if j['id']==job['split_parent'])
  for key in ['train_users','held_users','fold']+(['inner','outer_excluded_users'] if source else []):assert parent[key]==job[key]
  subset=dict(inp);excluded=job.get('outer_excluded_users',[]);subset['rows']=[r for r in inp['rows'] if r['user'] not in excluded]
  assert not set(job['train_users'])&(set(job['held_users'])|set(excluded)) and not set(job['held_users'])&set(excluded)
  assert {r['user'] for r in subset['rows']}==set(job['train_users'])|set(job['held_users'])
  path=out/job['id'];path.mkdir(exist_ok=True);bp=path/'JOB_BUDGET.json'
  if not bp.exists():atomic(bp,dict(protocol_sha256=ph,first_started_unix=time.time(),deadline_unix=min(bd['deadline_unix'],time.time()+h['job_wall_cap_seconds'])))
  jb=read(bp);assert jb['protocol_sha256']==ph;fg=Guard(path,jb['deadline_unix'],out/'TELEMETRY.json')
  fit(job,runtime,ph,path,subset,fg)
  metadata.append(dict(job=job['id'],kind=job['kind'],checkpoint_sha256=sha(path/'model_fp16.pt')))
 atomic(out/'ALL_FITS_FROZEN.json',dict(protocol_sha256=ph,jobs=metadata,held_evaluation_started=False,at_unix=time.time()))
 files={'ALL_FITS_FROZEN.json':sha(out/'ALL_FITS_FROZEN.json')}
 for job in jobs:
  subset=dict(inp);subset['rows']=[r for r in inp['rows'] if r['user'] not in job.get('outer_excluded_users',[])];path=out/job['id'];evaluate(job,runtime,ph,path,subset,guard)
  for n in ['TRAIN_DONE.json','model_fp16.pt','INITIAL.json','SPLIT.json','HISTORY.json','FIRST_UPDATE.json','FROZEN.json','predictions.npz','HELD_DONE.json']:files[job['id']+'/'+n]=sha(path/n)
 guard.check(force=True,phase='V2_BASE_COMPLETE');atomic(out/'DONE.json',dict(status='COMPLETE_V2_BASE_MATCHED_STAGE',protocol_sha256=ph,host=host,artifacts=files,no_Test=True,at_unix=time.time()))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--host',choices=['local','cloud'],required=True);a=p.parse_args()
 try:main(a.protocol,a.host)
 except BaseException as exc:
  s=read(a.protocol);atomic(Path(s['hosts'][a.host]['output_directory'])/'ERROR.json',dict(protocol_sha256=sha(a.protocol),error=str(exc),traceback=traceback.format_exc(),at_unix=time.time()));raise
'''
(E/'worker.py').write_text(w,encoding='utf-8')
co=(C/'videomaev2_small/controller.py').read_text(encoding='utf-8').replace('from dual_host_runtime.io import atomic,read','sys.path.insert(0,str(E))\nimport resilient_io;resilient_io.install()\nfrom dual_host_runtime.io import atomic,read').replace("h['wall_cap_seconds']<=1800","h['wall_cap_seconds']<=2700").replace("s['registered_at_unix']+1920","s['registered_at_unix']+2820").replace("COMPLETE_VIDEOMAE_FIT_AND_HELD","COMPLETE_V2_BASE_MATCHED_STAGE")
co=co.replace("nv=NVML();v=nv.sample();", "retired=h.get('retired_precheck')\n  if retired:\n   rr=Path(retired['directory']);assert sha(rr/'CONTROLLER_ERROR.json')==retired['error_sha256'];assert not alive(read(rr/'CONTROLLER.json')['worker'])\n  nv=NVML();v=nv.sample();")
(E/'controller.py').write_text(co,encoding='utf-8')
s=read(C/'videomaev2_small/protocol.json');now=time.time();s.update(id='V2_BASE_K710_TWO_OUTER_PLUS_ONE_PROVISIONAL_SOURCE',registered_at_unix=now,deadline_unix=now+2820,wall_cap_seconds=1800,trainable_parameters=86257960,checkpoint_cap_bytes=180000000,pretrained_sha256=q['public_weight_sha256'],admission_sha256=sha(E/'ADMISSION.json'),qa_protocol_sha256=sha(E/'QA_PROTOCOL.json'),qa_result_sha256=sha(E/'QA_RESULT.json'),inner_protocol_sha256=sha(O/'POSE_PRIVATE_INNER_PROTOCOL.json'),hypothesis='K400 Base improved matched Small standalone 840 to888; V2 Small improved to882 and deployed source-fitted ensemble900. Test whether combining Base capacity with V2 K710 vision-only pretraining improves complementary errors; effects need not add. Cloud performs both outer folds; local one source-inner fit provisionally overlaps to reduce follow-up latency. No claim that provisional source fit proves promotion.',jobs=[],hosts={})
for k in ['counterpart_status','control_protocol_sha256']:s.pop(k,None)
s['controls']={'primary':'Matched original K400 Base16 raw8-micro cloud F0/F3, same architecture/head seed/sampling/optimizer; change public pretrained body only. Compare V2Small standalone and latest qualified V2global900 separately.','old_base_protocol_sha256':sha(C/'videomae_base_kinetics/protocol.json'),'no_new_control_training':True,'source_unmatched_diagnostic_not_promotion':True}
s['evaluation']={'barrier':'No outer numerical comparison until both exact16 endpoints are frozen, complete, and checksummed. Provisional source remains separate.','report':['paired corrections and harms versus old Base','per user/class and NLL','rescue remaining errors of V2global900','single preregistered 0.8 latest qualified V2global +0.2 Base diagnostic'],'fusion_is_source_unmatched_diagnostic':True,'no_weight_or_threshold_scan':True,'actual50tensor_INT8_and_6source_fits_required_before_promotion':True,'no_guaranteed_Public_gain':True}
s['efficiency']='Cloud two outer fits with micro8/workers12. Local one smaller source-inner fit with proven micro4/workers4. Micro8 local source QA assertion failed and is not adopted. Existing90pause85resume thermal and eight-second Windows atomic-sharing retry retained.'
s['provisional_source_budget_note']='One fixed source fit performed before observing the new outer result; does not authorize the remaining five source fits or production. Local hard45min, cloud30min, global47min includes preparation and queues.'
op=read(O/'POSE_ADAPTER_PROTOCOL.json');ip=read(O/'POSE_PRIVATE_INNER_PROTOCOL.json')
for f in [0,3]:
 par=next(j for j in op['jobs'] if j['id']=='POSE_PRIVATE_F'+str(f));j={k:par[k] for k in ['fold','train_users','held_users']};j.update(id='V2_BASE_F'+str(f),host='cloud',split_parent=par['id'],kind='outer');s['jobs'].append(j)
par=next(j for j in ip['jobs'] if j['id']=='POSE_PRIVATE_F3_INNER2');j={k:par[k] for k in ['fold','inner','train_users','held_users','outer_excluded_users']};j.update(id='V2_BASE_SOURCE_F3_INNER2',host='local',split_parent=par['id'],kind='source_inner');s['jobs'].append(j)
basehost=read(C/'videomaev2_small/protocol.json')['hosts']['local'];localbindings={}
for name,hsh in basehost['input_bindings'].items():
 p=Path(name)
 if p.is_relative_to(C/'videomaev2_small'):continue
 assert sha(p)==hsh;localbindings[str(p)]=hsh
for p in [O/'POSE_PRIVATE_INNER_PROTOCOL.json']+[E/n for n in ['vit_b_k710_dl_from_giant.pth','upstream_modeling_finetune.py','upstream_LICENSE','source_v2.py','license.txt','metadata.json','QA_RESULT.json','ADMISSION.json']]:localbindings[str(p)]=sha(p)
def remote(p):
 rel=Path(p).relative_to(R)
 assert rel.parts[0]=='runs';return B+'/'+Path(*rel.parts[1:]).as_posix()
for host in ['cloud','local']:
 local=host=='local';h=copy.deepcopy(basehost);h.update(output_directory=str(E/'output/local') if local else RC+'/'+E.name+'/output/cloud',wall_cap_seconds=2700 if local else 1800,job_wall_cap_seconds=2520 if local else 780,workers=4 if local else 12,micro_clips=4 if local else 8,eval_micro=4,inner_protocol=str(O/'POSE_PRIVATE_INNER_PROTOCOL.json'),input_bindings=copy.deepcopy(localbindings))
 if not local:
  h['input_bindings']={remote(p):v for p,v in localbindings.items() if Path(p).relative_to(R).parts[0]=='runs'}
  for k in ['pose_inputs','original_protocol','inner_protocol','authorization_file']:h[k]=remote(h[k])
  h['data_root']='/home/featurize/cuhkx_dual_host_20260910_v1/dataset/Small-Model-Track'
 h['predecessor']={'directory':str(C/'v2_base_kd_pilot/output/local') if local else RC+'/v2_base_kd_pilot/output/cloud','protocol_sha256':sha(C/'v2_base_kd_pilot/protocol.json'),'success_only':True}
 if local:h['retired_precheck']={'directory':str(E/'gpu_qa/local'),'error_sha256':sha(E/'gpu_qa/local/CONTROLLER_ERROR.json'),'action':'Verify its owned process is dead. Preserve failed micro8 experiment; use existing proven micro4.'}
 s['hosts'][host]=h
paths=[C/'aux_public_regression_20260914/dynamic_interaction/photometric_candidate/photometric_data_numpy.py']+[E/n for n in ['model.py','worker.py','controller.py','resilient_io.py']]
s['code_sha256']={p.relative_to(C).as_posix():sha(p) for p in paths};s['worker_sha256']=sha(E/'worker.py');s['controller_sha256']=sha(E/'controller.py')
for p in paths:ast.parse(p.read_text(encoding='utf-8'))
write(E/'protocol.json',s)
for event,value in [('V2_BASE_LOCAL_MICRO8_QA_NOT_ADOPTED',{'error_sha256':sha(E/'gpu_qa/local/CONTROLLER_ERROR.json'),'evidence':str((E/'gpu_qa/local/worker.log').relative_to(R)),'micro4_finite_observed':True,'no_optimizer_updates':True}),('V2_BASE_MATCHED_DUAL_STAGE_REGISTERED',{'protocol_sha256':sha(E/'protocol.json'),'protocol':str((E/'protocol.json').relative_to(R)),'deadline_unix':s['deadline_unix'],'cloud_jobs':2,'local_provisional_source_jobs':1})]:
 with(R/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(event=event,at_unix=time.time(),**value))+'\n')
state=read(R/'PROJECT_STATE.json');state.update(updated_at_local=time.strftime('%Y-%m-%dT%H:%M:%S+08:00'),phase='V2_GLOBAL_CSV_READY_V2_BASE_MATCHED_STAGE_FROZEN',active_runs=[],next_frozen_candidate=str((E/'protocol.json').relative_to(R)))
state['current_authorization'].update(active_stage_protocols=[str((E/'protocol.json').relative_to(R))],active_stage_deadline_unix=s['deadline_unix'])
state['latest_base_kd_audit']={'result':str((C/'v2_base_kd_pilot/review/RESULT.json').relative_to(R)),'status':'COMPLETE','private_CE':878,'private_KD':878,'KD_vs_CE_prediction_changes':0,'source_unmatched_diagnostic_both':901,'promoted':False}
write(R/'PROJECT_STATE.json',state)
print(json.dumps({'protocol_sha256':sha(E/'protocol.json'),'deadline_unix':s['deadline_unix'],'jobs':[j['id'] for j in s['jobs']]}))
