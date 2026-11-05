"""Bind the one approved two-source pilot; does not dispatch either host."""
from pathlib import Path
import ast,hashlib,json,time
E=Path(__file__).resolve().parent;C=E.parent;R=C.parents[1];O=C.parent/'error_resolution_20260913'
REMOTE='/home/featurize/cuhkx_motion_transfer_20260911_v1/cloud_rethink_20260914'
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):assert not p.exists(),str(p);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n','utf8')

def main():
 assert not(E/'protocol.json').exists()
 prior=read(C/'v2_context_readout/protocol.json');window=read(C/'v2_window_cloud52910/protocol.json');inventory=read(C/'v2_base6_small8_source_completion/INPUTS.json')
 raw={j['id']:j for j in inventory['jobs']};jobs=[];uploads={};existing={};all8=[]
 for j in window['jobs']:
  if j['arm']!='CONTROL':continue
  d=(C/'v2_window_resume/output/local_v2'if j['host']=='local'else C/'v2_window_eval_tail/cloud')/j['id']
  fr=read(d/'FROZEN.json');pb=read(d/'PARENT_BINDING.json');sp=read(d/'SPLIT.json');a=raw[j['parent_job']]['parents']['Base'];rd=Path(a['parent']['path']).parent;rfr=read(rd/'FROZEN.json');rsp=read(rd/'SPLIT.json')
  assert pb['parent_checkpoint_sha256']==a['parent']['sha256']==j['parent_checkpoint_sha256']
  assert sp['train_users']==rsp['train_users']==j['train_users'] and sp['held_users']==rsp['held_users']==j['held_users']
  assert sp['train_ids']==rsp['train_ids'] and sp['held_ids']==rsp['held_ids']
  assert set(j['train_users']).isdisjoint(j['held_users']) and (set(j['train_users'])|set(j['held_users'])).isdisjoint(j.get('outer_excluded_users',[]))
  all8.append(dict(id=j['parent_job'],raw_fp16=a['parent'],control_fp16_sha256=fr['checkpoint_sha256'],control_metadata_directory=str(d),train_users=j['train_users'],held_users=j['held_users'],outer_excluded_users=j.get('outer_excluded_users',[]),cloud_status='historical_verified_not_live'if j['host']=='cloud'else'local_files_present'))
  if j['parent_job']in ('V2_BASE_SOURCE_F0_INNER0','V2_BASE_SOURCE_F3_INNER2'):continue
  host=j['host'];paths={}
  for kind,local,remote,meta in [('raw',rd,j['parent_checkpoint'].rsplit('/',1)[0]if host=='cloud'else str(rd),rfr),('control',d,REMOTE+'/v2_window_cloud52910/output/cloud/'+j['id']if host=='cloud'else str(d),fr)]:
   v=dict(directory=remote,protocol_sha256=meta['protocol_sha256'],checkpoint_sha256=meta['checkpoint_sha256'],frozen_sha256=sha(local/'FROZEN.json'),split_sha256=sha(local/'SPLIT.json'),train_done_sha256=sha(local/'TRAIN_DONE.json'))
   if kind=='control':v['parent_binding_sha256']=sha(local/'PARENT_BINDING.json')
   paths[kind]=v
   if host=='cloud':
    existing[remote+'/model_fp16.pt']=v['checkpoint_sha256']
    for n in ['FROZEN.json','SPLIT.json','TRAIN_DONE.json']+(['PARENT_BINDING.json']if kind=='control'else[]):
     q=local/n;target=remote+'/'+n;existing[target]=sha(q);uploads[target]=dict(local_path=str(q),sha256=sha(q),bytes=q.stat().st_size)
  jobs.append(dict(id=j['parent_job']+'_FIXED_HALF_AVERAGE',scope='source'if j['kind']=='source_inner'else'outer',host=host,fold=j['fold'],inner=j.get('inner'),records=len(sp['held_ids']),train_users=j['train_users'],held_users=j['held_users'],outer_excluded_users=j.get('outer_excluded_users',[]),**paths))
 own=['worker.py','data.py','controller.py','resilient_io.py','launch.py','prepare.py']
 names=[E.name+'/'+n for n in own]+['videomaev2_base/model.py','videomaev2_base/upstream_modeling_finetune.py','videomaev2_base_quant/codec.py','v2_window_resume/temporal_data.py','aux_public_regression_20260914/dynamic_interaction/photometric_candidate/photometric_data_numpy.py','../error_resolution_20260913/pose_view_data.py','../error_resolution_20260913/bundle/activity_view_data.py','../error_resolution_20260913/bundle/motion_video_runtime/data.py','../error_resolution_20260913/bundle/dual_host_runtime/io.py','../error_resolution_20260913/bundle/dual_host_runtime/health.py']
 codes={n:sha(C/n)for n in names}
 for n in names:
  q=C/n;ast.parse(q.read_text('utf8'));target=str(Path(REMOTE)/n).replace('\\','/');uploads[target]=dict(local_path=str(q),sha256=sha(q),bytes=q.stat().st_size);existing[target]=sha(q)
 hosts={}
 for host in ['local','cloud']:
  h=prior['hosts'][host];bh=dict(output_directory=str(E/'output/local')if host=='local'else REMOTE+'/'+E.name+'/output/cloud',wall_cap_seconds=5400,workers=2 if host=='local'else 12,eval_micro=4,data_root=h['data_root'],pose_inputs=h['pose_inputs'],authorization_file=h['authorization_file'],input_bindings={h['pose_inputs']:prior['pose_inputs_sha256'],h['authorization_file']:prior['source_authorization_sha256']})
  if host=='local':pd=C/'v2_true_fullframe_readout/output/local';pp='6cad58a63e3f1148cbaa492c96f70faa3e27d25efacf25920df890bb65af2de2'
  else:pd=Path(REMOTE)/'v2_true_fullframe_readout/output/cloud';pp='8c5a9cbdf0435d4c0175fb5c18b2bc3c849f982fe614f751b9844d90a1063e44'
  bh['predecessor']=dict(directory=str(pd).replace('\\','/')if host=='cloud'else str(pd),protocol_sha256=pp,success_only=True);hosts[host]=bh
  if host=='cloud':existing.update(bh['input_bindings'])
 jobs.sort(key=lambda j:(j['scope']=='outer',j['fold'],j.get('inner')or 0));now=time.time();s=dict(status='FROZEN_FOR_EXECUTION',id='V2_BASE_RAW16_CONTROL4_FIXED_HALF_WEIGHT_AVERAGE_REMAINING_SIX',registered_at_unix=now,deadline_unix=now+5400,wall_cap_seconds=5400,source_authorization_sha256=prior['source_authorization_sha256'],explicit_authorization='Root explicitly approved six remaining fixed-average endpoints plus full source global3/classwise3: cloud5/local1, exact audited local CONTROL4 retained, reuse pilot2. Queue behind successful true-fullframe per host.90min complete budget; zero updates, no Test or scan.',hypothesis='One fixed average can damp the endpoint perturbation of matched low-LR warm continuation while retaining one encoder. Prior Control4 classwise927 versus H928 is mixed evidence, not proof averaging improves.',alpha=.5,arithmetic='FP32 midpoint of two saved FP16 states, FP16 storage round, original50-matrix INT8 pack, actual dequantized FP32 forward',optimizer_updates=0,seed=20260915,storage_format='VIDEOMAE_BASE_ALL_50_WEIGHTS_OUTPUT_INT8_FP32_SCALE_OTHERS_FP16_V1',checkpoint_cap_bytes=87000000,pose_inputs_sha256=prior['pose_inputs_sha256'],jobs=jobs,hosts=hosts,code_sha256=codes,worker_sha256=codes[E.name+'/worker.py'],controller_sha256=codes[E.name+'/controller.py'],evaluation=dict(views='unchanged original five plus early/late midpoint halves',formula='.5*mean(first5)+.25*early+.25*late',source_groups=6,outer_arrays_opened=False,calibration_fits=4,comparators='exact existing raw16 INT8 seven-view and matched CONTROL4 INT8 seven-view source predictions',report='source global3/classwise3 same H recipe; all four frozen before outer arrays; compare fixed H928 and CONTROL927. GPU rawreadout has no metrics.'),limits=['Adaptive development source groups, not fresh confirmation.','CONTROL4 reset AdamW and LR warmup, not exact16to20 resume.','This completes fixed eightreadouts, reuses pilot2 and fits originalsource global3/classwise3. Full18 production is separate and not selected here.','No SWA claim: one uniform two-endpoint average, no cycle/snapshot scan.'],model_size_projection=dict(base_int8_prior_bytes=86770711,unchanged_TI_pose_bytes=12656523,total_before_calibrator_bytes=99427234,no_double_encoder_at_inference=True),references=['https://arxiv.org/abs/2203.05482','https://arxiv.org/abs/1803.05407'])
 qa=read(C/'v2_base_weight_average_pilot/CPU_QA.json');assert qa['status']=='PASS_FIXED_AVERAGE_CPU_QA';s['cpu_qa_sha256']=sha(C/'v2_base_weight_average_pilot/CPU_QA.json');s['pilot_protocol']=str(C/'v2_base_weight_average_pilot/protocol.json');s['pilot_protocol_sha256']=sha(C/'v2_base_weight_average_pilot/protocol.json');s['pilot_outputs']={h:str(C/'v2_base_weight_average_pilot'/('cloud'if h=='cloud'else'output/local'))for h in ['cloud','local']};s['pilot_audit_sha256']=sha(C/'v2_base_weight_average_pilot/audit/RESULT.json');s['fusion_recipe']={'branches':['T','I','BaseAverage'],'global':'original FP64 LBFGS max_iter500,sourceuserbalanced','classwise':'original H centered3x40 delta,penalty.1,FP64LBFGS500','outer_after_all_four_frozen':True,'reference_correct':[928,927]}
 write(E/'PARENT_INVENTORY.json',dict(status='PASS_EIGHT_MATCHED_ENDPOINT_METADATA',jobs=all8,no_live_cloud_check=True))
 write(E/'protocol.json',s);ph=sha(E/'protocol.json')
 for n in ['protocol.json','PARENT_INVENTORY.json']:
  q=E/n;target=REMOTE+'/'+E.name+'/'+n;uploads[target]=dict(local_path=str(q),sha256=sha(q),bytes=q.stat().st_size);existing[target]=sha(q)
 write(E/'TRANSFER_MANIFEST.json',dict(files=uploads,existing_checks=existing,checkpoint_uploads=0,raw_data_uploads=0))
 write(E/'REGISTRATION.json',dict(protocol_sha256=ph,at_unix=now,optimizer_updates=0,dispatch_started=False))
 with(R/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf8')as f:f.write(json.dumps(dict(event='REGISTERED_V2_BASE_FIXED_HALF_WEIGHT_AVERAGE_REMAINING_SIX',protocol=str(E/'protocol.json'),protocol_sha256=ph,at_unix=now),ensure_ascii=False)+'\n')
 print(json.dumps(dict(protocol_sha256=ph,cloud_jobs=5,local_jobs=1,deadline_unix=s['deadline_unix'])))
if __name__=='__main__':main()
