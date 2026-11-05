from pathlib import Path
import json,copy,hashlib,time
E=Path(__file__).resolve().parent;C=E.parent;REMOTE='/home/featurize/cuhkx_motion_transfer_20260911_v1/cloud_rethink_20260914'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
write=lambda p,v:p.write_text(json.dumps(v,indent=2),encoding='utf-8')
def main():
 p=read(C/'v2_window_cloud52910/protocol.json');qa=read(E/'CPU_QA.json');oldparent=C/'videomaev2_base_production/cloud';remote=REMOTE+'/'+E.name;pred=C/'v2_base_weight_average_completion/protocol.json'
 assert qa['status']=='PASS_FULL18_ORIGINAL_PARENT_AND_CONTROL_INTERFACE';h=copy.deepcopy(p['hosts']['cloud']);h.update(output_directory=remote+'/output/cloud',wall_cap_seconds=5400,job_wall_cap_seconds=5400,parent_checkpoint=remote+'/parent/model_fp16.pt')
 h['predecessor']=dict(directory=REMOTE+'/v2_base_weight_average_completion/output/cloud',protocol_file=REMOTE+'/v2_base_weight_average_completion/protocol.json',protocol_sha256=sha(pred),success_status='COMPLETE_WEIGHT_AVERAGE_COMPLETION')
 for n in ('model_fp16.pt','FROZEN.json','TRAIN_DONE.json','SPLIT.json'):h['input_bindings'][remote+'/parent/'+n]=sha(oldparent/n)
 h['input_bindings'][remote+'/CPU_QA.json']=sha(E/'CPU_QA.json')
 code={k:v for k,v in p['code_sha256'].items()if k not in ('v2_window_cloud52910/worker.py','v2_window_cloud52910/controller.py')}
 for n in ('worker.py','controller.py','freeze.py','deploy.py','resilient_io.py','cpu_qa.py'):code[E.name+'/'+n]=sha(E/n)
 for k in ('protected_local_bindings','registered_at_unix','deadline_unix'):p.pop(k,None)
 p.update(status='CPU_PREPARED_AWAITING_ROOT_LAUNCH_DECISION',id='FULL18_FRESH_CONTROL4_FIXED_MIDPOINT_RESERVE',prepared_at_unix=time.time(),execution_cap_seconds=5400,wall_cap_seconds=5400,queue_cap_seconds=3600,jobs=[dict(id='V2_BASE_FULL18_CONTROL4_RESERVE',host='cloud',fold=-1,train_users=qa['source_users'],held_users=[],outer_excluded_users=[],arm='CONTROL',kind='full18_reserve')],hosts={'cloud':h},code_sha256=code,worker_sha256=sha(E/'worker.py'),controller_sha256=sha(E/'controller.py'),hypothesis='Prepare full18 exact CONTROL4 and fixed raw16/control4 midpoint while complete development average is evaluated. This reserve is not a selected model and never uses Test.',parent_checkpoint_sha256=qa['parent_sha256'],parent_protocol_sha256=qa['parent_protocol_sha256'],parent_epochs=16,arm='CONTROL',average_alpha=.5,optimizer='freshAdamW',optimizer_updates=736,evaluation_points=[4],retained_aux_bytes=12656523,calibration_reserved_bytes=16000,no_Test=True,no_CSV=True,no_final_calibration=True,no_selection_claim=True,training_recipe='Exact videomaev2_base.worker.fit plus original WindowClips armCONTROL (epoch+16);8micro2accum=16batch;4epochs;freshoptimizer;whole-clip deterministic jitter/hflip',source_QA=dict(path=remote+'/CPU_QA.json',sha256=sha(E/'CPU_QA.json')),parent_manifest='Original saved full18 FP16, 18 Training users2931 clips; no held/Test',weight_average_metadata='Preserve epochs4 as freshCONTROL branch, explicit raw_epochs16,control_epochs4,freshAdamW,alpha.5,both protocols/both SHAs/full18users; never impersonate raw16.',budget='90min whole active stage including4epochtraining/checkpoints/average/storage/sourceQA; source completion predecessor and owned exit required',explicit_authorization='Root approved this exact full18CONTROL4 reserve and necessary172565307B full18parent transfer; not the prohibited F3INNER2 checkpoint.')
 write(E/'protocol.cloud.candidate.json',p)
 files=[]
 for f in sorted(E.glob('*.py')):files.append(dict(local=str(f),remote=remote+'/'+f.name,bytes=f.stat().st_size,sha256=sha(f)))
 for n in ('CPU_QA.json','protocol.cloud.candidate.json'):
  f=E/n;files.append(dict(local=str(f),remote=remote+'/'+n,bytes=f.stat().st_size,sha256=sha(f)))
 for n in ('model_fp16.pt','FROZEN.json','TRAIN_DONE.json','SPLIT.json'):
  f=oldparent/n;files.append(dict(local=str(f),remote=remote+'/parent/'+n,bytes=f.stat().st_size,sha256=sha(f)))
 write(E/'TRANSFER_MANIFEST.json',dict(candidate_sha256=sha(E/'protocol.cloud.candidate.json'),files=files,total_bytes=sum(x['bytes']for x in files),parent_bytes=172565307,no_raw_data=True,forbidden_F3INNER2_not_uploaded=True,authorization='Root explicitly approved exact full18CP transfer within broad user authorization',at_unix=time.time()))
 print(json.dumps(dict(status='EXACT_CLOUD_CONTROL4_CANDIDATE_READY',candidate_sha256=sha(E/'protocol.cloud.candidate.json'),parent_sha256=qa['parent_sha256'],total_transfer_bytes=sum(x['bytes']for x in files),predecessor_sha256=sha(pred),optimizer_updates=736)))
if __name__=='__main__':main()
