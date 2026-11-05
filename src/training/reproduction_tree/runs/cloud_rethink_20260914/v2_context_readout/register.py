"""Root-authorized single 60min context study registration; no device work."""
from pathlib import Path
import ast,hashlib,json,time
E=Path(__file__).resolve().parent;C=E.parent;ROOT=C.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
REMOTE='/home/featurize/cuhkx_motion_transfer_20260911_v1/cloud_rethink_20260914'
def main():
 assert not(E/'protocol.json').exists()
 s=read(E/'protocol_candidate.json');assert s['status']=='PREPARED_NOT_EXECUTABLE'
 for p in E.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
 assert read(E/'CPU_QA.json')['status']=='PASS_CONTEXT_SPATIAL_CPU_QA'
 assert read(E/'INTERFACE_QA.json')['status']=='PASS_CONTEXT_CPU_INTERFACE'
 replay=C/'v2_depth_residual_pilot/local_replay_protocol.json';r=read(replay)
 assert sha(replay)=='35a4f80de42f38cca7c1243559343402100cbb526797b82f4ab949e3dc9a0820'
 s['hosts']['local']['predecessor']=dict(directory=r['hosts']['local']['output_directory'],protocol_sha256=sha(replay),success_only=True)
 s['local_predecessor_engineering_note']='Original Depth fit completed; original evaluation failed before forward. Wait exact zero-update local replay success and both owned processes absent; never launch after original failure alone.'
 for n in ('data.py','worker.py','quant_identity.py','controller.py','resilient_io.py','watch.py','launch.py'):
  s['code_sha256']['v2_context_readout/'+n]=sha(E/n)
 for n in ('calibrate.py','classwise4.py'):s['calibration_code_sha256']['v2_context_readout/'+n]=sha(E/n)
 s['interface_QA_sha256']=sha(E/'INTERFACE_QA.json');s['worker_sha256']=sha(E/'worker.py');s['controller_sha256']=sha(E/'controller.py')
 s['registered_at_unix']=time.time();s['deadline_unix']=s['registered_at_unix']+3600;s['wall_cap_seconds']=3600;s['status']='FROZEN_FOR_EXECUTION'
 s['resource_authorization']='Root forwarded current user explicit local/cloud authorization; single finite60min stage including queues, two host readouts and CPU source calibration. No checkpoint upload/no Test/no CSV.'
 for p,h in s['code_sha256'].items():assert sha(C/p)==h
 for p,h in s['calibration_code_sha256'].items():assert sha(C/p)==h
 checks=dict(s['hosts']['cloud']['input_bindings']);checks.update({REMOTE+'/'+p:h for p,h in s['code_sha256'].items()})
 upload=read(E/'UPLOAD_CANDIDATE.json')['files'];assert read(E/'UPLOAD_CANDIDATE.json')['no_checkpoint_upload']
 for j in s['jobs']:
  if j['host']=='cloud':
   checks[j['parent_directory']+'/model_fp16.pt']=j['parent_fp16_sha256']
   for n,k in [('TRAIN_DONE.json','parent_train_done_sha256'),('FROZEN.json','parent_frozen_sha256'),('SPLIT.json','parent_split_sha256')]:checks[j['parent_directory']+'/'+n]=j[k]
   checks[j['identity_receipt']]=j['identity_receipt_sha256']
 # Bind all existing dependency hashes. Only missing small source/code metadata may transfer.
 def mapped(p):
  prefix='/home/featurize/cuhkx_motion_transfer_20260911_v1/'
  if p.startswith(prefix):return C.parent/p[len(prefix):].replace('/output/cloud','/cloud')
  raise AssertionError(p)
 for name,h in checks.items():
  if name in upload:continue
  local=mapped(name)
  if local.exists() and local.stat().st_size<2000000 and local.suffix in ('.py','.json','.txt',''):
   assert sha(local)==h,name;upload[name]=dict(local_path=str(local),sha256=h,bytes=local.stat().st_size)
 for p in E.glob('*.py'):upload[REMOTE+'/v2_context_readout/'+p.name]=dict(local_path=str(p),sha256=sha(p),bytes=p.stat().st_size)
 for n in ('CPU_QA.json','INTERFACE_QA.json'):
  p=E/n;upload[REMOTE+'/v2_context_readout/'+n]=dict(local_path=str(p),sha256=sha(p),bytes=p.stat().st_size)
 with(E/'protocol.json').open('x',encoding='utf-8')as f:json.dump(s,f,ensure_ascii=False,indent=2)
 p=E/'protocol.json';upload[REMOTE+'/v2_context_readout/protocol.json']=dict(local_path=str(p),sha256=sha(p),bytes=p.stat().st_size)
 assert all(v['bytes']<2000000 and Path(v['local_path']).suffix not in ('.pt','.pth','.npz') for v in upload.values())
 (E/'TRANSFER_MANIFEST.json').write_text(json.dumps(dict(status='FROZEN_SMALL_CODE_METADATA_ONLY',protocol_sha256=sha(p),files=upload,existing_checks=checks,no_raw_data=True,no_checkpoint_upload=True),ensure_ascii=False,indent=2),encoding='utf-8')
 event=dict(event='V2_CONTEXT_1P5_READOUT_REGISTERED',protocol=str(p.relative_to(ROOT)),protocol_sha256=sha(p),deadline_unix=s['deadline_unix'],jobs8=True,optimizer_updates=0,no_checkpoint_upload=True,no_Test=True,at_unix=time.time())
 with(ROOT/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf-8')as f:f.write(json.dumps(event)+'\n')
 print(json.dumps(event))
if __name__=='__main__':main()
