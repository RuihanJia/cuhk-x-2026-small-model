from pathlib import Path
import ast,copy,json,time,hashlib
E=Path(__file__).resolve().parent;C=E.parent;R='/home/featurize/cuhkx_motion_transfer_20260911_v1/cloud_rethink_20260914'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):assert not p.exists();p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def main():
 assert read(E/'CPU_QA.json')['status']=='PASS_MIRROR_DUAL_SOURCE_PIXELS_AND_LOGITS';old=read(C/'v2_base6_small8_source_completion/protocol.json');inp=read(old['inputs_file']);prior=read(C/'v2_hand_readout_source/protocol.json');jobs=copy.deepcopy(inp['jobs']);uploads={}
 def add(p,target):
  p=Path(p);uploads[target]=dict(local_path=str(p),sha256=sha(p),bytes=p.stat().st_size)
 for j in jobs:
  j['host']='local'if j['id']=='V2_BASE_SOURCE_F3_INNER2'else'cloud';host=j['host'];cp=Path(j.get('container',{}).get('path',C/'v2_base6_small_design/BASE6_SMALL8_F3_UNEVALUATED.bin'));j['container']=dict(path=str(cp)if host=='local'else R+'/'+E.name+'/containers/'+j['id']+'.bin',local_path=str(cp),sha256=sha(cp),bytes=cp.stat().st_size)
  root=Path(old['pilot_directory'])if j['reuse_pilot']else C/'v2_base6_small8_source_completion/output/local'/j['id'];done=read(root/'DONE.json');j['original']={}
  for kind in ['Base','Small']:
   p=root/(kind+'.npz');assert sha(p)==done['artifacts'][p.name];target=str(p)if host=='local'else R+'/'+E.name+'/original/'+j['id']+'/'+p.name
   j['original'][kind]=dict(path=target,local_path=str(p),sha256=sha(p),bytes=p.stat().st_size,done_sha256=sha(root/'DONE.json'),native_protocol_sha256=done['protocol_sha256'])
   if host=='cloud':add(p,target)
 jobs.sort(key=lambda j:(j['scope']!='source',j['fold'],j['inner']or 0))
 hosts=copy.deepcopy(prior['hosts'])
 for host,h in hosts.items():
  h.pop('hand_inputs',None);h['input_bindings']={k:v for k,v in h['input_bindings'].items()if 'HAND_ADAPTER'not in k};h['workers']=2 if host=='local'else 4;h['eval_micro']=3 if host=='local'else 7
  h['output_directory']=str(E/'output/local')if host=='local'else R+'/'+E.name+'/output/cloud'
  name='v2_temporal_residual_source_pilot'if host=='local'else'v2_temporal_residual_source_completion';pp=C/name/('protocol.local.json'if host=='local'else'protocol.json')
  h['predecessor']=dict(directory=str(C/name/'output/local')if host=='local'else R+'/'+name+'/output/cloud',protocol_sha256=sha(pp),success_only=True,success_status='COMPLETE_TEMPORAL_RESIDUAL_MATCHED4_TWO_SOURCE_PILOT'if host=='local'else'COMPLETE_TEMPORAL_RESIDUAL_SIX_REMAINING_SPLITS')
 names=[p.relative_to(C).as_posix()for p in E.glob('*.py')if p.name not in ['inventory.py','transfer_pairs.py']]
 for rel in ['../error_resolution_20260913/bundle/dual_host_runtime/io.py','../error_resolution_20260913/bundle/dual_host_runtime/health.py','../error_resolution_20260913/bundle/motion_video_runtime/data.py','../error_resolution_20260913/bundle/research_runtime/common.py']:names.append(rel)
 codes={n:sha(C/n)for n in names}
 for n in names:ast.parse((C/n).read_text(encoding='utf-8-sig'));add(C/n,str(Path(R)/n).replace('\\','/'))
 now=time.time();s=dict(status='FROZEN_FOR_EXECUTION',id='RTN6_BASE7_SMALL8_FIVE_FIXED_EQUAL_MIRROR_TTA_ALL8',registered_at_unix=now,deadline_unix=now+5400,wall_cap_seconds=5400,source_authorization_sha256=old['source_authorization_sha256'],explicit_authorization='Root authorized 8 exactoriginalPAIR readouts;cloud7/localF3INNER2;onlymirrorforwards;allfourcal frozen beforeouter;noTest/training/CSV.',hypothesis='Training50percenthorizontalflip is absent from current inference. Equalmirror TTA may reduce orientation sensitivity without extra weights.',optimizer_updates=0,seed=20260915,jobs=jobs,hosts=hosts,code_sha256=codes,worker_sha256=codes[E.name+'/worker.py'],controller_sha256=codes[E.name+'/controller.py'],cpu_qa_sha256=sha(E/'CPU_QA.json'),readout=dict(Base='Original7:0.5mean5+0.25early+0.25late;onlyfixedhorizontalmirrorofsameinputs',Small='Original5meanonly;neveraddhalves',mix='FP32(0.5originalFP32aggregate+0.5mirroredFP32aggregate)',TF32=False,training=False),source_recipe=dict(branches=['T','I','Base6mirror_equal','Small8mirror_equal'],global_iterations=500,classwise_iterations=500,centered_delta_penalty=.1,all4calfreeze_beforeouter=True),no_Test=True,no_CSV=True,no_Public_fit=True,original_protocol_sha256=sha(C/'v2_base6_small8_source_completion/protocol.json'),source_oof=old['source_oof'],source_oof_sha256=old['source_oof_sha256'],outer_users=old['outer_users'],pose_inputs_sha256=prior['pose_inputs_sha256'])
 write(E/'protocol.json',s)
 for n in ['protocol.json','CPU_QA.json']:add(E/n,R+'/'+E.name+'/'+n)
 write(E/'TRANSFER_MANIFEST.json',dict(files=uploads,existing_checks=hosts['cloud']['input_bindings'],pair_manifest_sha256=sha(E/'PAIR_TRANSFER_MANIFEST.json'),pairs_separate_byte_exact_streaming=True));print(json.dumps(dict(protocol_sha256=sha(E/'protocol.json'),metadata_transfer_bytes=sum(v['bytes']for v in uploads.values()))))
if __name__=='__main__':main()
