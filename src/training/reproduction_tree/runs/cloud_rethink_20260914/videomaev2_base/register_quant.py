"""Register fixed codec replay, queued behind the already managed training jobs."""
from pathlib import Path
import ast,copy,hashlib,json,shutil,time
W=Path(__file__).resolve().parent;C=W.parent;R=C.parents[1];O=C.parent/'error_resolution_20260913';S=C/'videomaev2_base_source';E=C/'videomaev2_base_quant';RC='/home/featurize/cuhkx_motion_transfer_20260911_v1/cloud_rethink_20260914';E.mkdir(exist_ok=True)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert not(E/'protocol.json').exists();parent=read(W/'protocol.json');sources=read(S/'protocol.json');assert (S/'LAUNCH.json').exists()
worker=(C/'videomaev2_quant/worker.py').read_text(encoding='utf-8').replace("W=C/'videomaev2_small'","W=C/'videomaev2_base'").replace('VideoMAESmall','VideoMAEBase').replace('stat().st_size<24000000','stat().st_size<87000000');(E/'worker.py').write_text(worker,encoding='utf-8')
for n in ['controller.py','resilient_io.py']:shutil.copyfile(C/'videomaev2_quant'/n,E/n)
shutil.copyfile(C/'videomae_base_quant/codec.py',E/'codec.py')
(E/'launch.py').write_text((C/'videomaev2_quant/launch.py').read_text(encoding='utf-8').replace('videomaev2_quant','videomaev2_base_quant').replace("state['active_runs']=[", "state['active_runs']=[r for r in state['active_runs'] if r['host']=='local' and r['stage']=='videomaev2_base']+ ["),encoding='utf-8')
(E/'watch.py').write_text((W/'watch.py').read_text(encoding='utf-8'),encoding='utf-8')
s=read(C/'videomae_base_quant/protocol.json');now=time.time();s.update(id='V2_BASE_FIXED50_INT8_TWO_OUTER_SIX_SOURCE_REPLAY',registered_at_unix=now,deadline_unix=now+2820,wall_cap_seconds=1800,jobs=[],hosts={},hypothesis='V2Base matched private918 warrants actual storage validation. Reuse already validated Base 50-tensor output-row INT8/FP32 scales codec exactly; no layer/precision search. Finish both outer and all six source replays before calibrated promotion. Local three queued behind existing source fit; cloud five queued behind five source fits.',resource_note='Zero optimizer updates. Cloud host45min including prior-fit wait; local30min including prior-fit wait; global47min. Each managed predecessor must be complete and owned worker dead. Original training budgets unchanged.',seed=20260915,optimizer_updates=0,no_Test=True,no_full18=True)
inp=read(O/'POSE_ADAPTER_INPUTS.json')
for p in parent['jobs']:
 source=p['kind']=='source_inner';j={k:p[k] for k in ['id','fold','train_users','held_users']};j.update(host='local',scope='source' if source else 'outer',outer_excluded_users=p.get('outer_excluded_users',[]),parent_directory=str(W/('output/local' if source else 'cloud')/p['id']),parent_protocol_sha256=sha(W/'protocol.json'),records=sum(r['user'] in p['held_users'] for r in inp['rows']))
 if source:j['inner']=p['inner']
 s['jobs'].append(j)
for p in sources['jobs']:
 j={k:p[k] for k in ['id','fold','inner','train_users','held_users','outer_excluded_users']};j.update(host='cloud',scope='source',parent_directory=sources['hosts']['cloud']['output_directory']+'/'+p['id'],parent_protocol_sha256=sha(S/'protocol.json'),records=sum(r['user'] in p['held_users'] for r in inp['rows']));s['jobs'].append(j)
assert len(s['jobs'])==8 and sum(j['host']=='local' for j in s['jobs'])==3
for host in ['local','cloud']:
 h=copy.deepcopy(parent['hosts'][host]);h.update(output_directory=str(E/'output/local') if host=='local' else RC+'/'+E.name+'/output/cloud',wall_cap_seconds=1800 if host=='local' else 2700)
 h.pop('retired_precheck',None)
 previous=parent if host=='local' else sources;stage=W if host=='local' else S
 h['predecessor']=dict(directory=previous['hosts'][host]['output_directory'],protocol_sha256=sha(stage/'protocol.json'),success_only=True)
 h['input_bindings'][str(W/'protocol.json') if host=='local' else RC+'/videomaev2_base/protocol.json']=sha(W/'protocol.json')
 if host=='cloud':h['input_bindings'][RC+'/videomaev2_base_source/protocol.json']=sha(S/'protocol.json')
 s['hosts'][host]=h
s['code_sha256']={k:v for k,v in parent['code_sha256'].items() if not k.endswith('controller.py') and not k.endswith('worker.py')}
for n in ['worker.py','controller.py','codec.py','resilient_io.py']:
 p=E/n;ast.parse(p.read_text(encoding='utf-8'));s['code_sha256'][p.relative_to(C).as_posix()]=sha(p)
s['worker_sha256']=sha(E/'worker.py');s['controller_sha256']=sha(E/'controller.py')
for n in ['launch.py','watch.py']:ast.parse((E/n).read_text(encoding='utf-8'))
write(E/'protocol.json',s)
files=[dict(name=n,bytes=(E/n).stat().st_size,sha256=sha(E/n)) for n in ['protocol.json','worker.py','controller.py','codec.py','resilient_io.py']]
write(E/'TRANSFER_MANIFEST.json',dict(status='READY',files=files,total_bytes=sum(v['bytes'] for v in files),remote_directory=RC+'/'+E.name,no_raw_data=True,no_credentials=True,at_unix=now))
ap=C/'post_aux_cloud_access.py';a=ap.read_text(encoding='utf-8');old="'videomaev2_base_source/protocol.json']";assert old in a;ap.write_text(a.replace(old,"'videomaev2_base_source/protocol.json','videomaev2_base_quant/protocol.json']"),encoding='utf-8')
with(R/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(event='V2_BASE_FIXED50_INT8_EIGHT_REPLAYS_REGISTERED',at_unix=now,protocol_sha256=sha(E/'protocol.json'),protocol=str((E/'protocol.json').relative_to(R)),deadline_unix=s['deadline_unix'],optimizer_updates=0))+'\n')
state=read(R/'PROJECT_STATE.json');state['current_authorization']['active_stage_protocols']+=[str((E/'protocol.json').relative_to(R))];state['current_authorization']['active_stage_deadline_unix']=s['deadline_unix'];state['next_frozen_candidate']=str((E/'protocol.json').relative_to(R));write(R/'PROJECT_STATE.json',state)
print(json.dumps(dict(protocol_sha256=sha(E/'protocol.json'),deadline_unix=s['deadline_unix'],bytes=sum(v['bytes'] for v in files),local_jobs=[j['id'] for j in s['jobs'] if j['host']=='local'])))
