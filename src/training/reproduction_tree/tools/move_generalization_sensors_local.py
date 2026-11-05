"""Repair the omitted accepted model module and move only unstarted sensor fits to the free local GPU."""
from pathlib import Path
from datetime import datetime
import ast,copy,importlib.util,json,shutil,subprocess,sys,time
import torch
import generalization_cloud as cloud
ROOT=cloud.ROOT;BASE=cloud.BASE;LOCAL=BASE/'local';CLOUD=BASE/'cloud';BUNDLE=LOCAL/'bundle';sys.path.insert(0,str(BUNDLE))
from dual_host_runtime.io import atomic,read
from dual_host_runtime.health import alive
from research_runtime.common import sha

def main():
    target=LOCAL/'sensor_local_protocol.json';assert not target.exists();assert not alive(read(LOCAL/'CONTROLLER.json'))
    old_done=read(LOCAL/'QUEUE_DONE.json');assert old_done['status']=='COMPLETE' and len(old_done['finished'])==27
    source=ROOT/'deliverables/mvit_sensor_20260908/sensor_model.py';spec=importlib.util.spec_from_file_location('accepted_sensor_model',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    torch.manual_seed(20260909);model=module.SensorTemporal();cp=torch.load(ROOT/'deliverables/mvit_sensor_20260908/checkpoints/sensor.pt',map_location='cpu',weights_only=True);model.load_state_dict(cp['state_dict'],strict=True);parameters=sum(v.numel() for v in model.parameters());del cp,model
    vendor=BUNDLE/'research_runtime/vendor';assert not vendor.exists();vendor.mkdir();(vendor/'__init__.py').write_text('',encoding='utf-8');shutil.copy2(source,vendor/'sensor.py')
    data=CLOUD/'bundle/data/measured_sensors.npz';shutil.copy2(data,BUNDLE/'data/measured_sensors.npz');assert sha(BUNDLE/'data/measured_sensors.npz')==sha(data)
    text=(CLOUD/'bundle/generalization_cloud_worker.py').read_text(encoding='utf-8');before='from dual_host_runtime.health import Guard';assert text.count(before)==1;text=text.replace(before,'from optional_time_budget import Guard')
    before="p.update(_sha=sha(args.protocol),_bundle=str(root/'bundle'))";assert text.count(before)==1;text=text.replace(before,"p.update(_sha=p.get('scientific_parent_sha256',sha(args.protocol)),_bundle=str(root/'bundle'))")
    ast.parse(text);worker=BUNDLE/'generalization_sensor_local.py';worker.write_text(text,encoding='utf-8');shutil.copy2(ROOT/'tools/generalization_sensor_publish.py',BUNDLE/'generalization_sensor_publish.py')
    original=read(CLOUD/'sensor_protocol.json');p=copy.deepcopy(original);p['id']='GENERALIZATION-MATCHED-SENSOR-LOCAL-DEPENDENCY-REPAIR-20260911-V1';p['scientific_parent_sha256']=sha(CLOUD/'sensor_protocol.json');p['jobs']={k:v for k,v in original['jobs'].items() if k.startswith('S_')};p['execution_amended_at_local']=datetime.now().astimezone().isoformat(timespec='seconds')
    assert len(p['jobs'])==8;p['budget']=dict(deadline_unix=None,local_wall_cap_enabled=False,max_new_fits=8,epochs=60,concurrency=2,expected_local_minutes='5-12',cloud_phase_deadline_unchanged=read(CLOUD/'QUEUE.json')['deadline_unix'])
    p['model_source']=dict(path=source.relative_to(ROOT).as_posix(),sha256=sha(source),parameters=parameters,accepted_serialized_checkpoint_strict_load=True)
    p['execution_delta']=dict(reason='Cloud import preflight confirmed missing research_runtime.vendor before any S launch. Reuse the existing accepted SensorTemporal definition, and use the free local GPU for already-registered jobs.',scientific_recipe_changed=False,host_changed=True,bitwise_cross_host_retraining_claim=False,existing_completed_jobs_retrained=False)
    p['code_sha256']={v.relative_to(BUNDLE).as_posix():sha(v) for v in BUNDLE.rglob('*.py')};atomic(target,p);ph=sha(target)
    publish=LOCAL/'sensor_publish_protocol.json';pp=dict(id='GENERALIZATION-LOCAL-SENSOR-PUBLISH-20260911-V1',source_jobs=list(p['jobs']),source_scientific_protocol_sha256=p['scientific_parent_sha256'],code_sha256=p['code_sha256']);atomic(publish,pp)
    m,c=cloud.access.connection()
    try:
        code=f'''import json,time,os,psutil
from pathlib import Path
root=Path({cloud.REMOTE!r});q=json.loads((root/'QUEUE.json').read_text());s=json.loads((root/'QUEUE_STATUS.json').read_text());keys={list(p['jobs'])!r}
assert set(s['active'])=={{'V_F0_outer','V_F3_outer'}},'Cloud work has changed; recheck before reallocating'
for key,v in s['active'].items():
 owner=v['owner'];process=psutil.Process(owner['pid']);assert abs(process.create_time()-owner['create_time'])<.01 and process.is_running()
for key in keys:assert not (root/'output'/key/'LAUNCH.json').exists(),key
assert {{j['id'] for j in q['jobs'] if j['id'].startswith('S_')}}==set(keys)
(root/'QUEUE_BEFORE_LOCAL_SENSOR_HANDOFF.json').write_text(json.dumps(q,indent=2));q['jobs']=[j for j in q['jobs'] if j['id'] not in keys]
tmp=root/'QUEUE.json.sensor.partial';tmp.write_text(json.dumps(q,indent=2));os.replace(tmp,root/'QUEUE.json')
print(json.dumps(dict(status='ONLY_EIGHT_UNSTARTED_SENSOR_JOBS_REMOVED',jobs=keys,active_preserved=list(s['active']),at_unix=time.time())))'''
        removed=cloud.access.execute(m,c,code)
    finally:c.close()
    cq=read(CLOUD/'QUEUE.json');cq['jobs']=[j for j in cq['jobs'] if j['id'] not in p['jobs']];atomic(CLOUD/'QUEUE.json',cq)
    fact=dict(event='E-GENERALIZATION-SENSOR-LOCAL-HANDOFF-20260911-V1',at_local=p['execution_amended_at_local'],status='REGISTERED_UNSTARTED_JOBS_REALLOCATED',protocol=target.relative_to(ROOT).as_posix(),protocol_sha256=ph,scientific_protocol_sha256=p['scientific_parent_sha256'],source_sha256=sha(source),parameters=parameters,cloud_removal=removed,max_new_fits=8,epochs=60,seed=20260909,cloud_phase_extended=False)
    atomic(BASE/'SENSOR_LOCAL_HANDOFF.json',fact)
    with (ROOT/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(fact,ensure_ascii=False)+'\n')
    atomic(LOCAL/'QUEUE_DONE_BEFORE_SENSOR_HANDOFF.json',old_done);(LOCAL/'QUEUE_DONE.json').unlink();q=read(LOCAL/'QUEUE.json');q['concurrency']=2
    q['jobs'].extend(dict(id=k,worker=worker.name,protocol=target.name,protocol_sha256=ph) for k in p['jobs'])
    q['jobs'].append(dict(id='SENSOR_PUBLISH',worker='generalization_sensor_publish.py',protocol=publish.name,protocol_sha256=sha(publish),requires=[f'output/{k}/DONE.json' for k in p['jobs']]))
    atomic(LOCAL/'QUEUE.json',q)
    with (LOCAL/'controller_sensor.stdout.log').open('a',encoding='utf-8') as out,(LOCAL/'controller_sensor.stderr.log').open('a',encoding='utf-8') as err:
        proc=subprocess.Popen([sys.executable,'-X','utf8','-B',str(BUNDLE/'native_queue_no_wall.py'),'--root',str(LOCAL)],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
    state=read(ROOT/'PROJECT_STATE.json');state['runtime']['local'].update(status='STARTED_EIGHT_MATCHED_SENSOR_FITS_AFTER_DEPENDENCY_REPAIR',current_gpu_training=True,execution_protocol=target.relative_to(ROOT).as_posix(),work_type='Eight previously unstarted matched sensor fits moved from cloud; two concurrent local jobs; native final publisher sends checked outputs for cloud fusion.')
    state['successor_preparation'].update(status='LOCAL_SENSOR_HANDOFF_ACTIVE_FUSION_PREFLIGHT_REPAIR',next='Complete local S8 fits and native artifact publication; finish fusion sample-ID alignment preflight and gate cloud CPU calibration on all required completed sources.',sensor_local_handoff='runs/generalization_training_20260910_v1/SENSOR_LOCAL_HANDOFF.json')
    atomic(ROOT/'PROJECT_STATE.json',state);print(json.dumps(dict(**fact,controller_spawn_pid=proc.pid),ensure_ascii=False))

if __name__=='__main__':main()
