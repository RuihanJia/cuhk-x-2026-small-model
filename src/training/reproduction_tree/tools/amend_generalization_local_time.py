"""Remove the assistant-imposed local wall cap at a completed-epoch checkpoint."""
from pathlib import Path
from datetime import datetime
import argparse, ast, copy, json, os, subprocess, sys, time
import generalization_cloud as cloud

ROOT=cloud.ROOT; BASE=cloud.BASE; LOCAL=BASE/'local'; BUNDLE=LOCAL/'bundle'
sys.path.insert(0,str(BUNDLE))
from dual_host_runtime.io import atomic,read
from dual_host_runtime.health import alive,stop_owned
from research_runtime.common import sha

POLICY=BASE/'LOCAL_TIME_POLICY.json'
JOB='T_F0_outer'

def now(): return datetime.now().astimezone().isoformat(timespec='seconds')

def copy_changed(source,destination,replacements):
    text=(BUNDLE/source).read_text(encoding='utf-8')
    for before,after in replacements:
        assert text.count(before)==1,(source,before)
        text=text.replace(before,after)
    ast.parse(text); path=BUNDLE/destination
    assert not path.exists(); path.write_text(text,encoding='utf-8')

def prepare():
    assert not POLICY.exists()
    q=read(LOCAL/'QUEUE.json'); owner=read(LOCAL/'CONTROLLER.json')
    assert alive(owner) and q['deadline_unix']>0
    s=read(LOCAL/'QUEUE_STATUS.json'); assert set(s['active'])=={JOB}
    assert all(v['status']=='COMPLETE' for v in s['finished'].values())
    pending=[j for j in q['jobs'] if j['id'] not in s['finished']]
    assert {j['id'] for j in pending}=={JOB}|{f'IMU_R_{a}_F{f}' for a in 'BCD' for f in [0,3]}
    assert all(not (LOCAL/'output'/j['id']/'LAUNCH.json').exists() for j in pending if j['id']!=JOB)
    guard='''"""Optional local wall cap; inherited telemetry and stop handling remain active."""
from dual_host_runtime.health import Guard as TimedGuard

class Guard(TimedGuard):
    def bounds(self):
        if self.deadline > 0:
            return super().bounds()
        if any((p/'STOP_REQUESTED').exists() for p in [self.out,self.out.parent]):
            raise InterruptedError('STOP_REQUESTED')
'''
    path=BUNDLE/'optional_time_budget.py'; assert not path.exists(); ast.parse(guard); path.write_text(guard,encoding='utf-8')
    copy_changed('native_queue.py','native_queue_no_wall.py',[
        ('while time.time()<deadline:', 'while deadline<=0 or time.time()<deadline:')])
    copy_changed('generalization_thermal_worker_solo.py','generalization_thermal_worker_no_wall.py',[
        ('from dual_host_runtime.health import Guard','from optional_time_budget import Guard'),
        ('if epoch<20 and time.time()+','if args.deadline>0 and epoch<20 and time.time()+')])
    copy_changed('generalization_imu_followup_worker.py','generalization_imu_followup_no_wall.py',[
        ('from dual_host_runtime.health import Guard','from optional_time_budget import Guard')])
    amendments={}
    for name,newname,worker in [
        ('thermal_solo_protocol.json','thermal_no_wall_protocol.json','generalization_thermal_worker_no_wall.py'),
        ('imu_followup_protocol.json','imu_followup_no_wall_protocol.json','generalization_imu_followup_no_wall.py')]:
        old=read(LOCAL/name); p=copy.deepcopy(old)
        p['id']+='-LOCAL-NO-WALL-CAP'; p['execution_amended_at_local']=now()
        p['execution_parent_protocol_sha256']=sha(LOCAL/name)
        p['budget']['deadline_unix']=None
        for field in ['local_phase_cap_seconds','existing_local_phase_cap_seconds']:
            if field in p['budget']:p['budget'][field]=None
        p['budget'].pop('if_deadline_reached',None)
        p['budget']['extension']=True
        p['budget']['policy']='Fixed registered jobs and epochs; no local hard wall deadline; runtime sentinel0'
        p['execution_delta']=dict(change='Remove local wall deadline and time-forecast stopping only',
            reason='User clarified local compute need not have an exaggerated elapsed-time cutoff; original3h was an uncalibrated assistant estimate.',
            scientific_changes=False,concurrency=1,cloud_budget_changed=False)
        p['code_sha256']={x.relative_to(BUNDLE).as_posix():sha(x) for x in BUNDLE.rglob('*.py')}
        atomic(LOCAL/newname,p)
        amendments[name]=dict(protocol=newname,protocol_sha256=sha(LOCAL/newname),worker=worker)
    baseline=read(LOCAL/'output'/JOB/'RESUME.json')
    value=dict(event='E-GENERALIZATION-LOCAL-TIME-POLICY-20260910-V1',status='PREPARED_WAITING_FOR_NEXT_COMPLETE_EPOCH',
        prepared_at_local=now(),user_preference='本机应该不必设过于夸张的耗时截止时间',
        previous_local_wall_cap_seconds=10800,previous_local_deadline_unix=q['deadline_unix'],
        local_deadline_unix=None,local_wall_cap_enabled=False,
        scope='One ongoing Thermal outer fit to fixed20epochs, then six previously registered IMU B/C/D fits at fixed60epochs',
        no_automatic_new_variants=True,unchanged_guards=['STOP_REQUESTED','PID/create_time ownership','telemetry freshness','AC power','87Cwarning/90Cpause/85Cresume','nonfinite/AMP checks'],
        cloud_deadline_unix=read(BASE/'cloud/QUEUE.json')['deadline_unix'],cloud_budget_changed=False,
        queue_before_sha256=sha(LOCAL/'QUEUE.json'),controller_before=owner,baseline_completed_epoch=baseline['completed_epoch'],
        baseline_resume_sha256=baseline['sha256'],amendments=amendments,
        original_estimate_source='runs/generalization_review_20260910_v1/REPORT.md:94-104')
    atomic(POLICY,value)
    print(json.dumps(value,ensure_ascii=False))

def apply():
    p=read(POLICY); assert p['status']=='PREPARED_WAITING_FOR_NEXT_COMPLETE_EPOCH'
    assert sha(LOCAL/'QUEUE.json')==p['queue_before_sha256']
    owner=read(LOCAL/'CONTROLLER.json'); assert owner==p['controller_before'] and alive(owner)
    resume=read(LOCAL/'output'/JOB/'RESUME.json'); assert resume['completed_epoch']>p['baseline_completed_epoch']
    assert resume['completed_epoch']<20
    for v in p['amendments'].values():
        assert sha(LOCAL/v['protocol'])==v['protocol_sha256']
        for name,expected in read(LOCAL/v['protocol'])['code_sha256'].items():assert sha(BUNDLE/name)==expected
    q=read(LOCAL/'QUEUE.json'); s=read(LOCAL/'QUEUE_STATUS.json')
    assert set(s['active'])=={JOB} and not any(v['status']!='COMPLETE' for v in s['finished'].values())
    atomic(LOCAL/'QUEUE_BEFORE_NO_WALL_CAP.json',q);atomic(LOCAL/'QUEUE_STATUS_BEFORE_NO_WALL_CAP.json',s)
    stop_owned(owner); assert not alive(owner)
    resume=read(LOCAL/'output'/JOB/'RESUME.json')
    assert sha(LOCAL/'output'/JOB/'resume.pt')==resume['sha256']
    atomic(LOCAL/'output'/JOB/'RESUME_AT_TIME_POLICY_TRANSITION.json',resume)
    for j in q['jobs']:
        if j['id'] in s['finished']:continue
        assert j['protocol'] in p['amendments']
        j.update(p['amendments'][j['protocol']])
    q['deadline_unix']=0;q['local_wall_cap_enabled']=False;q['execution_amendment']='../LOCAL_TIME_POLICY.json'
    s['active']={};s['deadline_unix']=0;s['status']='LOCAL_TIME_POLICY_REGISTERED'
    atomic(LOCAL/'QUEUE.json',q);atomic(LOCAL/'QUEUE_STATUS.json',s)
    for name in ['QUEUE_ERROR.json','QUEUE_DONE.json']:
        path=LOCAL/name
        if path.exists():atomic(LOCAL/(path.stem+'_BEFORE_NO_WALL_CAP.json'),read(path));path.unlink()
    command=[sys.executable,'-X','utf8','-B',str(BUNDLE/'native_queue_no_wall.py'),'--root',str(LOCAL)]
    with (LOCAL/'controller_no_wall.stdout.log').open('a',encoding='utf-8') as out,(LOCAL/'controller_no_wall.stderr.log').open('a',encoding='utf-8') as err:
        proc=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
    p.update(status='APPLIED_CONTROLLER_STARTED',applied_at_local=now(),retained_completed_epoch=resume['completed_epoch'],
        retained_checkpoint_sha256=resume['sha256'],retained_identity_hash=resume['identity_sha256'],new_controller_spawn_pid=proc.pid,
        controller_command=command,scientific_changes=False)
    atomic(POLICY,p)
    with (ROOT/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(p,ensure_ascii=False)+'\n')
    state=read(ROOT/'PROJECT_STATE.json');auth=state['current_authorization']['generalization_training']
    auth['local_phase_max_seconds']=None
    auth['local_time_policy_amendment']=dict(at_local=p['applied_at_local'],previous_seconds=10800,
        user_preference=p['user_preference'],current='Finite registered fits/epochs without a local hard wall deadline',
        evidence='runs/generalization_training_20260910_v1/LOCAL_TIME_POLICY.json')
    for run in state['active_runs']:
        if run['host']=='local':run['deadline_unix']=None;run['kind']='THERMAL_OUTER_THEN_IMU_SECOND_SEED_AND_DEVICE_DROPOUT'
    local=state['runtime']['local'];local['wall_time_cap_enabled']=False;local['deadline_unix']=None
    local['status']='RESUMING_FIXED_EPOCH_LOCAL_QUEUE_WITHOUT_WALL_CAP'
    local['time_policy']='runs/generalization_training_20260910_v1/LOCAL_TIME_POLICY.json'
    local['execution_protocol']='runs/generalization_training_20260910_v1/local/thermal_no_wall_protocol.json'
    local['work_type']='Six inner Thermal fits complete;T_F0_outer completes20epochs;then six registered IMU second-seed B/C/D fits at60epochs.'
    state['successor_preparation'].update(status='LOCAL_IMU_FOLLOWUP_REGISTERED_AND_QUEUED',
        local_imu_followup_basis='Positive B/A and opposing C/B fold signals; B/C second seed and fixed extra-device dropoutD are now frozen before fitting.',
        local_imu_followup_protocol='runs/generalization_training_20260910_v1/local/imu_followup_no_wall_protocol.json',
        next='Local: finish Thermal outer and registered IMU B/C/D followup, audit final FP32 predictions. Cloud: prepare registered inner-only CPU fusion before video/sensor/Thermal queue tail. Use measured ETA for scheduling, never local wall time as a scientific verdict.')
    state['runtime']['automation']['last_check_disposition']='LOCAL_TIME_POLICY_AMENDED_AT_USER_REQUEST;REGISTERED_IMU_FOLLOWUP_QUEUED'
    state['runtime']['automation']['last_check_artifact']='runs/generalization_training_20260910_v1/LOCAL_TIME_POLICY.json'
    atomic(ROOT/'PROJECT_STATE.json',state)
    print(json.dumps({k:p[k] for k in ['status','applied_at_local','retained_completed_epoch','new_controller_spawn_pid','local_deadline_unix','cloud_budget_changed']},ensure_ascii=False))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','apply']);args=a.parse_args()
    prepare() if args.mode=='prepare' else apply()
