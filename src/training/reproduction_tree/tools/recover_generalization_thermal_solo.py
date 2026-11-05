"""Resume the budget-stopped outer fit using fresh timing after its competing job has finished."""
from pathlib import Path
from datetime import datetime
import ast,copy,json,sys,time
import generalization_cloud as cloud
ROOT=cloud.ROOT;BASE=cloud.BASE;OUT=BASE/'local';BUNDLE=OUT/'bundle';sys.path.insert(0,str(BUNDLE))
from dual_host_runtime.io import atomic,read
from dual_host_runtime.health import alive,stop_owned
from research_runtime.common import sha
def main():
    job='T_F0_outer';done=read(OUT/'QUEUE_DONE.json');assert done['status']=='COMPLETED_WITH_FAILURES';error=read(OUT/'output'/job/'ERROR.json');assert 'THERMAL_FIXED_EPOCHS_EXCEED_REMAINING_CAP' in error['error'];resume=read(OUT/'output'/job/'RESUME.json');assert resume['completed_epoch']==1;assert sha(OUT/'output'/job/'resume.pt')==resume['sha256'];assert not (OUT/'thermal_solo_protocol.json').exists()
    source=BUNDLE/'generalization_thermal_worker_io2.py';text=source.read_text(encoding='utf-8');old="np.median([r['seconds'] for r in history[-3:]])";assert text.count(old)==1;text=text.replace('        for epoch in range(len(history)+1,21):','        forecast_start=len(history)\n        for epoch in range(len(history)+1,21):').replace(old,"np.median([r['seconds'] for r in history[forecast_start:][-3:]])");worker=BUNDLE/'generalization_thermal_worker_solo.py';ast.parse(text);worker.write_text(text,encoding='utf-8')
    p=copy.deepcopy(read(OUT/'thermal_io2_protocol.json'));p['id']='GENERALIZATION-THERMAL-F0-OUTER-SOLO-RECOVERY-20260910-V1';p['at_local']=datetime.now().astimezone().isoformat(timespec='seconds');p['jobs']={job:p['jobs'][job]};p['code_sha256']={x.relative_to(BUNDLE).as_posix():sha(x) for x in BUNDLE.rglob('*.py')};p['execution_delta']=dict(change='After resume, estimate remaining time only from new epochs on the now-uncontended GPU; keep absolute deadline and60second prediction reserve',reason='The first epoch170.66seconds was measured concurrently with the last inner fit, which is now complete; do not extrapolate that contention to all remaining epochs.',scientific_changes=False,completed_epoch_retained=1,concurrency=1,budget_extension=False)
    atomic(OUT/'thermal_solo_protocol.json',p);ph=sha(OUT/'thermal_solo_protocol.json');owner=read(OUT/'CONTROLLER.json')
    if alive(owner):stop_owned(owner)
    assert not alive(owner);atomic(OUT/'QUEUE_DONE_BEFORE_SOLO_RECOVERY.json',done);s=read(OUT/'QUEUE_STATUS.json');atomic(OUT/'QUEUE_STATUS_BEFORE_SOLO_RECOVERY.json',s);s['active']={};s['finished']={k:v for k,v in s['finished'].items() if k!=job};s['status']='SOLO_RECOVERY_REGISTERED';atomic(OUT/'QUEUE_STATUS.json',s)
    q=read(OUT/'QUEUE.json');q['concurrency']=1
    for item in q['jobs']:
        if item['id']==job:item.update(worker=worker.name,protocol='thermal_solo_protocol.json',protocol_sha256=ph)
    q['execution_amendment']='thermal_solo_protocol.json';atomic(OUT/'QUEUE.json',q);(OUT/'QUEUE_DONE.json').unlink();atomic(OUT/'output'/job/'INITIAL_BUDGET_PROJECTION_ERROR.json',error);(OUT/'output'/job/'ERROR.json').unlink()
    fact=dict(event='E-GENERALIZATION-THERMAL-SOLO-RECOVERY-20260910-V1',at_local=p['at_local'],status='FROZEN_READY_FOR_SAME_CHECKPOINT_RESUME',job=job,retained_epoch=1,retained_checkpoint_sha256=resume['sha256'],protocol='runs/generalization_training_20260910_v1/local/thermal_solo_protocol.json',protocol_sha256=ph,deadline_unix=q['deadline_unix'],scientific_changes=False,budget_extension=False)
    atomic(BASE/'THERMAL_SOLO_RECOVERY.json',fact)
    with (ROOT/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(fact,ensure_ascii=False)+'\n')
    print(json.dumps(fact,ensure_ascii=False))
if __name__=='__main__':main()
