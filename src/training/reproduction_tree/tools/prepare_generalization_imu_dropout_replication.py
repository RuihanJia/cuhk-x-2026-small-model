"""Replicate fixed device dropout with the original seed and its exact orientation schedule."""
from pathlib import Path
from datetime import datetime
import ast,copy,json,subprocess,sys
import generalization_cloud as cloud
ROOT=cloud.ROOT;BASE=cloud.BASE;LOCAL=BASE/'local';BUNDLE=LOCAL/'bundle';sys.path.insert(0,str(BUNDLE))
from dual_host_runtime.io import atomic,read
from dual_host_runtime.health import alive
from research_runtime.common import sha

def main():
    target=LOCAL/'imu_dropout_seed1_protocol.json';assert not target.exists()
    owner=read(LOCAL/'CONTROLLER.json');assert not alive(owner)
    status=read(LOCAL/'QUEUE_STATUS.json');done=read(LOCAL/'QUEUE_DONE.json');assert done['status']=='COMPLETE' and len(status['finished'])==19 and not status['active']
    audit=read(BASE/'IMU_FOLLOWUP_AUDIT.json');assert audit['status']=='PASS_FIXED_EPOCHS_AND_INDEPENDENT_FP32_RELOAD'
    source=BUNDLE/'generalization_imu_followup_no_wall.py';text=source.read_text(encoding='utf-8')
    before='imu-device-followup-20260910:2026091004:';after='imu-device-20260910:2026091003:';assert text.count(before)==1;text=text.replace(before,after)
    assert after in (BUNDLE/'generalization_imu_worker.py').read_text(encoding='utf-8')
    ast.parse(text);worker=BUNDLE/'generalization_imu_dropout_seed1.py';assert not worker.exists();worker.write_text(text,encoding='utf-8')
    p=copy.deepcopy(read(LOCAL/'imu_followup_no_wall_protocol.json'));original=read(LOCAL/'protocol.json')
    p['id']='GENERALIZATION-IMU-DEVICE-DROPOUT-ORIGINAL-SEED-20260911-V1';p['status']='FROZEN_BEFORE_TWO_REPLICATION_FITS';p['frozen_at_local']=datetime.now().astimezone().isoformat(timespec='seconds')
    p['training']['seed']=2026091003;p['training']['orientation']='Exact original seed2026091003 orientation tag and construction, shared with existing original B/C controls'
    p['jobs']={f'IMU_D1_F{f}':dict(original['jobs'][f'IMU_C_F{f}'],arm='D') for f in [0,3]}
    p['hypothesis']='The fixed0.25extra-device training dropout may reduce reliance on unstable added sensors. Replicate the observed second-seed D/C gain using the original seed and exact original orientation schedule.'
    p['controls']=dict(B='Existing IMU_B_F0/F3 original60epoch controls',C='Existing IMU_C_F0/F3 original60epoch controls',D='Two new fits only; same180264parameters,initialization,data,normalization,orientation,batches,optimizer,epochs as originalC; add fixed0.25extra-device dropout',selection='Previously used original seed; no new seed or dropout probability search')
    p['budget']=dict(max_new_fits=2,epochs_per_fit=60,expected_total_fit_seconds=40,expected_start_save_and_review_minutes='2-4',deadline_unix=None,local_wall_cap_enabled=False,scope='Two fixed replication fits only; finite queue and existing protection')
    p['prior_evidence']=dict(audit='runs/generalization_training_20260910_v1/IMU_FOLLOWUP_AUDIT.json',sha256=sha(BASE/'IMU_FOLLOWUP_AUDIT.json'),D_vs_C_net=24,D_vs_C_fold_net=[23,1],D_vs_B_net=34,interpretation='Positive but concentrated inF0; check exact originalseed before interpreting robustness')
    p['execution_delta']=dict(scientific_changes=True,change='Train fixedD on originalseed; original B/C controls reused after completed artifact audits',local_wall_cap_enabled=False,cloud_scope_changed=False)
    p['code_sha256']={v.relative_to(BUNDLE).as_posix():sha(v) for v in BUNDLE.rglob('*.py')}
    atomic(target,p);ph=sha(target)
    fact=dict(event='E-GENERALIZATION-IMU-DROPOUT-REPLICATION-FROZEN-20260911-V1',status='FROZEN_BEFORE_TRAINING',at_local=p['frozen_at_local'],protocol=target.relative_to(ROOT).as_posix(),protocol_sha256=ph,max_new_fits=2,epochs=60,seed=2026091003,dropout=.25,expected_total_fit_seconds=40,new_results_seen=False)
    atomic(BASE/'IMU_DROPOUT_REPLICATION_REGISTRATION.json',fact)
    with (ROOT/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(fact,ensure_ascii=False)+'\n')
    queue=read(LOCAL/'QUEUE.json');assert queue['deadline_unix']==0
    atomic(LOCAL/'QUEUE_DONE_BEFORE_DROPOUT_SEED1.json',done);(LOCAL/'QUEUE_DONE.json').unlink()
    queue['jobs'].extend(dict(id=k,worker=worker.name,protocol=target.name,protocol_sha256=ph,requires=['DEVICE_CACHE_READY.json']) for k in p['jobs']);atomic(LOCAL/'QUEUE.json',queue)
    with (LOCAL/'controller_dropout_seed1.stdout.log').open('a',encoding='utf-8') as out,(LOCAL/'controller_dropout_seed1.stderr.log').open('a',encoding='utf-8') as err:
        proc=subprocess.Popen([sys.executable,'-X','utf8','-B',str(BUNDLE/'native_queue_no_wall.py'),'--root',str(LOCAL)],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
    state=read(ROOT/'PROJECT_STATE.json');state['runtime']['local'].update(status='STARTED_TWO_ORIGINAL_SEED_DROPOUT_REPLICATION_FITS',execution_protocol=target.relative_to(ROOT).as_posix(),work_type='OriginalseedD,F0/F3 fixed60epochs; compare with already-audited original B/C; no local wallcap')
    state['successor_preparation'].update(status='IMU_DROPOUT_REPLICATION_REGISTERED_AND_STARTED',next='Audit originalseed D/C replication; prepare the already-registered inner-only CPU fusion for cloud models. Any further local variant requires a concrete hypothesis and registration.',imu_followup_audit='runs/generalization_training_20260910_v1/IMU_FOLLOWUP_AUDIT.json',local_imu_followup_protocol=target.relative_to(ROOT).as_posix())
    atomic(ROOT/'PROJECT_STATE.json',state)
    print(json.dumps(dict(**fact,controller_spawn_pid=proc.pid),ensure_ascii=False))

if __name__=='__main__':main()
