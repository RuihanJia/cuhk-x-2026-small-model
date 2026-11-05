"""Generate strictly user-excluded inner predictions for the replicated IMU dropout branch."""
from pathlib import Path
from datetime import datetime
import copy,json,subprocess,sys
import generalization_cloud as cloud
ROOT=cloud.ROOT;BASE=cloud.BASE;LOCAL=BASE/'local';BUNDLE=LOCAL/'bundle';sys.path.insert(0,str(BUNDLE))
from dual_host_runtime.io import atomic,read
from dual_host_runtime.health import alive
from research_runtime.common import sha

def main():
    target=LOCAL/'imu_dropout_crossfit_protocol.json';assert not target.exists()
    assert not alive(read(LOCAL/'CONTROLLER.json'));done=read(LOCAL/'QUEUE_DONE.json');assert done['status']=='COMPLETE' and len(done['finished'])==21
    for name in ['IMU_FOLLOWUP_AUDIT.json','IMU_DROPOUT_REPLICATION_AUDIT.json']:assert read(BASE/name)['status']=='PASS_FIXED_EPOCHS_AND_INDEPENDENT_FP32_RELOAD'
    original=read(LOCAL/'imu_dropout_seed1_protocol.json');parent=read(BASE/'cloud/protocol.json');p=copy.deepcopy(original)
    p['id']='GENERALIZATION-IMU-DROPOUT-INNER-CROSSFIT-20260911-V1';p['status']='FROZEN_BEFORE_SIX_INNER_FITS';p['frozen_at_local']=datetime.now().astimezone().isoformat(timespec='seconds')
    p['hypothesis']='Fixed extra-device dropout improved C in both already-used seeds. User-excluded inner D predictions permit assessing its incremental fusion contribution using matched originalseed outer D models, without fitting on any outer labels.'
    p['jobs']={}
    for fold in [0,3]:
        for group in range(3):
            source=parent['jobs'][f'V_F{fold}_{group}'];spec=dict(original['jobs'][f'IMU_D1_F{fold}'],train_users=source['train_users'],predict_users=source['predict_users'])
            assert not set(spec['train_users'])&set(spec['predict_users']);assert not set(spec['train_users'])&set(parent['folds'][str(fold)])
            p['jobs'][f'IMU_DX_F{fold}_{group}']=spec
    p['controls']=dict(outer_weights='Reuse originalseed D1_F0/F3 exact60epoch checkpoints; same architecture,augmentation,optimizer,seed and normalization recipe',fusion_control='The already-frozen video/sensor/Thermal variants remain; add prospective IMU-D convex/gated fusion comparisons using inner-only calibration',no_outer_refitting=True)
    p['budget']=dict(max_new_fits=6,epochs_per_fit=60,expected_total_fit_seconds=100,expected_start_save_and_audit_minutes='3-6',deadline_unix=None,local_wall_cap_enabled=False)
    p['prior_evidence']={name:dict(sha256=sha(BASE/name),D_vs_C=read(BASE/name)['comparisons']['D_vs_C']['net_correct']) for name in ['IMU_FOLLOWUP_AUDIT.json','IMU_DROPOUT_REPLICATION_AUDIT.json']}
    p['outer_checkpoint_sources']={str(f):dict(job=f'IMU_D1_F{f}',checkpoint_sha256=read(LOCAL/f'output/IMU_D1_F{f}/DONE.json')['checkpoint_sha256'],protocol_sha256=sha(LOCAL/'imu_dropout_seed1_protocol.json')) for f in [0,3]}
    p['code_sha256']={x.relative_to(BUNDLE).as_posix():sha(x) for x in BUNDLE.rglob('*.py')};p['execution_delta']=dict(change='Six inner user-excluded fits with exact originalseedD recipe; no new architecture,seed,probability or outer fit',scientific_changes='Different preregistered inner training user subsets',cloud_gpu_budget_changed=False)
    atomic(target,p);ph=sha(target)
    fact=dict(event='E-GENERALIZATION-IMU-DROPOUT-INNER-CROSSFIT-FROZEN-20260911-V1',status='FROZEN_BEFORE_TRAINING',at_local=p['frozen_at_local'],protocol=target.relative_to(ROOT).as_posix(),protocol_sha256=ph,max_new_fits=6,epochs=60,seed=2026091003,dropout=.25,no_outer_label_calibration=True,new_inner_results_seen=False)
    atomic(BASE/'IMU_DROPOUT_CROSSFIT_REGISTRATION.json',fact)
    with (ROOT/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(fact,ensure_ascii=False)+'\n')
    atomic(LOCAL/'QUEUE_DONE_BEFORE_IMU_CROSSFIT.json',done);(LOCAL/'QUEUE_DONE.json').unlink();q=read(LOCAL/'QUEUE.json');assert q['deadline_unix']==0
    q['jobs'].extend(dict(id=k,worker='generalization_imu_dropout_seed1.py',protocol=target.name,protocol_sha256=ph,requires=['DEVICE_CACHE_READY.json']) for k in p['jobs']);atomic(LOCAL/'QUEUE.json',q)
    with (LOCAL/'controller_imu_crossfit.stdout.log').open('a',encoding='utf-8') as out,(LOCAL/'controller_imu_crossfit.stderr.log').open('a',encoding='utf-8') as err:
        proc=subprocess.Popen([sys.executable,'-X','utf8','-B',str(BUNDLE/'native_queue_no_wall.py'),'--root',str(LOCAL)],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
    state=read(ROOT/'PROJECT_STATE.json');state['runtime']['local'].update(status='STARTED_SIX_INNER_IMU_DROPOUT_FITS',execution_protocol=target.relative_to(ROOT).as_posix(),work_type='Six strictly user-excluded inner fits for D fusion; fixed60epochs; originalseed outerD checkpoints reused')
    state['successor_preparation'].update(status='IMU_DROPOUT_CROSSFIT_STARTED_FUSION_PREPARATION_ACTIVE',next='Verify six IMU-D inner outputs; prepare fixed original three-branch and prospective four-branch inner-only CPU fusion, preserving matched controls.',local_imu_followup_protocol=target.relative_to(ROOT).as_posix(),imu_dropout_replication_audit='runs/generalization_training_20260910_v1/IMU_DROPOUT_REPLICATION_AUDIT.json')
    atomic(ROOT/'PROJECT_STATE.json',state);print(json.dumps(dict(**fact,controller_spawn_pid=proc.pid),ensure_ascii=False))

if __name__=='__main__':main()
