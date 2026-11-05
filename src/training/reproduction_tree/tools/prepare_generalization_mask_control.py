"""A fixed two-seed explicit-mask feature control, independent of the frozen fusion comparison."""
from pathlib import Path
from datetime import datetime
import ast,copy,json,subprocess,sys
import generalization_cloud as cloud
ROOT=cloud.ROOT;BASE=cloud.BASE;LOCAL=BASE/'local';BUNDLE=LOCAL/'bundle';sys.path.insert(0,str(BUNDLE))
from dual_host_runtime.io import atomic,read
from dual_host_runtime.health import alive
from research_runtime.common import sha

def main():
    assert not (BASE/'IMU_MASK_CONTROL_REGISTRATION.json').exists();assert not alive(read(LOCAL/'CONTROLLER.json'));done=read(LOCAL/'QUEUE_DONE.json');assert done['status']=='COMPLETE' and len(done['finished'])==36
    entries=[];protocols={};stamp=datetime.now().astimezone().isoformat(timespec='seconds')
    for index,seed,oldprotocol,oldworker in [(1,2026091003,'imu_dropout_seed1_protocol.json','generalization_imu_dropout_seed1.py'),(2,2026091004,'imu_followup_no_wall_protocol.json','generalization_imu_followup_no_wall.py')]:
        text=(BUNDLE/oldworker).read_text(encoding='utf-8');before='torch.cat([z.flatten(1),valid.float()],1)';assert text.count(before)==1;text=text.replace(before,'torch.cat([z.flatten(1),torch.zeros_like(valid,dtype=torch.float32)],1)')
        ast.parse(text);worker=BUNDLE/f'generalization_imu_nomask_seed{index}.py';assert not worker.exists();worker.write_text(text,encoding='utf-8')
        original=read(LOCAL/oldprotocol);p=copy.deepcopy(original);p['id']=f'GENERALIZATION-IMU-EXPLICIT-MASK-CONTROL-SEED{index}-20260911-V1';p['status']='FROZEN_BEFORE_TRAINING';p['frozen_at_local']=stamp
        p['hypothesis']='Device dropout D improved C in two seeds. Removing the five explicit availability features may reduce reliance on user-correlated missingness while retaining zero influence of missing raw inputs. Implicit availability through zero embeddings remains; this does not eliminate all missingness information.'
        p['jobs']={}
        for fold in [0,3]:
            oldjob=f'IMU_D1_F{fold}' if index==1 else f'IMU_R_D_F{fold}';p['jobs'][f'IMU_E_S{index}_F{fold}']=copy.deepcopy(original['jobs'][oldjob])
        p['controls']=dict(D=f'Existing exact seed{index}D checkpoint;same architecture and180264parameters',E='Only replace five explicit mask input columns to the head by zeros. Keep missing-input zeroing, masked embeddings,0.25extra-device dropout,original orientation schedule,seed,optimizer and60epochs unchanged.',interpretation='The unused five mask columns remain parameters to preserve exact shape and initialization. This tests explicit mask use,not all implicit missingness cues.')
        p['budget']=dict(max_new_fits=2,epochs_per_fit=60,expected_fit_seconds=40,deadline_unix=None,local_wall_cap_enabled=False)
        p['prior_evidence']=dict(seed1_D_vs_C=28,seed2_D_vs_C=24,source_audits=['IMU_DROPOUT_REPLICATION_AUDIT.json','IMU_FOLLOWUP_AUDIT.json'],fusion_stage_unchanged=True)
        p['execution_delta']=dict(scientific_changes=True,only_change='Zero explicit five-bit availability vector to the classifier head',new_seed_or_dropout_scan=False)
        p['code_sha256']={x.relative_to(BUNDLE).as_posix():sha(x) for x in BUNDLE.rglob('*.py')};target=LOCAL/f'imu_mask_control_seed{index}_protocol.json';atomic(target,p)
        protocols[str(index)]=dict(protocol=target.relative_to(ROOT).as_posix(),sha256=sha(target),worker=worker.name,seed=seed)
        entries.extend(dict(id=k,worker=worker.name,protocol=target.name,protocol_sha256=sha(target),requires=['DEVICE_CACHE_READY.json']) for k in p['jobs'])
    fact=dict(event='E-GENERALIZATION-IMU-EXPLICIT-MASK-CONTROL-FROZEN-20260911-V1',status='FROZEN_AND_QUEUED',at_local=stamp,protocols=protocols,max_new_fits=4,epochs=60,expected_total_fit_seconds=80,expected_total_minutes='2-4',frozen_fusion_protocol_unchanged=True,results_seen=False)
    atomic(BASE/'IMU_MASK_CONTROL_REGISTRATION.json',fact)
    with (ROOT/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(fact,ensure_ascii=False)+'\n')
    atomic(LOCAL/'QUEUE_DONE_BEFORE_MASK_CONTROL.json',done);(LOCAL/'QUEUE_DONE.json').unlink();q=read(LOCAL/'QUEUE.json');q['jobs'].extend(entries);atomic(LOCAL/'QUEUE.json',q)
    with (LOCAL/'controller_mask_control.stdout.log').open('a',encoding='utf-8') as out,(LOCAL/'controller_mask_control.stderr.log').open('a',encoding='utf-8') as err:
        proc=subprocess.Popen([sys.executable,'-X','utf8','-B',str(BUNDLE/'native_queue_no_wall.py'),'--root',str(LOCAL)],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
    state=read(ROOT/'PROJECT_STATE.json');state['runtime']['local'].update(status='STARTED_FOUR_EXPLICIT_MASK_CONTROL_FITS',current_gpu_training=True,work_type='Two already-used seeds,eachF0/F3;zero only explicit availability head features in fixedD;four60epoch fits;fusion comparison stays frozen')
    state['successor_preparation']['local_mask_control']='runs/generalization_training_20260910_v1/IMU_MASK_CONTROL_REGISTRATION.json';state['successor_preparation']['next']='Audit four local explicit-mask E controls against matching D checkpoints. Cloud native queue independently finishes video/Thermal and then the already-frozen eight fusion comparisons and summary.'
    atomic(ROOT/'PROJECT_STATE.json',state);print(json.dumps(dict(**fact,controller_spawn_pid=proc.pid),ensure_ascii=False))

if __name__=='__main__':main()
