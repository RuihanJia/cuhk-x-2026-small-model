"""A bounded second seed and one device-dropout control, registered before any new fit."""
from pathlib import Path
from datetime import datetime
import ast,copy,json,time
import generalization_cloud as cloud
ROOT=cloud.ROOT;BASE=cloud.BASE;OUT=BASE/'local';B=OUT/'bundle'
def read(p):return cloud.read(p)
def write(p,v):
    tmp=p.with_name(p.name+'.partial');cloud.write(tmp,v);tmp.replace(p)
def main():
    assert not (OUT/'imu_followup_protocol.json').exists();source=B/'generalization_imu_worker.py';text=source.read_text(encoding='utf-8');assert text.count("spec['arm']=='C'")==1;text=text.replace("spec['arm']=='C'","spec['arm'] in ['C','D']").replace('imu-device-20260910:2026091003:','imu-device-followup-20260910:2026091004:')
    anchor='            for k in range(0,len(order),64):';assert text.count(anchor)==1;text=text.replace(anchor,"            train_valid=valid\n            if spec['arm']=='D':\n                rng=np.random.default_rng(p['training']['seed']+spec['fold']*1000003+epoch);kept=torch.from_numpy(rng.random((len(ids),3))>=.25).cuda();train_valid=valid.clone();train_valid[:,2:]=train_valid[:,2:] & kept\n"+anchor);anchor="model(x,valid[ix])";assert text.count(anchor)==1;text=text.replace(anchor,"model(x,train_valid[ix])")
    worker=B/'generalization_imu_followup_worker.py';ast.parse(text);worker.write_text(text,encoding='utf-8');p=copy.deepcopy(read(OUT/'protocol.json'));p['id']='GENERALIZATION-IMU-SECOND-SEED-DEVICE-DROPOUT-20260910-V1';p['frozen_at_local']=datetime.now().astimezone().isoformat(timespec='seconds');p['parent_protocol_sha256']=cloud.access.sha(OUT/'protocol.json');p['prior_evidence']=dict(path='runs/generalization_training_20260910_v1/IMU_AUDIT.json',sha256=cloud.access.sha(BASE/'IMU_AUDIT.json'),B_vs_A_net=15,C_vs_B_net=-1,C_vs_B_fold_net=[7,-8],interpretation='Positive encoder signal and mixed extra-device signal; assess seed sensitivity and one mechanism-driven regularizer without closing the direction.')
    p['hypothesis']='A second seed tests whether B/C differences are fragile. Dropping each extra device independently with fixed probability0.25 during training may reduce reliance on unstable extra-device evidence while preserving both arms.';p['training']['seed']=2026091004;p['training']['orientation']='Same HaarSO3 construction and fixedprobability.5; new independent schedule tagseed2026091004 shared across B/C/D; no rotation at inference.';p['jobs']={}
    for fold in [0,3]:
        for arm in 'BCD':p['jobs'][f'IMU_R_{arm}_F{fold}']=dict(read(OUT/'protocol.json')['jobs'][f'IMU_B_F{fold}'],arm=arm)
    p['controls']=dict(B='Two-arm input only;shared five-slot DeviceIMU',C='Identical model,seed and schedule;include measured extras',D='Same C model and inputs;each extra-device training mask independently dropped withp.25;neverdrop either arm;fullobserved mask at inference',capacity='All3arms exactly180264parameters;initial states must match within this seed',selection='One seed and one fixed dropout probability; no sweep or result-based changes')
    p['budget']=dict(max_new_fits=6,epochs_per_fit=60,expected_total_fit_seconds=180,deadline_unix=read(OUT/'QUEUE.json')['deadline_unix'],existing_local_phase_cap_seconds=10800,extension=False,if_deadline_reached='Preserve completed/epoch recovery artifacts and mark incomplete; not a performance verdict.')
    p['code_sha256']={x.relative_to(B).as_posix():cloud.access.sha(x) for x in B.rglob('*.py')};p['data']['device_cache_sha256']=read(OUT/'DEVICE_CACHE_READY.json')['sha256'];p['device_dropout']=dict(probability=.25,scope='WTC,WTLL,WTRL only at training',seed_rule='independent PCG64(seed+fold*1000003+epoch)',applied_to='validity mask before zeroing inputs and encoder outputs',original_data_and_normalizer_unchanged=True)
    protocol=OUT/'imu_followup_protocol.json';write(protocol,p);ph=cloud.access.sha(protocol);q=read(OUT/'QUEUE.json');assert all(not j['id'].startswith('IMU_R_') for j in q['jobs']);q['jobs'].extend(dict(id=k,worker=worker.name,protocol=protocol.name,protocol_sha256=ph,requires=['DEVICE_CACHE_READY.json']) for k in p['jobs']);write(OUT/'QUEUE.json',q)
    fact=dict(event='E-GENERALIZATION-IMU-FOLLOWUP-FROZEN-20260910-V1',at_local=p['frozen_at_local'],status='REGISTERED_AND_QUEUED_AFTER_THERMAL',hypothesis=p['hypothesis'],fits=6,epochs=60,seed=2026091004,dropout_probability=.25,protocol=protocol.relative_to(ROOT).as_posix(),protocol_sha256=ph,deadline_unix=q['deadline_unix'],budget_extension=False,results_seen=False)
    write(BASE/'IMU_FOLLOWUP_REGISTRATION.json',fact)
    with (ROOT/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(fact,ensure_ascii=False)+'\n')
    print(json.dumps(fact,ensure_ascii=False))
if __name__=='__main__':main()
