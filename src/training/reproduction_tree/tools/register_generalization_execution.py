"""Record live evidence and two prospective execution corrections without changing frozen originals."""
from pathlib import Path
from datetime import datetime
import copy,json,sys,time
import generalization_cloud as cloud
ROOT=cloud.ROOT;BASE=cloud.BASE
def read(p):return cloud.read(p)
def write(p,v):cloud.write(p,v)
def main():
    now=datetime.now().astimezone().isoformat(timespec='seconds');local=BASE/'local';croot=BASE/'cloud';p=read(croot/'protocol.json');updated=copy.deepcopy(p);updated['id']='GENERALIZATION-CROSSFIT-SENSOR-SEED-CORRECTION-20260910-V1';updated['frozen_at_local']=now;updated['parent_protocol_sha256']=cloud.access.sha(croot/'protocol.json');updated['amendment']='Before any sensor fit: reused train_sensor explicitly seeds20260909. Register that actual seed for all matching inner/outer sensor fits; video protocol, runtime and active video jobs unchanged.';updated['jobs']={k:dict(v,seed=20260909) for k,v in p['jobs'].items() if v['kind']=='sensor'}
    assert not (croot/'sensor_protocol.json').exists();write(croot/'sensor_protocol.json',updated);ph=cloud.access.sha(croot/'sensor_protocol.json');q=read(croot/'QUEUE.json')
    for j in q['jobs']:
        if j['id'].startswith('S_'):j.update(protocol='sensor_protocol.json',protocol_sha256=ph)
    write(croot/'QUEUE.json',q);m,c=cloud.access.connection()
    try:
        r=cloud.access.execute(m,c,f'''import json
from pathlib import Path
p=Path({cloud.REMOTE!r});s=json.loads((p/'QUEUE_STATUS.json').read_text());assert not any(k.startswith('S_') for k in list(s['active'])+list(s['finished']));print(json.dumps(dict(sensor_training_started=False)))''')
        with c.open_sftp() as ftp:
            ftp.put(str(croot/'sensor_protocol.json'),cloud.REMOTE+'/sensor_protocol.json',confirm=True);ftp.put(str(croot/'QUEUE.json'),cloud.REMOTE+'/QUEUE.updated.json',confirm=True)
        cloud.access.execute(m,c,f'''import os,json,hashlib
from pathlib import Path
p=Path({cloud.REMOTE!r});assert hashlib.sha256((p/'sensor_protocol.json').read_bytes()).hexdigest().upper()=={ph!r};os.replace(p/'QUEUE.updated.json',p/'QUEUE.json');print(json.dumps(dict(status='SENSOR_SEED_CORRECTION_REGISTERED_BEFORE_TRAINING')))''')
    finally:c.close()
    lq=read(local/'QUEUE.json');sample=read(local/'TELEMETRY.json');assert sample['temperature_c']<85 and sample['memory_used_bytes']<5200000000;assert not (local/'THERMAL_EXECUTION_AMENDMENT.json').exists()
    amendment=dict(status='PROSPECTIVE_SCHEDULING_AMENDMENT',at_local=now,parent_protocol_sha256=cloud.access.sha(local/'thermal_protocol.json'),reason='First Thermal epoch85.36seconds implies8serial fits about3.8hours versus existing3hour local cap. Observed GPU utilization53percent and4.38GB total memory allow trying2owned fits with existing thermal guard; no scientific batch/seed/epoch changes.',concurrency_before=1,concurrency_after=2,deadline_unix=lq['deadline_unix'],budget_extension=False,admission_basis=sample,rollback='On verified OOM or sustained thermal pausing, return concurrency1 and reassign only unstarted remaining Thermal jobs within existing cloud6hour cap; retain interrupted artifacts.')
    write(local/'THERMAL_EXECUTION_AMENDMENT.json',amendment);lq['concurrency']=2;lq['execution_amendment']='THERMAL_EXECUTION_AMENDMENT.json';temp=local/'QUEUE.updated.json';write(temp,lq);temp.replace(local/'QUEUE.json')
    state=read(ROOT/'PROJECT_STATE.json');state.update(updated_at_local=now,phase='DUAL_HOST_GENERALIZATION_ACTIVE',next_frozen_candidate=dict(cloud_protocol=(croot/'protocol.json').relative_to(ROOT).as_posix(),cloud_sensor_protocol=(croot/'sensor_protocol.json').relative_to(ROOT).as_posix(),local_thermal_protocol=(local/'thermal_protocol.json').relative_to(ROOT).as_posix(),local_imu_protocol=(local/'protocol.json').relative_to(ROOT).as_posix()))
    state['active_runs']=[dict(host='cloud',kind='USER_CROSSFIT_VIDEO_SENSOR',remote_root=cloud.REMOTE,queue_status='runs/generalization_training_20260910_v1/CLOUD_SNAPSHOT.json',snapshot_command='tools/generalization_cloud.py snapshot',deadline_unix=q['deadline_unix']),dict(host='local',kind='MATCHED_USER_CROSSFIT_THERMAL',queue_status=(local/'QUEUE_STATUS.json').relative_to(ROOT).as_posix(),telemetry=(local/'TELEMETRY.json').relative_to(ROOT).as_posix(),controller=(local/'CONTROLLER.json').relative_to(ROOT).as_posix(),deadline_unix=lq['deadline_unix'])]
    state['runtime']['local'].update(status='TRAINING_THERMAL_CROSSFIT',current_gpu_training=True,last_snapshot=now,work_type='All6IMU60epochfits completed and independently replayed; local Thermal20epoch inner/outer fits use existing raw data.',snapshot_evidence=(local/'QUEUE_STATUS.json').relative_to(ROOT).as_posix(),queue=(local/'QUEUE.json').relative_to(ROOT).as_posix(),live_registered_identities=list(read(local/'QUEUE_STATUS.json')['active'].values()))
    state['runtime']['cloud'].update(status='TRAINING_VIDEO_CROSSFIT',current_gpu_training=True,last_snapshot_local=now,snapshot='runs/generalization_training_20260910_v1/CLOUD_SNAPSHOT.json',monitor_tool='tools/generalization_cloud.py',estimated_queue_complete_local='Approx4-6hours within03:06local harddeadline; refine from fixed epoch timings.',snapshot_observed_status=read(BASE/'CLOUD_SNAPSHOT.json'),live_registered_identities=list(read(BASE/'CLOUD_SNAPSHOT.json')['QUEUE_STATUS.json']['active'].values()))
    state['runtime']['automation'].update(status='ACTIVE',native_watchers='One bounded native queue controller per host, existing2second thermal protection',pause_after_both_csv_handoff=False,verification=now,last_check_disposition='NEW_USER_AUTHORIZED_GENERALIZATION_QUEUES_STARTED_AND_FIRST_IMU_STAGE_AUDITED',first_scheduled_execution_verified=False)
    state['latest_results']['generalization_imu']=dict(report='runs/generalization_training_20260910_v1/IMU_RESULT.md',audit='runs/generalization_training_20260910_v1/IMU_AUDIT.json',A=334,B=349,C=348,n=968,B_vs_A=15,C_vs_B=-1,status='COMPLETE_SIX_FITS_INDEPENDENT_FP32_REPLAY_PASS',promoted=False)
    state['successor_preparation']=dict(status='MATCHED_THERMAL_SUCCESSOR_ALREADY_LAUNCHED',run_root=BASE.relative_to(ROOT).as_posix(),next='Prepare frozen fusion implementation and asset joins20-40minutes before queue tail; use inner OOF for calibration only. Inspect local/cloud ETA and transfer only unstarted Thermal jobs if local3hourcap would be exceeded.',fusion_preregistration='runs/generalization_training_20260910_v1/cloud/protocol.json',local_preparation_has_no_cloud_weight_dependency=True,approval_needed=False,training_started=True)
    state['current_authorization']['generalization_training']['execution_allocation']=dict(at_local=now,local='IMU six fits complete; use free local GPU for already-authorized Thermal cross-fitting',cloud='MViT cross-fitting and matched measured sensors',scope_expansion=False,budget_extension=False)
    write(ROOT/'PROJECT_STATE.json',state)
    fact=dict(event='E-GENERALIZATION-ACTIVE-EXECUTION-20260910-V1',at_local=now,cloud_live_video_jobs=['V_F0_0','V_F3_0'],local_thermal_active=True,local_imu_completed=6,sensor_seed_correction_sha256=ph,thermal_scheduling_amendment=(local/'THERMAL_EXECUTION_AMENDMENT.json').relative_to(ROOT).as_posix(),automatic_budget_extension=False,automation='ACTIVE_20MIN_TWO_CONCISE_DUTIES')
    with (ROOT/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(fact,ensure_ascii=False)+'\n')
    write(BASE/'EXECUTION.json',fact);print(json.dumps(fact,ensure_ascii=False))
if __name__=='__main__':main()
