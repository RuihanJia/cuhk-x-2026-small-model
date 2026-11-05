"""Freeze and stage fixed fusion comparisons, then append gated CPU jobs to the existing cloud queue."""
from pathlib import Path
from datetime import datetime
import argparse,copy,json,sys
import generalization_cloud as cloud
from generalization_fusion_math import STRATEGIES
ROOT=cloud.ROOT;BASE=cloud.BASE;CLOUD=BASE/'cloud';LOCAL=BASE/'local';sys.path.insert(0,str(LOCAL/'bundle'))
from dual_host_runtime.io import atomic,read
from research_runtime.common import sha

def build():
    target=CLOUD/'fusion_protocol.json';assert not target.exists()
    assert read(BASE/'FUSION_NUMERICAL_CHECKS.json')['status']=='PASS'
    parent=read(CLOUD/'protocol.json');sensor=read(CLOUD/'sensor_protocol.json');thermal=read(LOCAL/'thermal_protocol.json');imu=read(LOCAL/'imu_dropout_crossfit_protocol.json');imu_outer=read(LOCAL/'imu_dropout_seed1_protocol.json')
    admission=dict(status='PASS_INDEPENDENT_CPU_REPLAY_FOR_ALL_EIGHT_IMU_D_SOURCES',jobs={})
    inner_audit=read(BASE/'IMU_DROPOUT_INNER_AUDIT.json');outer_audit=read(BASE/'IMU_DROPOUT_REPLICATION_AUDIT.json')
    for key,report in inner_audit['jobs'].items():admission['jobs'][key]=report
    for fold in [0,3]:admission['jobs'][f'IMU_D1_F{fold}']=outer_audit['arms'][f'D_F{fold}']
    atomic(BASE/'IMU_FUSION_ADMISSION.json',admission)
    for name in ['generalization_fusion_math.py','generalization_fusion_worker.py']:
        source=ROOT/'tools'/name;dst=CLOUD/'bundle'/name;assert not dst.exists();dst.write_bytes(source.read_bytes())
    ti=CLOUD/'bundle/data/thermal_index.json'
    if not ti.exists():ti.write_bytes((LOCAL/'bundle/data/thermal_index.json').read_bytes())
    assert sha(ti)==thermal['data']['index_sha256']
    p=dict(id='GENERALIZATION-INNER-ONLY-FUSION-20260911-V1',status='FROZEN_BEFORE_ANY_FUSION_FIT_OR_FUSED_OUTER_SCORE',frozen_at_local=datetime.now().astimezone().isoformat(timespec='seconds'),
        parent_protocol_sha256=sha(CLOUD/'protocol.json'),training_users=parent['training_users'],folds=parent['folds'],strategies=['matched_fixed_vs']+list(STRATEGIES),
        hypothesis='Calibrated and quality-gated late fusion may improve the matched video/sensor control. Independently compare Thermal addition and the replicated IMU-D branch using inner user-excluded predictions.',
        prospective_amendments=['Retain all original three-branch strategies; fully specify previously under-specified mixture math and gate features before fitting','Add two-branch calibrated convex control to distinguish calibration from Thermal contribution','Add no-availability-feature gate controls, retaining structural missing-branch exclusion','Add replicated IMU-D as a prospective fourth branch; its standalone development gains were seen before this amendment'],
        math=dict(fixed='FP32(.8*video_logits+.2*sensor_logits)',learned='sum_m softmax(masked_gate_score)_m * softmax(logits_m/T_m); save log-probabilities as FP32',temperature='.25+7.75*sigmoid(parameter),one scalar per branch',temperature_initial=1,gate_initial='equal branch weights; zero slopes',quality=['raw-softmax entropy/log40','raw-softmax top2 probability margin','Depth/IR Jensen-Shannon divergence/log2 for video;0for other branches'],quality_normalization='Mean/std from available innerOOF rows only,per branch;std floor1e-6;missing branch qualities zeroed',gate='branch intercept plus its own three quality features; full availability variants also use the centered vector of all branch availability flags',gate_no_availability='No availability coefficients; own quality features only; unavailable branches still receive exactly zero weight',l2='.01*mean(square(all quality and availability coefficients)); biases and temperatures unpenalized',objective='Inverse-user-count weighted NLL; equal total weight for each inner calibration user'),
        optimizer=dict(name='CPU FP64 LBFGS',max_iter=500,max_eval=750,lr=1,history_size=50,line_search='strong_wolfe',tolerance_grad=1e-7,tolerance_change=1e-9,initializations=1,hyperparameter_sweep=False),
        data_sha256=dict(parent['data_sha256'],**{'thermal_index.json':sha(ti)}),imu_admission=dict(path='fusion_inputs/imu_admission.json',sha256=sha(BASE/'IMU_FUSION_ADMISSION.json')),
        sources={},jobs={},budget=dict(deadline_unix=read(CLOUD/'QUEUE.json')['deadline_unix'],additional_GPU_fits=0,max_CPU_calibration_fits=14,max_iterations_per_fit=500,threads_per_job=2,expected_CPU_minutes='2-8',cloud_phase_extension=False),
        limitations=['F0/F3 are reused development users; user exclusion in fitting does not make the overall adaptive research process unbiased','IMU branch was selected after positive standalone development results; fusion effects remain unobserved at registration','Inner encoders train on fewer users than corresponding outer encoders; calibration distribution may shift','Two seed checks reuse the same seven assessment users','No Test,F2/F4 outer evaluation,all18 training or upload; no Public-score tuning'])
    upload={}
    def source(key,branch,spec,protocol,local=False):
        entry=dict(branch=branch,fold=spec['fold'],train_users=spec['train_users'],predict_users=spec['predict_users'],epochs=20 if branch in ['V','T'] else 60,batch=8 if branch in ['V','T'] else 64,protocol_sha256=protocol,path=('fusion_inputs/local/' if local else 'output/')+key)
        if local:
            out=LOCAL/'output'/key;done=read(out/'DONE.json');assert done['status']=='COMPLETE' and done['protocol_sha256']==protocol
            entry.update(checkpoint_sha256=done['checkpoint_sha256'],predictions_sha256=done['predictions_sha256'])
            if branch=='T':entry['execution_protocol_sha256']=done['execution_protocol_sha256']
            for name in ['DONE.json','PROVENANCE.json','SPLIT.json','model_fp16.pt','predictions.npz']+(['NORMALIZER.json'] if branch=='T' else []):upload[f"fusion_inputs/local/{key}/{name}"]=out/name
        elif branch=='T':entry['execution_protocol_sha256']=sha(CLOUD/'thermal_handoff_protocol.json')
        p['sources'][key]=entry
    for fold in [0,3]:
        groups=[]
        for tag in ['0','1','2','outer']:
            vk=f'V_F{fold}_{tag}';sk=f'S_F{fold}_{tag}';tk=f'T_F{fold}_{tag}';ik=f'IMU_D1_F{fold}' if tag=='outer' else f'IMU_DX_F{fold}_{tag}'
            source(vk,'V',parent['jobs'][vk],sha(CLOUD/'protocol.json'));source(sk,'S',sensor['jobs'][sk],sha(CLOUD/'sensor_protocol.json'))
            source(tk,'T',thermal['jobs'][tk],sha(LOCAL/'thermal_protocol.json'),local=tk!='T_F3_outer')
            ip=imu_outer if tag=='outer' else imu;source(ik,'I',ip['jobs'][ik],sha(LOCAL/('imu_dropout_seed1_protocol.json' if tag=='outer' else 'imu_dropout_crossfit_protocol.json')),local=True)
            groups.append([vk,sk,tk,ik])
        p['jobs'][f'FUSION_F{fold}']=dict(kind='CPU_FUSION',fold=fold,inner=groups[:3],outer=[groups[3]])
    p['jobs']['FUSION_SUMMARY']=dict(kind='CPU_SUMMARY')
    p['code_sha256']={v.relative_to(CLOUD/'bundle').as_posix():sha(v) for v in (CLOUD/'bundle').rglob('*.py')}
    atomic(target,p)
    upload['fusion_protocol.json']=target;upload['fusion_inputs/imu_admission.json']=BASE/'IMU_FUSION_ADMISSION.json'
    for name in ['generalization_fusion_math.py','generalization_fusion_worker.py']:upload['bundle/'+name]=CLOUD/'bundle'/name
    upload['bundle/data/thermal_index.json']=ti
    manifest=dict(files={rel:dict(local_path=str(path),sha256=sha(path),bytes=path.stat().st_size) for rel,path in upload.items()})
    atomic(BASE/'FUSION_UPLOAD_MANIFEST.json',manifest)
    # Validate every available local source against its real split, checkpoint and admission evidence.
    import generalization_fusion_worker as worker
    raw,master,cohorts=worker.load_cohorts(CLOUD,p);q=copy.deepcopy(p);q['imu_admission']['path']='IMU_FUSION_ADMISSION.json';checked=[]
    for key,entry in q['sources'].items():
        if entry['path'].startswith('fusion_inputs/local/'):
            entry['path']='local/output/'+key;worker.load_job(BASE,q,key,raw,master,cohorts);checked.append(key)
    atomic(BASE/'FUSION_SOURCE_CHECKS.json',dict(status='PASS_ALL15_LOCAL_CHECKPOINTS_PREDICTIONS_SPLITS_AND_ADMISSIONS',checked=checked,outer_local_validation='Identity and source checks only; no fused outer scoring'))
    fact=dict(event='E-GENERALIZATION-FUSION-FROZEN-20260911-V1',status='READY_FOR_GATED_CPU_QUEUE',at_local=p['frozen_at_local'],protocol=target.relative_to(ROOT).as_posix(),protocol_sha256=sha(target),strategies=p['strategies'],max_CPU_fits=14,additional_GPU_fits=0,cloud_deadline_unix=p['budget']['deadline_unix'],upload_bytes=sum(v['bytes'] for v in manifest['files'].values()),fused_outer_results_seen=False)
    atomic(BASE/'FUSION_REGISTRATION.json',fact)
    with (ROOT/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(fact,ensure_ascii=False)+'\n')
    print(json.dumps(fact,ensure_ascii=False))

def upload_and_queue():
    registration=read(BASE/'FUSION_REGISTRATION.json');protocol=ROOT/registration['protocol'];p=read(protocol);ph=sha(protocol);manifest=read(BASE/'FUSION_UPLOAD_MANIFEST.json');assert not (BASE/'FUSION_CLOUD_QUEUED.json').exists()
    assert read(BASE/'FUSION_SOURCE_CHECKS.json')['status'].startswith('PASS')
    jobs=[]
    for key,value in p['jobs'].items():
        if key=='FUSION_SUMMARY':requires=['output/FUSION_F0/DONE.json','output/FUSION_F3/DONE.json']
        else:requires=[p['sources'][s]['path']+'/DONE.json' for group in value['inner']+value['outer'] for s in group]
        jobs.append(dict(id=key,worker=p.get('runtime_worker','generalization_fusion_worker.py'),protocol=protocol.name,protocol_sha256=ph,requires=requires))
    m,c=cloud.access.connection()
    try:
        with c.open_sftp() as ftp:
            for rel,spec in manifest['files'].items():
                assert sha(spec['local_path'])==spec['sha256'];current=cloud.REMOTE
                for part in Path(rel).parent.as_posix().split('/'):
                    if part=='.':continue
                    current+='/'+part
                    try:ftp.stat(current)
                    except OSError:ftp.mkdir(current)
                ftp.put(spec['local_path'],cloud.REMOTE+'/'+rel,confirm=True)
        expected={rel:spec['sha256'] for rel,spec in manifest['files'].items()}
        code=f'''import json,hashlib,os,time,ast
from pathlib import Path
root=Path({cloud.REMOTE!r})
for rel,expected in {expected!r}.items():
 with (root/rel).open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest().upper()
 assert actual==expected,rel
for name in ['generalization_fusion_math.py','generalization_fusion_worker.py']:ast.parse((root/'bundle'/name).read_text())
q=json.loads((root/'QUEUE.json').read_text());assert q['deadline_unix']=={p['budget']['deadline_unix']!r};assert not any(j['id'].startswith('FUSION_') for j in q['jobs'])
q['jobs'].extend({jobs!r});tmp=root/'QUEUE.json.fusion.partial';tmp.write_text(json.dumps(q,indent=2));os.replace(tmp,root/'QUEUE.json')
print(json.dumps(dict(status='HASH_VERIFIED_AND_THREE_GATED_CPU_JOBS_QUEUED',at_unix=time.time(),jobs={[j['id'] for j in jobs]!r},files={len(expected)},deadline_unix=q['deadline_unix'])))'''
        result=cloud.access.execute(m,c,code,timeout=60)
    finally:c.close()
    local_queue=read(CLOUD/'QUEUE.json');assert not any(j['id'].startswith('FUSION_') for j in local_queue['jobs']);local_queue['jobs'].extend(jobs);atomic(CLOUD/'QUEUE.json',local_queue)
    atomic(BASE/'FUSION_CLOUD_QUEUED.json',result)
    with (ROOT/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(event='E-GENERALIZATION-FUSION-CLOUD-QUEUED-20260911-V1',**result,protocol_sha256=ph),ensure_ascii=False)+'\n')
    state=read(ROOT/'PROJECT_STATE.json');state['successor_preparation'].update(status='FUSION_CPU_MODELS_READY_AND_GATED_IN_NATIVE_CLOUD_QUEUE',fusion_execution_protocol=protocol.relative_to(ROOT).as_posix(),fusion_cloud_queued='runs/generalization_training_20260910_v1/FUSION_CLOUD_QUEUED.json',next='Local native queue completes eight matchedS fits and publishes checked outputs. Cloud finishes remaining video/Thermal,then automatically fits inner-only fusion and summary. Each fusion job waits for all exact source DONE markers; no assistant polling is required.')
    state['runtime']['automation'].update(last_check_disposition='IMU_D_GAIN_REPLICATED_AND_INNER_FITS_AUDITED;FUSION_SUCCESSOR_QUEUED',last_check_artifact='runs/generalization_training_20260910_v1/FUSION_CLOUD_QUEUED.json')
    atomic(ROOT/'PROJECT_STATE.json',state);print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['build','upload-and-queue']);args=a.parse_args()
    build() if args.action=='build' else upload_and_queue()
