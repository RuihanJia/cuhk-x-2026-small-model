"""Resolve measured Thermal cohort differences before launching any calibration job."""
from pathlib import Path
from datetime import datetime
import copy,json,sys
import generalization_cloud as cloud
ROOT=cloud.ROOT;BASE=cloud.BASE;CLOUD=BASE/'cloud';LOCAL=BASE/'local';sys.path.insert(0,str(LOCAL/'bundle'))
from dual_host_runtime.io import atomic,read
from research_runtime.common import sha

def main():
    old=CLOUD/'fusion_protocol.json';target=CLOUD/'fusion_protocol_align2.json';assert old.exists() and not target.exists()
    source=ROOT/'tools/generalization_fusion_worker.py';worker=CLOUD/'bundle/generalization_fusion_worker_align2.py';assert not worker.exists();worker.write_bytes(source.read_bytes())
    p=copy.deepcopy(read(old));p['id']+='-SOURCE-ALIGNMENT-REV2';p['execution_parent_protocol_sha256']=sha(old);p['runtime_worker']=worker.name;p['amended_at_local']=datetime.now().astimezone().isoformat(timespec='seconds')
    p['source_alignment']=dict(fusion_reference='All2931video/sensor records; per-fold corresponding rows',Thermal_valid_records=2891,Thermal_extra_vs_video=105,video_missing_Thermal=145,rule='Validate every model against its own full valid source cohort. Join predictions by exact clipID onto the video cohort; missing Thermal/IMU branches get zero weight and remaining weights renormalize. Do not use row position or fabricate missing predictions.')
    p['sensor_execution_amendment']=dict(protocol='runs/generalization_training_20260910_v1/local/sensor_local_protocol.json',sha256=sha(LOCAL/'sensor_local_protocol.json'),scientific_protocol_sha256=sha(CLOUD/'sensor_protocol.json'),normalizer_reconstruction_tolerance=dict(rtol=1e-6,atol=1e-7,reason='FP32 NumPy reduction across Windows training and Linux reconstruction; exact source hashes and normalizer-user/row exclusions are still checked'))
    p['code_sha256']={x.relative_to(CLOUD/'bundle').as_posix():sha(x) for x in (CLOUD/'bundle').rglob('*.py')};atomic(target,p)
    atomic(BASE/'FUSION_INITIAL_PREFLIGHT_ERROR.json',dict(status='RESOLVED_BEFORE_ANY_FUSION_FIT',original_protocol_sha256=sha(old),issue='Initial preflight assumed Thermal IDs were a subset of video/sensor IDs;105Thermal-onlyclips disproved that assumption. Original draft and code are preserved.',new_protocol_sha256=sha(target),fused_outer_results_seen=False))
    import generalization_fusion_worker as implementation
    raw,master,cohorts=implementation.load_cohorts(CLOUD,p);q=copy.deepcopy(p);q['imu_admission']['path']='IMU_FUSION_ADMISSION.json';checked=[]
    for key,entry in q['sources'].items():
        if entry['path'].startswith('fusion_inputs/local/'):
            entry['path']='local/output/'+key;implementation.load_job(BASE,q,key,raw,master,cohorts);checked.append(key)
    assert len(checked)==15
    atomic(BASE/'FUSION_SOURCE_CHECKS.json',dict(status='PASS_ALL15_LOCAL_CHECKPOINTS_PREDICTIONS_SPLITS_AND_ADMISSIONS',checked=checked,union_records=len(master),cohort_sizes={k:len(v) for k,v in cohorts.items()},outer_local_validation='Source identity checks only; no fitted fusion or fused outer score'))
    manifest=read(BASE/'FUSION_UPLOAD_MANIFEST.json')
    for rel,path in [('fusion_protocol_align2.json',target),('bundle/'+worker.name,worker)]:manifest['files'][rel]=dict(local_path=str(path),sha256=sha(path),bytes=path.stat().st_size)
    atomic(BASE/'FUSION_UPLOAD_MANIFEST.json',manifest)
    fact=dict(event='E-GENERALIZATION-FUSION-FROZEN-20260911-V1',status='READY_FOR_GATED_CPU_QUEUE',at_local=p['amended_at_local'],protocol=target.relative_to(ROOT).as_posix(),protocol_sha256=sha(target),strategies=p['strategies'],max_CPU_fits=14,additional_GPU_fits=0,cloud_deadline_unix=p['budget']['deadline_unix'],upload_bytes=sum(v['bytes'] for v in manifest['files'].values()),fused_outer_results_seen=False)
    atomic(BASE/'FUSION_REGISTRATION.json',fact)
    with (ROOT/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(fact,ensure_ascii=False)+'\n')
    print(json.dumps(fact,ensure_ascii=False))

if __name__=='__main__':main()
