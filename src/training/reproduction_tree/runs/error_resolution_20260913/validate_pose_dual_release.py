"""Independent NumPy FP64 fusion replay and complete cross-host artifact checks."""
from pathlib import Path
import json,csv,hashlib,time,shutil
import numpy as np
R=Path(__file__).resolve().parent;ROOT=R.parents[1];D=ROOT/'deliverables/pose_dual_context_candidate_20260914'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def logsumexp(a,axis):
    m=np.max(a,axis=axis,keepdims=True);return (m+np.log(np.exp(a-m).sum(axis=axis,keepdims=True))).squeeze(axis)
def main():
    manifest=read(D/'INFERENCE_MANIFEST.json')
    for n,q in manifest['files'].items():assert sha(D/n)==q['sha256'],n
    spec=read(D/'release.json');files={**spec['checkpoints'],**spec['extra_checkpoints']};total=(D/'calibrator.json').stat().st_size
    for k,q in files.items():assert sha(D/q['path']).upper()==q['sha256'].upper(),k;total+=(D/q['path']).stat().st_size
    assert total==82569140 and total<100000000
    for n,q in read(ROOT/'CORE_EVIDENCE_LOCK.json')['files'].items():assert sha(ROOT/n).upper()==q['sha256'] and (ROOT/n).stat().st_size==q['bytes'],n
    local=dict(np.load(D/'submission.csv.logits.npz',allow_pickle=False));cloud=dict(np.load(R/'release_cloud_mirror/cloud_submission.csv.logits.npz',allow_pickle=False));metrics={}
    for k in local:
        assert k in cloud and local[k].shape==cloud[k].shape,k
        if local[k].dtype.kind in 'UOSb':assert np.array_equal(local[k],cloud[k]),k
        else:
            err=float(np.max(np.abs(local[k].astype(np.float64)-cloud[k].astype(np.float64))));metrics[k]=err;assert err<.0003,(k,err)
    assert np.array_equal(local['fused'].argmax(1),cloud['fused'].argmax(1));assert sha(D/'submission.csv')==sha(R/'release_cloud_mirror/cloud_submission.csv')
    a=read(D/'submission.csv.geometry.json');b=read(R/'release_cloud_mirror/cloud_submission.csv.geometry.json');assert a['clip_ids']==b['clip_ids'];box_diff=sum(x['box']!=y['box'] for x,y in zip(a['regions'],b['regions']));assert box_diff==0,box_diff
    cal=read(D/'calibrator.json');assert cal['description']==dict(branches=list(range(6)),mode='convex',availability=False);state=cal['state'];assert set(state)=={'temperature_raw','bias'};T=.25+7.75/(1+np.exp(-np.asarray(state['temperature_raw'],np.float64)));assert np.max(np.abs(T-np.asarray(cal['temperature'])))<1e-12
    z=local['branch_logits'].astype(np.float64)/T[None,:,None];lp=z-logsumexp(z,2)[:,:,None];scores=np.broadcast_to(np.asarray(state['bias'],np.float64),local['available'].shape).copy();scores[~local['available']]=-np.inf;lw=scores-logsumexp(scores,1)[:,None];fused=logsumexp(lp+lw[:,:,None],1);weights=np.exp(lw);assert np.all(weights[~local['available']]==0);assert np.max(np.abs(np.exp(fused).sum(1)-1))<1e-12
    ferr=float(np.max(np.abs(fused.astype(np.float32)-local['fused'])));werr=float(np.max(np.abs(weights.astype(np.float32)-local['weights'])));assert ferr<2e-6 and werr<2e-7
    with (D/'submission.csv').open(encoding='utf-8-sig') as f:
        reader=csv.DictReader(f);assert reader.fieldnames==['path','prediction'];rows=list(reader)
    expected=read(D/'FROZEN_SAMPLE_PATHS.json')['normalized_output_paths'];assert len(rows)==405 and len(set(q['path'] for q in rows))==405;assert [q['path'] for q in rows]==expected==local['output_paths'].tolist();labels=np.asarray([int(q['prediction']) for q in rows]);assert ((labels>=0)&(labels<40)).all();assert np.array_equal(labels,fused.argmax(1))
    old=ROOT/'deliverables/submission_activity_token_stopgrad_fixed_r5_20260913.csv'
    with old.open(encoding='utf-8-sig') as f:before=list(csv.DictReader(f))
    assert [q['path'] for q in before]==expected;changed=sum(q['prediction']!=p['prediction'] for q,p in zip(before,rows))
    done=read(R/'release_local_output_v2/DONE.json');cd=read(R/'release_cloud_mirror/DONE.json');assert done['at_unix']<1789323192.0883653 and cd['at_unix']<1789323192.0883653;assert read(R/'release_local_output_v2/CONTROLLER_DONE.json')['exit_code']==read(R/'release_cloud_mirror/CONTROLLER_DONE.json')['exit_code']==0
    frozen=read(R/'production_calibration/FROZEN.json');assert frozen['at_unix']<done['at_unix'] and sha(R/'RELEASE_PROTOCOL.json')==frozen['protocol_sha256']
    result=dict(status='VERIFIED_FULL405_CROSS_HOST_AND_INDEPENDENT_NUMPY_FP64_REPLAY',at_unix=time.time(),rows=405,path_order_matches_original_template=True,csv_sha256=sha(D/'submission.csv'),inference_manifest_sha256=sha(D/'INFERENCE_MANIFEST.json'),frozen_files_verified=len(manifest['files']),serialized_model_and_calibrator_bytes=total,all_model_hashes_verified=True,cross_host_top1_mismatches=0,cross_host_csv_bytes_identical=True,cross_host_max_abs_errors=metrics,cross_host_crop_box_mismatches=box_diff,numpy_fp64_replay_fused_fp32_max_error=ferr,numpy_fp64_replay_weights_fp32_max_error=werr,changed_predictions_vs_submitted_R6=changed,changes_are_not_accuracy=True,local_elapsed_seconds=done['audit']['elapsed_seconds'],cloud_elapsed_seconds=cd['audit']['elapsed_seconds'],both_within_original_deadline=True,Test_labels_read=False,Kaggle_uploaded=False,calibration_frozen_before_Test_inference=True,limitations='Existing Windows/local and Linux/cloud dependency environments; no fresh pip installation claimed. Exposed development evidence is not Public confirmation.')
    (D/'ACCEPTANCE.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');target=ROOT/'deliverables/submission_pose_dual_context_20260914.csv';assert not target.exists() or sha(target)==sha(D/'submission.csv');shutil.copy2(D/'submission.csv',target)
    with (ROOT/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(event='POSE_DUAL6_VERIFIED_CSV',at_unix=time.time(),artifact=str(target.relative_to(ROOT)),acceptance=str((D/'ACCEPTANCE.json').relative_to(ROOT)),**{k:v for k,v in result.items() if k in ['rows','csv_sha256','serialized_model_and_calibrator_bytes','cross_host_top1_mismatches','changed_predictions_vs_submitted_R6']}))+'\n')
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
