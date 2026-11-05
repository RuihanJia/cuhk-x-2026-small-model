from pathlib import Path
import os,json,hashlib
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
E=Path(__file__).resolve().parent;C=E.parent;R=C.parents[1]
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def describe(path):
 q=read(path);b=np.asarray(q['global_bias']);d=np.asarray(q['delta']);d-=d.mean(0,keepdims=True);w=np.exp(b-b.max());w/=w.sum();c=b[:,None]+d;c=np.exp(c-c.max(0,keepdims=True));c/=c.sum(0,keepdims=True)
 return dict(path=str(path),sha256=sha(path),rows=q['calibration_rows'],users=q['calibration_users'],temperature=q['temperature'],global_weight=w.tolist(),classwise_weight_mean=c.mean(1).tolist(),classwise_weight_min=c.min(1).tolist(),classwise_weight_max=c.max(1).tolist(),centered_delta_rms=float(np.sqrt((d*d).mean())),penalty=q['penalty'],source_user_balanced_nll=q['source_user_balanced_nll'])
result=dict(status='SOURCE_FINAL_CALIBRATION_PARAMETER_COMPARISON_NO_FIT',branch_order=['T','I','BaseAverage'],development={str(f):describe(C/f'v2_base_weight_average_completion/fusion/F{f}_classwise.json')for f in (0,3)},final=describe(R/'deliverables/videomaev2_weight_average_candidate_20260915/calibrator.json'),interpretation='Final calibration intentionally uses all4768source rows and18users after development choice; it is not an outer refit and its in-sample NLL is not new validation.',no_Test=True)
(E/'CALIBRATION_COMPARISON.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
