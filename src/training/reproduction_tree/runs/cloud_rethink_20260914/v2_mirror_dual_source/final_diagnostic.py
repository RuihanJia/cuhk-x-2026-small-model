"""Only after the frozen four-source fit and independent replay have completed."""
from pathlib import Path
import json,hashlib,time,numpy as np
E=Path(__file__).resolve().parent;F=E/'fusion';O=F/'output'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert read(F/'CALLBACK_DONE.json')['status']=='COMPLETE_MIRROR_DUAL_SOURCE_FIT_AND_INDEPENDENT_AUDIT'
a=read(O/'INDEPENDENT_AUDIT.json');assert a['status']=='PASS_INDEPENDENT_MIRROR_DUAL_SOURCE_AND_OUTER_REPLAY'and a['result_sha256']==sha(O/'RESULT.json')
with np.load(O/'PAIRED.npz',allow_pickle=False)as v:p={k:v[k]for k in v.files}
y=p['targets'];u=p['users'];z=p['classwise4'].argmax(1);rows=[];refs={k:p[k].argmax(1)for k in ['RTN','Mean','H','SubmittedBase']}
for i in range(len(y)):
 if any(v[i]!=z[i]for v in refs.values()):rows.append(dict(clip_id=str(p['clip_ids'][i]),user=int(u[i]),target=int(y[i]),new=int(z[i]),references={k:int(v[i])for k,v in refs.items()},effect_vs_RTN='fix'if z[i]==y[i]and refs['RTN'][i]!=y[i]else'harm'if z[i]!=y[i]and refs['RTN'][i]==y[i]else'wrong_to_wrong'))
def stats(ids):
 out={}
 for k in ids:
  ix=y==k;pred=z[ix];label,counts=np.unique(pred,return_counts=True)
  out[str(k)]=dict(n=int(ix.sum()),mirror_correct=int((pred==k).sum()),before={n:int((v[ix]==k).sum())for n,v in refs.items()},new_prediction_histogram={str(int(t)):int(c)for t,c in zip(label,counts)})
 return out
r=dict(status='PASS_POST_FROZEN_MIRROR_CHANGED_ROWS_DIAGNOSTIC',result_sha256=sha(O/'RESULT.json'),independent_sha256=sha(O/'INDEPENDENT_AUDIT.json'),changed=rows,focus=stats([19,21,22,37]),all_classes=stats(sorted(set(y.tolist()))),no_fit=True,no_Test=True,at_unix=time.time())
(E/'FINAL_CHANGED_ROWS.json').write_text(json.dumps(r,indent=2),encoding='utf-8');print(json.dumps(dict(focus=r['focus'],changed_rows=len(rows))))
