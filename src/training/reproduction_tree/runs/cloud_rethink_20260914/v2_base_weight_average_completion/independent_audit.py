"""Independent NumPy probability-domain replay; no fit or GPU."""
from pathlib import Path
import hashlib,json,time
import numpy as np
E=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf8'))
def arr(p):
 with np.load(p,allow_pickle=False)as v:return {k:v[k]for k in v.files}
def softmax(z,axis):
 z=np.asarray(z,np.float64);p=np.exp(z-z.max(axis,keepdims=True));return p/p.sum(axis,keepdims=True)
def replay(z,mask,cal,kind):
 p=softmax(z.astype(np.float64)/np.array(cal['temperature'])[None,:,None],2)
 if kind=='global':bias=np.broadcast_to(np.array(cal['state']['bias'])[None,:,None],z.shape).copy()
 else:
  delta=np.array(cal['delta']);delta-=delta.mean(0,keepdims=True);bias=np.broadcast_to(np.array(cal['global_bias'])[None,:,None]+delta[None],z.shape).copy()
 bias[~mask]=-np.inf;value=(p*softmax(bias,1)).sum(1)
 if kind=='classwise':value/=value.sum(1,keepdims=True)
 return np.log(value).astype(np.float32)
def main():
 out=E/'fusion';ph=sha(E/'protocol.json');ap=read(E/'audit_protocol.json');assert ap['protocol_sha256']==ph and ap['audit_sha256']==sha(E/'independent_audit.py')
 d=read(out/'DONE.json');fr=read(out/'CALIBRATORS_FROZEN.json');assert d['status']=='COMPLETE'and d['protocol_sha256']==fr['protocol_sha256']==ph and fr['calibrators']==4 and fr['all_source_calibrators_before_outer_arrays']
 for n,h in d['artifacts'].items():assert sha(out/n)==h,n
 for n,h in fr['files'].items():assert sha(out/n)==h,n
 src=arr(out/'SOURCE_FITTED.npz');outer=arr(out/'PAIRED.npz');r=read(out/'RESULT.json');assert len(src['users'])==4768 and len(set(src['clip_ids']))==2931 and len(set(zip(src['source_fold'],src['clip_ids'])))==4768 and len(outer['users'])==1094
 errors={};outerusers={'0':[2,16,20],'3':[5,7,8,21]}
 for fold in [0,3]:
  for kind,key in [('global','global3'),('classwise','classwise3')]:
   cal=read(out/f'F{fold}_{kind}.json');assert not cal['outer_labels_used']and not set(cal['calibration_users'])&set(outerusers[str(fold)])
   for scope,v,fk in [('source',src,'source_fold'),('outer',outer,'fold')]:
    ix=v[fk]==fold;z=np.concatenate([v['TI'][ix],v['video'][ix,None]],1);q=replay(z,v['available'][ix],cal,kind);err=float(np.max(np.abs(q-v[key][ix])));assert err<=2e-6 and np.array_equal(q.argmax(1),v[key][ix].argmax(1));errors[f'{scope}_F{fold}_{kind}']=err
 for ref,resultkey,count in [('H','vs_H',928),('CONTROL','vs_CONTROL',927)]:
  x=outer[ref].argmax(1)==outer['targets'];assert int(x.sum())==count
  for key in ['video','global3','classwise3']:
   z=outer[key].argmax(1)==outer['targets'];q=r[resultkey][key];assert q['before']==count and q['after']==int(z.sum())and q['fix']==int((~x&z).sum())and q['harm']==int((x&~z).sum())
   for u in np.unique(outer['users']):
    ix=outer['users']==u;v=q['per_user'][str(u)];assert v['before']==int(x[ix].sum())and v['after']==int(z[ix].sum())and v['fix']==int((~x[ix]&z[ix]).sum())and v['harm']==int((x[ix]&~z[ix]).sum())
 result=dict(status='PASS_INDEPENDENT_FIXED_AVERAGE_SOURCE_AND_OUTER_REPLAY',protocol_sha256=ph,result_sha256=sha(out/'RESULT.json'),replay_maxabs=errors,source_rows=4768,source_unique_clips=2931,outer_rows=1094,calibration_fits_in_audit=0,GPU_forwards=0,no_Test=True,at_unix=time.time())
 assert not(out/'INDEPENDENT_AUDIT.json').exists();(out/'INDEPENDENT_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n','utf8');print(json.dumps(result))
if __name__=='__main__':main()
