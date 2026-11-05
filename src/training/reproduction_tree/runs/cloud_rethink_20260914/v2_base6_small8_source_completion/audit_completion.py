"""Single CPU terminal auditor; waits quietly, never polls GPU or fits models."""
from pathlib import Path
import os,json,hashlib,time,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import numpy as np
E=Path(__file__).resolve().parent;C=E.parent;O=E/'output/local'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrays(p):
 with np.load(p,allow_pickle=False)as f:return{k:f[k]for k in f.files}
def lse(z,axis,keep=True):
 m=np.max(z,axis=axis,keepdims=True);v=m+np.log(np.exp(z-m).sum(axis=axis,keepdims=True));return v if keep else v.squeeze(axis)
def apply(z,mask,g,c=None):
 z=z.astype(np.float64)/np.array(g['temperature'])[None,:,None];lp=z-lse(z,-1)
 bias=np.asarray(g['state']['bias'])[:,None]
 if c is not None:
  d=np.array(c['delta']);bias=bias+d-d.mean(0,keepdims=True)
 w=np.broadcast_to(bias,z.shape).copy();w[~mask]=-np.inf;w-=lse(w,1);p=lse(w+lp,1,False)
 if c is not None:p-=lse(p,-1)
 return p.astype(np.float32)
def align(v,target):
 idx={str(x):i for i,x in enumerate(v['clip_ids'])};assert len(idx)==len(v['clip_ids']);ix=np.array([idx[str(x)]for x in target['clip_ids']]);r={k:a[ix]for k,a in v.items()}
 for k in ['clip_ids','targets','users']:assert np.array_equal(r[k],target[k])
 return r
def main():
 s=read(E/'protocol.json');ph=sha(E/'protocol.json');deadline=s['deadline_unix']+60
 while not(O/'CONTROLLER_DONE.json').exists():
  assert time.time()<deadline,'AUDIT_WAIT_DEADLINE'
  if(O/'CONTROLLER_ERROR.json').exists():raise RuntimeError('MAIN_CONTROLLER_FAILED:'+read(O/'CONTROLLER_ERROR.json')['error'])
  time.sleep(30)
 ct=read(O/'CONTROLLER_DONE.json');d=read(O/'DONE.json');assert ct['status']=='COMPLETE'and ct['exit_code']==0 and ct['protocol_sha256']==d['protocol_sha256']==ph and ct['terminal_sha256']==sha(O/'DONE.json')
 for n,h in d['artifacts'].items():assert sha(O/n)==h,n
 for n,h in s['bindings'].items():assert sha(n)==h,n
 inputs=read(E/'INPUTS.json');source=arrays(O/'SOURCE_FITTED.npz');outer=arrays(O/'PAIRED.npz');freeze=read(O/'CALIBRATORS_FROZEN.json');result=read(O/'RESULT.json')
 assert freeze['calibrators']==4 and freeze['source_encoders']==6 and freeze['at_unix']<result['at_unix']
 for n,h in freeze['files'].items():assert sha(O/n)==h,n
 assert len(source['users'])==4768 and len(set(source['clip_ids']))==2931 and len(set(zip(source['source_fold'],source['clip_ids'])))==4768
 assert source['TI'].shape==(4768,2,40)and source['available'].shape==(4768,4)
 assert len(outer['users'])==1094 and len(set(outer['clip_ids']))==1094
 source_replay_error=0.;outer_replay_error=0.;newjob_counts={}
 fs=read(s['fixed_fusion_protocol'])
 for f in [0,3]:
  g=read(O/f'F{f}_global.json');c=read(O/f'F{f}_classwise.json');excluded=set(s['outer_users'][str(f)])
  assert len(g['temperature'])==4 and c['parameters']==160 and c['penalty']==.1
  assert not(g['outer_labels_used']or c['outer_labels_used'])and set(g['calibration_users']).isdisjoint(excluded)and set(c['calibration_users']).isdisjoint(excluded)
  part={k:v[source['source_fold']==f]for k,v in source.items()};assert set(part['users'])==set(g['calibration_users'])and len(part['users'])==g['calibration_records']
  z=np.concatenate([part['TI'],part['base6'][:,None],part['small8'][:,None]],1)
  for key,cal in [('global4',None),('classwise4',c)]:source_replay_error=max(source_replay_error,float(abs(apply(z,part['available'],g,cal)-part[key]).max()))
  opart={k:v[outer['fold']==f]for k,v in outer.items()};base=align(arrays(fs['inputs'][str(f)]['scores']),opart)
  z=np.concatenate([base['logits'][:,[2,3]],opart['base6'][:,None],opart['small8'][:,None]],1);mask=np.concatenate([base['available'][:,[2,3]],np.ones((len(opart['users']),2),bool)],1)
  for key,cal in [('global4',None),('classwise4',c)]:outer_replay_error=max(outer_replay_error,float(abs(apply(z,mask,g,cal)-opart[key]).max()))
  for j in [j for j in inputs['jobs']if j['fold']==f]:
   root=Path(s['pilot_directory'])if j['reuse_pilot']else O/j['id'];jd=read(root/'DONE.json')
   b=arrays(root/'Base.npz');sm=align(arrays(root/'Small.npz'),b)
   assert np.array_equal(b['five_view_logits'].mean(1),b['five_logits'])and np.array_equal(sm['five_view_logits'].mean(1),sm['five_logits'])
   assert np.array_equal(.5*b['five_logits']+.25*b['temporal_logits'][:,0]+.25*b['temporal_logits'][:,1],b['seven_logits'])
   assert len(b['users'])==j['records']and set(b['users'])==set(j['held_users'])
   assert set(j['train_users']).isdisjoint(set(j['held_users'])|set(j['outer_excluded_users']))
   target=part if j['scope']=='source'else opart;ids=set(b['clip_ids']);sub={k:v[np.array([x in ids for x in target['clip_ids']])]for k,v in target.items()};bound=align(b,sub);smallbound=align(sm,sub)
   assert np.array_equal(bound['seven_logits'],sub['base6'])and np.array_equal(smallbound['five_logits'],sub['small8'])
   newjob_counts[j['id']]=dict(records=j['records'],reuse_pilot=j['reuse_pilot'])
 assert source_replay_error<1e-6 and outer_replay_error<1e-6
 metrics={};y=outer['targets'];old=outer['current'].argmax(1)==y
 for key,reported in result['pooled'].items():
  z=outer[key].astype(np.float64);new=z.argmax(1)==y;lp=z-lse(z,-1);nll=float(-lp[np.arange(len(y)),y].mean())
  item=dict(correct=int(new.sum()),nll=nll,fixes=int((~old&new).sum()),harms=int((old&~new).sum()))
  assert item['correct']==reported['after']['correct']and abs(nll-reported['after']['nll'])<1e-10 and item['fixes']==reported['fixes']and item['harms']==reported['harms'];metrics[key]=item
 report=dict(status='PASS_INDEPENDENT_SOURCE_COMPLETION_CPU_AUDIT',protocol_sha256=ph,result_sha256=sha(O/'RESULT.json'),controller_done_sha256=sha(O/'CONTROLLER_DONE.json'),all_frozen_bindings_unchanged=True,source_replay_max_abs=source_replay_error,outer_replay_max_abs=outer_replay_error,source_rows=4768,source_unique=2931,source_encoder_isolation_verified=True,all_four_source_calibrators_frozen=True,readouts_raw_seven_five_replayed=True,jobs=newjob_counts,metrics=metrics,no_GPU=True,no_fit=True,no_Test=True,at_unix=time.time())
 (O/'INDEPENDENT_AUDIT.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
if __name__=='__main__':
 try:main()
 except BaseException as e:
  (O/'INDEPENDENT_AUDIT_ERROR.json').write_text(json.dumps(dict(error=str(e),at_unix=time.time()),indent=2),encoding='utf-8');raise
