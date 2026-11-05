"""One source-excluded quality gate per development fold; no outer-label fitting."""
from pathlib import Path
import json,sys,time,hashlib
import numpy as np
import torch
R=Path(__file__).resolve().parent;REPO=R.parents[1]
sys.path.insert(0,str(REPO/'deliverables/activity_token_stopgrad_candidate_20260913'))
import fusion_math as fm
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,j):Path(p).write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def arrays(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
class Guard:
 def __init__(self,deadline):self.deadline=deadline
 def check(self,**kwargs):assert time.time()<self.deadline,'CPU_TIME_BOUND'
def pair(y,u,a,b):
 a=a.argmax(1)==y;b=b.argmax(1)==y
 return dict(n=len(y),before=int(a.sum()),after=int(b.sum()),fix=int((~a&b).sum()),harm=int((a&~b).sum()),net=int(b.sum()-a.sum()),per_user={str(k):int(b[u==k].sum()-a[u==k].sum()) for k in np.unique(u)})
def main():
 torch.set_num_threads(2);out=R/'quality_gate_output';out.mkdir(exist_ok=True)
 assert not(out/'PROTOCOL.json').exists(),'NO_DUPLICATE_FITS'
 source=REPO/'runs/error_driven_overnight_20260913/imu/successors/calibration_export/output'
 sources={};inner={};controls={}
 for f,held in [(0,[2,16,20]),(3,[5,7,8,21])]:
  p=source/f'fold{f}';done=read(p/'DONE.json');assert done['status']=='COMPLETE'
  for n in ['control_inner.npz','control_calibrator.json','CALIBRATION_FROZEN.json']:assert sha(p/n).upper()==done['artifacts'][n]
  evidence=read(p/'CALIBRATION_FROZEN.json');data=arrays(p/'control_inner.npz');control=read(p/'control_calibrator.json')
  assert len(data['clip_ids'])=={0:2419,3:2349}[f] and not set(data['users'])&set(held)
  assert set(control['calibration_users'])==set(data['users']) and not control['outer_labels_used']
  for entry in evidence['sources'][str(f)]:
   part=entry['partition'];assert set(part['forbidden_outer_users'])==set(held)
   assert not (set(part['train_users'])|set(part['predict_users']))&set(held)
   assert not set(part['train_users'])&set(part['predict_users'])
   v=entry['video'];vp=Path(v['root'])/'output'/v['job'];assert sha(vp/'predictions.npz').upper()==v['predictions_sha256']
   assert sha(vp/'SPLIT.json') and sorted(read(vp/'SPLIT.json')['train_users'])==sorted(part['train_users'])
  inner[f]=data;controls[f]=control;sources[str(f)]={n:dict(path=str(p/n),sha256=sha(p/n)) for n in ['control_inner.npz','control_calibrator.json','CALIBRATION_FROZEN.json','DONE.json']}
 deadline=min(time.time()+600,read(R/'AUTHORIZATION.json')['execution_deadline_unix'])
 protocol=dict(id='R6_SOURCE_MATCHED_QUALITY_GATE_V1',at_unix=time.time(),hypothesis='Per-sample branch uncertainty and IR-Depth disagreement may recover existing complementary errors that global convex weights suppress.',strategy='gate_vsti_no_availability',control='existing R6 source-matched convex_vsti',fit_users='each outer fold completely excluded from all source models and calibrator fitting',maximum_CPU_fits=2,max_iterations=500,regularizer=.01,quality_features='class-agnostic raw entropy, probability margin, IR-Depth JS',no_class_or_user_routing=True,no_Test=True,no_Public_search=True,source=sources,code_sha256=sha(__file__),fusion_code_sha256=sha(fm.__file__),deadline_unix=deadline,prior_evidence='Older pre-activity gate and convex were equal on both folds; this single finite successor uses now-available exact R6 inner models, not a parameter scan.',post_audit_design=True)
 write(out/'PROTOCOL.json',protocol)
 with (REPO/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf-8') as fp:fp.write(json.dumps(dict(event='REGISTERED',experiment=protocol['id'],protocol=str((out/'PROTOCOL.json').relative_to(REPO)),protocol_sha256=sha(out/'PROTOCOL.json'),at_unix=time.time(),fits=2,CPU_seconds_cap=600))+'\n')
 guard=Guard(deadline)
 for f in [0,3]:
  m=fm.fit(inner[f],protocol['strategy'],guard,max_iter=500);write(out/f'F{f}_gate.json',m)
 write(out/'CALIBRATION_FROZEN.json',dict(at_unix=time.time(),outer_arrays_loaded=False,models={str(f):sha(out/f'F{f}_gate.json') for f in [0,3]}))
 allrows=[];result={};old=REPO/'runs/activity_token_stopgrad_20260913';binding=read(old/'PIPELINE_PROTOCOL.json')['bindings']
 for f in [0,3]:
  b=binding[str(f)];data=arrays(old/b['data']);stop=arrays(old/f'analysis/F{f}_FIXED_R3_WITH_STOPGRAD.npz');assert np.array_equal(data['clip_ids'],stop['clip_ids'])
  crop=arrays(R/f'crop_probe_output/F{f}_INTERVENTIONS.npz');lookup={k:i for i,k in enumerate(crop['clip_ids'])};ix=[lookup[k] for k in data['clip_ids']];phase=crop['baseline_phase'][ix]
  data['logits'][:,0]=.5*(phase[:,:,0].mean((1,2))+phase[:,:,1].mean((1,2)));data['depth']=.5*(phase[:,:,0,0].mean(1)+phase[:,:,1,0].mean(1));data['ir']=.5*(phase[:,:,0,1].mean(1)+phase[:,:,1,1].mean(1))
  a=fm.predict(data,controls[f]);g,w=fm.predict(data,read(out/f'F{f}_gate.json'),True)
  reference=arrays(source/f'fold{f}'/'predictions.npz');assert np.array_equal(reference['clip_ids'],data['clip_ids']);assert np.allclose(a,reference['control'],atol=2e-6,rtol=0)
  result[str(f)]=dict(vs_matched=pair(data['targets'],data['users'],a,g),vs_R6=pair(data['targets'],data['users'],stop['STOPGRAD'],g))
  np.savez_compressed(out/f'F{f}_predictions.npz',clip_ids=data['clip_ids'],targets=data['targets'],users=data['users'],matched_control=a,gate=g,weights=w,R6=stop['STOPGRAD'])
  allrows.append(dict(targets=data['targets'],users=data['users'],a=a,g=g,r6=stop['STOPGRAD']))
 z={k:np.concatenate([v[k] for v in allrows]) for k in allrows[0]};final=dict(status='COMPLETE',at_unix=time.time(),protocol_sha256=sha(out/'PROTOCOL.json'),per_fold=result,pooled_vs_matched=pair(z['targets'],z['users'],z['a'],z['g']),pooled_vs_R6=pair(z['targets'],z['users'],z['r6'],z['g']),no_Test=True,limitation='Seven repeatedly exposed development users; not unbiased Public forecast.')
 write(out/'RESULT.json',final)
 with (REPO/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf-8') as fp:fp.write(json.dumps(dict(event='COMPLETED',experiment=protocol['id'],artifact=str((out/'RESULT.json').relative_to(REPO)),result=final))+'\n')
 print(json.dumps(final))
if __name__=='__main__':main()
