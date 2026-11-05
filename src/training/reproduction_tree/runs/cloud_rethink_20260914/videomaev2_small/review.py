"""Audit both fixed V2 endpoints before considering any source-fit expansion."""
from pathlib import Path
import importlib.util,json,sys,time
E=Path(__file__).resolve().parent;C=E.parent;R=C.parents[1];F=C/'videomae_kinetics_fusion';sys.path[:0]=[str(F),str(C)]
sp=importlib.util.spec_from_file_location('v2_endpoint_helpers',F/'worker.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);a=m.a;np=m.np
def main():
 reg=a.read(E/'REVIEW_REGISTRATION.json');assert a.sha(E/'review.py')==reg['worker_sha256'];out=E/'review';out.mkdir(exist_ok=True);assert not(out/'DONE.json').exists()
 fs=a.read(C/'trajectory_fixed_fusion_protocol.json');parts=[];sampling={}
 for f,pn in [(0,'protocol_cloud.json'),(3,'protocol.json')]:
  s=a.read(a.check(E/pn,reg['protocols'][pn]));root=E/('cloud' if f==0 else 'output/local');m.terminal(root,reg['protocols'][pn])
  job=s['jobs'][0];fr=a.read(root/'FROZEN.json');assert fr['epochs']==16 and fr['train_users']==job['train_users'] and fr['held_users']==job['held_users'] and fr['FP16_reload_exact']
  hist=a.read(root/'HISTORY.json');oldroot=C/'videomae_kinetics'/('cloud' if f==0 else 'output/local');oldhist=a.read(oldroot/'HISTORY.json')
  assert len(hist)==len(oldhist)==16 and [x['sampling_sha256'] for x in hist]==[x['sampling_sha256'] for x in oldhist];sampling[str(f)]=True
  inp=fs['inputs'][str(f)];base=a.arr(a.check(inp['scores'],inp['scores_sha256']));new=a.align(a.arr(root/'predictions.npz'),base);old=a.align(a.arr(oldroot/'predictions.npz'),base)
  cal=a.read(a.check(inp['calibrator'],inp['calibrator_sha256']));oldpose=a.fusion(base['logits'][:,:6],base['available'][:,:6],cal);soft=lambda z:np.exp(z-m.cw.logsumexp(z,-1,True))
  v={k:base[k] for k in ['clip_ids','targets','users']};v.update(k400=old['videomae_logits'],v2=new['videomae_logits'],old_pose6=oldpose)
  v['fixed_diagnostic']=np.log(.8*soft(oldpose.astype(np.float64))+.2*soft(new['videomae_logits'].astype(np.float64))).astype(np.float32);parts.append(v)
 v={k:np.concatenate([x[k] for x in parts]) for k in parts[0]};ref=a.align(a.arr(C/'videomae_kinetics_joint_quant/output/PAIRED.npz'),v);v['current7']=ref['classwise7']
 result=dict(status='COMPLETE_V2_SMALL_MATCHED16_AUDIT',private=m.a.pair(v['targets'],v['users'],v['k400'],v['v2']),fixed_diagnostic=m.a.pair(v['targets'],v['users'],v['current7'],v['fixed_diagnostic']),source_sampling_exact=sampling,no_Test=True,no_source_fits=True,no_CSV=True,limitations='FP16storage FP32forward, repeated development; source-unmatched fixeddiagnostic is not eligible deployment fusion. No Public guarantee.',at_unix=time.time())
 np.savez_compressed(out/'PAIRED.npz',**v);a.atomic(out/'RESULT.json',result);a.atomic(out/'DONE.json',dict(status='PASS',review_registration_sha256=a.sha(E/'REVIEW_REGISTRATION.json'),artifacts={n:a.sha(out/n) for n in ['RESULT.json','PAIRED.npz']}))
 with(R/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(event='V2_SMALL_MATCHED_TWO_FOLD_COMPLETE',result=str((out/'RESULT.json').relative_to(R)),result_sha256=a.sha(out/'RESULT.json'),at_unix=time.time()))+'\n')
 print(json.dumps({k:{n:v for n,v in x.items() if n not in ['per_user','per_class']} for k,x in result.items() if isinstance(x,dict)}))
if __name__=='__main__':main()
