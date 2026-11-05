"""Two frozen source-only fusion alternatives, fit before either outer fold is read."""
from pathlib import Path
import os,sys,importlib.util,json,time,argparse
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
E=Path(__file__).resolve().parent;C=E.parent;P=C/'aux_public_regression_20260914/dynamic_interaction/photometric_candidate'
sys.path[:0]=[str(E),str(C)]
sp=importlib.util.spec_from_file_location('source_fusion_audit',P/'recovery_independent_audit.py');a=importlib.util.module_from_spec(sp);sp.loader.exec_module(a)
import numpy as np
import torch
import frozen_fusion_math as fm
import gate
spc=importlib.util.spec_from_file_location('classwise_calibration',C/'videomae_classwise/gate.py');cw=importlib.util.module_from_spec(spc);spc.loader.exec_module(cw)

class Clock:
 def __init__(self,deadline):self.deadline=deadline
 def check(self,**kw):assert time.time()<self.deadline,'FUSION_CPU_CAP'

def values(p):return a.arr(p)
def subset(v,ix):return {k:x[ix].copy() for k,x in v.items() if x.ndim and len(x)==len(ix)}
def terminal(root,ph):
 d=a.read(root/'DONE.json');ct=a.read(root/'CONTROLLER_DONE.json');assert d['protocol_sha256']==ct['protocol_sha256']==ph and ct['status']=='COMPLETE' and ct['exit_code']==0 and ct['terminal_sha256']==a.sha(root/'DONE.json')
 for n,h in d['artifacts'].items():a.check(root/n,h)

def convex_replay(data,cal):
 model=fm.Calibrator(cal['description']);model.load_state_dict({k:torch.tensor(v,dtype=torch.float64) for k,v in cal['state'].items()},strict=True);model.eval()
 batch=dict(logits=torch.tensor(data['logits'],dtype=torch.float64),mask=torch.tensor(data['available'],dtype=torch.bool))
 with torch.no_grad():z=model(batch).numpy().astype(np.float32)
 return z

def main(protocol,mode):
 start=time.time();s=a.read(protocol);ph=a.sha(protocol);assert s['status']=='FROZEN_FOR_EXECUTION' and mode in ['raw','quant']
 for n,h in s['code_sha256'].items():a.check(C/n,h)
 out=E/mode;out.mkdir(exist_ok=True);assert not(out/'DONE.json').exists();clock=Clock(start+180);torch.set_num_threads(1)
 fm.STRATEGIES['convex_POSE6_VIDEOMAE7']=dict(branches=list(range(7)),mode='convex',availability=False)
 audit=C/'videomae_kinetics_source_rest/audit';d=a.read(a.check(audit/'DONE.json',s['source_audit_done_sha256']));assert d['status']=='PASS'
 for n,h in d['artifacts'].items():a.check(audit/n,h)
 fs=a.read(a.check(C/'trajectory_fixed_fusion_protocol.json',s['fixed_fusion_protocol_sha256']));source=values(a.check(s['source_oof_file'],s['source_oof_sha256']));assert source['logits'].shape==(4768,10,40) and len(set(source['original_clip_ids']))==2931
 qjobs=[]
 if mode=='quant':
  qs=a.read(a.check(C/'videomae_kinetics_quant/protocol.json',s['quant_replay_protocol_sha256']));terminal(C/'videomae_kinetics_quant/cloud',s['quant_replay_protocol_sha256']);qjobs=qs['jobs']
 cfiles={};sourceparts=[];sourcesummary={}
 for f in (0,3):
  ix=source['source_fold']==f;base=subset(source,ix);base['clip_ids']=base['original_clip_ids'];base['logits']=base['logits'][:,:6].copy();base['available']=base['available'][:,:6].copy();binding=s['folds'][str(f)]
  snap=values(a.check(binding['source_scores'],binding['source_scores_sha256']))
  for k in ['clip_ids','targets','users']:assert np.array_equal(base[k],snap[k])
  assert np.array_equal(base['logits'],snap['logits'][:,:6]) and np.array_equal(base['available'],snap['available'][:,:6]) and sorted(set(base['users']))==binding['source_users']
  raw=a.align(values(audit/f'F{f}_SOURCE_PAIRED.npz'),base);assert np.array_equal(raw['old_private'],base['logits'][:,5])
  new=raw['videomae_private']
  if mode=='quant':
   parts=[]
   for j in qjobs:
    if j['scope']=='source' and j['fold']==f:
     path=C/'videomae_kinetics_quant/cloud'/j['id'];jd=a.read(path/'DONE.json');assert jd['no_optimizer'] and jd['outer_excluded_users']==binding['outer_users'];parts.append(values(a.check(path/'predictions.npz',jd['predictions_sha256'])))
   q=a.align({k:np.concatenate([v[k] for v in parts]) for k in parts[0]},base);new=q['videomae_logits']
  data=dict(base,logits=np.concatenate([base['logits'],new[:,None]],1),available=np.concatenate([base['available'],np.ones((len(new),1),bool)],1))
  oldinfo=fs['inputs'][str(f)];oldcal=a.read(a.check(oldinfo['calibrator'],oldinfo['calibrator_sha256']));assert oldcal['calibration_users']==binding['source_users'] and not oldcal['outer_labels_used']
  old=a.fusion(base['logits'],base['available'],oldcal);a.near(old,fm.predict(base,oldcal))
  globalcal=(a.read(a.check(out/f'F{f}_global.json',a.read(out/'CALIBRATORS_FROZEN.json')['files'][f'F{f}_global.json'])) if (out/'CALIBRATORS_FROZEN.json').exists() else fm.fit(data,'convex_POSE6_VIDEOMAE7',clock,500));assert globalcal['calibration_users']==binding['source_users'] and not globalcal['outer_labels_used']
  globalpred=convex_replay(data,globalcal);a.near(globalpred,a.fusion(data['logits'],data['available'],globalcal))
  gated=(a.read(a.check(out/f'F{f}_gate.json',a.read(out/'CALIBRATORS_FROZEN.json')['files'][f'F{f}_gate.json'])) if (out/'CALIBRATORS_FROZEN.json').exists() else gate.fit(gate.softmax(old),new,base['logits'][:,1:4],base['available'][:,1:4],base['targets'],base['users'],globalcal['temperature'][-1],clock))
  gatepred,alpha=gate.predict(gate.softmax(old),new,base['logits'][:,1:4],base['available'][:,1:4],gated)
  classcal=cw.fit(data['logits'],data['available'],base['targets'],base['users'],globalcal,clock);classpred=cw.predict(data['logits'],data['available'],classcal);assert classcal['calibration_users']==binding['source_users']
  for name,cal in [('global',globalcal),('gate',gated),('classwise',classcal)]:
   p=out/f'F{f}_{name}.json'
   if not p.exists():a.atomic(p,cal)
   cfiles[p.name]=a.sha(p)
  v={k:base[k] for k in ['clip_ids','targets','users']};v.update(source_fold=np.full(len(new),f),old_pose6=old,new_videomae=new,global7=globalpred,confidence_gate=gatepred,classwise7=classpred,alpha=alpha);sourceparts.append(v)
  sourcesummary[str(f)]=dict(global_nll=globalcal['final_user_balanced_nll'],global_temperature=globalcal['temperature'],global_weights=gate.softmax(np.asarray(globalcal['state']['bias'])[None])[0].tolist(),gate_alpha_mean=float(alpha.mean()),source_fit_metrics_not_validation={n:a.pair(base['targets'],base['users'],old,z) for n,z in [('global7',globalpred),('confidence_gate',gatepred),('classwise7',classpred)]})
 assert not set(a.OPENED)&{str(a.mapped(v['scores'])) for v in fs['inputs'].values()}
 if not (out/'CALIBRATORS_FROZEN.json').exists():a.atomic(out/'CALIBRATORS_FROZEN.json',dict(protocol_sha256=ph,mode=mode,files=cfiles,calibrators=6,all_calibrators_frozen_before_this_stage_outer_arrays=True,source_audit_done_sha256=s['source_audit_done_sha256'],at_unix=time.time()))
 parts=[];per={}
 for f in (0,3):
  inp=fs['inputs'][str(f)];base=values(a.check(inp['scores'],inp['scores_sha256']));base['logits']=base['logits'][:,:6];base['available']=base['available'][:,:6];oldcal=a.read(a.check(inp['calibrator'],inp['calibrator_sha256']));old=a.fusion(base['logits'],base['available'],oldcal);a.near(old,base['pose_baseline'])
  raw=a.align(values(a.check(s['outer_predictions'][str(f)]['path'],s['outer_predictions'][str(f)]['sha256'])),base);new=raw['videomae_logits']
  if mode=='quant':
   j=next(j for j in qjobs if j['scope']=='outer' and j['fold']==f);root=C/'videomae_kinetics_quant/cloud'/j['id'];jd=a.read(root/'DONE.json');q=a.align(values(a.check(root/'predictions.npz',jd['predictions_sha256'])),base);new=q['videomae_logits']
  data=dict(base,logits=np.concatenate([base['logits'],new[:,None]],1),available=np.concatenate([base['available'],np.ones((len(new),1),bool)],1));globalcal=a.read(out/f'F{f}_global.json');gated=a.read(out/f'F{f}_gate.json');assert not set(globalcal['calibration_users'])&set(base['users'])
  globalpred=convex_replay(data,globalcal);a.near(globalpred,a.fusion(data['logits'],data['available'],globalcal));gatepred,alpha=gate.predict(gate.softmax(old),new,base['logits'][:,1:4],base['available'][:,1:4],gated);classcal=a.read(out/f'F{f}_classwise.json');assert not set(classcal['calibration_users'])&set(base['users']);classpred=cw.predict(data['logits'],data['available'],classcal)
  v={k:base[k] for k in ['clip_ids','targets','users']};v.update(old_pose6=old,new_videomae=new,global7=globalpred,confidence_gate=gatepred,classwise7=classpred,alpha=alpha);parts.append(v);per[str(f)]={n:a.pair(base['targets'],base['users'],old,z) for n,z in [('global7',globalpred),('confidence_gate',gatepred),('classwise7',classpred)]};np.savez_compressed(out/f'F{f}_OUTER.npz',**v)
 allv={k:np.concatenate([v[k] for v in parts]) for k in parts[0]};assert len(set(allv['clip_ids']))==len(allv['clip_ids'])==1094 and (allv['old_pose6'].argmax(1)==allv['targets']).sum()==855
 src={k:np.concatenate([v[k] for v in sourceparts]) for k in sourceparts[0]};np.savez_compressed(out/'SOURCE_FITTED.npz',**src);np.savez_compressed(out/'OUTER_PAIRED.npz',**allv)
 pooled={n:a.pair(allv['targets'],allv['users'],allv['old_pose6'],allv[n]) for n in ['global7','confidence_gate','classwise7']}
 result=dict(status='COMPLETE_SOURCE_ONLY_VIDEOMAE_FUSION_COMPARISON',protocol_sha256=ph,mode=mode,pooled=pooled,per_fold=per,source_fit_summary=sourcesummary,source_rows=4768,source_unique=2931,source_calibration_fit_metrics_are_not_validation=True,no_Test=True,no_GPU=True,no_CSV=True,all_calibrators_frozen_before_outer=True,uncertainty={n:a.boot(allv['targets'],allv['users'],allv['old_pose6'],allv[n]) for n in ['global7','confidence_gate','classwise7']},limitations=s['limitations'],seconds=time.time()-start)
 clock.check();a.atomic(out/'RESULT.json',result);a.atomic(out/'DONE.json',dict(status='COMPLETE',protocol_sha256=ph,mode=mode,artifacts={n:a.sha(out/n) for n in ['RESULT.json','CALIBRATORS_FROZEN.json','SOURCE_FITTED.npz','OUTER_PAIRED.npz','F0_OUTER.npz','F3_OUTER.npz',*cfiles]},at_unix=time.time()))
 print(json.dumps(dict(mode=mode,pooled={n:{k:v for k,v in d.items() if k not in ['per_user','per_class']} for n,d in pooled.items()},per_fold={f:{n:{k:d[k] for k in ['before','after','fix','harm']} for n,d in ds.items()} for f,ds in per.items()},source_fit_summary={f:{k:v for k,v in d.items() if k!='source_fit_metrics_not_validation'} for f,d in sourcesummary.items()},seconds=result['seconds'])))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--mode',choices=['raw','quant'],required=True);main(**vars(p.parse_args()))
