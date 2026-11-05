"""Four source-only calibrators; new outer context arrays open after barrier."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,sys,time
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
E=Path(__file__).resolve().parent;C=E.parent;O=C.parent/'error_resolution_20260913'
sys.path[:0]=[str(E),str(C),str(O/'bundle')]
import numpy as np,torch
import frozen_fusion_math as fm
import classwise4 as cw4
sp=importlib.util.spec_from_file_location('context_old_classwise',C/'v2_temporal_halves_fusion/classwise3.py');cw3=importlib.util.module_from_spec(sp);sp.loader.exec_module(cw3)
from dual_host_runtime.io import read,atomic,retry
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(p,h):assert h and sha(p)==h,str(p);return Path(p)
def arr(p):
 with np.load(p,allow_pickle=False) as a:return {k:a[k] for k in a.files}
def subset(a,ix):return {k:v[ix] for k,v in a.items() if v.ndim and len(v)==len(ix)}
def align(a,b):
 assert len(set(a['clip_ids'].tolist()))==len(a['clip_ids'])
 lookup={str(c):i for i,c in enumerate(a['clip_ids'])};ix=np.array([lookup[str(c)] for c in b['clip_ids']])
 result={k:v[ix] for k,v in a.items() if v.ndim and len(v)==len(a['clip_ids'])}
 for key in ('clip_ids','targets','users'):assert np.array_equal(result[key],b[key]),key
 return result
def near(a,b):assert np.max(np.abs(np.asarray(a)-np.asarray(b)))<1e-6
class Clock:
 def __init__(self,deadline):self.deadline=deadline
 def check(self,**kwargs):assert time.time()<self.deadline,'CONTEXT_CPU_CAP'
def convex(data,g):
 model=fm.Calibrator(g['description']);model.load_state_dict({k:torch.tensor(v,dtype=torch.float64) for k,v in g['state'].items()},strict=True);model.eval()
 with torch.no_grad():return model(dict(logits=torch.tensor(data['logits'],dtype=torch.float64),mask=torch.tensor(data['available'],dtype=torch.bool))).numpy().astype(np.float32)
def pair(y,u,a,b):
 p=a.argmax(1);q=b.argmax(1);ok=p==y;nk=q==y
 def nll(z):z=z.astype(np.float64);z=z-z.max(1,keepdims=True);return -z[np.arange(len(y)),y]+np.log(np.exp(z).sum(1))
 an=nll(a);bn=nll(b)
 result=dict(n=len(y),old_correct=int(ok.sum()),new_correct=int(nk.sum()),fix=int((~ok&nk).sum()),harm=int((ok&~nk).sum()),old_nll=float(an.mean()),new_nll=float(bn.mean()))
 result['per_user']={str(int(k)):dict(n=int((u==k).sum()),old_correct=int(ok[u==k].sum()),new_correct=int(nk[u==k].sum()),fix=int((~ok&nk)[u==k].sum()),harm=int((ok&~nk)[u==k].sum())) for k in np.unique(u)}
 result['per_class']={str(int(k)):dict(n=int((y==k).sum()),old_correct=int(ok[y==k].sum()),new_correct=int(nk[y==k].sum()),fix=int((~ok&nk)[y==k].sum()),harm=int((ok&~nk)[y==k].sum())) for k in np.unique(y)}
 return result

def stage_gate(s,ph):
 result={}
 for host in ('local','cloud'):
  root=Path(s['calibration_inputs']['context_outputs'][host]);d=read(root/'DONE.json');ct=read(root/'CONTROLLER_DONE.json')
  assert d['protocol_sha256']==ct['protocol_sha256']==ph and d['status']=='COMPLETE_CONTEXT_READOUT'
  assert ct['status']=='COMPLETE' and ct['exit_code']==0 and ct['terminal_sha256']==sha(root/'DONE.json')
  for n,v in d['artifacts'].items():check(root/n,v)
  for j in [j for j in s['jobs'] if j['host']==host]:
   p=root/j['id'];jd=read(p/'DONE.json');qi=read(p/'QUANT_IDENTITY.json')
   assert jd['protocol_sha256']==qi['protocol_sha256']==ph and jd['status']=='COMPLETE_CONTEXT3_READOUT'
   assert jd['original_quant_checkpoint_sha256']==j['original_quant_checkpoint_sha256'] and jd['optimizer_updates']==0
   for k in ('train_users','held_users','outer_excluded_users','scope','fold'):assert jd[k]==j[k]
   assert qi['all_quantized_tensors_exact'] and qi['checkpoint_metadata_exact']
   check(p/'predictions.npz',jd['predictions_sha256']);check(p/'QUANT_IDENTITY.json',jd['identity_sha256'])
   result[j['id']]=p
 return result

def main(protocol,expected_protocol_sha):
 start=time.time();s=read(check(protocol,expected_protocol_sha));ph=expected_protocol_sha
 assert s['status']=='FROZEN_FOR_EXECUTION' and s['calibration_recipe']['steps']==500 and s['calibration_recipe']['classwise_penalty']==.1
 assert s['calibration_recipe']['branches']==['T','I','BaseSevenFixed','ContextThreeMean']
 for p,h in s['calibration_code_sha256'].items():check(C/p,h)
 out=E/'calibration';out.mkdir(exist_ok=True);assert not(out/'DONE.json').exists()
 clock=Clock(min(start+s['proposal_cpu_calibration_cap_seconds'],s['deadline_unix']));clock.check();torch.set_num_threads(1)
 roots=stage_gate(s,ph);inp=s['calibration_inputs'];bound=inp['bound_files']
 for p,h in bound.items():check(p,h)
 oldbar=read(inp['old_calibrator_barrier']);assert oldbar['protocol_sha256']==inp['old_protocol_sha256'] and oldbar['calibrators']==4
 source=arr(inp['source_fitted']);assert len(source['targets'])==4768 and len(set(source['clip_ids']))==2931
 raw=arr(inp['source_oof']);assert len(raw['targets'])==4768
 sourceparts=[];files={};fitted={};oldcal={}
 fm.STRATEGIES['context_convex4']=dict(branches=[0,1,2,3],mode='convex',availability=False)
 for f in (0,3):
  base=subset(source,source['source_fold']==f);r=subset(raw,raw['source_fold']==f);r['clip_ids']=r['original_clip_ids'];r=align(r,base)
  assert np.array_equal(base['TI'],r['logits'][:,[2,3]]) and np.array_equal(base['available'][:,:2],r['available'][:,[2,3]])
  parts=[]
  for j in [j for j in s['jobs'] if j['scope']=='source' and j['fold']==f]:
   v=arr(roots[j['id']]/'predictions.npz');assert set(v['users'])==set(j['held_users'])
   assert not (set(j['train_users'])|set(v['users']))&set(j['outer_excluded_users'])
   assert v['context_logits'].shape==(j['records'],3,40);near(v['context_mean'],v['context_logits'].mean(1));parts.append(v)
  q=align({k:np.concatenate([v[k] for v in parts]) for k in parts[0]},base)
  outerusers=next(j['held_users'] for j in s['jobs'] if j['scope']=='outer' and j['fold']==f)
  assert set(base['users']).isdisjoint(outerusers)
  olddata=dict(logits=np.concatenate([base['TI'],base['quant_base'][:,None]],1),available=base['available'])
  og=read(inp['old_calibrators'][str(f)]['global']['path']);oc=read(inp['old_calibrators'][str(f)]['classwise']['path']);oldcal[f]=(og,oc)
  for cal in (og,oc):assert cal['outer_labels_used'] is False and cal['calibration_users']==sorted(set(base['users']))
  near(convex(olddata,og),base['global3']);near(cw3.predict(olddata['logits'],olddata['available'],oc),base['classwise3'])
  data=dict(logits=np.concatenate([olddata['logits'],q['context_mean'][:,None]],1),available=np.column_stack([olddata['available'],np.ones(len(q['users']),bool)]),targets=base['targets'],users=base['users'],depth=r['depth'],ir=r['ir'])
  g=fm.fit(data,'context_convex4',clock,500);c=cw4.fit(data['logits'],data['available'],data['targets'],data['users'],g,clock)
  for mode,cal in [('global',g),('classwise',c)]:
   assert cal['outer_labels_used'] is False and set(cal['calibration_users']).isdisjoint(outerusers)
   dest=out/f'F{f}_{mode}.json';atomic(dest,cal);files[dest.name]=sha(dest)
  fitted[f]=(g,c)
  v={k:base[k] for k in ('clip_ids','targets','users','source_fold')};v.update(old_global3=base['global3'],old_classwise3=base['classwise3'],base_seven=base['quant_base'],context=q['context_mean'],global4=convex(data,g),classwise4=cw4.predict(data['logits'],data['available'],c));sourceparts.append(v)
 # No new context outer arrays have been opened above this immutable barrier.
 atomic(out/'CALIBRATORS_FROZEN.json',dict(protocol_sha256=ph,files=files,calibrators=4,source_rows=4768,source_unique=2931,all_four_before_new_outer_context_arrays=True,old_three_branch_calibrators_unchanged=True,at_unix=time.time()))
 for name,h in files.items():check(out/name,h)
 old=arr(inp['old_paired']);assert len(old['targets'])==1094
 parts=[];per={}
 for f in (0,3):
  j=next(j for j in s['jobs'] if j['scope']=='outer' and j['fold']==f);base=subset(old,np.isin(old['users'],j['held_users']));q=align(arr(roots[j['id']]/'predictions.npz'),base)
  r=align(arr(inp['outer_scores'][str(f)]['path']),base)
  z=np.concatenate([r['logits'][:,[2,3]],base['quant_base'][:,None]],1);mask=np.column_stack([r['available'][:,[2,3]],np.ones(len(base['users']),bool)])
  og,oc=oldcal[f];near(convex(dict(logits=z,available=mask),og),base['global3']);near(cw3.predict(z,mask,oc),base['classwise3'])
  data=dict(logits=np.concatenate([z,q['context_mean'][:,None]],1),available=np.column_stack([mask,np.ones(len(base['users']),bool)]));g,c=fitted[f]
  assert set(g['calibration_users']).isdisjoint(base['users']) and set(c['calibration_users']).isdisjoint(base['users'])
  v={k:base[k] for k in ('clip_ids','targets','users')};v.update(old_global3=base['global3'],old_classwise3=base['classwise3'],base_seven=base['quant_base'],context=q['context_mean'],global4=convex(data,g),classwise4=cw4.predict(data['logits'],data['available'],c));parts.append(v)
  per[str(f)]={mode:pair(v['targets'],v['users'],v['old_classwise3'],v[mode]) for mode in ('global4','classwise4')}
 allv={k:np.concatenate([v[k] for v in parts]) for k in parts[0]};src={k:np.concatenate([v[k] for v in sourceparts]) for k in sourceparts[0]}
 assert len(set(allv['clip_ids']))==1094
 np.savez_compressed(out/'PAIRED.npz',**allv);np.savez_compressed(out/'SOURCE_FITTED.npz',**src)
 result=dict(status='COMPLETE_CONTEXT_SOURCE_FUSION',protocol_sha256=ph,old_global_correct=int((allv['old_global3'].argmax(1)==allv['targets']).sum()),old_classwise_correct=int((allv['old_classwise3'].argmax(1)==allv['targets']).sum()),pooled={mode:pair(allv['targets'],allv['users'],allv['old_classwise3'],allv[mode]) for mode in ('global4','classwise4','context')},per_fold=per,source_fit_not_validation={mode:pair(src['targets'],src['users'],src['old_classwise3'],src[mode]) for mode in ('global4','classwise4','context')},all_four_calibrators_before_new_outer=True,no_GPU=True,no_Test=True,no_CSV=True,source_rows=4768,source_unique=2931,limitations=s['limitations'],seconds=time.time()-start)
 clock.check();atomic(out/'RESULT.json',result)
 files.update({n:sha(out/n) for n in ('CALIBRATORS_FROZEN.json','RESULT.json','PAIRED.npz','SOURCE_FITTED.npz')});atomic(out/'DONE.json',dict(status='COMPLETE',protocol_sha256=ph,artifacts=files,at_unix=time.time()))
 print(json.dumps({m:{k:v for k,v in z.items() if k not in ('per_user','per_class')} for m,z in result['pooled'].items()}))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--expected-protocol-sha',required=True);main(**vars(p.parse_args()))
