"""Fixed half original/half mirrored RTN6+Small8; original four-way source recipe."""
from pathlib import Path
import hashlib,importlib.util,json,os,sys,time
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
E=Path(__file__).resolve().parent;C=E.parents[1]
sys.path[:0]=[str(C),str(C.parent/'error_resolution_20260913/bundle')]
import numpy as np,torch
import frozen_fusion_math as fm
from dual_host_runtime.io import read,atomic
sp=importlib.util.spec_from_file_location('fixed_RTN_classwise4',C/'v2_base6_small8_source_completion/classwise4.py');cw=importlib.util.module_from_spec(sp);sp.loader.exec_module(cw)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(p,h):assert sha(p)==h,str(p);return Path(p)
def mirror(p):
 prefix='/home/featurize/cuhkx_motion_transfer_20260911_v1/'
 return C.parent/p[len(prefix):]if p.startswith(prefix)else Path(p)
def arr(p):
 with np.load(p,allow_pickle=False)as v:return {k:v[k]for k in v.files}
def subset(v,ix):return {k:x[ix]for k,x in v.items()}
def align(v,ids,y=None,u=None):
 assert len(set(v['clip_ids']))==len(v['clip_ids']);look={cid:i for i,cid in enumerate(v['clip_ids'])};ix=np.array([look[cid]for cid in ids]);q=subset(v,ix)
 assert np.array_equal(q['clip_ids'],ids)
 if y is not None:assert np.array_equal(q['targets'],y)
 if u is not None:assert np.array_equal(q['users'],u)
 return q
def lse(x,axis,keep=False):
 m=x.max(axis=axis,keepdims=True);v=m+np.log(np.exp(x-m).sum(axis=axis,keepdims=True));return v if keep else np.squeeze(v,axis)
def convex(z,mask,cal):
 z=np.asarray(z,np.float64);lp=z/np.asarray(cal['temperature'])[None,:,None];lp-=lse(lp,-1,True)
 w=np.broadcast_to(np.asarray(cal['state']['bias'])[None,:],mask.shape).copy();w[~mask]=-np.inf;w-=lse(w,1,True)
 return lse(w[:,:,None]+lp,1).astype(np.float32)
def torch_convex(z,mask,cal):
 model=fm.Calibrator(cal['description']);model.load_state_dict({k:torch.tensor(v,dtype=torch.float64)for k,v in cal['state'].items()})
 with torch.no_grad():return model(dict(logits=torch.tensor(z,dtype=torch.float64),mask=torch.tensor(mask))).numpy().astype(np.float32)
def pair(y,u,a,b):
 x=a.argmax(1)==y;z=b.argmax(1)==y
 def nll(v):v=np.asarray(v,np.float64);return float((lse(v,1)-v[np.arange(len(y)),y]).mean())
 return dict(n=len(y),before=int(x.sum()),after=int(z.sum()),fix=int((~x&z).sum()),harm=int((x&~z).sum()),net=int(z.sum()-x.sum()),NLL_before=nll(a),NLL_after=nll(b),prediction_changes=int((a.argmax(1)!=b.argmax(1)).sum()),per_user={str(q):dict(n=int((u==q).sum()),before=int(x[u==q].sum()),after=int(z[u==q].sum()),fix=int((~x[u==q]&z[u==q]).sum()),harm=int((x[u==q]&~z[u==q]).sum()))for q in sorted(set(u.tolist()))},per_class={str(q):dict(n=int((y==q).sum()),before=int(x[y==q].sum()),after=int(z[y==q].sum()),fix=int((~x[y==q]&z[y==q]).sum()),harm=int((x[y==q]&~z[y==q]).sum()))for q in sorted(set(y.tolist()))})
class Clock:
 def __init__(self,end):self.end=end
 def check(self,**kwargs):assert time.time()<self.end,'CPU_CALIBRATION_CAP'


def gates(fs):
 import psutil
 def live(v):
  try:return psutil.pid_exists(v['pid']) and abs(psutil.Process(v['pid']).create_time()-v['create_time'])<.01
  except psutil.NoSuchProcess:return False
 inv=read(fs['inputs']['parent_inventory']['path']);lookup={j['id']:j for j in inv['jobs']};jobs=[];receipts=[]
 for origin in fs['origins']:
  proto=read(check(origin['protocol_path'],origin['protocol_sha256']));ph=origin['protocol_sha256'];root=Path(origin['output_root']);d=read(root/'DONE.json');ct=read(root/'CONTROLLER_DONE.json');owner=read(root/'CONTROLLER.json')
  assert ct['protocol_sha256']==d['protocol_sha256']==ph and ct['status']=='COMPLETE' and ct['exit_code']==0 and ct['terminal_sha256']==sha(root/'DONE.json') and d['status']==origin['success_status']
  if origin['host']=='local':assert not live(owner['controller']) and not live(owner['worker'])
  else:
   receipt=read(root/'DOWNLOAD_DONE.json');assert receipt['all_owned_dead'] and receipt['protocol_sha256']==ph
   for n in ['DONE.json','CONTROLLER_DONE.json','CONTROLLER.json']:check(root/n,receipt['files'][n]['sha256'])
  for n,h in d['artifacts'].items():check(root/n,h['sha256'] if isinstance(h,dict) else h)
  receipts.append(dict(host=origin['host'],protocol_sha256=ph,done_sha256=sha(root/'DONE.json'),controller_done_sha256=sha(root/'CONTROLLER_DONE.json'),all_owned_dead=True))
  for j in [j for j in proto['jobs'] if j['host']==origin['host']]:
   original=lookup[j['id']]
   for k in ['scope','fold','inner','records','train_users','held_users','outer_excluded_users']:assert j[k]==original[k],(j['id'],k)
   r=root/j['id'];jd=read(r/'DONE.json');fr=read(r/'BACKBONE_FROZEN.json');assert jd['status']=='COMPLETE_MIRROR_DUAL_JOB_READOUT' and jd['protocol_sha256']==fr['protocol_sha256']==ph and jd['optimizer_updates']==fr['optimizer_updates']==0
   assert jd['container_sha256']==j['container']['sha256']==fs['expected_containers'][j['id']] and fr['container']==j['container'] and fr['parents']==j['parents'] and not fr['weight_values_changed']
   for model in ['Base','Small']:
    for key in ['parent','frozen','split']:assert j['parents'][model][key]['sha256']==original['parents'][model][key]['sha256']
   for n,h in jd['artifacts'].items():check(r/n,h['sha256'] if isinstance(h,dict) else h)
   parent_splits={}
   for model in ['Base','Small']:
    p=original['parents'][model];sp=read(check(p['split']['path'],p['split']['sha256']));fr=read(check(p['frozen']['path'],p['frozen']['sha256']))
    assert fr['checkpoint_sha256']==p['parent']['sha256'] and fr['epochs']==16
    for k in ['train_users','held_users']:assert sp[k]==fr[k]==j[k]
    parent_splits[model]=sp
   assert set(parent_splits['Base']['held_ids'])==set(parent_splits['Small']['held_ids'])
   assert set(j['train_users']).isdisjoint(j['held_users']) and (set(j['train_users'])|set(j['held_users'])).isdisjoint(j['outer_excluded_users'])
   jobs.append(dict(j,root=str(r),origin_protocol_sha256=ph,held_ids=parent_splits['Base']['held_ids'],predictions_sha256=sha(r/'predictions.npz'),base_views_sha256=sha(r/'Base.npz'),small_views_sha256=sha(r/'Small.npz'),done_sha256=sha(r/'DONE.json')))
 assert len(jobs)==8 and len({(j['scope'],j['fold'],j['inner']) for j in jobs})==8 and sum(j['scope']=='source' for j in jobs)==6
 return jobs,receipts

def read_job(j):
 root=Path(j['root']);v=arr(check(root/'predictions.npz',j['predictions_sha256']));bv=arr(check(root/'Base.npz',j['base_views_sha256']));sm=arr(check(root/'Small.npz',j['small_views_sha256']))
 assert len(v['users'])==j['records'] and set(v['users'])==set(j['held_users']) and set(v['clip_ids'])==set(j['held_ids'])
 for q in [bv,sm]:
  for k in ['clip_ids','targets','users']:assert np.array_equal(q[k],v[k])
 b=bv['seven_view_logits'];t=sm['five_view_logits'];assert b.dtype==t.dtype==np.float32 and b.shape==(j['records'],7,40) and t.shape==(j['records'],5,40) and np.isfinite(b).all() and np.isfinite(t).all()
 assert np.array_equal((.5*b[:,:5].mean(1)+.25*b[:,5]+.25*b[:,6]).astype(np.float32),v['base6_mirror'])
 assert np.array_equal(t.mean(1).astype(np.float32),v['small8_mirror'])
 for k in ['base6','small8','base6_original','small8_original','base6_mirror','small8_mirror']:assert v[k].dtype==np.float32 and v[k].shape==(j['records'],40) and np.isfinite(v[k]).all()
 for label,views in [('base6',bv),('small8',sm)]:
  assert np.array_equal(v[label+'_original'],views['original_logits']) and np.array_equal(v[label+'_mirror'],views['mirror_logits']) and np.array_equal(v[label],views['mixed_logits'])
  assert np.array_equal(v[label],(.5*v[label+'_original']+.5*v[label+'_mirror']).astype(np.float32))
 q={k:v[k] for k in ['clip_ids','targets','users','base6_mirror','small8_mirror']}
 q.update(base6=v['base6_original'],small8=v['small8_original'],base_mixed=v['base6'],small_mixed=v['small8'])
 return q

def main(protocol,expected_protocol_sha):
 start=time.time();fs=read(check(protocol,expected_protocol_sha));ph=expected_protocol_sha;assert fs['status']=='FROZEN_FOR_EXECUTION' and time.time()<fs['deadline_unix']
 for p,h in fs['code_bindings'].items():check(p,h)
 for v in fs['inputs'].values():check(v['path'],v['sha256'])
 for v in fs['references'].values():check(v['path'],v['sha256'])
 out=E/'output';out.mkdir(exist_ok=True);claim=out/'CLAIM'
 with claim.open('x') as f:f.write(ph)
 clock=Clock(min(fs['deadline_unix'],start+fs['cpu_fit_cap_seconds']));torch.set_num_threads(1)
 jobs,receipts=gates(fs);atomic(out/'ALL_EIGHT_READOUTS_FROZEN.json',dict(protocol_sha256=ph,receipts=receipts,jobs=jobs,outer_arrays_opened=False,at_unix=time.time()))
 source=arr(fs['inputs']['source_oof']['path']);rawsource=arr(fs['inputs']['raw_source']['path']);assert len(source['users'])==4768 and len(set(source['original_clip_ids']))==2931
 fm.STRATEGIES['convex_TI_BASE_SMALL4']=dict(branches=[0,1,2,3],mode='convex',availability=False)
 fits={};files={};sources=[];source_comparison={}
 for fold in [0,3]:
  b=subset(source,source['source_fold']==fold);b['clip_ids']=b['original_clip_ids'];vv=[read_job(j) for j in jobs if j['scope']=='source' and j['fold']==fold];assert len(vv)==3
  q=align({k:np.concatenate([v[k] for v in vv]) for k in vv[0]},b['clip_ids'],b['targets'],b['users']);raw=align(subset(rawsource,rawsource['source_fold']==fold),q['clip_ids'],q['targets'],q['users'])
  assert len(q['users'])==len(b['users']) and not set(q['users'])&set(fs['outer_users'][str(fold)])
  for k in ['base6','small8']:assert np.array_equal(q[k],raw[k]),('original_readout_reuse',fold,k)
  assert np.array_equal(b['logits'][:,[2,3]],raw['TI']) and np.array_equal(b['available'][:,[2,3]],raw['available'][:,:2])
  z=np.concatenate([b['logits'][:,[2,3]],q['base_mixed'][:,None],q['small_mixed'][:,None]],1);mask=np.concatenate([b['available'][:,[2,3]],np.ones((len(q['users']),2),bool)],1)
  data=dict(b,logits=z,available=mask);g=fm.fit(data,'convex_TI_BASE_SMALL4',clock,500);c=cw.fit(z,mask,q['targets'],q['users'],g,clock)
  for name,cal in [('global',g),('classwise',c)]:
   assert not cal['outer_labels_used'] and not set(cal['calibration_users'])&set(fs['outer_users'][str(fold)])
   p=out/f'F{fold}_{name}.json';atomic(p,cal);files[p.name]=sha(p)
  gp=convex(z,mask,g);assert np.max(np.abs(gp-torch_convex(z,mask,g)))<1e-6
  cq=cw.predict(z,mask,c);sources.append(dict(q,source_fold=np.full(len(q['users']),fold),TI=b['logits'][:,[2,3]],available=mask,global4=gp,classwise4=cq,raw_global4=raw['global4'],raw_classwise4=raw['classwise4']));fits[fold]=(g,c)
  source_comparison[str(fold)]={name:pair(q['targets'],q['users'],raw[old],new) for name,old,new in [('global4','global4',gp),('classwise4','classwise4',cq),('Base','base6',q['base_mixed']),('Small','small8',q['small_mixed'])]}
 atomic(out/'CALIBRATORS_FROZEN.json',dict(protocol_sha256=ph,files=files.copy(),calibrators=4,all_source_calibrators_before_outer_arrays=True,at_unix=time.time()))
 for n,h in files.items():check(out/n,h)
 first_outer=time.time();rawouter=arr(fs['references']['RTN']['path']);parts=[]
 for fold in [0,3]:
  b=arr(fs['inputs'][f'outer_scores_{fold}']['path']);j=next(j for j in jobs if j['scope']=='outer' and j['fold']==fold);q=align(read_job(j),b['clip_ids'],b['targets'],b['users']);raw=align(rawouter,q['clip_ids'],q['targets'],q['users']);g,c=fits[fold]
  assert set(q['users'])==set(fs['outer_users'][str(fold)])
  for k in ['base6','small8']:assert np.array_equal(q[k],raw[k]),('outer_original_reuse',fold,k)
  z=np.concatenate([b['logits'][:,[2,3]],q['base_mixed'][:,None],q['small_mixed'][:,None]],1);mask=np.concatenate([b['available'][:,[2,3]],np.ones((len(q['users']),2),bool)],1)
  parts.append(dict(q,fold=np.full(len(q['users']),fold),TI=b['logits'][:,[2,3]],available=mask,global4=convex(z,mask,g),classwise4=cw.predict(z,mask,c)))
 av={k:np.concatenate([v[k] for v in parts]) for k in parts[0]};sv={k:np.concatenate([v[k] for v in sources]) for k in sources[0]};y=av['targets'];u=av['users'];assert len(y)==1094 and len(set(zip(sv['source_fold'],sv['clip_ids'])))==4768
 comparisons={}
 for name,ref in fs['references'].items():
  b=align(arr(ref['path']),av['clip_ids'],y,u);av[name]=b[ref['field']];assert int((av[name].argmax(1)==y).sum())==ref['correct']
  comparisons[name]={key:pair(y,u,av[name],av[key]) for key in ['base_mixed','small_mixed','global4','classwise4']}
 np.savez_compressed(out/'PAIRED.npz',**av);np.savez_compressed(out/'SOURCE_FITTED.npz',**sv)
 result=dict(status='COMPLETE_MIRROR_DUAL_EIGHT_SOURCE_FUSION',protocol_sha256=ph,primary='classwise4',comparisons=comparisons,source_comparison=source_comparison,source_rows=4768,source_unique_clips=2931,source_composite_unique=4768,source_calibration_fits=4,optimizer_updates=0,original_readouts_reused=8,readout_origins=receipts,source_recipe=fs['source_recipe'],first_outer_arrays_started_unix=first_outer,no_Test=True,no_CSV=True,limitations='Adaptive repeated development. Fixed original/mirrored half-half logits, no mirror-weight scan; source fits are not validation; held gains do not establish Public gains.',seconds=time.time()-start,at_unix=time.time())
 atomic(out/'RESULT.json',result)
 for n in ['ALL_EIGHT_READOUTS_FROZEN.json','CALIBRATORS_FROZEN.json','PAIRED.npz','SOURCE_FITTED.npz','RESULT.json']:files[n]=sha(out/n)
 atomic(out/'DONE.json',dict(status='COMPLETE',protocol_sha256=ph,artifacts=files,at_unix=time.time()));print(json.dumps({k:{x:v for x,v in q['classwise4'].items() if x not in ['per_user','per_class']} for k,q in comparisons.items()}))
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--expected-protocol-sha',required=True);main(**vars(p.parse_args()))
