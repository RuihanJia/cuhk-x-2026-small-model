"""Original H global3/classwise3 source fits, all four frozen before outer arrays."""
from pathlib import Path
import hashlib,importlib.util,json,os,sys,time
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
E=Path(__file__).resolve().parent;C=E.parent
sys.path[:0]=[str(C),str(C.parent/'error_resolution_20260913/bundle')]
import numpy as np,torch
import frozen_fusion_math as fm
from dual_host_runtime.io import read,atomic
sp=importlib.util.spec_from_file_location('fixed_H_classwise3',C/'videomaev2_base_fusion/classwise3.py');cw=importlib.util.module_from_spec(sp);sp.loader.exec_module(cw)
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

def gates(s,ph):
 pilot=read(check(s['pilot_protocol'],s['pilot_protocol_sha256']));jobs=[];receipts=[]
 for origin,proto,p in [('pilot',pilot,s['pilot_protocol_sha256']),('completion',s,ph)]:
  for host in ['cloud','local']:
   root=Path(s['pilot_outputs'][host])if origin=='pilot'else (E/'cloud'if host=='cloud'else E/'output/local')
   d=read(root/'DONE.json');ct=read(root/'CONTROLLER_DONE.json');assert d['protocol_sha256']==ct['protocol_sha256']==p and ct['status']=='COMPLETE'and ct['terminal_sha256']==sha(root/'DONE.json')
   assert d['status']==('COMPLETE_WEIGHT_AVERAGE_PILOT'if origin=='pilot'else'COMPLETE_WEIGHT_AVERAGE_COMPLETION')
   for n,h in d['artifacts'].items():
    if not n.endswith('.pt'):check(root/n,h)
   receipts.append(dict(origin=origin,host=host,protocol_sha256=p,controller_done_sha256=sha(root/'CONTROLLER_DONE.json'),done_sha256=sha(root/'DONE.json')))
   for j in [j for j in proto['jobs']if j['host']==host]:
    r=root/j['id'];fr=read(r/'AVERAGE_FROZEN.json');jd=read(r/'DONE.json')
    assert jd['protocol_sha256']==fr['protocol_sha256']==p and jd['optimizer_updates']==0 and fr['alpha']==.5 and fr['quantized_matrices']==50
    assert fr['raw']==j['raw']and fr['control']==j['control']and fr['checkpoint_sha256']==jd['artifacts']['model_int8.pt']
    assert set(j['train_users']).isdisjoint(j['held_users'])and(set(j['train_users'])|set(j['held_users'])).isdisjoint(j['outer_excluded_users'])
    jobs.append(dict(j,root=str(r),origin_protocol_sha256=p,predictions_sha256=sha(r/'predictions.npz')))
 assert len(jobs)==8 and len({(j['scope'],j['fold'],j['inner'])for j in jobs})==8
 return jobs,receipts
def read_job(j):
 v=arr(check(Path(j['root'])/'predictions.npz',j['predictions_sha256']));assert len(v['users'])==j['records'] and set(v['users'])==set(j['held_users'])
 z=v['seven_view_logits'];assert z.shape==(j['records'],7,40)and np.isfinite(z).all()
 assert np.array_equal((.5*z[:,:5].mean(1)+.25*z[:,5]+.25*z[:,6]).astype(np.float32),v['videomae_logits'])
 return {k:v[k]for k in ['clip_ids','targets','users','videomae_logits']}
def main():
 start=time.time();s=read(E/'protocol.json');ph=sha(E/'protocol.json');fs=read(E/'fusion_protocol.json');assert fs['protocol_sha256']==ph and time.time()<s['deadline_unix']
 for p,h in fs['code_bindings'].items():check(p,h)
 for k,v in fs['inputs'].items():check(v['path'],v['sha256'])
 out=E/'fusion';out.mkdir(exist_ok=True);assert not(out/'DONE.json').exists();clock=Clock(min(s['deadline_unix'],start+240));torch.set_num_threads(1)
 jobs,receipts=gates(s,ph);atomic(out/'ALL_EIGHT_READOUTS_FROZEN.json',dict(protocol_sha256=ph,receipts=receipts,jobs=jobs,outer_arrays_opened=False))
 source=arr(fs['inputs']['source_oof']['path']);assert len(source['users'])==4768 and len(set(source['original_clip_ids']))==2931
 fm.STRATEGIES['convex_TI_BASE3']=dict(branches=[0,1,2],mode='convex',availability=False)
 fits={};files={};sources=[]
 for f in [0,3]:
  base=subset(source,source['source_fold']==f);base['clip_ids']=base['original_clip_ids'];vv=[read_job(j)for j in jobs if j['scope']=='source'and j['fold']==f]
  q=align({k:np.concatenate([v[k]for v in vv])for k in vv[0]},base['clip_ids'],base['targets'],base['users']);assert len(q['users'])==len(base['users'])
  assert not set(q['users'])&set(fs['outer_users'][str(f)])
  data=dict(base,logits=np.concatenate([base['logits'][:,[2,3]],q['videomae_logits'][:,None]],1),available=np.concatenate([base['available'][:,[2,3]],np.ones((len(q['users']),1),bool)],1))
  g=fm.fit(data,'convex_TI_BASE3',clock,500);c=cw.fit(data['logits'],data['available'],data['targets'],data['users'],g,clock)
  for name,cal in [('global',g),('classwise',c)]:
   assert not cal['outer_labels_used']and not set(cal['calibration_users'])&set(fs['outer_users'][str(f)])
   p=out/f'F{f}_{name}.json';atomic(p,cal);files[p.name]=sha(p)
  gp=convex(data['logits'],data['available'],g);assert np.max(np.abs(gp-torch_convex(data['logits'],data['available'],g)))<1e-6
  sources.append(dict(clip_ids=q['clip_ids'],targets=q['targets'],users=q['users'],source_fold=np.full(len(q['users']),f),video=q['videomae_logits'],TI=base['logits'][:,[2,3]],available=data['available'],global3=gp,classwise3=cw.predict(data['logits'],data['available'],c)));fits[f]=(g,c)
 atomic(out/'CALIBRATORS_FROZEN.json',dict(protocol_sha256=ph,files=files,calibrators=4,all_source_calibrators_before_outer_arrays=True,at_unix=time.time()))
 for n,h in files.items():check(out/n,h)
 fixed=read(fs['inputs']['fixed_fusion_protocol']['path']);parts=[]
 for f in [0,3]:
  spec=fixed['inputs'][str(f)];base=arr(check(mirror(spec['scores']),spec['scores_sha256']));j=next(j for j in jobs if j['scope']=='outer'and j['fold']==f);q=align(read_job(j),base['clip_ids'],base['targets'],base['users']);g,c=fits[f]
  assert set(q['users'])==set(fs['outer_users'][str(f)])
  z=np.concatenate([base['logits'][:,[2,3]],q['videomae_logits'][:,None]],1);mask=np.concatenate([base['available'][:,[2,3]],np.ones((len(q['users']),1),bool)],1)
  parts.append(dict(clip_ids=q['clip_ids'],targets=q['targets'],users=q['users'],fold=np.full(len(q['users']),f),video=q['videomae_logits'],TI=base['logits'][:,[2,3]],available=mask,global3=convex(z,mask,g),classwise3=cw.predict(z,mask,c)))
 allv={k:np.concatenate([v[k]for v in parts])for k in parts[0]};sv={k:np.concatenate([v[k]for v in sources])for k in sources[0]};h=align(arr(fs['inputs']['H_reference']['path']),allv['clip_ids'],allv['targets'],allv['users']);control=align(arr(fs['inputs']['CONTROL_reference']['path']),allv['clip_ids'],allv['targets'],allv['users'])
 y=allv['targets'];u=allv['users'];allv['H']=h['classwise3'];allv['CONTROL']=control['CONTROL_classwise3'];assert len(y)==1094 and (allv['H'].argmax(1)==y).sum()==928 and (allv['CONTROL'].argmax(1)==y).sum()==927
 np.savez_compressed(out/'PAIRED.npz',**allv);np.savez_compressed(out/'SOURCE_FITTED.npz',**sv)
 result=dict(status='COMPLETE_FIXED_AVERAGE_EIGHT_SOURCE_FUSION',protocol_sha256=ph,vs_H={k:pair(y,u,allv['H'],allv[k])for k in ['video','global3','classwise3']},vs_CONTROL={k:pair(y,u,allv['CONTROL'],allv[k])for k in ['video','global3','classwise3']},source_rows=4768,source_unique_clips=2931,source_composite_unique=len(set(zip(sv['source_fold'].tolist(),sv['clip_ids'].tolist()))),optimizer_updates=0,source_calibration_fits=4,no_Test=True,no_CSV=True,limitations='Adaptive repeated development; source fit is not held validation. No weight or precision scan. Full18 CONTROL4 is separate and not selected by this execution.',seconds=time.time()-start,at_unix=time.time())
 atomic(out/'RESULT.json',result)
 for n in ['ALL_EIGHT_READOUTS_FROZEN.json','CALIBRATORS_FROZEN.json','PAIRED.npz','SOURCE_FITTED.npz','RESULT.json']:files[n]=sha(out/n)
 atomic(out/'DONE.json',dict(status='COMPLETE',protocol_sha256=ph,artifacts=files,at_unix=time.time()));print(json.dumps({k:{x:v for x,v in q.items()if x not in ['per_user','per_class']}for k,q in result['vs_H'].items()}))
if __name__=='__main__':main()
