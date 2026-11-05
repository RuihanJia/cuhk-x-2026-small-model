"""Seven remaining paired readouts, then four strictly source-only calibrators."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,sys,time,traceback
E=Path(__file__).resolve().parent;C=E.parent;D=C/'v2_base6_small_design'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
sys.path[:0]=[str(E),str(D)]
def module(path,name):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
p=module(D/'pilot_worker.py','bound_base6_small8_pilot')
torch=p.torch;np=p.np;read=p.read;atomic=p.atomic;sha=p.sha;check=p.check;arrays=p.arrays;align=p.align
from dual_host_runtime.health import Guard,identity,owned_tree
from loader import load_state
fm=module(C/'frozen_fusion_math.py','completion_frozen_fusion_math')
cw=module(E/'classwise4.py','completion_classwise4')

def save_npz(path,value):
 temp=path.with_suffix('.partial.npz');np.savez_compressed(temp,**value);p.retry(lambda:os.replace(temp,path))
def subset(a,ix):return {k:v[ix]for k,v in a.items()}
def concatenate(parts):return {k:np.concatenate([v[k]for v in parts])for k in parts[0]}
def pair(y,u,old,new):
 a=old.argmax(1)==y;b=new.argmax(1)==y
 def stats(z):
  d=z.astype(np.float64);d-=d.max(1,keepdims=True);lp=d-np.log(np.exp(d).sum(1,keepdims=True))
  return dict(correct=int((z.argmax(1)==y).sum()),accuracy=float((z.argmax(1)==y).mean()),nll=float(-lp[np.arange(len(y)),y].mean()))
 return dict(records=len(y),before=stats(old),after=stats(new),fixes=int((~a&b).sum()),harms=int((a&~b).sum()),changed=int((old.argmax(1)!=new.argmax(1)).sum()),per_user={str(int(k)):dict(records=int((u==k).sum()),before=int((a&(u==k)).sum()),after=int((b&(u==k)).sum()),fixes=int((~a&b&(u==k)).sum()),harms=int((a&~b&(u==k)).sum()))for k in np.unique(u)})
def bootstrap(y,u,old,new):
 delta=(new.argmax(1)==y).astype(float)-(old.argmax(1)==y);uu=np.unique(u);r=np.random.default_rng(20260915);vals=[]
 for _ in range(2000):
  ids=r.choice(uu,len(uu),replace=True);vals.append(sum(delta[u==i].sum()for i in ids)/sum((u==i).sum()for i in ids))
 return dict(kind='2000_user_cluster_bootstrap',accuracy_delta=float(delta.mean()),interval95=np.quantile(vals,[.025,.975]).tolist(),users=len(uu),not_new_independent_validation=True)
def global_replay(data,g):
 z=data['logits'].astype(np.float64)/np.asarray(g['temperature'])[None,:,None];z-=cw.logsumexp(z,-1,True)
 bias=np.broadcast_to(np.asarray(g['state']['bias'])[None,:,None],z.shape).copy();bias[~data['available']]=-np.inf
 bias-=cw.logsumexp(bias,1,True);v=cw.logsumexp(z+bias,1);return v.astype(np.float32)

def validate_pilot(s):
 prior=Path(s['pilot_directory']);d=read(prior/'DONE.json');ct=read(prior/'CONTROLLER_DONE.json')
 assert ct['status']=='COMPLETE'and ct['exit_code']==0 and ct['terminal_sha256']==sha(prior/'DONE.json')
 assert d['status']=='COMPLETE_BASE6_SMALL8_PILOT'and d['protocol_sha256']==ct['protocol_sha256']==s['pilot_protocol_sha256']
 for n,h in d['artifacts'].items():check(prior/n,h)
 b=arrays(prior/'Base.npz');sm=align(arrays(prior/'Small.npz'),b);result=read(prior/'RESULT.json');verified={}
 for name,v,a,z in [('Base_five',b,b['old_five_logits'],b['five_logits']),('Base_seven',b,b['old_seven_logits'],b['seven_logits']),('Small_five',sm,sm['old_five_logits'],sm['five_logits'])]:
  computed=p.pair(v['targets'],v['users'],a,z);assert computed==result['metrics'][name];verified[name]=computed
 assert np.array_equal(b['five_view_logits'].mean(1),b['five_logits'])and np.array_equal(sm['five_view_logits'].mean(1),sm['five_logits'])
 assert np.array_equal(.5*b['five_logits']+.25*b['temporal_logits'][:,0]+.25*b['temporal_logits'][:,1],b['seven_logits'])
 return dict(status='PASS_PILOT_CPU_RAW_METRIC_REPLAY',protocol_sha256=s['pilot_protocol_sha256'],done_sha256=sha(prior/'DONE.json'),controller_done_sha256=sha(prior/'CONTROLLER_DONE.json'),metrics=verified,no_fit=True,no_Test=True,at_unix=time.time())

def readout(j,s,ph,out,inp,guard):
 root=out/j['id'];root.mkdir(exist_ok=True)
 if(root/'DONE.json').exists():
  d=read(root/'DONE.json');assert d['protocol_sha256']==ph and d['container_sha256']==j['container']['sha256']
  for n,h in d['artifacts'].items():check(root/n,h)
  return
 rows=[r for r in inp['rows']if r['user']in j['held_users']]
 assert len(rows)==j['records'];assert all(r['streams'][0]['directory'].startswith('Training/')for r in rows)
 assert set(j['train_users']).isdisjoint(set(j['held_users'])|set(j['outer_excluded_users']))
 check(j['container']['path'],j['container']['sha256']);files={};started=time.time()
 atomic(root/'STARTED.json',dict(protocol_sha256=ph,job=j['id'],container=j['container'],train_users=j['train_users'],held_users=j['held_users'],outer_excluded_users=j['outer_excluded_users'],optimizer_updates=0,at_unix=time.time()))
 for label in ['Base','Small']:
  state,origin=load_state(j['container']['path'],label);assert origin['train_users']==j['train_users']and origin['held_users']==j['held_users']and origin['epochs']==16
  config=s['models'][label];m=module(config['model_code'],'completion_model_'+label);cls=getattr(m,config['constructor']);model=cls(classes=40,fast=True,seed=s['seed']);model.load_state_dict(state,strict=True)
  assert all(torch.equal(v,state[k])for k,v in model.state_dict().items());del state
  model.cuda().eval();five=p.replay(model,p.PoseFiveViews(rows,s['data_root'],inp['regions']),5,guard,j['id']+'_'+label+'_FIVE_FP32')
  v={k:five[k]for k in ['clip_ids','targets','users']};v.update(five_view_logits=five['values'],five_logits=five['values'].mean(1))
  if label=='Base':
   half=p.replay(model,p.TemporalHalves(rows,s['data_root'],inp['regions']),2,guard,j['id']+'_BASE6_HALVES_FP32')
   for k in ['clip_ids','targets','users']:assert np.array_equal(half[k],five[k])
   v.update(temporal_logits=half['values'],seven_logits=.5*v['five_logits']+.25*half['values'][:,0]+.25*half['values'][:,1])
  save_npz(root/(label+'.npz'),v);files[label+'.npz']=sha(root/(label+'.npz'));del model,m,cls,five,v;torch.cuda.empty_cache()
 atomic(root/'DONE.json',dict(status='COMPLETE_PAIRED_BASE6_SMALL8_READOUT',protocol_sha256=ph,container_sha256=j['container']['sha256'],artifacts=files,records=len(rows),optimizer_updates=0,metrics_computed=False,no_Test=True,seconds=time.time()-started,at_unix=time.time()))

def get_values(j,s,out):
 root=Path(s['pilot_directory'])if j['reuse_pilot']else out/j['id'];d=read(root/'DONE.json')
 b=arrays(check(root/'Base.npz',d['artifacts']['Base.npz']));sm=align(arrays(check(root/'Small.npz',d['artifacts']['Small.npz'])),b)
 assert set(b['users'])==set(j['held_users'])and len(b['users'])==j['records']
 return dict(clip_ids=b['clip_ids'],targets=b['targets'],users=b['users'],base6=b['seven_logits'],small8=sm['five_logits'])

def fuse(s,ph,out,inputs,guard):
 source=arrays(check(s['source_oof'],s['source_oof_sha256']));assert source['logits'].shape==(4768,10,40)
 fm.STRATEGIES['convex_TI_BASE_SMALL4']=dict(branches=[0,1,2,3],mode='convex',availability=False)
 fitted={};files={};sources=[]
 # All four source-only calibration states must freeze before opening any outer readout here.
 for f in [0,3]:
  base=subset(source,source['source_fold']==f);base['clip_ids']=base['original_clip_ids']
  js=[j for j in inputs['jobs']if j['scope']=='source'and j['fold']==f];assert len(js)==3
  v=align(concatenate([get_values(j,s,out)for j in js]),base)
  assert set(v['users']).isdisjoint(s['outer_users'][str(f)])and len(set(v['clip_ids']))==len(v['clip_ids'])
  mask=np.concatenate([base['available'][:,[2,3]],np.ones((len(v['users']),2),bool)],1);z=np.concatenate([base['logits'][:,[2,3]],v['base6'][:,None],v['small8'][:,None]],1)
  data=dict(base,logits=z,available=mask);g=fm.fit(data,'convex_TI_BASE_SMALL4',guard,500);c=cw.fit(z,mask,v['targets'],v['users'],g,guard)
  for name,cal in [('global',g),('classwise',c)]:
   assert not cal['outer_labels_used']and set(cal['calibration_users']).isdisjoint(s['outer_users'][str(f)])
   path=out/f'F{f}_{name}.json';atomic(path,cal);files[path.name]=sha(path)
  gp=global_replay(data,g);assert np.max(abs(gp-fm.predict(data,g)))<1e-6
  sources.append(dict(v,source_fold=np.full(len(v['users']),f,np.int64),TI=base['logits'][:,[2,3]],available=mask,global4=gp,classwise4=cw.predict(z,mask,c)));fitted[f]=(g,c)
 sourcev=concatenate(sources);assert len(sourcev['clip_ids'])==4768 and len(set(sourcev['clip_ids']))==2931
 assert len(set(zip(sourcev['source_fold'].tolist(),sourcev['clip_ids'].tolist())))==4768
 save_npz(out/'SOURCE_FITTED.npz',sourcev);files['SOURCE_FITTED.npz']=sha(out/'SOURCE_FITTED.npz')
 atomic(out/'CALIBRATORS_FROZEN.json',dict(protocol_sha256=ph,files=files,calibrators=4,source_encoders=6,branches=['T','I','Base6','Small8'],all_source_calibrators_frozen_before_new_outer_combinations=True,at_unix=time.time()))
 fs=read(s['fixed_fusion_protocol']);parts=[]
 for f in [0,3]:
  inp=fs['inputs'][str(f)];base=arrays(check(inp['scores'],inp['scores_sha256']));j=next(j for j in inputs['jobs']if j['scope']=='outer'and j['fold']==f);v=align(get_values(j,s,out),base)
  assert set(v['users'])==set(s['outer_users'][str(f)]);mask=np.concatenate([base['available'][:,[2,3]],np.ones((len(v['users']),2),bool)],1);z=np.concatenate([base['logits'][:,[2,3]],v['base6'][:,None],v['small8'][:,None]],1)
  data=dict(base,logits=z,available=mask);g,c=fitted[f];gp=global_replay(data,g);assert np.max(abs(gp-fm.predict(data,g)))<1e-6
  parts.append(dict(v,fold=np.full(len(v['users']),f,np.int64),global4=gp,classwise4=cw.predict(z,mask,c)))
 allv=concatenate(parts);old=align(arrays(check(s['reference'],s['reference_sha256'])),allv);allv['current']=old['classwise3'];assert len(allv['targets'])==1094
 save_npz(out/'PAIRED.npz',allv)
 result=dict(status='COMPLETE_BASE6_SMALL8_SOURCE_COMPLETION',protocol_sha256=ph,branches=['T','I','Base6','Small8'],pooled={k:pair(allv['targets'],allv['users'],allv['current'],allv[k])for k in ['base6','small8','global4','classwise4']},uncertainty={k:bootstrap(allv['targets'],allv['users'],allv['current'],allv[k])for k in ['global4','classwise4']},source_rows=4768,source_unique_clips=2931,new_readout_jobs=7,reused_pilot_jobs=1,optimizer_updates=0,no_Test=True,no_CSV=True,limitations='Repeatedly exposed adaptive7user development. No independent Public guarantee. Source calibrator metrics are in-sample. Final18 fullpackage still requires qualification.',at_unix=time.time())
 atomic(out/'RESULT.json',result);files.update({n:sha(out/n)for n in ['PAIRED.npz','RESULT.json','CALIBRATORS_FROZEN.json']});return files

def main(protocol,host):
 s=read(protocol);ph=sha(protocol);assert host=='local'and s['status']=='FROZEN_FOR_EXECUTION'and s['optimizer_updates']==0
 out=Path(s['hosts'][host]['output_directory']);bd=read(out/'BUDGET.json');ct=read(out/'CONTROLLER.json');me=identity(os.getpid());assert bd['protocol_sha256']==ct['protocol_sha256']==ph
 assert any(v['pid']==me['pid']and abs(v['create_time']-me['create_time'])<.01 for v in owned_tree(ct['worker']))
 for path,h in s['bindings'].items():check(path,h)
 guard=Guard(out,bd['deadline_unix'],out/'TELEMETRY.json');guard.check(force=True,phase='BASE6_SMALL8_ALLSOURCE_PREFLIGHT');torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 pilot=validate_pilot(s);atomic(out/'PILOT_CPU_REPLAY.json',pilot);inp=read(s['pose_inputs']);inputs=read(s['inputs_file']);start=time.time()
 for j in sorted([j for j in inputs['jobs']if not j['reuse_pilot']],key=lambda j:(j['scope']!='source',j['fold'],j.get('inner')or 0)):
  readout(j,s,ph,out,inp,guard)
 files=fuse(s,ph,out,inputs,guard);files['PILOT_CPU_REPLAY.json']=sha(out/'PILOT_CPU_REPLAY.json')
 for j in inputs['jobs']:
  if not j['reuse_pilot']:files[j['id']+'/DONE.json']=sha(out/j['id']/'DONE.json')
 atomic(out/'DONE.json',dict(status='COMPLETE_BASE6_SMALL8_SOURCE_COMPLETION',protocol_sha256=ph,artifacts=files,optimizer_updates=0,no_Test=True,seconds=time.time()-start,at_unix=time.time()));print(json.dumps(dict(status='COMPLETE',result_sha256=sha(out/'RESULT.json'))))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--protocol',required=True);ap.add_argument('--host',required=True);a=ap.parse_args()
 try:main(a.protocol,a.host)
 except BaseException as e:
  s=read(a.protocol);atomic(Path(s['hosts'][a.host]['output_directory'])/'ERROR.json',dict(protocol_sha256=sha(a.protocol),error=str(e),traceback=traceback.format_exc(),at_unix=time.time()));raise
