"""Independent recovery CPU fixed-six-branch audit; immutable original four fits.
No fit, feature-model forward, Test, state mutation, or checkpoint modification.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json,hashlib,time,math,argparse,csv
import numpy as np
P=Path(__file__).resolve().parent;C=P.parents[2];ARMS=('RAW_CONTROL','CLIP_STRETCH')
BEGIN=time.time();CHECKED={};OPENED=[]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def mapped(p):
 p=Path(p)
 if p.exists():return p
 text=str(p).replace('\\','/');mark='/cuhkx_motion_transfer_20260911_v1/'
 if mark in text:
  q=C.parent/text.split(mark,1)[1]
  if q.exists():return q
  for i,k in enumerate(q.parts):
   if k=='cloud':
    v=Path(*q.parts[:i+1])/'supervisor/FINAL_CLOUD_EVIDENCE'/Path(*q.parts[i+1:])
    if v.exists():return v
  return q
 return p
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for z in iter(lambda:f.read(1048576),b''):h.update(z)
 return h.hexdigest()
def check(p,h=None):
 p=mapped(p);a=CHECKED.get(str(p)) or sha(p)
 if h is not None:assert a==h,(str(p),a,h)
 CHECKED[str(p)]=a;return p
def arr(p):
 p=Path(p);assert not any(v in str(p).lower() for v in ('submission','test','release_cloud_mirror'));OPENED.append(str(p))
 with np.load(p,allow_pickle=False) as a:return {k:a[k] for k in a.files}
def align(a,ref):
 q={str(v):i for i,v in enumerate(a['clip_ids'])};assert len(q)==len(a['clip_ids']) and set(q)==set(map(str,ref['clip_ids']));ix=[q[str(v)] for v in ref['clip_ids']]
 a={k:v[ix] for k,v in a.items() if v.ndim and len(v)==len(ix)}
 for k in ('targets','users'):assert np.array_equal(a[k],ref[k]),k
 return a
def softmax(z,axis=-1):
 z=np.asarray(z,np.float64);e=np.exp(z-z.max(axis=axis,keepdims=True));return e/e.sum(axis=axis,keepdims=True)
def fusion(z,av,c):
 n=z.shape[1];assert n in (6,7) and c['description']==dict(branches=list(range(n)),mode='convex',availability=False) and c['outer_labels_used'] is False
 assert set(c['state'])=={'temperature_raw','bias'} and z.shape[1:]==(n,40) and av.shape==z.shape[:2] and av.dtype==bool and av.any(1).all()
 t=.25+7.75/(1+np.exp(-np.asarray(c['state']['temperature_raw'],np.float64)));assert np.allclose(t,c['temperature'],atol=1e-12,rtol=0)
 b=np.broadcast_to(np.asarray(c['state']['bias'],np.float64),av.shape).copy();b[~av]=-np.inf;w=softmax(b,1)
 # Independent probability-domain mixture; producer used log-sum-exp.
 p=(w[:,:,None]*softmax(z.astype(np.float64)/t[None,:,None])).sum(1);assert np.all(p>0) and np.allclose(p.sum(1),1,atol=1e-12)
 return np.log(p).astype(np.float32)
def near(a,b):
 e=float(np.abs(a.astype(np.float64)-b.astype(np.float64)).max());assert e<=1e-6 and np.array_equal(a.argmax(-1),b.argmax(-1)),e;return e
def nll(z,y):
 z=z.astype(np.float64);m=z.max(1);return m+np.log(np.exp(z-m[:,None]).sum(1))-z[np.arange(len(y)),y]
def pair(y,u,a,b,group=True):
 aok=a.argmax(1)==y;bok=b.argmax(1)==y;la=nll(a,y);lb=nll(b,y)
 d=dict(n=len(y),before=int(aok.sum()),after=int(bok.sum()),fix=int((~aok&bok).sum()),harm=int((aok&~bok).sum()),net=int(bok.sum()-aok.sum()),nll_before=float(la.mean()),nll_after=float(lb.mean()),prediction_changes=int((a.argmax(1)!=b.argmax(1)).sum()),wrong_to_wrong=int((~aok&~bok&(a.argmax(1)!=b.argmax(1))).sum()))
 if group:
  for k,v in [('per_user',u),('per_class',y)]:d[k]={str(x):pair(y[v==x],u[v==x],a[v==x],b[v==x],False) for x in np.unique(v)}
  for k,v in [('macro_accuracy_before',aok),('macro_accuracy_after',bok),('user_balanced_nll_before',la),('user_balanced_nll_after',lb)]:d[k]=float(np.mean([v[u==x].mean() for x in np.unique(u)]))
 return d
def match(a,b):
 for k in ('n','before','after','fix','harm','net','nll_before','nll_after'):assert abs(a[k]-b[k])<1e-9,(k,a[k],b[k])
 for g in ('per_user','per_class'):
  if g in b:
   for k in b[g]:match(a[g][k],b[g][k])
def boot(y,u,a,b):
 keys=np.unique(u);n=np.array([(u==k).sum() for k in keys]);v=(b.argmax(1)==y).astype(float)-(a.argmax(1)==y);v=np.array([v[u==k].sum() for k in keys]);rng=np.random.default_rng(20260914);ix=rng.integers(len(keys),size=(10000,len(keys)))
 return dict(user_clusters=len(keys),accuracy_delta_CI95=np.quantile(v[ix].sum(1)/n[ix].sum(1),[.025,.975]).tolist(),scope='Descriptive whole-user resampling; repeated development exposure and selection not corrected.')


def atomic(p,d):
 p=Path(p);tmp=p.with_name(p.name+'.partial');data=json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)
 with tmp.open('w',encoding='utf-8') as f:f.write(data);f.flush();os.fsync(f.fileno())
 for attempt in range(8):
  try:os.replace(tmp,p);return
  except PermissionError:
   if attempt==7:raise
   time.sleep(.15*(attempt+1))

def bounded():
 assert time.time()<AUDIT_DEADLINE,'INDEPENDENT_CPU_AUDIT_CAP'

def phase(root,kind,ph):
 d=read(check(root/kind/'DONE.json'));ct=read(check(root/kind/'CONTROLLER_DONE.json'));b=read(check(root/kind/'CONTROLLER_BUDGET.json'));controller=read(check(root/kind/'CONTROLLER.json'))
 assert d['status']==('COMPLETE_TRAIN_PHASE' if kind=='train' else 'COMPLETE_EVALUATE_PHASE') and d['protocol_sha256']==ct['protocol_sha256']==b['protocol_sha256']==controller['protocol_sha256']==ph and ct['exit_code']==0 and ct['terminal_sha256']==sha(root/kind/'DONE.json')
 for n,h in d['artifacts'].items():check(root/kind/n,h['sha256'] if isinstance(h,dict) else h)
 return d,ct,b,controller

def bootstrap_group(a,b,y,u,ids):
 result={'user':boot(y,u,a,b)}
 # Not official recording boundaries: sensitivity grouping only.
 keys=np.asarray([f'{int(y[i])}:{int(u[i])}:'+str(cid).rsplit('-',1)[0] for i,cid in enumerate(ids)]);_,code=np.unique(keys,return_inverse=True)
 result['clip_prefix_proxy']=boot(y,code,a,b);result['clip_prefix_proxy']['scope']='Sensitivity only: (class,user,clip string minus final hyphen segment), not verified independent recording groups.'
 return result

def main(args):
 global AUDIT_DEADLINE
 rp=Path(args.protocol).resolve();rph=sha(rp);assert rph==args.expected_protocol_sha.lower();rs=read(check(rp));assert rs['status']=='FROZEN_FOR_EXECUTION' and rs['wall_cap_seconds']==900;pp=check(rs['original_protocol'],rs['original_protocol_sha256']);ph=sha(pp);s=read(pp);assert ph=='3de7290de17d4e8c9c7339fbcd73c97d3bfa0c4f280e109968c5715eaf60bf4d';assert s['status']=='FROZEN_FOR_EXECUTION' and s['epochs']==4 and s['original_pose_epochs']==8 and s['seed']==20260914 and s['micro_clips']==s['effective_clip_batch']==8
 assert sha(__file__)==rs['independent_audit_sha256'];check(P/'independent_audit.py',s['independent_audit_sha256']);assert [(j['fold'],j['arm'],j['host']) for j in s['jobs']]==[(f,a,h) for f,h in ((0,'cloud'),(3,'local')) for a in ARMS]
 roots={'cloud':P/'supervisor/cloud_mirror','local':mapped(s['hosts']['local']['output_directory'])};recovered=mapped(rs['hosts']['local']['output_directory']);evalroots={'cloud':roots['cloud']/'evaluate','local':recovered/'evaluate'};globalbar=P/'recovery_20260915/ALL_FOUR_READOUTS_READY.json';out=P/'recovery_20260915/audit'
 required=[globalbar]
 for host,root in roots.items():
  required += [root/'HOST_BUDGET.json',root/'HOST_TRAIN_DONE.json']
  for phasepath in (root/'train',evalroots[host]):required += [phasepath/n for n in ('DONE.json','CONTROLLER_DONE.json','CONTROLLER_BUDGET.json','CONTROLLER.json')]
  for j in s['jobs']:
   if j['host']==host:required += [root/j['id']/'TRAIN_DONE.json',root/j['id']/'FROZEN.json',evalroots[host]/j['id']/'DONE.json',evalroots[host]/j['id']/'predictions.npz']
 required += [recovered/'HOST_BUDGET.json'];missing=[str(p) for p in required if not p.exists()]
 if missing:return dict(status='NOT_READY',missing=missing,arrays_read=0)
 assert not(out/'DONE.json').exists() and not(out/'RESULT.json').exists();AUDIT_DEADLINE=min(BEGIN+180,float(rs['deadline_unix'])+360);bounded();gb=read(check(globalbar));assert gb['protocol_sha256']==ph and gb['recovery_protocol_sha256']==rph and gb['all_owned_dead'] and gb['host_states']=={'cloud':'done','local':'done'}
 manifest=read(check(P/'supervisor/DOWNLOAD_MANIFEST.json'))
 for n,v in manifest.items():check(roots['cloud']/n,v['sha256']);assert (roots['cloud']/n).stat().st_size==v['bytes']
 for rel,h in s['code_sha256'].items():check(C/rel,h)
 for rel,h in rs['code_sha256'].items():check(C/rel,h)
 failure=roots['local']/'evaluate';err=read(check(failure/'ERROR.json','190a19bf04fc8363d2c14695d447d400b0794bae67de2c59e91788ad8c80fcf5'));check(failure/'CONTROLLER_ERROR.json','f54d5d469d0aebefa4720a89ba44e68a2f99f07a2c45cad20e0d7697b09d4fe3');assert err['protocol_sha256']==ph and err['error']=='CLOUD_ONLY_NO_LOCAL_DATA_ANALYSIS'
 assert not any((failure/j['id']/'predictions.npz').exists() for j in s['jobs'] if j['host']=='local')
 rowmap=None;metadata={};hostproof={}
 for host,root in roots.items():
  h=s['hosts'][host];hb=read(check(root/'HOST_BUDGET.json'));assert hb['protocol_sha256']==ph and hb['deadline_unix']<=min(s['deadline_unix'],hb['first_started_unix']+1800)
  train,tc,tb,tctrl=phase(root,'train',ph);evalph=ph if host=='cloud' else rph;evalbase=root if host=='cloud' else recovered;ev,ec,eb,ectrl=phase(evalbase,'evaluate',evalph)
  assert ev['global_metrics_not_computed'] and ev['calibration_fit_steps']==0 and ev['all_four_jobs_protocol_frozen_before_training']
  if host=='local':assert ev['training_protocol_sha256']==ph and ev['training_updates']==0
  assert tb['first_started_unix']==hb['first_started_unix'] and tb['deadline_unix']==hb['deadline_unix'] and tb['host_budget_sha256']==sha(root/'HOST_BUDGET.json') and tc['at_unix']<=hb['deadline_unix']
  evalbudget=hb if host=='cloud' else read(check(recovered/'HOST_BUDGET.json'))
  if host=='local':assert evalbudget['protocol_sha256']==rph and evalbudget['first_started_unix']>=rs['registered_at_unix'] and evalbudget['deadline_unix']<=min(rs['deadline_unix'],evalbudget['first_started_unix']+900)
  assert eb['first_started_unix']==evalbudget['first_started_unix'] and eb['deadline_unix']==evalbudget['deadline_unix'] and eb['host_budget_sha256']==sha(evalbase/'HOST_BUDGET.json')
  assert tc['at_unix']<=eb['phase_started_unix']<=ec['at_unix']<=evalbudget['deadline_unix'] and ec['at_unix']<=gb['at_unix']
  hd=read(check(root/'HOST_TRAIN_DONE.json'));assert hd['status']=='COMPLETE_HOST_TRAINING' and hd['protocol_sha256']==ph and hd['host']==host and not hd['held_evaluation_started']
  assert sha(root/'HOST_TRAIN_DONE.json')==sha(root/'train/HOST_TRAIN_DONE.json')==train['artifacts']['HOST_TRAIN_DONE.json']
  for n,v in hd['artifacts'].items():check(root/n,v)
  inputs=read(check(s['hosts']['local']['pose_inputs'],s['pose_inputs_sha256']));rows=inputs['rows'];rowmap={v['clip_id']:v for v in rows};assert len(rows)==len(rowmap)==2931
  for j in [j for j in s['jobs'] if j['host']==host]:
   original=read(check(h['original_protocol'],s['original_protocol_sha256']));parent=next(v for v in original['jobs'] if v['id']==j['parent_job'])
   for k in ('fold','train_users','held_users','source_sha256'):assert parent[k]==j[k]
   assert parent['source']['cloud']==j['canonical_source_cloud'];paths=j['paths'][host];od=read(check(paths['original_done'],j['original_done_sha256']));check(paths['original_private_checkpoint'],j['original_private_sha256']);check(paths['original_predictions'],j['original_predictions_sha256'])
   assert od['protocol_sha256']==s['original_protocol_sha256'] and od['epochs']==8 and od['checkpoint_sha256']==j['original_private_sha256'] and od['source_sha256']==j['source_sha256'] and od['frozen_base_hash_before']==j['original_frozen_base_digest'] and od['frozen_base_unchanged']
   canonical=j['canonical_source_cloud'];marker='/activity_token_stopgrad_20260913/cloud/output/';assert marker in canonical;sourcepath=(C.parent/'activity_token_stopgrad_20260913/cloud_mirror/output'/canonical.split(marker,1)[1]) if os.name=='nt' else Path(canonical);check(sourcepath,j['source_sha256'])
   d=root/j['id'];td=read(d/'TRAIN_DONE.json');fr=read(d/'FROZEN.json');sp=read(d/'SPLIT.json');ini=read(d/'INITIAL.json');hi=read(d/'HISTORY.json');fb=read(d/'FIRST_BATCH.json');jd=read(evalroots[host]/j['id']/'DONE.json');desc=hd['jobs'][j['id']]
   for v in (td,fr,sp,ini,fb,jd):assert v['protocol_sha256']==ph
   if host=='local':assert jd['recovery_protocol_sha256']==rph
   link=gb['jobs'][j['id']];assert link['training_protocol_sha256']==ph and link['evaluation_protocol_sha256']==evalph and link['checkpoint_sha256']==fr['checkpoint_sha256'] and link['done_sha256']==sha(evalroots[host]/j['id']/'DONE.json') and link['predictions_sha256']==jd['predictions_sha256'] and link['photometric_sha256']==sha(evalroots[host]/j['id']/'PHOTOMETRIC.json')
   assert td['status']=='COMPLETE_TRAIN_ONLY' and not td['held_evaluation_started'] and not fr['held_evaluation_started'];assert jd['status']=='COMPLETE_PHOTOMETRIC_HELD' and jd['job']==j['id'] and jd['fold']==j['fold'] and jd['arm']==j['arm'] and jd['global_metrics_not_computed'] and jd['no_fit'] and jd['no_Test']
   assert sha(d/'TRAIN_DONE.json')==desc['train_done_sha256'] and jd['host_training_barrier_sha256']==sha(root/'HOST_TRAIN_DONE.json')
   for n,v in desc['metadata'].items():check(d/n,v)
   for n,v in td['artifacts'].items():check(d/n,v)
   assert fr['checkpoint_sha256']==jd['checkpoint_sha256']==desc['checkpoint_sha256']==td['artifacts']['adapter_fp16.pt'];check(d/'adapter_fp16.pt',fr['checkpoint_sha256']);check(evalroots[host]/j['id']/'predictions.npz',jd['predictions_sha256'])
   assert fr['epochs']==len(hi)==4 and fr['FP16_reload_exact'] and fr['changed_private_parameters'] and fr['initial_state_digest']==ini['initial_state_digest'] and fr['frozen_base_digest']==j['original_frozen_base_digest'] and fr['source_sha256']==j['source_sha256'] and fr['original_private_sha256']==j['original_private_sha256']
   for k in ('train_users','held_users'):assert fr[k]==sp[k]==j[k]
   assert not set(sp['train_users'])&set(sp['held_users']) and not sp['held_labels_used_in_fit']
   for kind in ('train','held'):assert sp[kind+'_ids']==[v['clip_id'] for v in rows if v['user'] in j[kind+'_users']]
   updates=math.ceil(len(sp['train_ids'])/8);assert fr['updates']==4*updates and fr['at_unix']<=tc['at_unix']<=jd['at_unix']<=ec['at_unix']
   for e,v in enumerate(hi,1):assert v['epoch']==e and v['records']==len(sp['train_ids']) and v['updates']==e*updates and np.isfinite(v['loss']) and np.isfinite(v['denominator_min']) and len(v['sampling_sha256'])==64
   assert fb['sampling_seed']==20260914 and fb['sampling_epoch']==1 and fb['AMP_scale']>=128 and all(cid in sp['train_ids'] for cid in fb['clip_ids'])
   photo=read(evalroots[host]/j['id']/'PHOTOMETRIC.json');
   if host=='local':assert photo['recovery_protocol_sha256']==rph
   assert photo['protocol_sha256']==ph and photo['arm']==j['arm'] and photo['per_view16_shared_scalar'] and len(photo['records'])==jd['records']==len(sp['held_ids'])
   for v in photo['records']:
    assert v['clip_id'] in sp['held_ids'] and all(np.asarray(v[k]).shape==(5,) and np.isfinite(v[k]).all() for k in ('stretch_low','stretch_high','denominator'))
   if j['arm']=='RAW_CONTROL':
    for record in [fb]+photo['records']:
     for k,expected in [('stretch_low',0.),('stretch_high',1.),('denominator',1.)]:assert np.all(np.asarray(record[k])==expected)
   else:
    for record in [fb]+photo['records']:assert (np.asarray(record['denominator'])>=1/255-1e-8).all() and (np.asarray(record['stretch_high'])>=np.asarray(record['stretch_low'])).all()
   metadata[j['id']]=(j,sp,fr,hi,fb,jd);bounded()
  pairjobs=[j for j in s['jobs'] if j['host']==host];a=metadata[pairjobs[0]['id']];b=metadata[pairjobs[1]['id']];assert a[2]['initial_state_digest']==b[2]['initial_state_digest'] and [v['sampling_sha256'] for v in a[3]]==[v['sampling_sha256'] for v in b[3]]
  for k in ('clip_ids','frame_indices','flip','sampling_seed','sampling_epoch'):assert a[4][k]==b[4][k]
  hostproof[host]=dict(host_budget_sha256=sha(root/'HOST_BUDGET.json'),deadline_unix=hb['deadline_unix'],train_controller_done_sha256=sha(root/'train/CONTROLLER_DONE.json'),evaluate_controller_done_sha256=sha(evalroots[host]/'CONTROLLER_DONE.json'),evaluate_protocol_sha256=evalph,evaluate_budget=dict(first_started_unix=evalbudget['first_started_unix'],deadline_unix=evalbudget['deadline_unix'],host_budget_sha256=sha(evalbase/'HOST_BUDGET.json')),host_train_barrier_sha256=sha(root/'HOST_TRAIN_DONE.json'),paired_sampling_identical=True)
 assert not OPENED # Four raw outputs terminal before metrics or numerical payload reads.
 pieces=[];perfold={};replay={};calhashes={}
 for host,root in roots.items():
  h=s['hosts'][host];fold=0 if host=='cloud' else 3;fs=read(check(s['fixed_fusion_protocol'],s['fixed_fusion_protocol_sha256']));inp=fs['inputs'][str(fold)];assert fs['fit_steps']==0 and fs['replace_branch']==5
  a=arr(check(inp['scores'],inp['scores_sha256']));cal=read(check(inp['calibrator'],inp['calibrator_sha256']));fr=read(check(inp['calibrator_frozen'],inp['calibrator_frozen_sha256']));assert fr['model_sha256']==inp['calibrator_sha256'] and fr['protocol_sha256']==fs['old_calibration_protocol_sha256'] and not fr['outer_labels_used_for_fit'];calhashes[str(fold)]=inp['calibrator_sha256']
  z6=a['logits'][:,:6];av=a['available'][:,:6];old=fusion(z6,av,cal);replay[str(fold)]=near(old,a['pose_baseline']);assert int((old.argmax(1)==a['targets']).sum())=={0:409,3:446}[fold]
  value={k:a[k] for k in ('clip_ids','targets','users')};value.update(old_pose6=old,old_private=z6[:,5]);stats={}
  for j in [j for j in s['jobs'] if j['host']==host]:
   raw=align(arr(evalroots[host]/j['id']/'predictions.npz'),a);arm=j['arm'];assert set(raw['clip_ids'])==set(metadata[j['id']][1]['held_ids']) and cal['calibration_users']==j['train_users'] and sorted(set(raw['users']))==j['held_users']
   for i,cid in enumerate(raw['clip_ids']):assert raw['users'][i]==rowmap[str(cid)]['user'] and raw['targets'][i]==rowmap[str(cid)]['label']
   assert raw['logits'].dtype==raw['pose_logits'].dtype==raw['headarm_five_view_logits'].dtype==np.float32 and raw['headarm_five_view_logits'].shape==(len(raw['users']),5,40)
   near(raw['logits'],raw['pose_logits']);near(raw['headarm_five_view_logits'].mean(1),raw['pose_logits']);assert np.array_equal(raw['old_private_logits'],z6[:,5]);changed=z6.copy();changed[:,5]=raw['pose_logits'];assert np.array_equal(changed[:,:5],z6[:,:5]);fused=fusion(changed,av,cal)
   value[arm+'_private']=raw['pose_logits'];value[arm+'_fused']=fused;stats[arm+'_vs_old_private']=pair(a['targets'],a['users'],z6[:,5],raw['pose_logits']);stats[arm+'_vs_old_POSE6']=pair(a['targets'],a['users'],old,fused)
  stats['CLIP_STRETCH_vs_RAW_CONTROL']=pair(a['targets'],a['users'],value['RAW_CONTROL_fused'],value['CLIP_STRETCH_fused']);stats['CLIP_STRETCH_vs_RAW_CONTROL_private']=pair(a['targets'],a['users'],value['RAW_CONTROL_private'],value['CLIP_STRETCH_private']);perfold[str(fold)]=stats;pieces.append(value);bounded()
 a={k:np.concatenate([v[k] for v in pieces]) for k in pieces[0]};y=a['targets'];u=a['users'];assert len(y)==len(set(a['clip_ids']))==1094 and int((a['old_pose6'].argmax(1)==y).sum())==855 and int((a['old_private'].argmax(1)==y).sum())==820
 stats={}
 for arm in ARMS:
  for name,key in [('private','old_private'),('fused','old_pose6')]:stats[arm+'_'+name+'_vs_old']=pair(y,u,a[key],a[arm+'_'+name])
 for name in ('private','fused'):stats['CLIP_STRETCH_vs_RAW_CONTROL_'+name]=pair(y,u,a['RAW_CONTROL_'+name],a['CLIP_STRETCH_'+name])
 uncertainty={key:bootstrap_group(a[before],a[after],y,u,a['clip_ids']) for key,before,after in [('matched_fused','RAW_CONTROL_fused','CLIP_STRETCH_fused'),('stretch_vs_old','old_pose6','CLIP_STRETCH_fused')]}
 out.mkdir(parents=True,exist_ok=True);casepath=out/'CASES.csv';assert not casepath.exists()
 with casepath.open('x',encoding='utf-8-sig',newline='') as f:
  names=['clip_id','target','user']+[k+'_prediction' for k,v in a.items() if v.ndim==2];w=csv.DictWriter(f,fieldnames=names);w.writeheader()
  for i in range(len(y)):w.writerow(dict(clip_id=str(a['clip_ids'][i]),target=int(y[i]),user=int(u[i]),**{k+'_prediction':int(v[i].argmax()) for k,v in a.items() if v.ndim==2}))
 npz=out/'FIXED_FUSION.npz'
 with npz.open('xb') as f:np.savez_compressed(f,**a)
 result=dict(status='PASS_INDEPENDENT_PHOTOMETRIC_RECOVERY_FIXED_FUSION_AUDIT',protocol_sha256=rph,original_protocol_sha256=ph,auditor_sha256=sha(__file__),per_fold=perfold,pooled=stats,uncertainty=uncertainty,host_budget_and_barriers=hostproof,global_raw_barrier_sha256=sha(globalbar),fixed_calibrator_sha256=calhashes,original_replay_maxabs=replay,original_private_correct820=True,original_POSE6_correct855=True,unchanged_branches_0_through_4_and_availability=True,source_train_held_isolation=True,paired_initial_and_sampling_identical=True,all_four_raw_terminal_before_metrics=True,calibration_fit_steps=0,no_model_forward=True,no_Test=True,no_production=True,screening_only=True,verified_files=CHECKED,arrays_opened=OPENED,seconds=time.time()-BEGIN,limitations=['This is a fixed-old-calibrator initial screen, not source-calibrated production evidence.','Original four fits and cloud evaluation retain the original protocol/budgets. Only local inference used a separately registered900s recovery window; it is not charged to or represented as the original1800s window.','Only seven repeatedly exposed outer users; bootstrap does not correct adaptive selection.','User/class clip-prefix groups are sensitivity proxies, not official recording boundaries.','Stored checkpoints verified by byte SHA and frozen metadata; no model or optimizer replay.'])
 bounded();atomic(out/'RESULT.json',result);atomic(out/'DONE.json',dict(status='COMPLETE',protocol_sha256=rph,original_protocol_sha256=ph,artifacts={n:sha(out/n) for n in ('RESULT.json','CASES.csv','FIXED_FUSION.npz')},no_fit=True,at_unix=time.time()));return dict(status=result['status'],output=str(out),pooled={k:{n:v[n] for n in ('before','after','fix','harm','net','nll_before','nll_after')} for k,v in stats.items()},seconds=result['seconds'])

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--expected-protocol-sha',required=True);r=main(p.parse_args());print(json.dumps(r,ensure_ascii=False));raise SystemExit(3 if r['status']=='NOT_READY' else 0)
