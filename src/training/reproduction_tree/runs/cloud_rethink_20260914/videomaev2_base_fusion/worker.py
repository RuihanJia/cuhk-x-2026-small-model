"""Source-isolated T/I/Base calibration after actual fixed50 INT8 replay."""
from pathlib import Path
import argparse,importlib.util,json,os,sys,time
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
E=Path(__file__).resolve().parent;C=E.parent;R=C.parents[1];F=C/'videomae_kinetics_fusion'
sys.path[:0]=[str(F),str(C)]
sp=importlib.util.spec_from_file_location('base_source_helpers',F/'worker.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
a=m.a;np=m.np;torch=m.torch;fm=m.fm
sp=importlib.util.spec_from_file_location('base_classwise3',E/'classwise3.py');cw=importlib.util.module_from_spec(sp);sp.loader.exec_module(cw)
REMOTE='/home/featurize/cuhkx_motion_transfer_20260911_v1/cloud_rethink_20260914/'

def mirror(p):
 if p.startswith(REMOTE):return C/p[len(REMOTE):].replace('/output/cloud','/cloud')
 return Path(p)

def convex_torch(data,g):
 assert g['description']['mode']=='convex' and not g['description']['availability']
 model=fm.Calibrator(g['description']);model.load_state_dict({k:torch.tensor(v,dtype=torch.float64) for k,v in g['state'].items()},strict=True);model.eval()
 with torch.no_grad():v=model(dict(logits=torch.tensor(data['logits'],dtype=torch.float64),mask=torch.tensor(data['available'],dtype=torch.bool)))
 return v.numpy().astype(np.float32)

def main():
 s=a.read(E/'protocol.json');ph=a.sha(E/'protocol.json');start=time.time();assert start<s['activation_deadline_unix']
 out=E/'output';out.mkdir(exist_ok=True);assert not(out/'DONE.json').exists();clock=m.Clock(min(start+240,s['activation_deadline_unix']));torch.set_num_threads(1)
 for k,v in s['code_sha256'].items():a.check(C/k,v)
 qs=a.read(a.check(C/'videomaev2_base_quant/protocol.json',s['quant_protocol_sha256']))
 for host in ['cloud','local']:m.terminal(C/'videomaev2_base_quant'/('cloud' if host=='cloud' else 'output/local'),s['quant_protocol_sha256'])
 for stage in ['videomaev2_base','videomaev2_base_source']:
  ps=a.read(a.check(C/stage/'protocol.json',s['parent_protocols'][stage]))
  for host in ps['hosts']:m.terminal(C/stage/('cloud' if host=='cloud' else 'output/local'),s['parent_protocols'][stage])
 source=a.arr(a.check(s['source_oof'],s['source_oof_sha256']));assert source['logits'].shape==(4768,10,40) and len(set(source['original_clip_ids']))==2931
 fs=a.read(a.check(C/'trajectory_fixed_fusion_protocol.json',s['fixed_fusion_protocol_sha256']))
 fm.STRATEGIES['convex_TI_BASE3']=dict(branches=[0,1,2],mode='convex',availability=False)
 parents={};sourceparts=[];files={};fitted={}
 # All source calibration is completed and checksummed before either outer score array is opened.
 for f in [0,3]:
  ix=source['source_fold']==f;base=m.subset(source,ix);base['clip_ids']=base['original_clip_ids'];parts=[];rawparts=[]
  for j in [j for j in qs['jobs'] if j['scope']=='source' and j['fold']==f]:
   parent=mirror(j['parent_directory']);split=a.read(parent/'SPLIT.json');fr=a.read(parent/'FROZEN.json')
   assert split['protocol_sha256']==fr['protocol_sha256']==j['parent_protocol_sha256']
   assert split['train_users']==fr['train_users']==j['train_users'] and split['held_users']==fr['held_users']==j['held_users']
   assert not set(j['train_users'])&(set(j['held_users'])|set(j['outer_excluded_users'])) and not set(j['held_users'])&set(j['outer_excluded_users'])
   qroot=C/'videomaev2_base_quant'/('cloud' if j['host']=='cloud' else 'output/local')/j['id'];d=a.read(qroot/'DONE.json');binding=a.read(qroot/'EXACT_PARENT_BINDING.json')
   assert d['protocol_sha256']==s['quant_protocol_sha256'] and d['parent_checkpoint_sha256']==binding['checkpoint_sha256']==fr['checkpoint_sha256']
   assert d['quantized_matrices']==50 and d['outer_excluded_users']==j['outer_excluded_users'] and d['no_optimizer']
   v=a.arr(a.check(qroot/'predictions.npz',d['predictions_sha256']));raw=a.align(a.arr(parent/'predictions.npz'),v)
   assert set(v['users'])==set(j['held_users']) and set(v['clip_ids'])==set(split['held_ids']) and not set(split['train_ids'])&set(v['clip_ids'])
   assert np.array_equal(v['targets'],raw['targets']);parts.append(v);rawparts.append(raw);parents[j['id']]=dict(parent_checkpoint_sha256=fr['checkpoint_sha256'],quant_checkpoint_sha256=d['checkpoint_sha256'],split_sha256=a.sha(parent/'SPLIT.json'))
  q=a.align({k:np.concatenate([v[k] for v in parts]) for k in parts[0]},base);raw=a.align({k:np.concatenate([v[k] for v in rawparts]) for k in rawparts[0]},base)
  assert len(set(q['clip_ids']))==len(q['clip_ids']) and not set(base['users'])&set(s['outer_users'][str(f)])
  data=dict(base,logits=np.concatenate([base['logits'][:,[2,3]],q['videomae_logits'][:,None]],1),available=np.concatenate([base['available'][:,[2,3]],np.ones((len(q['users']),1),bool)],1))
  g=fm.fit(data,'convex_TI_BASE3',clock,500);c=cw.fit(data['logits'],data['available'],data['targets'],data['users'],g,clock)
  for name,cal in [('global',g),('classwise',c)]:
   assert not cal['outer_labels_used'] and not set(cal['calibration_users'])&set(s['outer_users'][str(f)])
   p=out/f'F{f}_{name}.json';a.atomic(p,cal);files[p.name]=a.sha(p)
  fitted[f]=(g,c);v={k:base[k] for k in ['clip_ids','targets','users']};v.update(source_fold=np.full(len(q['users']),f),raw_base=raw['videomae_logits'],quant_base=q['videomae_logits'],TI=base['logits'][:,[2,3]],available=data['available'],global3=m.convex_replay(data,g),classwise3=cw.predict(data['logits'],data['available'],c));sourceparts.append(v)
 a.atomic(out/'CALIBRATORS_FROZEN.json',dict(protocol_sha256=ph,files=files,calibrators=4,source_parents=parents,all_source_fits_and_calibrators_frozen_before_outer_arrays=True,at_unix=time.time()))
 parts=[];per={}
 for f in [0,3]:
  inp=fs['inputs'][str(f)];base=a.arr(a.check(inp['scores'],inp['scores_sha256']));j=next(j for j in qs['jobs'] if j['scope']=='outer' and j['fold']==f)
  qroot=C/'videomaev2_base_quant'/('cloud' if j['host']=='cloud' else 'output/local')/j['id'];d=a.read(qroot/'DONE.json');q=a.align(a.arr(a.check(qroot/'predictions.npz',d['predictions_sha256'])),base);raw=a.align(a.arr(mirror(j['parent_directory'])/'predictions.npz'),base)
  assert set(base['users'])==set(s['outer_users'][str(f)])
  data=dict(base,logits=np.concatenate([base['logits'][:,[2,3]],q['videomae_logits'][:,None]],1),available=np.concatenate([base['available'][:,[2,3]],np.ones((len(q['users']),1),bool)],1))
  g,c=fitted[f];assert not set(g['calibration_users'])&set(base['users']);globalp=m.convex_replay(data,g);a.near(globalp,convex_torch(data,g))
  classp=cw.predict(data['logits'],data['available'],c);v={k:base[k] for k in ['clip_ids','targets','users']};v.update(raw_base=raw['videomae_logits'],quant_base=q['videomae_logits'],global3=globalp,classwise3=classp)
  parts.append(v)
 allv={k:np.concatenate([v[k] for v in parts]) for k in parts[0]};reference=a.align(a.arr(a.check(s['current_reference'],s['current_reference_sha256'])),allv);allv['current7']=reference['global7'];assert len(allv['targets'])==1094 and (allv['current7'].argmax(1)==allv['targets']).sum()==900
 sourcev={k:np.concatenate([v[k] for v in sourceparts]) for k in sourceparts[0]};assert len(sourcev['targets'])==4768
 np.savez_compressed(out/'PAIRED.npz',**allv);np.savez_compressed(out/'SOURCE_FITTED.npz',**sourcev)
 result=dict(status='COMPLETE_V2_BASE_TI_INT8_SOURCE_ONLY_FUSION',protocol_sha256=ph,pooled={k:a.pair(allv['targets'],allv['users'],allv['current7'],allv[k]) for k in ['quant_base','global3','classwise3']},quant_precision=a.pair(allv['targets'],allv['users'],allv['raw_base'],allv['quant_base']),source_precision=a.pair(sourcev['targets'],sourcev['users'],sourcev['raw_base'],sourcev['quant_base']),uncertainty={k:a.boot(allv['targets'],allv['users'],allv['current7'],allv[k]) for k in ['global3','classwise3']},source_rows=4768,source_unique=2931,no_Test=True,no_CSV=True,limitations='Repeated adaptive7-user development; source fit metrics are not validation. No Public guarantee. Final18model/wholepackage not yet qualified.',seconds=time.time()-start,at_unix=time.time())
 a.atomic(out/'RESULT.json',result);files.update({n:a.sha(out/n) for n in ['RESULT.json','PAIRED.npz','SOURCE_FITTED.npz','CALIBRATORS_FROZEN.json']});a.atomic(out/'DONE.json',dict(status='COMPLETE',protocol_sha256=ph,artifacts=files,at_unix=time.time()))
 event=dict(event='V2_BASE_TI_INT8_SOURCE_ONLY_FUSION_COMPLETE',result=str((out/'RESULT.json').relative_to(R)),result_sha256=a.sha(out/'RESULT.json'),protocol_sha256=ph,at_unix=time.time())
 with(R/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf-8') as fp:fp.write(json.dumps(event)+'\n')
 print(json.dumps({k:{n:v for n,v in x.items() if n not in ['per_user','per_class']} for k,x in result['pooled'].items()}))
if __name__=='__main__':main()
