"""Independent NumPy checks of mirrored logits, fixed half mixing, four calibrators and comparisons. No fit/model import."""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
E=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def check(p,h):assert sha(p)==h,str(p);return Path(p)
def arr(p):
 with np.load(p,allow_pickle=False) as v:return {k:v[k] for k in v.files}
def subset(v,ix):return {k:x[ix] for k,x in v.items()}
def aligned(v,ids,y,u):
 assert len(v['clip_ids'])==len(set(v['clip_ids']));lookup={c:i for i,c in enumerate(v['clip_ids'])};q=subset(v,np.array([lookup[c] for c in ids]));assert np.array_equal(q['targets'],y) and np.array_equal(q['users'],u);return q
def softmax(z,axis):
 z=np.asarray(z,np.float64);p=np.exp(z-z.max(axis,keepdims=True));return p/p.sum(axis,keepdims=True)
def replay(z,mask,cal,kind):
 p=softmax(z/np.array(cal['temperature'],np.float64)[None,:,None],2)
 if kind=='global':bias=np.broadcast_to(np.array(cal['state']['bias'])[None,:,None],z.shape).copy()
 else:
  delta=np.array(cal['delta']);delta-=delta.mean(0,keepdims=True);bias=np.broadcast_to(np.array(cal['global_bias'])[None,:,None]+delta[None],z.shape).copy()
 bias[~mask]=-np.inf;value=(p*softmax(bias,1)).sum(1)
 if kind=='classwise':value/=value.sum(1,keepdims=True)
 return np.log(value).astype(np.float32)
def nll(z,y):
 z=np.asarray(z,np.float64);m=z.max(1);return float((m+np.log(np.exp(z-m[:,None]).sum(1))-z[np.arange(len(y)),y]).mean())
def verify_pair(q,y,u,a,b):
 before=a.argmax(1)==y;after=b.argmax(1)==y
 assert q['n']==len(y) and q['before']==int(before.sum()) and q['after']==int(after.sum()) and q['fix']==int((~before&after).sum()) and q['harm']==int((before&~after).sum())
 assert q['net']==int(after.sum()-before.sum()) and q['prediction_changes']==int((a.argmax(1)!=b.argmax(1)).sum())
 assert abs(q['NLL_before']-nll(a,y))<1e-9 and abs(q['NLL_after']-nll(b,y))<1e-9
 for field,group in [('per_user',u),('per_class',y)]:
  for g in np.unique(group):
   ix=group==g;v=q[field][str(g)];assert v['n']==int(ix.sum()) and v['before']==int(before[ix].sum()) and v['after']==int(after[ix].sum()) and v['fix']==int((~before[ix]&after[ix]).sum()) and v['harm']==int((before[ix]&~after[ix]).sum())
def main(protocol,expected_protocol_sha):
 fs=read(check(protocol,expected_protocol_sha));ph=expected_protocol_sha;assert time.time()<fs['deadline_unix']
 for p,h in fs['code_bindings'].items():check(p,h)
 for v in list(fs['inputs'].values())+list(fs['references'].values()):check(v['path'],v['sha256'])
 out=E/'output';d=read(out/'DONE.json');fr=read(out/'CALIBRATORS_FROZEN.json');bar=read(out/'ALL_EIGHT_READOUTS_FROZEN.json');r=read(out/'RESULT.json')
 assert d['status']=='COMPLETE' and d['protocol_sha256']==fr['protocol_sha256']==bar['protocol_sha256']==r['protocol_sha256']==ph and fr['calibrators']==4 and fr['all_source_calibrators_before_outer_arrays'] and not bar['outer_arrays_opened']
 assert len(bar['jobs'])==8 and all(x['all_owned_dead'] for x in bar['receipts']) and fr['at_unix']<=r['first_outer_arrays_started_unix']
 for n,h in d['artifacts'].items():check(out/n,h)
 for n,h in fr['files'].items():check(out/n,h)
 src=arr(out/'SOURCE_FITTED.npz');outer=arr(out/'PAIRED.npz');original=arr(fs['inputs']['raw_source']['path']);oof=arr(fs['inputs']['source_oof']['path']);rawouter=arr(fs['references']['RTN']['path'])
 assert len(src['users'])==4768 and len(set(src['clip_ids']))==2931 and len(set(zip(src['source_fold'],src['clip_ids'])))==4768 and len(outer['users'])==len(set(outer['clip_ids']))==1094
 readout_checks=[]
 for j in bar['jobs']:
  root=Path(j['root']);v=arr(check(root/'predictions.npz',j['predictions_sha256']));b=arr(check(root/'Base.npz',j['base_views_sha256']));t=arr(check(root/'Small.npz',j['small_views_sha256']));check(root/'DONE.json',j['done_sha256'])
  assert len(v['users'])==j['records'] and set(v['users'])==set(j['held_users']) and set(v['clip_ids'])==set(j['held_ids'])
  for other in [b,t]:
   for k in ['clip_ids','targets','users']:assert np.array_equal(v[k],other[k])
  bv=b['seven_view_logits'];tv=t['five_view_logits'];assert bv.shape==(len(v['users']),7,40) and tv.shape==(len(v['users']),5,40)
  qb=(.5*bv[:,:5].mean(1)+.25*bv[:,5]+.25*bv[:,6]).astype(np.float32);qt=tv.mean(1).astype(np.float32)
  assert np.array_equal(qb,v['base6_mirror']) and np.array_equal(qt,v['small8_mirror'])
  dest=subset(src,src['source_fold']==j['fold']) if j['scope']=='source' else subset(outer,outer['fold']==j['fold']);actual=aligned(dest,v['clip_ids'],v['targets'],v['users'])
  for k in ['base6','small8']:
   assert np.array_equal(actual[k],v[k+'_original']) and np.array_equal(actual[k+'_mirror'],v[k+'_mirror'])
  assert np.array_equal(actual['base_mixed'],v['base6']) and np.array_equal(actual['small_mixed'],v['small8'])
  for label,views in [('base6',b),('small8',t)]:
   assert np.array_equal(v[label+'_original'],views['original_logits']) and np.array_equal(v[label+'_mirror'],views['mirror_logits']) and np.array_equal(v[label],views['mixed_logits'])
  ref=subset(original,original['source_fold']==j['fold']) if j['scope']=='source' else rawouter;ref=aligned(ref,v['clip_ids'],v['targets'],v['users'])
  assert np.array_equal(ref['base6'],v['base6_original']) and np.array_equal(ref['small8'],v['small8_original'])
  readout_checks.append(dict(id=j['id'],originals_exact=True,mirrored_perview_aggregation_exact=True))
 errors={}
 for scope,v,fk in [('source',src,'source_fold'),('outer',outer,'fold')]:
  assert np.array_equal(v['base_mixed'],(.5*v['base6']+.5*v['base6_mirror']).astype(np.float32)) and np.array_equal(v['small_mixed'],(.5*v['small8']+.5*v['small8_mirror']).astype(np.float32))
  for fold in [0,3]:
   ix=v[fk]==fold;q=subset(v,ix)
   if scope=='source':
    b=subset(oof,oof['source_fold']==fold);b['clip_ids']=b['original_clip_ids'];b=aligned(b,q['clip_ids'],q['targets'],q['users']);assert not set(q['users'])&set(fs['outer_users'][str(fold)])
   else:b=aligned(arr(fs['inputs'][f'outer_scores_{fold}']['path']),q['clip_ids'],q['targets'],q['users']);assert set(q['users'])==set(fs['outer_users'][str(fold)])
   assert np.array_equal(q['TI'],b['logits'][:,[2,3]]) and np.array_equal(q['available'][:,:2],b['available'][:,[2,3]]) and q['available'][:,2:].all()
   z=np.concatenate([q['TI'],q['base_mixed'][:,None],q['small_mixed'][:,None]],1)
   for kind,key in [('global','global4'),('classwise','classwise4')]:
    cal=read(out/f'F{fold}_{kind}.json');assert not cal['outer_labels_used'] and set(cal['calibration_users'])==set(src['users'][src['source_fold']==fold])
    got=replay(z,q['available'],cal,kind);err=float(np.max(np.abs(got-q[key])));assert err<2e-6 and np.array_equal(got.argmax(1),q[key].argmax(1));errors[f'{scope}_F{fold}_{kind}']=err
    if kind=='classwise':assert cal['penalty']==.1 and cal['parameters']==160
    if scope=='source':verify_pair(r['source_comparison'][str(fold)][key],q['targets'],q['users'],q['raw_'+key],q[key])
   if scope=='source':
    for label,before,after in [('Base','base6','base_mixed'),('Small','small8','small_mixed')]:verify_pair(r['source_comparison'][str(fold)][label],q['targets'],q['users'],q[before],q[after])
 for name,reference in fs['references'].items():
  ref=aligned(arr(reference['path']),outer['clip_ids'],outer['targets'],outer['users']);assert np.array_equal(outer[name],ref[reference['field']]) and int((outer[name].argmax(1)==outer['targets']).sum())==reference['correct']
  for key in ['base_mixed','small_mixed','global4','classwise4']:verify_pair(r['comparisons'][name][key],outer['targets'],outer['users'],outer[name],outer[key])
 result=dict(status='PASS_INDEPENDENT_MIRROR_DUAL_SOURCE_AND_OUTER_REPLAY',protocol_sha256=ph,result_sha256=sha(out/'RESULT.json'),calibrators_frozen_sha256=sha(out/'CALIBRATORS_FROZEN.json'),source_rows=4768,source_unique_clips=2931,outer_rows=1094,readout_checks=readout_checks,replay_maxabs=errors,calibration_fits_in_audit=0,GPU_forwards=0,no_Test=True,at_unix=time.time())
 p=out/'INDEPENDENT_AUDIT.json'
 with p.open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--expected-protocol-sha',required=True);main(**vars(p.parse_args()))
