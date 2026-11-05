"""Registered small factorial: preserve both visual specialists, add temporal pose."""
from pathlib import Path
import json,time,hashlib,csv
import numpy as np
import torch
import calibrate_pose as p
import calibrate_posec3d as pc
c=p.c;R=p.R;ROOT=p.ROOT

def c3d(f,inner):
 proto=R/'POSEC3D_FINETUNE_PROTOCOL.json';parts=[];bindings=[]
 for j in c.read(proto)['jobs']:
  if j['fold']!=f or (j['kind']=='inner')!=inner:continue
  d=R/'posec3d_finetune_output'/j['host']/j['id']
  if not(d/'DONE.json').exists():return None,None
  z=c.read(d/'DONE.json');assert z['protocol_sha256']==c.sha(proto) and z['checkpoint_sha256']==c.sha(d/'model_fp16.pt') and z['predictions_sha256']==c.sha(d/'predictions.npz');assert not set(j['train_users'])&set(j['held_users']);assert not (set(j['train_users'])|set(j['held_users']))&set(j.get('outer_excluded_users',[]));sp=c.read(d/'SPLIT.json');assert sp['train_users']==j['train_users'] and sp['held_users']==j['held_users'];a=c.arrays(d/'predictions.npz');assert set(a['users'])==set(j['held_users']);parts.append(a);bindings.append(dict(job=j['id'],done_sha256=c.sha(d/'DONE.json')))
 assert len(parts)==(3 if inner else 1)
 return c.concat(parts,['clip_ids','users','targets','logits','available']),bindings

def source(f,recipe):
 _,crop,_,cb=c.sources(f,c.read(R/'CROP_THERMAL_CALIBRATION_PROTOCOL.json'));z,zb=p.pose_parts(f,False);data=p.add_pose(crop,z);bindings=dict(crop=[q for q in cb if q['component']=='CROP'],pose=zb)
 if recipe['private']:
  z,zb=p.pose_parts(f,True)
  if z is None:return None,None
  data=p.add_pose(data,z);bindings['private']=zb
 if recipe['temporal']:
  z,zb=c3d(f,True)
  if z is None:return None,None
  data=pc.add(data,z);bindings['temporal']=zb
 return data,bindings

def outer(f,recipe):
 data,_,ref,zero=c.outer(f);a=c.arrays(R/'pose_crop_output'/f'F{f}'/'predictions.npz');a['pose']=a['phase_spatial_modality_logits'][:,:,1,1].mean(1);data=p.add_pose(data,a)
 if recipe['private']:
  a=c.arrays(R/'pose_adapter_output/cloud'/f'POSE_PRIVATE_F{f}'/'predictions.npz');a['pose']=a['phase_spatial_modality_logits'][:,:,1,1].mean(1);data=p.add_pose(data,a)
 if recipe['temporal']:
  a,_=c3d(f,False)
  if a is None:return None,None,None
  data=pc.add(data,a)
 prior=c.arrays(R/'pose_calibration'/f'F{f}_ZERO_POSE5'/'predictions.npz');assert np.array_equal(prior['clip_ids'],data['clip_ids'])
 return data,ref,prior['fused']

def main():
 torch.set_num_threads(2);proto=R/'POSE_COMPLEMENT_CALIBRATION_PROTOCOL.json';s=c.read(proto);assert c.sha(__file__)==s['worker_sha256'];out=R/'pose_complement_calibration';out.mkdir(exist_ok=True);guard=c.Guard(min(s['deadline_unix'],time.time()+600));stats={};packed={};bindings={};rows=[]
 for recipe in s['recipes']:
  key=recipe['id'];strategy='convex_'+key;nb=5+int(recipe['private'])+int(recipe['temporal']);c.fm.STRATEGIES[strategy]=dict(branches=list(range(nb)),mode='convex',availability=False);packed[key]={}
  for f in [0,3]:
   data,bind=source(f,recipe)
   if data is None:continue
   dst=out/f'F{f}_{key}';dst.mkdir(exist_ok=True);mp=dst/'calibrator.json';fr=dst/'CALIBRATION_FROZEN.json';bh=hashlib.sha256(json.dumps(bind,sort_keys=True).encode()).hexdigest()
   if not fr.exists():
    assert not mp.exists();m=c.fm.fit(data,strategy,guard,500);c.write(mp,m);c.write(fr,dict(protocol_sha256=c.sha(proto),bindings=bind,bindings_sha256=bh,model_sha256=c.sha(mp),at_unix=time.time(),outer_labels_used_for_fit=False))
   q=c.read(fr);assert q['protocol_sha256']==c.sha(proto) and q['bindings_sha256']==bh and q['model_sha256']==c.sha(mp)
   data,ref,prior=outer(f,recipe)
   if data is None:continue
   z,w=c.fm.predict(data,c.read(mp),True);v=dict(targets=data['targets'],users=data['users'],R6=ref,POSE5=prior,fused=z);packed[key][f]=v;np.savez_compressed(dst/'predictions.npz',clip_ids=data['clip_ids'],weights=w,**v);stats[f'F{f}_{key}']=dict(vs_R6=c.paired(v['targets'],v['users'],ref,z),vs_pose5=c.paired(v['targets'],v['users'],prior,z));bindings[f'F{f}_{key}']=dict(frozen_sha256=c.sha(fr),model_sha256=c.sha(mp))
   for i,k in enumerate(data['clip_ids']):rows.append(dict(recipe=key,clip_id=str(k),fold=f,user=int(data['users'][i]),target=int(data['targets'][i]),R6=int(ref[i].argmax()),POSE5=int(prior[i].argmax()),prediction=int(z[i].argmax())))
 pooled={}
 for key,folds in packed.items():
  if len(folds)!=2:continue
  z={k:np.concatenate([folds[f][k] for f in [0,3]]) for k in folds[0]};pooled[key]=dict(vs_R6=c.paired(z['targets'],z['users'],z['R6'],z['fused']),vs_pose5=c.paired(z['targets'],z['users'],z['POSE5'],z['fused']))
 result=dict(status='COMPLETE' if len(bindings)==len(s['recipes'])*2 else 'PARTIAL',protocol_sha256=c.sha(proto),at_unix=time.time(),bindings=bindings,per_fold=stats,pooled=pooled,limits='Repeated exposed developmentusers; no independent Public proof.',no_Test=True);h=hashlib.sha256(json.dumps(bindings,sort_keys=True).encode()).hexdigest();rec=out/(h+'.json')
 if not rec.exists():
  c.write(rec,result)
  with(ROOT/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf-8') as fp:fp.write(json.dumps(dict(event='POSE_COMPLEMENT_CALIBRATION_RESULT',at_unix=time.time(),artifact=str(rec.relative_to(ROOT)),status=result['status'],pooled=pooled))+'\n')
 c.write(out/'RESULT.json',result)
 if rows:
  with(out/'ALL_PAIRED_CASES.csv').open('w',encoding='utf-8-sig',newline='') as fp:w=csv.DictWriter(fp,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps(dict(status=result['status'],per_fold=stats,pooled=pooled)))
if __name__=='__main__':main()
