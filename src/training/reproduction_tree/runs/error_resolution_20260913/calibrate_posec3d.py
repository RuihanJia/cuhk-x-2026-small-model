"""One fixed source-only test: add temporal pose to the crop+IR-pose system."""
from pathlib import Path
import json,time,hashlib,csv
import numpy as np
import torch
import calibrate_pose as p
c=p.c;R=p.R;ROOT=p.ROOT
c.fm.STRATEGIES['convex_vsti_pose_c3d']=dict(branches=[0,1,2,3,4,5],mode='convex',availability=False)
def branch(f,inner):
 spec=c.read(R/'POSEC3D_LINEAR_PROTOCOL.json');parts=[];bind=[]
 for j in spec['jobs']:
  if j['fold']!=f or (j['kind']=='inner')!=inner:continue
  d=R/'posec3d_linear_output'/j['id']
  if not(d/'DONE.json').exists():return None,None
  q=c.read(d/'DONE.json');assert q['protocol_sha256']==c.sha(R/'POSEC3D_LINEAR_PROTOCOL.json') and q['checkpoint_sha256']==c.sha(d/'linear.pt') and q['predictions_sha256']==c.sha(d/'predictions.npz');assert not set(j['train_users'])&set(j['held_users']) and not(set(j['train_users'])|set(j['held_users']))&set(j.get('outer_excluded_users',[]));a=c.arrays(d/'predictions.npz');parts.append(a);bind.append(dict(job=j['id'],DONE_sha256=c.sha(d/'DONE.json')))
 assert len(parts)==(3 if inner else 1)
 return c.concat(parts,['clip_ids','users','targets','logits','available']),bind
def add(data,expert):
 a=c.reorder(expert,data['clip_ids']);assert np.array_equal(a['targets'],data['targets']) and np.array_equal(a['users'],data['users']);data=c.copydata(data);data['logits']=np.concatenate([data['logits'],a['logits'][:,None]],1);data['available']=np.concatenate([data['available'],a['available'][:,None]],1);return data
def main():
 torch.set_num_threads(2);proto=R/'POSEC3D_CALIBRATION_PROTOCOL.json';s=c.read(proto);assert c.sha(__file__)==s['worker_sha256'];out=R/'posec3d_calibration';out.mkdir(exist_ok=True);guard=c.Guard(min(s['deadline_unix'],time.time()+600));stats={};packed={};bindings={};rows=[]
 for f in [0,3]:
  expert,bind=branch(f,True)
  if expert is None:continue
  _,crop,_,_=c.sources(f,c.read(R/'CROP_THERMAL_CALIBRATION_PROTOCOL.json'));pose,pb=p.pose_parts(f,False);src=add(p.add_pose(crop,pose),expert);dst=out/f'F{f}';dst.mkdir(exist_ok=True);mp=dst/'calibrator.json';fr=dst/'CALIBRATION_FROZEN.json';bh=hashlib.sha256(json.dumps(dict(pose=pb,c3d=bind),sort_keys=True).encode()).hexdigest()
  if not fr.exists():
   assert not mp.exists();model=c.fm.fit(src,'convex_vsti_pose_c3d',guard,500);c.write(mp,model);c.write(fr,dict(protocol_sha256=c.sha(proto),source_binding_sha256=bh,source_bindings=dict(pose=pb,c3d=bind),model_sha256=c.sha(mp),at_unix=time.time(),outer_labels_used_for_fit=False))
  q=c.read(fr);assert q['protocol_sha256']==c.sha(proto) and q['source_binding_sha256']==bh and q['model_sha256']==c.sha(mp)
  expert,_=branch(f,False)
  if expert is None:continue
  crop,_,ref,zero=c.outer(f);pose=c.arrays(R/'pose_crop_output'/f'F{f}'/'predictions.npz');pose['pose']=pose['phase_spatial_modality_logits'][:,:,1,1].mean(1);data=add(p.add_pose(crop,pose),expert);z,w=c.fm.predict(data,c.read(mp),True);old=c.arrays(R/'pose_calibration'/f'F{f}_ZERO_POSE5'/'predictions.npz');assert np.array_equal(old['clip_ids'],data['clip_ids']);prior=old['fused'];stats[str(f)]=dict(vs_R6=c.paired(data['targets'],data['users'],ref,z),vs_pose5=c.paired(data['targets'],data['users'],prior,z));bindings[str(f)]=dict(frozen_sha256=c.sha(fr),model_sha256=c.sha(mp));packed[f]=dict(targets=data['targets'],users=data['users'],R6=ref,POSE5=prior,POSE_C3D6=z);np.savez_compressed(dst/'predictions.npz',clip_ids=data['clip_ids'],weights=w,**packed[f])
  for i,k in enumerate(data['clip_ids']):rows.append(dict(clip_id=str(k),fold=f,user=int(data['users'][i]),target=int(data['targets'][i]),R6=int(ref[i].argmax()),POSE5=int(prior[i].argmax()),POSE_C3D6=int(z[i].argmax())))
 if not packed:print(json.dumps(dict(status='WAITING')));return
 z={k:np.concatenate([packed[f][k] for f in sorted(packed)]) for k in packed[next(iter(packed))]};pooled=dict(vs_R6=c.paired(z['targets'],z['users'],z['R6'],z['POSE_C3D6']),vs_pose5=c.paired(z['targets'],z['users'],z['POSE5'],z['POSE_C3D6']));result=dict(status='COMPLETE' if len(bindings)==2 else 'PARTIAL',at_unix=time.time(),protocol_sha256=c.sha(proto),bindings=bindings,per_fold=stats,pooled=pooled,no_Test=True,limits='Repeated exposed developmentusers; no independentPublicproof.');h=hashlib.sha256(json.dumps(bindings,sort_keys=True).encode()).hexdigest();rec=out/(h+'.json')
 if not rec.exists():
  c.write(rec,result)
  with(ROOT/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf-8') as fp:fp.write(json.dumps(dict(event='POSEC3D_SOURCE_CALIBRATION_RESULT',at_unix=time.time(),artifact=str(rec.relative_to(ROOT)),status=result['status'],pooled=pooled))+'\n')
 c.write(out/'RESULT.json',result)
 with(out/'ALL_PAIRED_CASES.csv').open('w',encoding='utf-8-sig',newline='') as fp:w=csv.DictWriter(fp,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps(dict(status=result['status'],per_fold=stats,pooled=pooled)))
if __name__=='__main__':main()
