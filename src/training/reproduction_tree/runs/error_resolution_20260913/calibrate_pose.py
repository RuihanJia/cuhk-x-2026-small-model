"""Keep old crop V/S/T/I; add one source-calibrated IR pose expert."""
from pathlib import Path
import json,time,hashlib,csv
import numpy as np
import torch
import calibrate_crop_thermal as c
R,ROOT=c.R,c.ROOT
c.fm.STRATEGIES['convex_vsti_pose']=dict(branches=[0,1,2,3,4],mode='convex',availability=False)

def pose_parts(f,private):
 protocol=R/('POSE_PRIVATE_INNER_PROTOCOL.json' if private else 'POSE_INNER_PROTOCOL.json');s=c.read(protocol);parts=[];bindings=[]
 for j in s['jobs']:
  if j['fold']!=f:continue
  p=R/('pose_private_inner_output/cloud' if private else 'pose_inner_output')/j['id']
  if not(p/'DONE.json').exists():return None,None
  d=c.read(p/'DONE.json');assert d['protocol_sha256']==c.sha(protocol) and d['source_sha256']==j['source_sha256'] and c.sha(p/'predictions.npz')==d['predictions_sha256'];assert not(set(j['train_users'])|set(j['held_users']))&set(j['outer_excluded_users']);assert not set(j['train_users'])&set(j['held_users'])
  a=c.arrays(p/'predictions.npz');assert set(a['users'])==set(j['held_users'])
  if private:
   sp=c.read(p/'SPLIT.json');assert sp['train_users']==j['train_users'] and sp['outer_excluded_users']==j['outer_excluded_users'];assert c.sha(p/'adapter_fp16.pt')==d['checkpoint_sha256'] and d['epochs']==8
  a['pose']=a['phase_spatial_modality_logits'][:,:,1,1].mean(1);parts.append(a);bindings.append(dict(job=j['id'],done_sha256=c.sha(p/'DONE.json')))
 assert len(parts)==3
 return c.concat(parts,['clip_ids','targets','users','pose']),bindings

def add_pose(data,a):
 a=c.reorder(a,data['clip_ids']);assert np.array_equal(a['targets'],data['targets']) and np.array_equal(a['users'],data['users']);data=c.copydata(data);data['logits']=np.concatenate([data['logits'],a['pose'][:,None]],1);data['available']=np.concatenate([data['available'],np.ones((len(a['pose']),1),bool)],1);return data

def main():
 torch.set_num_threads(2);p=R/'POSE_CALIBRATION_PROTOCOL.json';s=c.read(p);assert s['analysis_code_sha256']==c.sha(__file__);out=R/'pose_calibration';out.mkdir(exist_ok=True);guard=c.Guard(min(time.time()+600,s['deadline_unix']));stats={};packed={};bindings={};rows=[]
 for f in [0,3]:
  _,crop,_,cb=c.sources(f,c.read(R/'CROP_THERMAL_CALIBRATION_PROTOCOL.json'));fit={};partbind={}
  for key,priv in [('ZERO_POSE5',False),('PRIVATE_POSE5',True)]:
   a,b=pose_parts(f,priv)
   if a is not None:fit[key]=add_pose(crop,a);partbind[key]=b
  for key,data in fit.items():
   dest=out/f'F{f}_{key}';dest.mkdir(exist_ok=True);fr=dest/'CALIBRATION_FROZEN.json';mp=dest/'calibrator.json';bind=dict(crop=[x for x in cb if x['component']=='CROP'],pose=partbind[key]);bh=hashlib.sha256(json.dumps(bind,sort_keys=True).encode()).hexdigest()
   if not fr.exists():
    assert not mp.exists();m=c.fm.fit(data,'convex_vsti_pose',guard,500);c.write(mp,m);c.write(fr,dict(protocol_sha256=c.sha(p),bindings=bind,bindings_sha256=bh,model_sha256=c.sha(mp),at_unix=time.time(),outer_labels_used_for_fit=False))
   d=c.read(fr);assert d['protocol_sha256']==c.sha(p) and d['bindings_sha256']==bh and d['model_sha256']==c.sha(mp);bindings[f'F{f}_{key}']=dict(frozen_sha256=c.sha(fr),calibrator_sha256=c.sha(mp))
  oc,_,ref,zero=c.outer(f);preds={}
  for key in fit:
   ap=(R/'pose_adapter_output/cloud'/f'POSE_PRIVATE_F{f}') if key=='PRIVATE_POSE5' else (R/'pose_crop_output'/f'F{f}')
   # Use the validated endpoint only after its source calibrator was frozen.
   a=c.arrays(ap/'predictions.npz');d=c.read(ap/'DONE.json');assert c.sha(ap/'predictions.npz')==d['predictions_sha256'];a['pose']=a['phase_spatial_modality_logits'][:,:,1,1].mean(1);data=add_pose(oc,a);z,w=c.fm.predict(data,c.read(out/f'F{f}_{key}'/'calibrator.json'),True);preds[key]=z;np.savez_compressed(out/f'F{f}_{key}'/'predictions.npz',clip_ids=data['clip_ids'],targets=data['targets'],users=data['users'],fused=z,weights=w,R6=ref,zero_crop=zero);stats[f'F{f}_{key}']=dict(vs_R6=c.paired(data['targets'],data['users'],ref,z),vs_zero_crop=c.paired(data['targets'],data['users'],zero,z))
  packed[f]=dict(targets=oc['targets'],users=oc['users'],R6=ref,zero_crop=zero,**preds)
  for i,k in enumerate(oc['clip_ids']):rows.append(dict(clip_id=str(k),fold=f,user=int(oc['users'][i]),target=int(oc['targets'][i]),R6=int(ref[i].argmax()),zero_crop=int(zero[i].argmax()),**{k:int(v[i].argmax()) for k,v in preds.items()}))
 common=set(packed[0])&set(packed[3]);z={k:np.concatenate([packed[f][k] for f in [0,3]]) for k in common};pooled={k:dict(vs_R6=c.paired(z['targets'],z['users'],z['R6'],z[k]),vs_zero_crop=c.paired(z['targets'],z['users'],z['zero_crop'],z[k])) for k in common if k not in ['targets','users','R6','zero_crop']};result=dict(status='COMPLETE' if len(bindings)==4 else 'PARTIAL',at_unix=time.time(),bindings=bindings,per_fold=stats,pooled=pooled,protocol_sha256=c.sha(p),no_Test=True,limits='Repeated exposed developmentusers; no Public forecast.')
 h=hashlib.sha256(json.dumps(bindings,sort_keys=True).encode()).hexdigest();rec=out/(h+'.json')
 if not rec.exists():
  c.write(rec,result)
  with(ROOT/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf-8') as fp:fp.write(json.dumps(dict(event='POSE_SOURCE_CALIBRATION_AUDIT',at_unix=time.time(),artifact=str(rec.relative_to(ROOT)),status=result['status'],pooled=pooled))+'\n')
 c.write(out/'RESULT.json',result);fields=list(dict.fromkeys(k for r in rows for k in r))
 with(out/'ALL_PAIRED_CASES.csv').open('w',encoding='utf-8-sig',newline='') as fp:w=csv.DictWriter(fp,fieldnames=fields);w.writeheader();w.writerows(rows)
 print(json.dumps(dict(status=result['status'],per_fold=stats,pooled=pooled)))
if __name__=='__main__':main()
