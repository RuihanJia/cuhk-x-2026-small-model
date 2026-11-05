"""Source-only calibrations of preregistered crop and private Thermal recipes."""
from pathlib import Path
import json,time,hashlib,sys,csv
import numpy as np
import torch
R=Path(__file__).resolve().parent;ROOT=R.parents[1];OLD=ROOT/'runs/activity_token_stopgrad_20260913';CONTROL=ROOT/'runs/error_driven_overnight_20260913/imu/successors/calibration_export/output'
sys.path.insert(0,str(ROOT/'deliverables/activity_token_stopgrad_candidate_20260913'))
import fusion_math as fm
fm.STRATEGIES['convex_vsti_private_t']=dict(branches=[0,1,2,3,4],mode='convex',availability=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def arrays(p):
 with np.load(p,allow_pickle=False) as f:return {k:f[k] for k in f.files}
def reorder(a,ids):
 lookup={k:i for i,k in enumerate(a['clip_ids'])};assert len(lookup)==len(a['clip_ids']);ix=[lookup[k] for k in ids]
 return {k:v[ix] for k,v in a.items() if hasattr(v,'shape') and len(v.shape)>0 and v.shape[0]==len(a['clip_ids'])}
def replace_video(data,a):
 a=reorder(a,data['clip_ids']);assert np.array_equal(a['targets'],data['targets']) and np.array_equal(a['users'],data['users']);ph=a['phase_spatial_modality_logits']
 data['logits'][:,0]=.5*(ph[:,:,0].mean((1,2))+ph[:,:,1].mean((1,2)));data['depth']=.5*(ph[:,:,0,0].mean(1)+ph[:,:,1,0].mean(1));data['ir']=.5*(ph[:,:,0,1].mean(1)+ph[:,:,1,1].mean(1))
 return data
def add_t(data,a,replace=False):
 lookup={k:i for i,k in enumerate(a['clip_ids'])};assert len(lookup)==len(a['clip_ids']);value=np.zeros_like(data['logits'][:,2]);mask=np.zeros(len(value),bool)
 for i,k in enumerate(data['clip_ids']):
  if k in lookup:
   j=lookup[k];assert a['targets'][j]==data['targets'][i] and a['users'][j]==data['users'][i];value[i]=a['logits'][j];mask[i]=True
 assert np.array_equal(mask,data['available'][:,2]),'THERMAL_AVAILABILITY_MISMATCH'
 if replace:data['logits'][:,2]=value
 else:data['logits']=np.concatenate([data['logits'],value[:,None]],1);data['available']=np.concatenate([data['available'],mask[:,None]],1)
 return data
def copydata(d):return {k:v.copy() for k,v in d.items()}
def concat(parts,keys):return {k:np.concatenate([p[k] for p in parts]) for k in keys}
def paired(y,u,a,b):
 aa=a.argmax(1)==y;bb=b.argmax(1)==y
 return dict(n=len(y),before=int(aa.sum()),after=int(bb.sum()),fix=int((~aa&bb).sum()),harm=int((aa&~bb).sum()),net=int(bb.sum()-aa.sum()),per_user={str(k):int(bb[u==k].sum()-aa[u==k].sum()) for k in np.unique(u)})
class Guard:
 def __init__(self,d):self.deadline=d
 def check(self,**kw):assert time.time()<self.deadline,'CPU_BUDGET'
def sources(f,spec):
 data=arrays(CONTROL/f'fold{f}'/'control_inner.npz');done=read(CONTROL/f'fold{f}'/'DONE.json')
 assert sha(CONTROL/f'fold{f}'/'control_inner.npz').upper()==done['artifacts']['control_inner.npz'];held=spec['outer_users'][str(f)];assert not set(data['users'])&set(held)
 cp=read(R/'CROP_INNER_PROTOCOL.json');tp=read(R/'THERMAL_INNER_PROTOCOL.json');cparts=[];tparts=[];binding=[]
 for cj,tj in zip([j for j in cp['jobs'] if j['fold']==f],[j for j in tp['jobs'] if j['fold']==f]):
  assert cj['train_users']==tj['train_users'] and cj['held_users']==tj['held_users'];assert cj['outer_excluded_users']==tj['outer_excluded_users']==held;assert not (set(tj['train_users'])|set(tj['held_users']))&set(held);assert not set(tj['train_users'])&set(tj['held_users'])
  p=R/'crop_inner_output'/cj['id'];d=read(p/'DONE.json');assert d['protocol_sha256']==sha(R/'CROP_INNER_PROTOCOL.json') and d['source_sha256']==cj['source_sha256'] and sha(p/'predictions.npz')==d['predictions_sha256'];cparts.append(arrays(p/'predictions.npz'));binding.append(dict(component='CROP',job=cj['id'],done_sha256=sha(p/'DONE.json')))
  p=R/'thermal_inner_output'/tj['host']/tj['id']
  if not (p/'DONE.json').exists():continue
  d=read(p/'DONE.json');split=read(p/'SPLIT.json');assert d['protocol_sha256']==sha(R/'THERMAL_INNER_PROTOCOL.json') and d['source_sha256']==tj['source_sha256'];assert sha(p/'predictions.npz')==d['predictions_sha256'] and sha(p/'adapter_fp16.pt')==d['checkpoint_sha256'];assert d['frozen_base_unchanged'] and d['epochs']==8
  assert split['train_users']==tj['train_users'] and split['held_users']==tj['held_users'] and split['outer_excluded_users']==held
  a=arrays(p/'predictions.npz');assert a['clip_ids'].tolist()==split['held_ids'];tparts.append(a);binding.append(dict(component='THERMAL',job=tj['id'],done_sha256=sha(p/'DONE.json')))
 cropped=replace_video(copydata(data),concat(cparts,['clip_ids','targets','users','phase_spatial_modality_logits']))
 return data,cropped,concat(tparts,['clip_ids','targets','users','logits']) if len(tparts)==3 else None,binding
def outer(f):
 b=read(OLD/'PIPELINE_PROTOCOL.json')['bindings'][str(f)];assert sha(OLD/b['data']).upper()==b['data_sha256'];data=arrays(OLD/b['data']);z=arrays(R/f'crop_probe_output/F{f}_INTERVENTIONS.npz');a=dict(clip_ids=z['clip_ids'],targets=z['targets'],users=z['users'],phase_spatial_modality_logits=z['depth_box_phase']);cropped=replace_video(copydata(data),a)
 ref=arrays(OLD/f'analysis/F{f}_FIXED_R3_WITH_STOPGRAD.npz');assert np.array_equal(ref['clip_ids'],data['clip_ids'])
 a['phase_spatial_modality_logits']=z['baseline_phase'];original=replace_video(copydata(data),a);model=read(ROOT/b['models'])['convex_vsti'];assert np.array_equal(fm.predict(original,model),ref['STOPGRAD'])
 p=R/'thermal_adapter_output/local'/f'THERMAL_ADAPT_F{f}';thermal=None
 if (p/'DONE.json').exists():
  d=read(p/'DONE.json');assert d['protocol_sha256']==sha(R/'THERMAL_ADAPTER_PROTOCOL.json') and sha(p/'predictions.npz')==d['predictions_sha256'];thermal=arrays(p/'predictions.npz')
 return cropped,thermal,ref['STOPGRAD'],fm.predict(cropped,model)
def main():
 torch.set_num_threads(2);p=R/'CROP_THERMAL_CALIBRATION_PROTOCOL.json';s=read(p);assert s['analysis_code_sha256']==sha(__file__) and s['fusion_code_sha256']==sha(fm.__file__);out=R/'source_calibration';out.mkdir(exist_ok=True);guard=Guard(min(time.time()+600,s['deadline_unix']));completed={};packed={};perfold={};rows=[]
 for f in [0,3]:
  old,crop,t,binding=sources(f,s);fit_inputs={'CROP4':crop}
  if t is not None:fit_inputs.update(CROP5=add_t(copydata(crop),t),CROP_T_REPLACE4=add_t(copydata(crop),t,True))
  for key,d in fit_inputs.items():
   dest=out/f'F{f}_{key}';dest.mkdir(exist_ok=True);modelpath=dest/'calibrator.json';frozen=dest/'CALIBRATION_FROZEN.json';bh=hashlib.sha256(json.dumps(binding if key!='CROP4' else [x for x in binding if x['component']=='CROP'],sort_keys=True).encode()).hexdigest()
   if not frozen.exists():
    assert not modelpath.exists(),'INCOMPLETE_CALIBRATION_REQUIRES_REVIEW';m=fm.fit(d,'convex_vsti_private_t' if key=='CROP5' else 'convex_vsti',guard,500);write(modelpath,m);write(frozen,dict(protocol_sha256=sha(p),source_binding_sha256=bh,source_bindings=binding if key!='CROP4' else [x for x in binding if x['component']=='CROP'],calibrator_sha256=sha(modelpath),at_unix=time.time(),outer_labels_loaded_for_fit=False,calibration_users=m['calibration_users']))
   fr=read(frozen);assert fr['protocol_sha256']==sha(p) and fr['source_binding_sha256']==bh and fr['calibrator_sha256']==sha(modelpath);completed[f'F{f}_{key}']=dict(path=str(modelpath),sha256=sha(modelpath),temperature=read(modelpath)['temperature'])
  # Calibrators above are already frozen before loading any outer prediction.
  oc,ot,ref,zero=outer(f);preds={};data_inputs={'CROP4':oc}
  if ot is not None and t is not None:data_inputs.update(CROP5=add_t(copydata(oc),ot),CROP_T_REPLACE4=add_t(copydata(oc),ot,True))
  for key,d in data_inputs.items():
   dest=out/f'F{f}_{key}';m=read(dest/'calibrator.json');z,w=fm.predict(d,m,True);preds[key]=z;np.savez_compressed(dest/'predictions.npz',clip_ids=d['clip_ids'],targets=d['targets'],users=d['users'],fused=z,weights=w,R6=ref,zero_crop=zero);perfold[f'F{f}_{key}']=dict(vs_R6=paired(d['targets'],d['users'],ref,z),vs_zero_crop=paired(d['targets'],d['users'],zero,z))
  packed[f]=dict(targets=oc['targets'],users=oc['users'],R6=ref,zero_crop=zero,**preds)
  for i,k in enumerate(oc['clip_ids']):rows.append(dict(clip_id=str(k),fold=f,user=int(oc['users'][i]),target=int(oc['targets'][i]),R6=int(ref[i].argmax()),zero_crop=int(zero[i].argmax()),**{k:int(v[i].argmax()) for k,v in preds.items()}))
 common=set(packed[0])&set(packed[3]);z={k:np.concatenate([packed[f][k] for f in [0,3]]) for k in common};pooled={k:dict(vs_R6=paired(z['targets'],z['users'],z['R6'],z[k]),vs_zero_crop=paired(z['targets'],z['users'],z['zero_crop'],z[k])) for k in common if k not in ['targets','users','R6','zero_crop']};result=dict(status='COMPLETE' if len(completed)==6 else 'PARTIAL',at_unix=time.time(),protocol_sha256=sha(p),calibrations=completed,per_fold=perfold,pooled=pooled,no_Test=True,limits='Repeatedly exposed F0/F3 development; no unbiased0.75 Public guarantee. F2/F4 not reopened.')
 h=hashlib.sha256(json.dumps(completed,sort_keys=True).encode()).hexdigest();artifact=out/(h+'.json')
 if not artifact.exists():
  write(artifact,result)
  with(ROOT/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf-8') as fp:fp.write(json.dumps(dict(event='SOURCE_CALIBRATION_RESULT',artifact=str(artifact.relative_to(ROOT)),result=result))+'\n')
 write(out/'RESULT.json',result);fields=list(dict.fromkeys(k for row in rows for k in row))
 with(out/'ALL_PAIRED_CASES.csv').open('w',encoding='utf-8-sig',newline='') as fp:w=csv.DictWriter(fp,fieldnames=fields);w.writeheader();w.writerows(rows)
 print(json.dumps(dict(status=result['status'],completed=list(completed),pooled=pooled),ensure_ascii=False))
if __name__=='__main__':main()
