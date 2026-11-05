"""Replay only the preregistered original fusion after automatic geometry."""
from pathlib import Path
import json,time,hashlib,csv
import numpy as np
import torch
from analyze_private_endpoints import read,sha,arrays,pair,R,ROOT,OLD,predict

def main():
 torch.set_num_threads(2)
 spec=read(R/'POSE_ADAPTER_PROTOCOL.json'); assert spec['id']=='R6_POSE_IR_PRIVATE_CE_ADAPTER_V1'
 out=R/'pose_private_analysis';out.mkdir(exist_ok=True);packed={};stats={};bindings={};rows=[]
 for job in spec['jobs']:
  f=job['fold'];p=R/'pose_adapter_output/cloud'/job['id']
  if not(p/'DONE.json').exists():continue
  d=read(p/'DONE.json');assert d['status']=='COMPLETE' and d['protocol_sha256']==sha(R/'POSE_ADAPTER_PROTOCOL.json') and d['source_sha256']==job['source_sha256'] and d['frozen_base_unchanged'] and d['epochs']==8
  assert sha(p/'predictions.npz')==d['predictions_sha256'] and sha(p/'adapter_fp16.pt')==d['checkpoint_sha256'];pose=arrays(p/'predictions.npz')
  b=read(OLD/'PIPELINE_PROTOCOL.json')['bindings'][str(f)];assert sha(OLD/b['data']).upper()==b['data_sha256'] and sha(ROOT/b['models']).upper()==b['models_sha256'];data=arrays(OLD/b['data']);model=read(ROOT/b['models'])['convex_vsti'];ref=arrays(OLD/f'analysis/F{f}_FIXED_R3_WITH_STOPGRAD.npz')['STOPGRAD']
  ids=data['clip_ids'];y=data['targets'];u=data['users'];crop=arrays(R/f'crop_probe_output/F{f}_INTERVENTIONS.npz');ci=[{k:i for i,k in enumerate(crop['clip_ids'])}[k] for k in ids];pi=[{k:i for i,k in enumerate(pose['clip_ids'])}[k] for k in ids]
  assert np.array_equal(pose['targets'][pi],y) and np.array_equal(pose['users'][pi],u)
  phases={'R6':crop['baseline_phase'][ci],'DEPTH_CROP':crop['depth_box_phase'][ci],'POSE_PRIVATE':pose['phase_spatial_modality_logits'][pi]};preds={}
  for k,ph in phases.items():
   assert np.array_equal(ph[:,:,0],phases['R6'][:,:,0]) and np.array_equal(ph[:,:,1,0],phases['R6'][:,:,1,0]),'UNCHANGED_OTHER_ROUTES'
   a={k:v.copy() for k,v in data.items()};a['logits'][:,0]=.5*(ph[:,:,0].mean((1,2))+ph[:,:,1].mean((1,2)));a['depth']=.5*(ph[:,:,0,0].mean(1)+ph[:,:,1,0].mean(1));a['ir']=.5*(ph[:,:,0,1].mean(1)+ph[:,:,1,1].mean(1));preds[k]=predict(a,model)
  assert np.array_equal(preds['R6'],ref),'R6_BITWISE_REPLAY'
  stats[str(f)]={k:pair(y,u,v,preds['POSE_PRIVATE']) for k,v in preds.items() if k!='POSE_PRIVATE'};bindings[str(f)]=dict(done_sha256=sha(p/'DONE.json'),predictions_sha256=d['predictions_sha256'],private_sha256=d['checkpoint_sha256'],model_sha256=job['source_sha256'])
  packed[f]=dict(targets=y,users=u,**preds);np.savez_compressed(out/f'F{f}_COMPARISONS.npz',clip_ids=ids,**packed[f])
  for i,k in enumerate(ids): rows.append(dict(clip_id=str(k),fold=f,user=int(u[i]),target=int(y[i]),**{k:int(v[i].argmax()) for k,v in preds.items()}))
 if not packed: print(json.dumps(dict(status='WAITING')));return
 z={k:np.concatenate([packed[f][k] for f in sorted(packed)]) for k in ['targets','users','R6','DEPTH_CROP','POSE_PRIVATE']};pooled={k:pair(z['targets'],z['users'],z[k],z['POSE_PRIVATE']) for k in ['R6','DEPTH_CROP']}
 result=dict(status='COMPLETE' if len(packed)==2 else 'PARTIAL',at_unix=time.time(),protocol_sha256=sha(R/'POSE_ADAPTER_PROTOCOL.json'),analysis_sha256=sha(__file__),bindings=bindings,per_fold=stats,pooled=pooled,no_fit=True,no_Test=True,no_Public_selection=True,interpretation='Repeated exposed development users. Geometry and development gain alone do not establish Public >0.75.')
 bh=hashlib.sha256(json.dumps(bindings,sort_keys=True).encode()).hexdigest();rec=out/(bh+'.json')
 if not rec.exists():
  rec.write_text(json.dumps(result,indent=2)+'\n')
  with(ROOT/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf-8') as fp: fp.write(json.dumps(dict(event='POSE_PRIVATE_FIXED_FUSION_AUDIT',at_unix=time.time(),artifact=str(rec.relative_to(ROOT)),status=result['status'],pooled=pooled))+'\n')
 (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
 with(out/'ALL_PAIRED_CASES.csv').open('w',encoding='utf-8-sig',newline='') as fp: w=csv.DictWriter(fp,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps(dict(status=result['status'],per_fold=stats,pooled=pooled)))

if __name__=='__main__':main()
