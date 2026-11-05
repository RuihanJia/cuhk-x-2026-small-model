"""Recompute paired development metrics from only completed serialized endpoints."""
from pathlib import Path
import sys,json,time,hashlib,csv
import numpy as np
R=Path(__file__).resolve().parent;ROOT=R.parents[1];OLD=ROOT/'runs/activity_token_stopgrad_20260913'
sys.path.insert(0,str(ROOT/'deliverables/activity_token_stopgrad_candidate_20260913'))
from fusion_math import predict
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def arrays(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def pair(y,u,a,b):
 a=a.argmax(1)==y;b=b.argmax(1)==y
 return dict(n=len(y),before=int(a.sum()),after=int(b.sum()),fix=int((~a&b).sum()),harm=int((a&~b).sum()),net=int(b.sum()-a.sum()),per_user={str(k):dict(n=int((u==k).sum()),net=int(b[u==k].sum()-a[u==k].sum())) for k in np.unique(u)})
def prob(x):
 q=np.exp(x.astype(np.float64)-x.max(1,keepdims=True));return q/q.sum(1,keepdims=True)
def main():
 p=read(OLD/'PIPELINE_PROTOCOL.json');out=R/'training_analysis';out.mkdir(exist_ok=True);results={};packed={};cases=[];bindings={}
 for f in [0,3]:
  b=p['bindings'][str(f)];assert sha(OLD/b['data']).upper()==b['data_sha256'];assert sha(ROOT/b['models']).upper()==b['models_sha256']
  data=arrays(OLD/b['data']);model=read(ROOT/b['models'])['convex_vsti'];old=arrays(OLD/f'analysis/F{f}_FIXED_R3_WITH_STOPGRAD.npz');crop=arrays(R/f'crop_probe_output/F{f}_FROZEN_FUSION.npz')
  assert np.array_equal(data['clip_ids'],old['clip_ids']) and np.array_equal(crop['clip_ids'],data['clip_ids']);ref=old['STOPGRAD'];y=data['targets'];u=data['users'];preds={'R6':ref,'CROP_ZERO':crop['depth_box_phase']}
  for arm in ['CONTROL','CONFLICT']:
   job=f'VIDEO_F{f}_{arm}';src=R/'video_conflict_output'/job
   if not(src/'DONE.json').exists():continue
   d=read(src/'DONE.json');assert d['status']=='COMPLETE' and d['protocol_sha256']==sha(R/'VIDEO_CONFLICT_PROTOCOL.json');assert sha(src/'predictions.npz')==d['predictions_sha256']
   split=read(src/'SPLIT.json');assert not set(split['train_users'])&set(split['held_users']);a=arrays(src/'predictions.npz');ix=[{k:i for i,k in enumerate(a['clip_ids'])}[k] for k in data['clip_ids']];assert np.array_equal(a['targets'][ix],y)
   phase=a['phase_spatial_modality_logits'][ix];candidate={k:v.copy() for k,v in data.items()};candidate['logits'][:,0]=a['logits'][ix];candidate['depth']=.5*(phase[:,:,0,0].mean(1)+phase[:,:,1,0].mean(1));candidate['ir']=.5*(phase[:,:,0,1].mean(1)+phase[:,:,1,1].mean(1))
   z=predict(candidate,model);key='VIDEO_'+arm;preds[key]=z;bindings[job]=dict(DONE_sha256=sha(src/'DONE.json'),predictions_sha256=sha(src/'predictions.npz'));results[job]=dict(vs_R6=pair(y,u,ref,z),vs_crop_zero=pair(y,u,crop['depth_box_phase'],z),standalone_video_correct=int((a['logits'][ix].argmax(1)==y).sum()),seconds=d['seconds'])
  if 'VIDEO_CONTROL' in preds and 'VIDEO_CONFLICT' in preds:results[f'VIDEO_F{f}_WEIGHT_EFFECT']=pair(y,u,preds['VIDEO_CONTROL'],preds['VIDEO_CONFLICT'])
  job=f'DINO_ADAPT_F{f}';src=R/'detail_finetune_output'/job
  if(src/'DONE.json').exists():
   d=read(src/'DONE.json');assert d['status']=='COMPLETE' and d['protocol_sha256']==sha(R/'DETAIL_FINETUNE_PROTOCOL.json');assert sha(src/'predictions.npz')==d['predictions_sha256']
   assert sha(src/'dino_int8.pt')==d['backbone_sha256'] and sha(src/'head_fp16.pt')==d['head_sha256'];split=read(src/'SPLIT.json');assert not set(split['train_users'])&set(split['held_users'])
   a=arrays(src/'predictions.npz');lookup={k:i for i,k in enumerate(a['clip_ids'])};ix=[lookup[k] for k in data['clip_ids']];assert np.array_equal(a['targets'][ix],y);z=a['logits'][ix];fusion=.8*prob(ref)+.2*prob(z);preds['DINO_FIXED_FUSION']=fusion
   results[job]=dict(vs_R6=pair(y,u,ref,fusion),independent_correct=int((z.argmax(1)==y).sum()),independent_R6_rescue=int(((ref.argmax(1)!=y)&(z.argmax(1)==y)).sum()),serialized_added_bytes=d['serialized_added_bytes'],seconds=d['seconds']);bindings[job]=dict(DONE_sha256=sha(src/'DONE.json'),predictions_sha256=sha(src/'predictions.npz'))
  packed[f]=dict(targets=y,users=u,**preds);np.savez_compressed(out/f'F{f}_COMPARISONS.npz',clip_ids=data['clip_ids'],**packed[f])
  for i,k in enumerate(data['clip_ids']):
   row=dict(clip_id=str(k),fold=f,user=int(u[i]),target=int(y[i]));row.update({key:int(z[i].argmax()) for key,z in preds.items()});cases.append(row)
 common=set(packed[0])&set(packed[3]);z={k:np.concatenate([packed[f][k] for f in [0,3]]) for k in common};pooled={k:pair(z['targets'],z['users'],z['R6'],z[k]) for k in common if k not in ['targets','users','R6']}
 if 'VIDEO_CONTROL' in common and 'VIDEO_CONFLICT' in common:pooled['WEIGHT_EFFECT']=pair(z['targets'],z['users'],z['VIDEO_CONTROL'],z['VIDEO_CONFLICT'])
 if all(f'DINO_ADAPT_F{f}' in results for f in [0,3]):pooled['DINO_INDEPENDENT']=dict(n=1094,correct=sum(results[f'DINO_ADAPT_F{f}']['independent_correct'] for f in [0,3]),R6_rescues=sum(results[f'DINO_ADAPT_F{f}']['independent_R6_rescue'] for f in [0,3]))
 result=dict(at_unix=time.time(),status='PARTIAL_COMPLETED_ENDPOINTS' if len(bindings)<6 else 'ALL_SIX_ENDPOINTS_COMPLETE',completed=list(bindings),bindings=bindings,per_job=results,pooled=pooled,no_Test=True,limits='F0/F3 repeatedly exposed development. Frozen calibrator transfer for video; fixed .2 DINO fusion; no posthoc weight tuning. No Public forecast.')
 fingerprint=hashlib.sha256(json.dumps(bindings,sort_keys=True).encode()).hexdigest();obs=out/(fingerprint+'.json')
 if not obs.exists():
  obs.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
  with(ROOT/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf-8') as fp:fp.write(json.dumps(dict(event='ERROR_RESOLUTION_ENDPOINT_AUDIT',artifact=str(obs.relative_to(ROOT)),completed=result['completed'],pooled=pooled,at_unix=time.time()))+'\n')
 (out/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 fields=list(dict.fromkeys(k for row in cases for k in row))
 with(out/'ALL_PAIRED_CASES.csv').open('w',encoding='utf-8-sig',newline='') as fp:w=csv.DictWriter(fp,fieldnames=fields);w.writeheader();w.writerows(cases)
 print(json.dumps({k:v for k,v in result.items() if k!='bindings'},ensure_ascii=False))
if __name__=='__main__':main()
