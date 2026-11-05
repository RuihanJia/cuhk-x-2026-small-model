"""Frozen factorial assessment of private IR and Thermal routes, with no fitting."""
from pathlib import Path
import json,time,hashlib,sys,csv
import numpy as np
import torch
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
def private(which,f):
 if which=='IR':p=R/'ir_adapter_output/cloud'/f'IR_PRIVATE_F{f}';protocol=R/'IR_ADAPTER_PROTOCOL.json'
 else:p=R/'thermal_adapter_output/local'/f'THERMAL_ADAPT_F{f}';protocol=R/'THERMAL_ADAPTER_PROTOCOL.json'
 if not(p/'DONE.json').exists():return None
 d=read(p/'DONE.json');s=read(protocol);j=next(j for j in s['jobs'] if j['fold']==f);assert d['status']=='COMPLETE' and d['epochs']==8 and d['protocol_sha256']==sha(protocol)
 assert d['source_sha256']==j['source_sha256'] and d['frozen_base_unchanged'];assert sha(p/'predictions.npz')==d['predictions_sha256'];assert sha(p/'adapter_fp16.pt')==d['checkpoint_sha256']
 cp=torch.load(p/'adapter_fp16.pt',map_location='cpu',weights_only=True);assert cp['train_users']==j['train_users'] and cp['source_sha256']==j['source_sha256'] and cp['protocol_sha256']==sha(protocol);assert all(v.dtype==torch.float16 and torch.isfinite(v).all() for v in cp['private_state'].values());assert(p/'adapter_fp16.pt').stat().st_size==d['checkpoint_bytes']<1500000
 split=read(p/'SPLIT.json');assert split['train_users']==j['train_users'] and split['held_users']==j['held_users'] and not set(split['train_users'])&set(split['held_users']);assert not set(split['train_ids'])&set(split['held_ids']);a=arrays(p/'predictions.npz');assert a['clip_ids'].tolist()==split['held_ids']
 return a,dict(path=str(p),DONE_sha256=sha(p/'DONE.json'),checkpoint_sha256=d['checkpoint_sha256'],source_sha256=d['source_sha256'],bytes=d['checkpoint_bytes'],seconds=d['seconds'],n=d['n'],standalone_correct=d['correct'])
def main():
 torch.set_num_threads(2);protocol=read(R/'PRIVATE_FUSION_PROTOCOL.json');assert protocol['analysis_code_sha256']==sha(__file__);out=R/'private_analysis';out.mkdir(exist_ok=True);bindings={};stats={};packed={};rows=[]
 bp=read(OLD/'PIPELINE_PROTOCOL.json')['bindings']
 for f in [0,3]:
  b=bp[str(f)];assert sha(OLD/b['data']).upper()==b['data_sha256'] and sha(ROOT/b['models']).upper()==b['models_sha256'];data=arrays(OLD/b['data']);model=read(ROOT/b['models'])['convex_vsti'];ref=arrays(OLD/f'analysis/F{f}_FIXED_R3_WITH_STOPGRAD.npz')['STOPGRAD'];y=data['targets'];u=data['users'];ids=data['clip_ids']
  crop=arrays(R/f'crop_probe_output/F{f}_INTERVENTIONS.npz');ii=[{k:i for i,k in enumerate(crop['clip_ids'])}[k] for k in ids];phases={'ORIGINAL':crop['baseline_phase'][ii],'CROP':crop['depth_box_phase'][ii]};t_data={'OLD':(data['logits'][:,2].copy(),data['available'][:,2].copy())}
  ir=private('IR',f);thermal=private('T',f)
  for which,value in [('IR',ir),('T',thermal)]:
   if value:bindings[f'{which}_F{f}']=value[1]
  if ir:
   a,_=ir;lookup={k:i for i,k in enumerate(a['clip_ids'])};ix=[lookup[k] for k in ids];assert np.array_equal(a['targets'][ix],y);ph=a['phase_spatial_modality_logits'][ix]
   assert np.array_equal(ph[:,:,:1],phases['CROP'][:,:,:1]) and np.array_equal(ph[:,:,1,0],phases['CROP'][:,:,1,0]),'UNCHANGED3ROUTES'
   phases['PRIVATE']=ph
  if thermal:
   a,e=thermal;lookup={k:i for i,k in enumerate(a['clip_ids'])};logits=np.zeros_like(data['logits'][:,2]);mask=np.zeros(len(y),bool)
   for i,k in enumerate(ids):
    if k in lookup:
     j=lookup[k];assert a['targets'][j]==y[i] and a['users'][j]==u[i];logits[i]=a['logits'][j];mask[i]=True
   assert np.array_equal(mask,data['available'][:,2]);t_data['PRIVATE']=(logits,mask)
   old_good=data['logits'][mask,2].argmax(1)==y[mask];new_good=logits[mask].argmax(1)==y[mask];stats[f'T_COMPONENT_F{f}']=dict(n=int(mask.sum()),old=int(old_good.sum()),new=int(new_good.sum()),fix=int((~old_good&new_good).sum()),harm=int((old_good&~new_good).sum()),net=int(new_good.sum()-old_good.sum()),R6_errors_rescued=int(((ref[mask].argmax(1)!=y[mask])&new_good).sum()))
  preds={}
  for vname,phase in phases.items():
   for tname,(tl,tm) in t_data.items():
    key=vname+'_T_'+tname;assert key in protocol['recipes'];a={k:v.copy() for k,v in data.items()};a['logits'][:,0]=.5*(phase[:,:,0].mean((1,2))+phase[:,:,1].mean((1,2)));a['depth']=.5*(phase[:,:,0,0].mean(1)+phase[:,:,1,0].mean(1));a['ir']=.5*(phase[:,:,0,1].mean(1)+phase[:,:,1,1].mean(1));a['logits'][:,2]=tl;a['available'][:,2]=tm;preds[key]=predict(a,model)
  assert np.array_equal(preds['ORIGINAL_T_OLD'],ref),'R6_BITWISE_REPLAY'
  for k,z in preds.items():stats[f'F{f}_{k}']=dict(vs_R6=pair(y,u,ref,z),vs_zero_crop=pair(y,u,preds['CROP_T_OLD'],z))
  if ir and thermal:assert ir[1]['source_sha256']==thermal[1]['source_sha256'],'SHARED_BASE_SOURCE'
  packed[f]=dict(targets=y,users=u,**preds);np.savez_compressed(out/f'F{f}_COMPARISONS.npz',clip_ids=ids,**packed[f])
  for i,k in enumerate(ids):rows.append(dict(clip_id=str(k),fold=f,user=int(u[i]),target=int(y[i]),**{key:int(z[i].argmax()) for key,z in preds.items()}))
 common=set(packed[0])&set(packed[3]);z={k:np.concatenate([packed[f][k] for f in [0,3]]) for k in common};pooled={k:dict(vs_R6=pair(z['targets'],z['users'],z['ORIGINAL_T_OLD'],z[k]),vs_zero_crop=pair(z['targets'],z['users'],z['CROP_T_OLD'],z[k])) for k in common if k not in ['targets','users']}
 result=dict(status='COMPLETE' if len(bindings)==4 else 'PARTIAL',at_unix=time.time(),protocol_sha256=sha(R/'PRIVATE_FUSION_PROTOCOL.json'),bindings=bindings,per_fold=stats,pooled=pooled,no_fit=True,no_Test=True,limits='Exposed developmentusers; frozenfactorial comparison, noPublic forecast. Strongcandidate needs source-safe calibration and finalserialized replay.')
 h=hashlib.sha256(json.dumps(bindings,sort_keys=True).encode()).hexdigest();p=out/(h+'.json')
 if not p.exists():
  p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
  with(ROOT/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf-8') as fp:fp.write(json.dumps(dict(event='PRIVATE_ADAPTER_ENDPOINT_AUDIT',artifact=str(p.relative_to(ROOT)),status=result['status'],completed=list(bindings),pooled=pooled,at_unix=time.time()))+'\n')
 (out/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');fields=list(dict.fromkeys(k for row in rows for k in row))
 with(out/'ALL_PAIRED_CASES.csv').open('w',encoding='utf-8-sig',newline='') as fp:w=csv.DictWriter(fp,fieldnames=fields);w.writeheader();w.writerows(rows)
 print(json.dumps(dict(status=result['status'],completed=list(bindings),per_fold={k:v for k,v in stats.items() if k.startswith('T_COMPONENT')},pooled=pooled),ensure_ascii=False))
if __name__=='__main__':main()
