from pathlib import Path
import json,time,hashlib,csv,sys
import numpy as np
import torch
from detail_model import DetailHead
R=Path(__file__).resolve().parent;ROOT=R.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def softmax(z):
 a=z.astype(np.float64);a-=a.max(1,keepdims=True);a=np.exp(a);return a/a.sum(1,keepdims=True)
spec=json.loads((R/'DETAIL_PROTOCOL.json').read_text(encoding='utf-8'));proto=sha(R/'DETAIL_PROTOCOL.json');inputs=json.loads((R/'DETAIL_INPUTS.json').read_text(encoding='utf-8'));positions={r['clip_id']:i for i,r in enumerate(inputs)}
features=[np.load(R/'detail_output'/host/'features.npy',mmap_mode='r') for host in ['cloud','local']];available=np.concatenate([np.load(R/'detail_output'/host/'available.npy') for host in ['cloud','local']],1)
torch.set_num_threads(2);allrows=[];replay=[];results={}
for fold,host in [(0,'cloud'),(3,'local')]:
 base=dict(np.load(ROOT/f'runs/activity_token_stopgrad_20260913/analysis/F{fold}_FIXED_R3_WITH_STOPGRAD.npz',allow_pickle=False));y=base['targets'];bp=softmax(base['STOPGRAD']);old=bp.argmax(1);ids=[str(v) for v in base['clip_ids']]
 for arm in spec['head_arms']:
  p=R/'detail_output'/host/f'DETAIL_F{fold}_{arm}';done=json.loads((p/'DONE.json').read_text());assert done['protocol_sha256']==proto and done['head_sha256']==sha(p/'head_fp16.pt') and done['predictions_sha256']==sha(p/'predictions.npz')
  z=dict(np.load(p/'predictions.npz',allow_pickle=False));mapping={str(v):i for i,v in enumerate(z['clip_ids'])};ix=[mapping[v] for v in ids];assert np.array_equal(z['targets'][ix],y) and np.array_equal(z['users'][ix],base['users'])
  cp=torch.load(p/'head_fp16.pt',map_location='cpu',weights_only=True);assert cp['train_users']==spec['folds'][str(fold)]['train_users'];assert not set(cp['train_users'])&set(base['users'])
  model=DetailHead(cp['ordered']).eval();model.load_state_dict(cp['state_dict']);sample=ids[:4];ii=[positions[v] for v in sample];xx=torch.from_numpy(np.concatenate([np.asarray(x[ii]) for x in features],1)).float()
  with torch.no_grad():actual=model(xx,torch.from_numpy(available[ii]))[0].numpy()
  err=float(np.abs(actual-z['logits'][[mapping[v] for v in sample]]).max());assert err<1e-4,err;replay.append(dict(fold=fold,arm=arm,max_error=err,source_ids=sample))
  hp=softmax(z['logits'][ix]);fused=.8*bp+.2*hp;new=fused.argmax(1);own=hp.argmax(1)
  key=f'F{fold}_{arm}';results[key]=dict(n=len(y),head_correct=int((own==y).sum()),r6_correct=int((old==y).sum()),fusion_correct=int((new==y).sum()),fix=int(((new==y)&(old!=y)).sum()),harm=int(((new!=y)&(old==y)).sum()),independent_rescue=int(((own==y)&(old!=y)).sum()),head_source=done)
  for k,cid in enumerate(ids):allrows.append(dict(clip_id=cid,fold=fold,arm=arm,user=int(base['users'][k]),target=int(y[k]),r6_prediction=int(old[k]),head_prediction=int(own[k]),fusion_prediction=int(new[k]),delta=int(new[k]==y[k])-int(old[k]==y[k])))
pooled={}
for arm in spec['head_arms']:
 rows=[r for r in allrows if r['arm']==arm];pooled[arm]=dict(n=len(rows),r6_correct=sum(r['r6_prediction']==r['target'] for r in rows),head_correct=sum(r['head_prediction']==r['target'] for r in rows),fusion_correct=sum(r['fusion_prediction']==r['target'] for r in rows),fix=sum(r['delta']==1 for r in rows),harm=sum(r['delta']==-1 for r in rows),head_rescue=sum(r['r6_prediction']!=r['target'] and r['head_prediction']==r['target'] for r in rows),per_user={str(u):sum(r['delta'] for r in rows if r['user']==u) for u in sorted({r['user'] for r in rows})})
out=dict(status='COMPLETE_REPLAY_AND_PAIRED_AUDIT',at_unix=time.time(),protocol_sha256=proto,script_sha256=sha(Path(__file__)),per_fold=results,pooled=pooled,replay=replay,scope='F0/F3 exposed development, no Test or Public tuning; single-seed head screen, frozen backbone. Cannot establish unbiased generalization or a Public threshold.')
(R/'detail_output/RESULT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
with (R/'detail_output/ALL_PAIRED_CASES.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
with (ROOT/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(event='COMPLETE',experiment=spec['id'],at_unix=time.time(),protocol_sha256=proto,result='runs/error_resolution_20260913/detail_output/RESULT.json',result_sha256=sha(R/'detail_output/RESULT.json'),fits=6,pooled=pooled,no_Test_access=True))+'\n')
print(json.dumps(pooled))
