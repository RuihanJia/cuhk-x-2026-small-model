"""Replay the already frozen R6 fusion, then report every paired intervention."""
from pathlib import Path
import json,sys,hashlib,time,csv
import numpy as np
R=Path(__file__).resolve().parent;ROOT=R.parents[1];OLD=ROOT/'runs/activity_token_stopgrad_20260913'
sys.path.insert(0,str(ROOT/'deliverables/activity_token_stopgrad_candidate_20260913'))
from fusion_math import predict
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
protocol=read(OLD/'PIPELINE_PROTOCOL.json');done=read(R/'crop_probe_output/DONE.json')
assert done['protocol_sha256']==sha(R/'CROP_PROBE_PROTOCOL.json')
cases=[];result={};bindings={};offset=0
for fold in [0,3]:
    b=protocol['bindings'][str(fold)]
    bp=OLD/b['data'];mp=ROOT/b['models'];assert sha(bp)==b['data_sha256'] and sha(mp)==b['models_sha256']
    base=dict(np.load(bp));model=read(mp)['convex_vsti'];assert not set(model['calibration_users'])&set(base['users'].tolist())
    assert model['description']['mode']=='convex'
    source=R/f'crop_probe_output/F{fold}_INTERVENTIONS.npz';a=dict(np.load(source))
    ids={str(x):i for i,x in enumerate(a['clip_ids'])};ix=[ids[str(x)] for x in base['clip_ids']]
    assert np.array_equal(a['targets'][ix],base['targets']) and np.array_equal(a['users'][ix],base['users'])
    outputs={};weights=None
    for key in ['baseline_phase','depth_box_phase','depth_box_gamma_phase']:
        z=a[key][ix];v=(z[:,:,0].mean((1,2))+z[:,:,1].mean((1,2)))*.5
        d=(z[:,:,0,0].mean(1)+z[:,:,1,0].mean(1))*.5;ir=(z[:,:,0,1].mean(1)+z[:,:,1,1].mean(1))*.5
        logits=base['logits'].copy();logits[:,0]=v
        outputs[key],weights=predict(dict(logits=logits,depth=d,ir=ir,available=base['available']),model,True)
    expected=dict(np.load(OLD/f'analysis/F{fold}_FIXED_R3_WITH_STOPGRAD.npz'))
    assert np.array_equal(expected['clip_ids'],base['clip_ids'])
    error=float(np.abs(outputs['baseline_phase']-expected['STOPGRAD']).max());assert error<2e-6 and np.array_equal(outputs['baseline_phase'].argmax(1),expected['STOPGRAD'].argmax(1)),error
    pp={k:v.argmax(1) for k,v in outputs.items()};y=base['targets'];ref=pp['baseline_phase']==y
    result[str(fold)]={k:dict(correct=int((p==y).sum()),fix=int(((p==y)&~ref).sum()),harm=int(((p!=y)&ref).sum()),prediction_changes=int((p!=pp['baseline_phase']).sum())) for k,p in pp.items()}
    bindings[str(fold)]=dict(fusion_model_sha256=sha(mp),source_npz_sha256=sha(source),baseline_max_abs_error=error,held_users=np.unique(base['users']).tolist())
    for i,clip in enumerate(base['clip_ids']):
        row=dict(clip_id=str(clip),fold=fold,user=int(base['users'][i]),target=int(y[i]))
        for k,p in pp.items():row[k]=int(p[i]);row[k+'_true_rank']=int((outputs[k][i]>outputs[k][i,y[i]]).sum()+1)
        row.update({f'weight_{name}':float(weights[i,j]) for j,name in enumerate(['V','S','T','I'])});cases.append(row)
    np.savez_compressed(R/f'crop_probe_output/F{fold}_FROZEN_FUSION.npz',clip_ids=base['clip_ids'],targets=y,users=base['users'],**outputs)
with (R/'crop_probe_output/ALL_PAIRED_CASES.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(cases[0]));w.writeheader();w.writerows(cases)
pooled={};rng=np.random.default_rng(20260913)
for key in ['depth_box_phase','depth_box_gamma_phase']:
    n=len(cases);correct=sum(r[key]==r['target'] for r in cases);ref=sum(r['baseline_phase']==r['target'] for r in cases)
    fix=sum(r[key]==r['target'] and r['baseline_phase']!=r['target'] for r in cases);harm=sum(r[key]!=r['target'] and r['baseline_phase']==r['target'] for r in cases)
    userstats={str(u):dict(n=sum(r['user']==u for r in cases),net=sum(int(r[key]==r['target'])-int(r['baseline_phase']==r['target']) for r in cases if r['user']==u)) for u in sorted({r['user'] for r in cases})}
    draws=np.zeros(20000);totals=np.zeros(20000)
    for fold in [0,3]:
        users=sorted({r['user'] for r in cases if r['fold']==fold});idx=rng.integers(len(users),size=(20000,len(users)))
        net=np.array([userstats[str(u)]['net'] for u in users]);ns=np.array([userstats[str(u)]['n'] for u in users]);draws+=net[idx].sum(1);totals+=ns[idx].sum(1)
    pooled[key]=dict(n=n,baseline_correct=ref,correct=correct,accuracy=correct/n,fix=fix,harm=harm,net=correct-ref,per_user=userstats,paired_user_bootstrap_delta_ci95=np.quantile(draws/totals,[.025,.975]).tolist(),bootstrap_caveat='Exploratory 7 repeatedly exposed users; small-cluster CI may under-cover; no unbiased Public forecast.')
final=dict(status='FIXED_R6_FUSION_REPLAY_PASS',at_unix=time.time(),protocol_sha256=sha(R/'CROP_PROBE_PROTOCOL.json'),code_sha256=sha(Path(__file__)),no_fit=True,no_Test=True,bindings=bindings,per_fold=result,pooled=pooled)
(R/'crop_probe_output/FROZEN_FUSION_RESULT.json').write_text(json.dumps(final,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(final,ensure_ascii=False))
