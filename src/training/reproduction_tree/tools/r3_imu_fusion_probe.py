"""One preregistered V/S/I subset of the existing inner-only convex calibrator."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
import torch

def main():
    a=argparse.ArgumentParser();a.add_argument('--root',required=True);a.add_argument('--release-root',required=True);args=a.parse_args();root=Path(args.root);out=Path(args.release_root);sys.path.insert(0,str(root/'bundle'))
    from dual_host_runtime.io import atomic,read
    from dual_host_runtime.health import Guard
    from research_runtime.common import sha
    import generalization_fusion_worker_align2 as f
    import generalization_fusion_math as math
    torch.set_num_threads(2);p=read(root/'fusion_protocol_align2.json');r=read(out/'protocol.json');assert sha(root/'fusion_protocol_align2.json')==r['parent_fusion_protocol_sha256']
    for name,expected in p['code_sha256'].items():assert sha(root/'bundle'/name)==expected
    for name,expected in r['source_sha256'].items():assert sha(out/name)==expected
    math.STRATEGIES['convex_vsi']=r['probe']['description'];raw,master,cohorts=f.load_cohorts(root,p);guard=Guard(out,p['budget']['deadline_unix'],root/'TELEMETRY.json');parts=[];inners=[];fold_results={}
    def assemble(fold,stage):
        batches=[];evidence=[]
        for group in p['jobs'][f'FUSION_F{fold}'][stage]:
            loaded={}
            for key in group:
                branch=p['sources'][key]['branch']
                if branch=='T':continue
                z,e=f.load_job(root,p,key,raw,master,cohorts);loaded[branch]=z;evidence.append(e)
            ids=loaded['V']['clip_ids'];logits=np.zeros((len(ids),4,40),np.float32);mask=np.zeros((len(ids),4),bool)
            for branch,j in [('V',0),('S',1),('I',3)]:
                z=loaded[branch];lookup={str(k):i for i,k in enumerate(z['clip_ids'])}
                for i,k in enumerate(ids):
                    if str(k) in lookup:logits[i,j]=z['logits'][lookup[str(k)]];mask[i,j]=True
            assert mask[:,:2].all();v=loaded['V'];batches.append(dict(clip_ids=ids,users=v['users'],targets=v['targets'],logits=logits,available=mask,depth=v['depth_logits'],ir=v['ir_logits']))
        data={key:np.concatenate([b[key] for b in batches]) for key in batches[0]};assert len(set(data['clip_ids']))==len(data['clip_ids'])
        return data,evidence
    for fold in [0,3]:
        inner,evidence=assemble(fold,'inner');assert not set(inner['users'])&set(p['folds'][str(fold)])
        model=math.fit(inner,'convex_vsi',guard);atomic(out/f'F{fold}_fit_trace.json',model.pop('trace'));atomic(out/f'F{fold}_model.json',model)
        atomic(out/f'F{fold}_FROZEN.json',dict(at_unix=time.time(),model_sha256=sha(out/f'F{fold}_model.json'),outer_loaded=False,source_evidence=evidence))
        outer,outer_evidence=assemble(fold,'outer');matched=math.predict(outer,model);baseline=.8*outer['logits'][:,0]+.2*outer['logits'][:,1]
        existing=dict(np.load(out/f'existing_F{fold}.npz',allow_pickle=False));lookup={str(k):i for i,k in enumerate(existing['clip_ids'])};ix=np.array([lookup[str(k)] for k in outer['clip_ids']]);assert np.array_equal(existing['targets'][ix],outer['targets']) and np.array_equal(existing['users'][ix],outer['users'])
        transfer={k:v.copy() for k,v in outer.items()};transfer['logits'][:,0]=existing['video'][ix];transfer['logits'][:,1]=existing['sensor'][ix]
        # The convex strategy has no quality slopes. Depth/IR features therefore cannot affect this transfer.
        candidate=math.predict(transfer,model);old=.8*existing['video'][ix]+.2*existing['sensor'][ix]
        result=dict(clip_ids=outer['clip_ids'],users=outer['users'],targets=outer['targets'],matched_baseline=baseline,matched_vsi=matched,existing_baseline=old,existing_vsi=candidate)
        np.savez_compressed(out/f'F{fold}_predictions.npz',**result);parts.append(result);inners.append(inner)
        fold_results[str(fold)]={name:f.scores(result[name],outer['targets'],outer['users']) for name in ['matched_baseline','matched_vsi','existing_baseline','existing_vsi']}
    allz={key:np.concatenate([v[key] for v in parts]) for key in parts[0]};np.savez_compressed(out/'predictions.npz',**allz)
    inner_all={key:np.concatenate([v[key] for v in inners]) for key in inners[0]};np.savez_compressed(out/'production_inner_oof.npz',**inner_all)
    pooled={name:f.scores(allz[name],allz['targets'],allz['users']) for name in ['matched_baseline','matched_vsi','existing_baseline','existing_vsi']};pairs={}
    for a,b in [('matched_vsi','matched_baseline'),('existing_vsi','existing_baseline')]:
        ca=allz[a].argmax(1)==allz['targets'];cb=allz[b].argmax(1)==allz['targets'];pairs[a]=dict(net=int(ca.sum()-cb.sum()),fixed=int((ca&~cb).sum()),harmed=int((cb&~ca).sum()),per_user_net={str(u):int(ca[allz['users']==u].sum()-cb[allz['users']==u].sum()) for u in np.unique(allz['users'])})
    report=dict(status='COMPLETE_FIXED_VSI_AND_EXISTING_RELEASE_TRANSFER_CHECK',at_unix=time.time(),folds=fold_results,pooled=pooled,comparisons=pairs,production_inner_records=len(inner_all['users']),production_inner_users=len(set(inner_all['users'])),Test_access=False,unbiased_unseen_claim=False)
    atomic(out/'RESULT.json',report);atomic(out/'DONE.json',dict(status='COMPLETE',result_sha256=sha(out/'RESULT.json'),predictions_sha256=sha(out/'predictions.npz'),inner_sha256=sha(out/'production_inner_oof.npz'),at_unix=time.time()));print(json.dumps(report))

if __name__=='__main__':main()
