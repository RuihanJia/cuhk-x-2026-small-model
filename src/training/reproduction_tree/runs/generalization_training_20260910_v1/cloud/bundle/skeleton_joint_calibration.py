"""Frozen, user-excluded calibration and second-seed assessment for skeletal encoding."""
from pathlib import Path
import argparse,copy,json,sys,time,traceback
import numpy as np


def main():
    a=argparse.ArgumentParser()
    for name in ['root','job','protocol']:a.add_argument('--'+name,required=True)
    a.add_argument('--deadline',type=float,required=True);args=a.parse_args()
    root=Path(args.root);sys.path.insert(0,str(root/'bundle'))
    import torch
    from dual_host_runtime.io import atomic,read
    from dual_host_runtime.health import Guard
    from research_runtime.common import sha
    from generalization_fusion_math import fit,predict
    import generalization_fusion_worker_align2 as f
    torch.set_num_threads(2);q=read(args.protocol);ph=sha(args.protocol);spec=q['jobs'][args.job]
    out=root/'output'/args.job;out.mkdir(parents=True,exist_ok=True);guard=Guard(out,min(args.deadline,q['budget']['stage_deadline_unix']),root/'TELEMETRY.json')
    try:
        for name,expected in q['code_sha256'].items():assert sha(root/'bundle'/name)==expected,name
        if spec['kind']=='summary':
            parts=[]
            for fold in [0,3]:
                path=root/'output'/f'SJC_CAL_F{fold}';done=read(path/'DONE.json')
                assert done['protocol_sha256']==ph and sha(path/'predictions.npz')==done['predictions_sha256']
                parts.append(dict(np.load(path/'predictions.npz',allow_pickle=False)))
            z={k:np.concatenate([v[k] for v in parts]) for k in parts[0]};assert len(set(z['clip_ids']))==1094
            for arm in ['flat','joint_identity']:
                zs=[]
                for fold in [0,3]:
                    path=root/'output'/f'SJC_seed2_{arm}_F{fold}';done=read(path/'DONE.json');assert done['protocol_sha256']==ph and sha(path/'predictions.npz')==done['predictions_sha256']
                    saved=dict(np.load(path/'predictions.npz',allow_pickle=False));lookup={str(k):i for i,k in enumerate(saved['clip_ids'])};ids=parts[0 if fold==0 else 1]['clip_ids'];ix=[lookup[str(k)] for k in ids]
                    np.testing.assert_array_equal(saved['targets'][ix],parts[0 if fold==0 else 1]['targets']);zs.append(saved['logits'][ix])
                z['seed2_'+arm]=np.concatenate(zs);z['seed2_'+arm+'_fixed_vs']=.8*z['video']+.2*z['seed2_'+arm]
            report={k:f.scores(v,z['targets'],z['users']) for k,v in z.items() if v.shape==(1094,40)}
            comparisons={}
            for a,b in [('joint_identity_calibrated','flat_calibrated'),('joint_identity_fixed_vs','flat_fixed_vs'),('seed2_joint_identity','seed2_flat'),('seed2_joint_identity_fixed_vs','seed2_flat_fixed_vs')]:
                aa=z[a].argmax(1)==z['targets'];bb=z[b].argmax(1)==z['targets'];u=z['users']
                comparisons[a+'_vs_'+b]=dict(net=int(aa.sum()-bb.sum()),fixed=int((aa&~bb).sum()),harmed=int((bb&~aa).sum()),per_user_net={str(k):int(aa[u==k].sum()-bb[u==k].sum()) for k in np.unique(u)})
            result=dict(status='COMPLETE_FIXED_CALIBRATION_AND_SECOND_SEED_DEVELOPMENT',protocol_sha256=ph,at_unix=time.time(),pooled=report,comparisons=comparisons,Test_access=False,unbiased_unseen_claim=False)
        else:
            parent=read(root/'fusion_protocol_align2.json');assert sha(root/'fusion_protocol_align2.json')==q['parent_fusion_sha256']
            raw,master,cohorts=f.load_cohorts(root,parent);models={};sources={};modified={};fold=spec['fold']
            for arm in ['flat','joint_identity']:
                p=copy.deepcopy(parent)
                for key,source in p['sources'].items():
                    if source['branch']!='S':continue
                    replacement=q['sensor_source_map'][arm][key]
                    source.update(path='output/'+replacement['job'],protocol_sha256=ph if replacement['protocol_sha256']=='SELF' else replacement['protocol_sha256'])
                    for name in ['checkpoint_sha256','predictions_sha256']:source.pop(name,None)
                inner,evidence=f.assemble(root,p,fold,'inner',raw,master,cohorts,guard)
                models[arm]=fit(inner,'convex_vsti',guard);atomic(out/(arm+'_fit_trace.json'),models[arm].pop('trace'));sources[arm]=evidence;modified[arm]=p
            atomic(out/'models.json',models);atomic(out/'CALIBRATION_FROZEN.json',dict(protocol_sha256=ph,models_sha256=sha(out/'models.json'),at_unix=time.time(),outer_loaded=False,fit_sources=sources))
            z={};outer_sources={}
            for arm in ['flat','joint_identity']:
                data,evidence=f.assemble(root,modified[arm],fold,'outer',raw,master,cohorts,guard);outer_sources[arm]=evidence
                if not z:z={key:data[key] for key in ['clip_ids','targets','users']};z['video']=data['logits'][:,0]
                for key in ['clip_ids','targets','users']:np.testing.assert_array_equal(z[key],data[key])
                z[arm+'_calibrated']=predict(data,models[arm]);z[arm+'_fixed_vs']=.8*data['logits'][:,0]+.2*data['logits'][:,1]
            result=dict(status='COMPLETE_TWO_FROZEN_INNER_ONLY_CALIBRATORS',protocol_sha256=ph,at_unix=time.time(),fold=fold,
                        metrics={k:f.scores(v,z['targets'],z['users']) for k,v in z.items() if v.shape==(len(z['users']),40)},sources=outer_sources,Test_access=False)
        np.savez_compressed(out/'predictions.npz',**z);atomic(out/'RESULT.json',result)
        atomic(out/'DONE.json',dict(status='COMPLETE',protocol_sha256=ph,result_sha256=sha(out/'RESULT.json'),predictions_sha256=sha(out/'predictions.npz'),at_unix=time.time()))
        print(json.dumps(result),flush=True)
    except BaseException as e:atomic(out/'ERROR.json',dict(error=repr(e),traceback=traceback.format_exc(),at_unix=time.time()));raise


if __name__=='__main__':main()
