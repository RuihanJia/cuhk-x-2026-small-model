"""Verify user-crossfit sources, freeze inner-only calibrators, then assess fixed outer outputs."""
from pathlib import Path
import argparse,json,math,sys,time,traceback
import numpy as np
import torch

def load_cohorts(root,p):
    from research_runtime.common import sha,read
    for name,expected in p['data_sha256'].items():assert sha(root/'bundle/data'/name)==expected
    raw=dict(np.load(root/'bundle/data/measured_sensors.npz',allow_pickle=False));video=read(root/'bundle/data/frame_index.json');thermal=read(root/'bundle/data/thermal_index.json')
    master={str(k):(int(y),int(u)) for k,y,u in zip(raw['clip_ids'],raw['targets'],raw['users'])};assert len(master)==len(raw['clip_ids'])
    video_ids={r['clip_id'] for r in video};sensor_ids=set(master);thermal_ids={r['clip_id'] for r in thermal}
    assert len(video_ids)==len(video) and len(thermal_ids)==len(thermal) and video_ids==sensor_ids
    for r in video:assert master[r['clip_id']]==(r['label'],r['user'])
    # Thermal has105 valid clips absent from the video/sensor cohort; verify its complete own cohort,
    # then join predictions onto the video cohort. Missing Thermal in145 video clips remains masked.
    for r in thermal:
        value=(r['label'],r['user'])
        if r['clip_id'] in master:assert master[r['clip_id']]==value
        else:master[r['clip_id']]=value
    return raw,master,dict(V=video_ids,S=sensor_ids,T=thermal_ids,I=set(raw['clip_ids'][raw['imu_valid']].tolist()))

def load_job(root,p,key,raw,master,cohorts):
    from research_runtime.common import read,sha
    src=p['sources'][key];path=root/src['path'];done=read(path/'DONE.json');prov=read(path/'PROVENANCE.json');split=read(path/'SPLIT.json')
    assert done['status']=='COMPLETE' and done['protocol_sha256']==src['protocol_sha256'] and prov['protocol_sha256']==src['protocol_sha256']
    assert set(src['train_users'])==set(prov['train_users'])==set(split['train_users']) and set(src['predict_users'])==set(split['predict_users'])
    assert not set(src['train_users'])&set(src['predict_users']);assert not set(src['train_users'])&set(p['folds'][str(src['fold'])])
    assert not prov.get('target_pretrained_users',[])
    cohort=cohorts[src['branch']];train_ids={k for k in cohort if master[k][1] in src['train_users']};predict_ids={k for k in cohort if master[k][1] in src['predict_users']}
    assert set(split['train_ids'])==train_ids and len(split['train_ids'])==len(train_ids)
    if 'predict_ids' in split:assert set(split['predict_ids'])==predict_ids and len(split['predict_ids'])==len(predict_ids)
    assert done['epochs']==src['epochs'] and done['updates']==src['epochs']*math.ceil(len(train_ids)/src['batch'])
    for file,field in [('model_fp16.pt','checkpoint_sha256'),('predictions.npz','predictions_sha256')]:
        assert sha(path/file)==done[field],(key,file)
        if field in src:assert done[field]==src[field]
    assert (path/'model_fp16.pt').stat().st_size==done['checkpoint_bytes']
    cp=torch.load(path/'model_fp16.pt',map_location='cpu',weights_only=True);assert cp['protocol_sha256']==src['protocol_sha256'] and set(cp['train_users'])==set(src['train_users'])
    if src['branch']=='T':
        norm=read(path/'NORMALIZER.json');assert set(norm['train_users'])==set(src['train_users']) and set(norm['train_ids'])==train_ids
        assert cp['mean']==norm['mean'] and cp['std']==norm['std']
        assert done['execution_protocol_sha256']==src['execution_protocol_sha256']
    if src['branch']=='S':
        fit=np.isin(raw['users'],src['train_users'])&raw['imu_valid'];expected=raw['imu'][fit]
        assert set(split['normalizer_fit_ids'])==set(raw['clip_ids'][fit]);assert set(prov['normalizer_users'])==set(src['train_users'])
        np.testing.assert_allclose(cp['mean'].numpy(),expected.mean((0,2),keepdims=True),rtol=1e-6,atol=1e-7)
        np.testing.assert_allclose(cp['std'].numpy(),expected.std((0,2),keepdims=True).clip(1e-4),rtol=1e-6,atol=1e-7)
    if src['branch']=='I':
        admission=read(root/p['imu_admission']['path']);assert sha(root/p['imu_admission']['path'])==p['imu_admission']['sha256']
        accepted=admission['jobs'][key];assert accepted['checkpoint_sha256']==done['checkpoint_sha256'] and accepted['predictions_sha256']==done['predictions_sha256'] and accepted['top1_mismatches']==0
        assert set(prov['normalizer_users'])==set(src['train_users']) and done['parameters']==180264
    del cp
    z=dict(np.load(path/'predictions.npz',allow_pickle=False));assert len(z['clip_ids'])==len(predict_ids) and set(z['clip_ids'])==predict_ids
    assert z['logits'].shape==(len(predict_ids),40) and z['logits'].dtype==np.float32 and np.isfinite(z['logits']).all()
    for k,y,u in zip(z['clip_ids'],z['targets'],z['users']):assert master[str(k)]==(int(y),int(u))
    if src['branch']=='V':
        v=z['view_modality_logits'];assert v.shape==(len(predict_ids),5,2,40) and v.dtype==np.float32 and np.isfinite(v).all()
        assert np.array_equal(z['logits'],v.mean((1,2))) and np.array_equal(z['depth_logits'],v[:,:,0].mean(1)) and np.array_equal(z['ir_logits'],v[:,:,1].mean(1))
    return z,dict(job=key,branch=src['branch'],train_users=src['train_users'],predict_users=src['predict_users'],checkpoint_sha256=done['checkpoint_sha256'],checkpoint_bytes=done['checkpoint_bytes'],predictions_sha256=done['predictions_sha256'],protocol_sha256=done['protocol_sha256'])

def assemble(root,p,fold,stage,raw,master,cohorts,guard):
    sources=p['jobs'][f'FUSION_F{fold}'][stage];parts=[];evidence=[]
    for group in sources:
        loaded=[]
        for key in group:
            guard.check(phase='VERIFY_'+stage.upper()+'_SOURCES',source=key)
            z,e=load_job(root,p,key,raw,master,cohorts);loaded.append(z);evidence.append(e)
        assert [p['sources'][k]['branch'] for k in group]==['V','S','T','I']
        ids=loaded[0]['clip_ids'];logits=np.zeros((len(ids),4,40),np.float32);mask=np.zeros((len(ids),4),bool)
        for j,z in enumerate(loaded):
            lookup={str(k):i for i,k in enumerate(z['clip_ids'])}
            for i,k in enumerate(ids):
                if str(k) in lookup:logits[i,j]=z['logits'][lookup[str(k)]];mask[i,j]=True
        assert mask[:,:2].all()
        parts.append(dict(clip_ids=ids,users=loaded[0]['users'],targets=loaded[0]['targets'],logits=logits,available=mask,depth=loaded[0]['depth_logits'],ir=loaded[0]['ir_logits']))
    combined={key:np.concatenate([v[key] for v in parts]) for key in parts[0]};assert len(set(combined['clip_ids']))==len(combined['clip_ids'])
    allowed=set(p['training_users'])-set(p['folds'][str(fold)]) if stage=='inner' else set(p['folds'][str(fold)])
    assert set(combined['users'])==allowed and set(combined['clip_ids'])=={k for k in cohorts['V'] if master[k][1] in allowed}
    order=np.argsort(combined['clip_ids']);combined={k:v[order] for k,v in combined.items()}
    return combined,evidence

def scores(z,y,users):
    predicted=z.argmax(1);correct=predicted==y;cm=np.zeros((40,40),np.int64);np.add.at(cm,(y,predicted),1);den=cm.sum(0)+cm.sum(1)
    return dict(correct=int(correct.sum()),n=len(y),accuracy=float(correct.mean()),macro_f1=float(np.divide(2*cm.diagonal(),den,out=np.zeros(40),where=den>0).mean()),per_user={str(u):dict(correct=int(correct[users==u].sum()),n=int((users==u).sum())) for u in np.unique(users)})

def summarize(root,p,ph,out,guard):
    from research_runtime.common import read,sha
    from dual_host_runtime.io import atomic
    values=[];bundles={}
    for fold in [0,3]:
        path=root/'output'/f'FUSION_F{fold}';done=read(path/'DONE.json');assert done['protocol_sha256']==ph and sha(path/'predictions.npz')==done['predictions_sha256']
        values.append(dict(np.load(path/'predictions.npz',allow_pickle=False)));bundles[str(fold)]=done['prototype_bundle_bytes']
    z={key:np.concatenate([v[key] for v in values]) for key in values[0]};assert len(set(z['clip_ids']))==len(z['clip_ids'])
    report={name:scores(z[name],z['targets'],z['users']) for name in p['strategies']};comparisons={};users=z['users'];y=z['targets'];order=np.unique(users);rng=np.random.default_rng(2026091101)
    pairs=[(name,'matched_fixed_vs') for name in p['strategies'] if name!='matched_fixed_vs']+[(a,b) for a,b in [('convex_vst','convex_vs'),('convex_vsti','convex_vst'),('gate_vsti','gate_vst'),('gate_vst','gate_vst_no_availability'),('gate_vsti','gate_vsti_no_availability')]]
    for newer,older in pairs:
        a=z[newer].argmax(1)==y;b=z[older].argmax(1)==y;gains=np.array([int(a[users==u].sum())-int(b[users==u].sum()) for u in order]);counts=np.array([(users==u).sum() for u in order]);ix=rng.integers(0,len(order),(10000,len(order)));boot=gains[ix].sum(1)/counts[ix].sum(1)
        comparisons[newer+'_vs_'+older]=dict(net_correct=int(a.sum()-b.sum()),fixed=int((a&~b).sum()),harmed=int((b&~a).sum()),accuracy_delta_pp=100*float((a.astype(float)-b).mean()),per_user_net=dict(zip(map(str,order),gains.tolist())),user_cluster_bootstrap95_delta_pp=(100*np.quantile(boot,[.025,.975])).tolist())
    result=dict(status='COMPLETE_FIXED_INNER_CALIBRATION_AND_OUTER_ASSESSMENT',protocol_sha256=ph,at_unix=time.time(),strategies=report,comparisons=comparisons,prototype_bundle_bytes=bundles,independent_unseen_user_claim=False,selection_during_calibrator_fit=False,limitations=p['limitations'])
    atomic(out/'RESULT.json',result);np.savez_compressed(out/'predictions.npz',**z)
    lines=['# 按用户交叉拟合的融合比较','', '所有融合参数和特征标准化只由各外层训练用户的内层 OOF 预测拟合；外层预测在参数冻结后读取并评分。','', '| 策略 | 正确数 | 准确率 | Macro F1 |','|---|---:|---:|---:|']
    for name,r in report.items():lines.append(f"| {name} | {r['correct']}/{r['n']} | {100*r['accuracy']:.3f}% | {r['macro_f1']:.4f} |")
    lines+=['','三分支 V/S/T 为原始登记比较；四分支额外加入已复核的 IMU 设备丢弃分支。no_availability 对照去掉可学习的可用性特征，同时保留缺失分支退出及权重重新归一化。','', 'F0/F3 是重复使用的开发用户。本次消除了融合拟合中的外层用户标签交叉使用，但不能据此声称整个研究选择过程无偏，也不能代表公榜或新的未见用户收益。未读取 Test、未进行 F2/F4 外层评估、未训练全量发布权重、未上传 Kaggle。','', '逐用户变化、纠错/伤害和描述性用户聚类区间见 RESULT.json。大小记录对应每个外层原型的单套分支，不是把两个外层模型合并成一个提交。']
    (out/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    atomic(out/'DONE.json',dict(status='COMPLETE',protocol_sha256=ph,results_sha256=sha(out/'RESULT.json'),predictions_sha256=sha(out/'predictions.npz'),at_unix=time.time()))

def main():
    a=argparse.ArgumentParser();a.add_argument('--root',required=True);a.add_argument('--job',required=True);a.add_argument('--protocol',required=True);a.add_argument('--deadline',type=float,required=True);args=a.parse_args();root=Path(args.root);sys.path.insert(0,str(root/'bundle'))
    from dual_host_runtime.io import atomic,read
    from dual_host_runtime.health import Guard
    from research_runtime.common import sha
    from generalization_fusion_math import STRATEGIES,fit,predict
    torch.set_num_threads(2);p=read(args.protocol);ph=sha(args.protocol);out=root/'output'/args.job;out.mkdir(parents=True,exist_ok=True);guard=Guard(out,args.deadline,root/'TELEMETRY.json')
    try:
        for name,expected in p['code_sha256'].items():assert sha(root/'bundle'/name)==expected
        assert p['strategies']==['matched_fixed_vs']+list(STRATEGIES)
        if args.job=='FUSION_SUMMARY':summarize(root,p,ph,out,guard);return
        fold=p['jobs'][args.job]['fold'];raw,master,cohorts=load_cohorts(root,p);inner,evidence=assemble(root,p,fold,'inner',raw,master,cohorts,guard)
        models={}
        for name in STRATEGIES:
            value=fit(inner,name,guard);atomic(out/(name+'_fit_trace.json'),value.pop('trace'));models[name]=value
        atomic(out/'models.json',models)
        # This immutable checkpoint precedes any outer prediction/target file loading in this job.
        atomic(out/'CALIBRATION_FROZEN.json',dict(protocol_sha256=ph,at_unix=time.time(),fit_users=sorted(set(inner['users'].tolist())),outer_users=p['folds'][str(fold)],models_sha256=sha(out/'models.json'),sources=evidence,outer_predictions_loaded=False))
        outer,outer_evidence=assemble(root,p,fold,'outer',raw,master,cohorts,guard)
        results={key:outer[key] for key in ['clip_ids','targets','users','available']};results['matched_fixed_vs']=.8*outer['logits'][:,0]+.2*outer['logits'][:,1]
        for name,value in models.items():results[name]=predict(outer,value)
        np.savez_compressed(out/'predictions.npz',**results)
        assert sha(out/'models.json')==read(out/'CALIBRATION_FROZEN.json')['models_sha256']
        size=sum(v['checkpoint_bytes'] for v in outer_evidence)+(out/'models.json').stat().st_size;assert size<=100_000_000
        metrics={name:scores(results[name],outer['targets'],outer['users']) for name in p['strategies']}
        atomic(out/'RESULT.json',dict(protocol_sha256=ph,fold=fold,strategies=metrics,coverage={name:int(outer['available'][:,j].sum()) for j,name in enumerate(['V','S','T','I'])},sources=outer_evidence,prototype_bundle_bytes=size,outer_fit_users_overlap=False))
        atomic(out/'DONE.json',dict(status='COMPLETE',protocol_sha256=ph,predictions_sha256=sha(out/'predictions.npz'),models_sha256=sha(out/'models.json'),results_sha256=sha(out/'RESULT.json'),prototype_bundle_bytes=size,at_unix=time.time()))
    except BaseException as e:atomic(out/'ERROR.json',dict(error=repr(e),traceback=traceback.format_exc(),at_unix=time.time()));raise

if __name__=='__main__':main()
