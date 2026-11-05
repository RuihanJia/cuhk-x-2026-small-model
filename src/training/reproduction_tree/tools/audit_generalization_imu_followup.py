"""Independent CPU reconstruction of new IMU fits and paired development-user comparisons."""
from pathlib import Path
from datetime import datetime
import argparse,json,math,sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/generalization_training_20260910_v1';LOCAL=BASE/'local'
sys.path.insert(0,str(LOCAL/'bundle'))
from generalization_imu_worker import DeviceIMU,state_sha
from dual_host_runtime.io import atomic,read
from research_runtime.common import sha

def evaluate(protocol,name,replay=True,model_class=DeviceIMU):
    p=read(LOCAL/protocol);ph=sha(LOCAL/protocol);spec=p['jobs'][name];out=LOCAL/'output'/name
    done=read(out/'DONE.json');assert done['protocol_sha256']==ph and done['epochs']==60
    assert sha(out/'model_fp16.pt')==done['checkpoint_sha256'] and sha(out/'predictions.npz')==done['predictions_sha256']
    z=dict(np.load(LOCAL/'bundle/data/arm_acceleration_t256.npz',allow_pickle=False));assert sha(LOCAL/'bundle/data/arm_acceleration_t256.npz')==p['data']['arm_sha256']
    ready=read(LOCAL/'DEVICE_CACHE_READY.json');assert sha(LOCAL/ready['path'])==ready['sha256'];d=dict(np.load(LOCAL/ready['path'],allow_pickle=False))
    assert np.array_equal(d['x'][:,:2].reshape(-1,6,256),z['x']) and np.array_equal(d['clip_ids'],z['clip_ids'])
    keep=z['valid'];ids=z['clip_ids'][keep];users=z['users'][keep];ys=z['targets'][keep]
    train=np.flatnonzero(np.isin(users,spec['train_users']));test=np.flatnonzero(np.isin(users,spec['predict_users']))
    assert not set(users[train])&set(users[test]);assert set(users[test])==set(spec['predict_users'])
    assert not set(users[train])&set(p['folds'][str(spec['fold'])])
    pred=dict(np.load(out/'predictions.npz',allow_pickle=False))
    for key,value in [('clip_ids',ids[test]),('users',users[test]),('targets',ys[test])]:assert np.array_equal(pred[key],value)
    assert np.isfinite(pred['logits']).all() and pred['logits'].dtype==np.float32
    assert done['updates']==60*math.ceil(len(train)/64) and done['parameters']==180264
    raw=np.zeros((len(ids),5,3,256),np.float32);raw[:,:2]=z['x'][keep].reshape(-1,2,3,256);mask=np.zeros((len(ids),5),bool);mask[:,:2]=True
    if spec['arm'] in ['C','D']:raw=d['x'][keep];mask=d['valid'][keep]
    cp=torch.load(out/'model_fp16.pt',map_location='cpu',weights_only=True);mu=cp['mean'].numpy();sd=cp['std'].numpy()
    for dev in range(5):
        ix=train[mask[train,dev]]
        if len(ix):
            a=raw[ix,dev].astype(np.float64)
            assert np.array_equal(mu[:,dev],a.mean((0,2))[None,:,None].astype(np.float32))
            assert np.array_equal(sd[:,dev],a.std((0,2)).clip(1e-4)[None,:,None].astype(np.float32))
    torch.manual_seed(p['training']['seed']);model=model_class();assert state_sha(model.state_dict())==done['initial_state_sha256']
    error=None;mismatches=None
    if replay:
        model.load_state_dict(cp['state_dict'],strict=True);model.eval();values=[]
        with torch.no_grad():
            for i in range(0,len(test),64):
                ix=test[i:i+64];values.append(model(torch.from_numpy((raw[ix]-mu)/sd),torch.from_numpy(mask[ix])).numpy())
        replayed=np.concatenate(values);error=float(np.max(np.abs(replayed-pred['logits'])));mismatches=int(np.sum(replayed.argmax(1)!=pred['logits'].argmax(1)))
        assert error<1e-4 and mismatches==0,(name,error,mismatches)
    correct=pred['logits'].argmax(1)==pred['targets'];cm=np.zeros((40,40),np.int64);np.add.at(cm,(pred['targets'],pred['logits'].argmax(1)),1);den=cm.sum(0)+cm.sum(1)
    report=dict(job=name,protocol_sha256=ph,seed=p['training']['seed'],correct=int(correct.sum()),n=len(correct),macro_f1=float(np.divide(2*cm.diagonal(),den,out=np.zeros(40),where=den>0).mean()),
        per_user={str(u):dict(correct=int(correct[pred['users']==u].sum()),n=int((pred['users']==u).sum())) for u in np.unique(pred['users'])},parameters=done['parameters'],checkpoint_bytes=done['checkpoint_bytes'],checkpoint_sha256=done['checkpoint_sha256'],predictions_sha256=done['predictions_sha256'],initial_state_sha256=done['initial_state_sha256'],CPU_FP32_replay_maxabs=error,top1_mismatches=mismatches,previous_replay_evidence=None if replay else 'IMU_AUDIT.json')
    return report,dict(correct=correct,users=pred['users'],ids=pred['clip_ids'])

def comparison(new,old):
    assert np.array_equal(new['users'],old['users']) and np.array_equal(new['ids'],old['ids'])
    a,b,u=new['correct'],old['correct'],new['users'];order=np.unique(u)
    gains=np.array([int(a[u==v].sum())-int(b[u==v].sum()) for v in order]);counts=np.array([(u==v).sum() for v in order]);rng=np.random.default_rng(2026091101);ix=rng.integers(0,len(order),(10000,len(order)));boot=gains[ix].sum(1)/counts[ix].sum(1)
    return dict(fixed=int((a&~b).sum()),harmed=int((b&~a).sum()),net_correct=int(a.sum()-b.sum()),accuracy_delta_pp=100*float((a.astype(float)-b).mean()),per_user_net=dict(zip(map(str,order),gains.tolist())),user_cluster_bootstrap95_delta_pp=(100*np.quantile(boot,[.025,.975])).tolist(),uncertainty_limit='Only seven reused development users; no independent unseen-user or Public-score claim')

def main():
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['second-seed','dropout-replication']);args=a.parse_args();torch.set_num_threads(2)
    replication=args.mode=='dropout-replication';tag='IMU_DROPOUT_REPLICATION' if replication else 'IMU_FOLLOWUP'
    assert not (BASE/(tag+'_AUDIT.json')).exists(),'Audit already exists; avoid duplicate result-table rows'
    reports={};joined={}
    for arm in 'BCD':
        pairs=[]
        for fold in [0,3]:
            protocol='imu_dropout_seed1_protocol.json' if replication and arm=='D' else 'protocol.json' if replication else 'imu_followup_no_wall_protocol.json'
            name=f'IMU_D1_F{fold}' if replication and arm=='D' else f'IMU_{arm}_F{fold}' if replication else f'IMU_R_{arm}_F{fold}'
            report,values=evaluate(protocol,name,replay=not replication or arm=='D');reports[f'{arm}_F{fold}']=report;pairs.append(values)
        joined[arm]={key:np.concatenate([v[key] for v in pairs]) for key in pairs[0]}
    assert len({v['initial_state_sha256'] for v in reports.values()})==1
    comparisons={f'{a}_vs_{b}':comparison(joined[a],joined[b]) for a,b in [('C','B'),('D','B'),('D','C')]}
    totals={a:sum(reports[f'{a}_F{f}']['correct'] for f in [0,3]) for a in 'BCD'}
    result=dict(status='PASS_FIXED_EPOCHS_AND_INDEPENDENT_FP32_RELOAD',at_local=datetime.now().astimezone().isoformat(timespec='seconds'),mode=args.mode,arms=reports,pooled_correct=totals,n=968,comparisons=comparisons,equal_capacity_and_initialization=True,held_users_excluded_from_training_and_normalization=True,original_arm_tensors_unchanged=True,promotion=False,limitations=['Seven reused F0/F3 development users; repeated seeds do not create new independent users','D has full observed device masks at inference; device dropout is training-only','No Test, F2/F4 outer evaluation, all18 training or Kaggle upload'])
    target=BASE/(tag+'_AUDIT.json');atomic(target,result)
    lines=['# IMU 设备丢弃控制复核' if replication else '# 第二组随机种子的 IMU 对照','',f"核查时间：{result['at_local']}。固定 60 轮训练，FP16 保存后按 FP32 独立重载预测。",'','| 方案 | F0 | F3 | 合计 |','|---|---:|---:|---:|']
    for arm in 'BCD':lines.append(f"| {arm} | {reports[f'{arm}_F0']['correct']}/470 | {reports[f'{arm}_F3']['correct']}/498 | {totals[arm]}/968 |")
    lines+=['','B：只用双臂；C：同结构加入有效额外设备；D：在 C 的基础上，仅在训练时以固定 0.25 概率独立丢弃每个额外设备，双臂保留。三组结构和初始权重一致。','']
    for name,v in comparisons.items():lines.append(f"{name}：纠正 {v['fixed']} 条、伤害 {v['harmed']} 条，净变化 {v['net_correct']:+d}；用户聚类 bootstrap 95% 区间 {v['user_cluster_bootstrap95_delta_pp'][0]:+.2f} 至 {v['user_cluster_bootstrap95_delta_pp'][1]:+.2f} 个百分点。")
    lines+=['','结果仅用于反复使用的 F0/F3 开发比较。随机种子复核不能替代新的未见用户证据；当前不据此替换提交。逐用户变化和精确文件哈希见配套审计 JSON。']
    (BASE/(tag+'_RESULT.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fact=dict(event='E-GENERALIZATION-'+tag+'-RESULT-20260911-V1',at_local=result['at_local'],status=result['status'],audit=target.relative_to(ROOT).as_posix(),audit_sha256=sha(target),pooled_correct=totals,n=968,promoted=False)
    with (ROOT/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(fact,ensure_ascii=False)+'\n')
    print(json.dumps(dict(status=result['status'],totals=totals,comparisons=comparisons,max_replay_error=max(v['CPU_FP32_replay_maxabs'] or 0 for v in reports.values())),ensure_ascii=False))

if __name__=='__main__':main()
