"""Independent CPU reload and paired user-level readout of the six frozen IMU fits."""
from pathlib import Path
import hashlib,json,sys,time
from datetime import datetime
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/generalization_training_20260910_v1';LOCAL=BASE/'local';sys.path.insert(0,str(LOCAL/'bundle'))
from generalization_imu_worker import ArmIMU,DeviceIMU,state_sha
from dual_host_runtime.io import atomic,read
from research_runtime.common import sha

def main():
    torch.set_num_threads(2);p=read(LOCAL/'protocol.json');ph=sha(LOCAL/'protocol.json');z=dict(np.load(LOCAL/'bundle/data/arm_acceleration_t256.npz'));ready=read(LOCAL/'DEVICE_CACHE_READY.json');assert sha(LOCAL/ready['path'])==ready['sha256'];device=dict(np.load(LOCAL/ready['path']));keep=z['valid'];users=z['users'][keep];ids=z['clip_ids'][keep];ys=z['targets'][keep];reports={};joined={}
    assert np.array_equal(device['x'][:,:2].reshape(-1,6,256),z['x'])
    for name,spec in p['jobs'].items():
        out=LOCAL/'output'/name;done=read(out/'DONE.json');assert done['protocol_sha256']==ph and done['epochs']==60;checkpoint=out/'model_fp16.pt';assert sha(checkpoint)==done['checkpoint_sha256'] and sha(out/'predictions.npz')==done['predictions_sha256'];pred=dict(np.load(out/'predictions.npz'));test=np.flatnonzero(np.isin(users,spec['predict_users']));train=np.flatnonzero(np.isin(users,spec['train_users']));assert np.array_equal(pred['clip_ids'],ids[test]) and np.array_equal(pred['targets'],ys[test]);assert done['updates']==60*((len(train)+63)//64)
        raw=np.zeros((len(ids),5,3,256),np.float32);raw[:,:2]=z['x'][keep].reshape(-1,2,3,256);mask=np.zeros((len(ids),5),bool);mask[:,:2]=True
        if spec['arm']=='C':raw=device['x'][keep];mask=device['valid'][keep]
        cp=torch.load(checkpoint,map_location='cpu',weights_only=True);mu=cp['mean'].numpy();sd=cp['std'].numpy()
        for dev in range(5):
            ix=train[mask[train,dev]]
            if len(ix):
                a=raw[ix,dev].astype(np.float64);assert np.array_equal(mu[:,dev],a.mean((0,2))[None,:,None].astype(np.float32));assert np.array_equal(sd[:,dev],a.std((0,2)).clip(1e-4)[None,:,None].astype(np.float32))
        torch.manual_seed(p['training']['seed']);model=ArmIMU() if spec['arm']=='A' else DeviceIMU();assert state_sha(model.state_dict())==done['initial_state_sha256'];model.load_state_dict(cp['state_dict'],strict=True);model.eval();replayed=[]
        with torch.no_grad():
            for i in range(0,len(test),64):
                ix=test[i:i+64];value=model(torch.from_numpy((raw[ix]-mu)/sd),torch.from_numpy(mask[ix]));replayed.append(value.numpy())
        replay=np.concatenate(replayed);error=float(np.abs(replay-pred['logits']).max());mismatches=int((replay.argmax(1)!=pred['logits'].argmax(1)).sum());assert error<=1e-4 and mismatches==0,(name,error,mismatches);correct=pred['logits'].argmax(1)==pred['targets'];cm=np.zeros((40,40),np.int64);np.add.at(cm,(pred['targets'],pred['logits'].argmax(1)),1);den=cm.sum(0)+cm.sum(1);f1=np.divide(2*cm.diagonal(),den,out=np.zeros(40),where=den>0).mean()
        reports[name]=dict(correct=int(correct.sum()),n=len(correct),macro_f1=float(f1),per_user={str(u):dict(correct=int(correct[pred['users']==u].sum()),n=int((pred['users']==u).sum())) for u in sorted(set(pred['users']))},parameters=done['parameters'],checkpoint_bytes=done['checkpoint_bytes'],initial_state_sha256=done['initial_state_sha256'],CPU_FP32_replay_maxabs=error,top1_mismatches=mismatches)
        joined[name]=dict(correct=correct,users=pred['users'])
    comparisons={};user_order=sorted(set(users[np.isin(users,[2,16,20,5,7,8,21])].tolist()));rng=np.random.default_rng(2026091003)
    for newer,older in [('B','A'),('C','A'),('C','B')]:
        new=np.concatenate([joined[f'IMU_{newer}_F{f}']['correct'] for f in [0,3]]);old=np.concatenate([joined[f'IMU_{older}_F{f}']['correct'] for f in [0,3]]);u=np.concatenate([joined[f'IMU_{newer}_F{f}']['users'] for f in [0,3]]);gains=np.array([int(new[u==v].sum())-int(old[u==v].sum()) for v in user_order]);counts=np.array([int((u==v).sum()) for v in user_order]);ix=rng.integers(0,len(gains),(10000,len(gains)));boot=gains[ix].sum(1)/counts[ix].sum(1)
        comparisons[f'{newer}_vs_{older}']=dict(fixed=int((new&~old).sum()),harmed=int((old&~new).sum()),net_correct=int(new.sum()-old.sum()),accuracy_delta_pp=100*float((new.astype(float)-old).mean()),per_user_net=dict(zip(map(str,user_order),gains.tolist())),user_cluster_bootstrap95_delta_pp=(100*np.quantile(boot,[.025,.975])).tolist(),uncertainty_limit='Only7reused development users; descriptive bootstrap, not proof of unseen-user improvement')
    result=dict(status='PASS_COMPLETE_SIX_FITS_AND_INDEPENDENT_REPLAY',at_local=datetime.now().astimezone().isoformat(timespec='seconds'),protocol_sha256=ph,arms=reports,comparisons=comparisons,B_C_exact_initialization_and_capacity=True,original_arm_tensors_unchanged=True,data_coverage=ready['valid_devices'],promotion=False,interpretation='Shared device encoder is positive against A; adding other devices is mixed and pooled-1 versus B. Preserve both directions; no automatic rejection. The same fusion experiment Thermal fits now occupy the local GPU. F0/F3 are reused development folds.')
    target=BASE/'IMU_AUDIT.json';atomic(target,result);lines=['# 首批 IMU 三组对照','',f"核查时间：{result['at_local']}。六次固定 60 轮训练已完成，独立 CPU FP32 重载全量预测通过（最大绝对误差不超过 1e-4，预测类别零差异）。",'', '| 组别 | F0 | F3 | 合计 | 参数量 |','|---|---:|---:|---:|---:|']
    for a in 'ABC':x=reports[f'IMU_{a}_F0'];y=reports[f'IMU_{a}_F3'];lines.append(f"| {a} | {x['correct']}/470 | {y['correct']}/498 | {x['correct']+y['correct']}/968 | {x['parameters']} |")
    lines+=['','A：原双臂网络 + 方向增强。B：共享设备编码器，只用双臂。C：与 B 完全相同的结构和初始权重，加入通过有效性检查的其余设备。A/B 参数量不同；B/C 才是隔离新增信号影响的对照。','', 'B 相比 A 净增 15 条；C 相比 B 净减 1 条（F0 +7、F3 -8）。新增设备的效益仍不明确。两类结构保留，不依据这一次结果关闭方向。当前先完成已获授权的 Thermal 交叉拟合，之后结合互补错误和用户差异准备有限后继。','','统计覆盖相同的 968 条有效 IMU 开发样本；不等于全部 1094 条视频开发样本。所有原始双臂张量保持逐值一致，均值/方差只由该次训练用户计算。F0/F3 均为反复使用的开发用户，不能把本结果表述为独立未见用户或比赛成绩改善。','','详细用户差异、纠错/伤害、聚类自助区间及文件核查见 `IMU_AUDIT.json`。']
    (BASE/'IMU_RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');fact=dict(event='E-GENERALIZATION-IMU-RESULT-20260910-V1',at_local=result['at_local'],status=result['status'],audit=target.relative_to(ROOT).as_posix(),audit_sha256=sha(target),pooled_correct={'A':334,'B':349,'C':348},n=968,promoted=False)
    with (ROOT/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(fact,ensure_ascii=False)+'\n')
    print(json.dumps(dict(status=result['status'],comparisons=comparisons,max_replay_error=max(v['CPU_FP32_replay_maxabs'] for v in reports.values())),ensure_ascii=False))
if __name__=='__main__':main()
