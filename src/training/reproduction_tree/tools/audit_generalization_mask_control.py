"""Independent reconstruction of explicit-mask E controls versus their exact D controls."""
from datetime import datetime
import importlib.util,json
import numpy as np
import audit_generalization_imu_followup as a

def main():
    target=a.BASE/'IMU_MASK_CONTROL_AUDIT.json';assert not target.exists();a.torch.set_num_threads(2);seeds={}
    for index in [1,2]:
        path=a.LOCAL/f'bundle/generalization_imu_nomask_seed{index}.py';spec=importlib.util.spec_from_file_location(f'nomask_{index}',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);reports={};joined={'D':[],'E':[]}
        for fold in [0,3]:
            djob=f'IMU_D1_F{fold}' if index==1 else f'IMU_R_D_F{fold}';dprotocol='imu_dropout_seed1_protocol.json' if index==1 else 'imu_followup_no_wall_protocol.json'
            dr,d=a.evaluate(dprotocol,djob,replay=False);er,e=a.evaluate(f'imu_mask_control_seed{index}_protocol.json',f'IMU_E_S{index}_F{fold}',model_class=module.DeviceIMU)
            assert dr['initial_state_sha256']==er['initial_state_sha256'];reports[str(fold)]=dict(D=dr,E=er);joined['D'].append(d);joined['E'].append(e)
        pairs={arm:{k:np.concatenate([v[k] for v in parts]) for k in parts[0]} for arm,parts in joined.items()}
        seeds[str(index)]=dict(folds=reports,E_vs_D=a.comparison(pairs['E'],pairs['D']),totals={arm:int(v['correct'].sum()) for arm,v in pairs.items()},n=968)
    result=dict(status='PASS_FOUR_FIXED_FITS_AND_INDEPENDENT_REPLAY',at_local=datetime.now().astimezone().isoformat(timespec='seconds'),seeds=seeds,fusion_protocol_unchanged=True,limitations=['Only explicit availability features are removed; zero embeddings retain implicit missingness information','Same seven reused development users; no independent unseen-user claim'])
    a.atomic(target,result)
    lines=['# IMU 显式可用性特征对照','', 'E 保留 D 的缺失输入清零、编码后 mask 和训练期设备丢弃，只把分类头收到的五个显式可用性特征置零。参数形状及初始化不变。','', '| 种子组 | D | E | E-D |','|---|---:|---:|---:|']
    for seed,v in seeds.items():lines.append(f"| {seed} | {v['totals']['D']}/968 | {v['totals']['E']}/968 | {v['E_vs_D']['net_correct']:+d} |")
    lines+=['','当前融合比较已提前冻结，仍使用 D。本控制用于后续机制判断，不回写或重选正在进行的融合协议。七位开发用户反复使用；不能把此结果解释为独立未见用户收益。']
    (a.BASE/'IMU_MASK_CONTROL_RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fact=dict(event='E-GENERALIZATION-IMU-MASK-CONTROL-RESULT-20260911-V1',status=result['status'],at_local=result['at_local'],audit=target.relative_to(a.ROOT).as_posix(),sha256=a.sha(target),net_by_seed={k:v['E_vs_D']['net_correct'] for k,v in seeds.items()},frozen_fusion_unchanged=True)
    with (a.ROOT/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(fact,ensure_ascii=False)+'\n')
    print(json.dumps(fact,ensure_ascii=False))

if __name__=='__main__':main()
