"""One native completion event followed by a bounded paired Thermal audit."""
from pathlib import Path
from datetime import datetime
import argparse,json,math,subprocess,sys,time,traceback
import numpy as np
import psutil
import generalization_cloud as c
from audit_r3_development import scores,pair
ROOT=c.ROOT;BASE=c.BASE;LOCAL=BASE/'local';RUN=ROOT/'runs/thermal_contrast_20260911_v1'
R3=ROOT/'runs/r3_imu_fusion_release_20260911_v1'
sys.path.insert(0,str(LOCAL/'bundle'))
from dual_host_runtime.io import atomic,read
from dual_host_runtime.health import identity,alive
from research_runtime.common import sha


def audit():
    index=read(LOCAL/'bundle/data/thermal_index.json');master={r['clip_id']:(r['label'],r['user']) for r in index}
    parts=[];checks={};folds={}
    for fold in [0,3]:
        dirs={'control':LOCAL/'output/T_F0_outer','candidate':LOCAL/'output/T_CONTRAST_F0'} if fold==0 else {
            'control':R3/'cloud_backup/T_F3_outer','candidate':R3/'cloud_backup/T_CONTRAST_F3'}
        values={}
        for arm,path in dirs.items():
            done=read(path/'DONE.json');split=read(path/'SPLIT.json');norm=read(path/'NORMALIZER.json');prov=read(path/'PROVENANCE.json')
            assert done['status']=='COMPLETE' and done['epochs']==20
            assert done['updates']==20*math.ceil(len(split['train_ids'])/8)
            assert sha(path/'model_fp16.pt')==done['checkpoint_sha256'] and sha(path/'predictions.npz')==done['predictions_sha256']
            assert done['protocol_sha256']==prov['protocol_sha256'] and set(split['train_users'])==set(prov['train_users'])
            assert set(norm['train_ids'])==set(split['train_ids']) and set(norm['train_users'])==set(split['train_users'])
            assert not set(split['train_users'])&set(split['predict_users'])
            z=dict(np.load(path/'predictions.npz',allow_pickle=False));assert z['logits'].dtype==np.float32 and np.isfinite(z['logits']).all()
            assert len(z['clip_ids'])==len(set(z['clip_ids'])) and set(z['clip_ids'])=={k for k,(y,u) in master.items() if u in split['predict_users']}
            for k,y,u in zip(z['clip_ids'],z['targets'],z['users']):assert master[str(k)]==(int(y),int(u))
            order=np.argsort(z['clip_ids']);values[arm]={k:v[order] for k,v in z.items()};checks[f'{fold}_{arm}']=done
        a=values['candidate'];b=values['control']
        for key in ['clip_ids','targets','users']:np.testing.assert_array_equal(a[key],b[key])
        folds[str(fold)]=dict(control=scores(b['logits'],b['targets'],b['users']),candidate=scores(a['logits'],a['targets'],a['users']),paired=pair(a['logits'],b['logits'],a['targets'],a['users']))
        parts.append(dict(clip_ids=a['clip_ids'],targets=a['targets'],users=a['users'],candidate=a['logits'],control=b['logits']))
    z={k:np.concatenate([v[k] for v in parts]) for k in parts[0]};np.savez_compressed(RUN/'predictions.npz',**z)
    now=datetime.now().astimezone().isoformat(timespec='seconds');result=dict(status='BOTH_REGISTERED_THERMAL_CONTRAST_FOLDS_COMPLETE_PAIRED_AUDIT_PASS',at_local=now,
        folds=folds,pooled={key:scores(z[key],z['targets'],z['users']) for key in ['control','candidate']},paired=pair(z['candidate'],z['control'],z['targets'],z['users']),
        source_checks=checks,F0_recovered_after_shutdown=True,new_GPU_training=False,Test_access=False,direction_abandoned=False,
        evaluation_scope='Independent source hash,split andmetric recomputation; no newrawThermalprediction replay in this audit. CurrentCSV remainsfrozen.')
    atomic(RUN/'RESULT.json',result)
    event=dict(event='E-THERMAL-CONTRAST-TWO-FOLD-RESULT-20260911-V1',at_local=now,result=RUN.relative_to(ROOT).as_posix()+'/RESULT.json',sha256=sha(RUN/'RESULT.json'),
        pooled_correct={k:v['correct'] for k,v in result['pooled'].items()},paired=result['paired'],direction_abandoned=False)
    ledger=ROOT/'EXPERIMENT_RESULTS.jsonl'
    if not any(json.loads(line).get('event')==event['event'] for line in ledger.read_text(encoding='utf-8').splitlines()):
        with ledger.open('a',encoding='utf-8') as f:f.write(json.dumps(event,ensure_ascii=False)+'\n')
    state=read(ROOT/'PROJECT_STATE.json');state['updated_at_local']=now;state['latest_results']['thermal_contrast']=event
    state['runtime']['local'].update(status='REGISTERED_THERMAL_CONTRAST_COMPLETE',current_gpu_training=False,current_gpu_inference=False,latest_completion=event['result'])
    state['active_runs']=[v for v in state.get('active_runs',[]) if v.get('run_id')!='T_CONTRAST_F0']
    if not state['active_runs']:state['phase']='JOINT_CSV_DELIVERED_REGISTERED_RESEARCH_COMPLETE'
    atomic(ROOT/'PROJECT_STATE.json',state);atomic(RUN/'FINISH_DONE.json',result);return event


def run():
    p=read(RUN/'FINISH_PROTOCOL.json');assert sha(__file__)==p['source_sha256']
    owner=p['worker'];out=LOCAL/'output/T_CONTRAST_F0'
    if not (out/'DONE.json').exists():
        assert alive(owner),'REGISTERED_THERMAL_WORKER_NOT_ALIVE'
        # Native Windows process event; this is not a polling loop or an active Codex wait.
        psutil.Process(owner['pid']).wait()
    assert (out/'DONE.json').exists(),'THERMAL_WORKER_EXITED_WITHOUT_COMPLETE'
    return audit()


def launch():
    assert not (RUN/'FINISH_PROTOCOL.json').exists();owner=read(LOCAL/'output/T_CONTRAST_F0/LAUNCH.json');assert alive(owner)
    p=dict(source_sha256=sha(__file__),worker=owner,at_local=datetime.now().astimezone().isoformat(timespec='seconds'),
        behavior='Waitonce on exactownedworker;thenoneCPU source/metric audit andfact update. Existing native90/85Cthermalprotection continues separately.',new_training=False,upload=False)
    atomic(RUN/'FINISH_PROTOCOL.json',p)
    with (RUN/'finish.stdout.log').open('w') as so,(RUN/'finish.stderr.log').open('w') as se:
        proc=subprocess.Popen([sys.executable,'-X','utf8','-B',str(Path(__file__).resolve()),'run'],cwd=ROOT,stdout=so,stderr=se,creationflags=subprocess.CREATE_NO_WINDOW)
    value=dict(**identity(proc.pid),protocol_sha256=sha(RUN/'FINISH_PROTOCOL.json'));atomic(RUN/'FINISH_LAUNCH.json',value);return value


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['launch','run']);args=a.parse_args()
    try:print(json.dumps(globals()[args.action](),ensure_ascii=False))
    except BaseException as e:
        if args.action=='run':atomic(RUN/'FINISH_ERROR.json',dict(error=repr(e),traceback=traceback.format_exc(),at_unix=time.time()))
        raise
