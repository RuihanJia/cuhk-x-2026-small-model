"""Register one full-data IMU fit and a fixed, low-cost video/sensor/IMU transfer check."""
from pathlib import Path
from datetime import datetime
import ast,copy,json,subprocess,sys
import numpy as np
import generalization_cloud as cloud
ROOT=cloud.ROOT;BASE=cloud.BASE;LOCAL=BASE/'local';BUNDLE=LOCAL/'bundle';RUN=ROOT/'runs/r3_imu_fusion_release_20260911_v1'
sys.path.insert(0,str(BUNDLE))
from dual_host_runtime.io import atomic,read
from dual_host_runtime.health import alive
from research_runtime.common import sha

def main():
    assert not RUN.exists();assert not alive(read(LOCAL/'CONTROLLER.json'))
    done=read(LOCAL/'QUEUE_DONE.json');assert done['status']=='COMPLETE'
    RUN.mkdir();state=read(ROOT/'PROJECT_STATE.json');parent=read(BASE/'cloud/fusion_protocol_align2.json')
    now=datetime.now().astimezone().isoformat(timespec='seconds')
    # Preserve original development video and sensor predictions for a deployment-recipe transfer check.
    for fold in [0,3]:
        v=dict(np.load(ROOT/f'runs/r1_fiveview_release_20260910_v1/development/fold{fold}.npz',allow_pickle=False))
        s=dict(np.load(ROOT/state['assets'][f'fixed_f{fold}_reference'],allow_pickle=False))
        assert np.array_equal(v['clip_ids'],s['clip_ids'])
        np.savez_compressed(RUN/f'existing_F{fold}.npz',clip_ids=v['clip_ids'],users=s['users'],targets=s['targets'],video=v['logits'],sensor=s['sensor'])
    p=dict(id='R3-LOCAL-IMU-FUSION-RELEASE-20260911-V1',status='FROZEN_BEFORE_VSI_FITS_OR_ALL18_IMU_TRAINING',frozen_at_local=now,
        authorization='User at2026-09-11 01:02 explicitly requested an improved candidate CSV to personally submit toKaggle. This authorizes the necessary bounded production fit and Test inference; no assistant upload.',
        parent_fusion_protocol_sha256=sha(BASE/'cloud/fusion_protocol_align2.json'),
        hypothesis='Replicated IMU-D may add complementary information to the existing five-view video/sensor release. One convex probability calibrator provides a fast deployable three-branch check without needing a new full-data Thermal model.',
        probe=dict(strategies=['convex_vsi'],description=dict(branches=[0,1,3],mode='convex',availability=False),folds=[0,3],
            inner_only=True,matched_outer_check=True,existing_video_sensor_outer_transfer_check=True,
            selection='Use paired pooled and fold/user changes to judge whether the IMU candidate warrants a submission. No numeric minimum-gain gate and no Public-driven scan; preserve any mixed or negative results.',
            math='Exact previously frozen convex probability mixture, scalar temperature and user-balanced NLL; only branch subset changes toV/S/I; no gate or coefficient sweep'),
        production_calibration='If candidate is retained, fit the same single convex_vsi recipe to the union of six verified inner-OOF batches. Repeated clips from different user-excluded fits remain separate draws; inverse-user-row-count loss gives each of18users equal total weight. No Test labels or predictions used for fitting.',
        all18_imu=dict(max_fits=1,epochs=60,seed=2026091003,orientation_fold_tag=18,dropout=.25,architecture_parameters=180264,
            recipe='Exact replicatedD training except all18 training users and a prospectively fixed full-data orientation tag18; no heldout evaluation or checkpoint selection'),
        deployment=dict(video='Exact existing all18 five-view checkpoint',sensor='Exact existing all18 measured sensor checkpoint',imu='One new full18D checkpoint',serialization_limit_bytes=100000000),
        budget=dict(new_GPU_fits=1,new_CPU_development_calibrators=2,new_CPU_production_calibrators=1,max_IMU_epochs=60,
            expected_IMU_fit_seconds=45,expected_total_release_minutes=[15,30],new_cloud_GPU_fits=0,cloud_deadline_unix=parent['budget']['deadline_unix'],local_wall_cap=False),
        boundaries=dict(Test_inference_authorized=True,Test_labels_read=False,F2_F4_outer_evaluation=False,assistant_Kaggle_upload=False,Public_score_tuning=False),
        source_sha256={f'existing_F{f}.npz':sha(RUN/f'existing_F{f}.npz') for f in [0,3]})
    worker_source=BUNDLE/'generalization_imu_dropout_seed1.py';code=worker_source.read_text(encoding='utf-8')
    old='assert len(train)>0 and len(test)>0 and not set(users[train])&set(users[test]);'
    assert code.count(old)==1;code=code.replace(old,'assert len(train)>0 and len(test)==0 and not set(users[train])&set(users[test]);')
    assert code.count('logits=np.concatenate(pred)')==1;code=code.replace('logits=np.concatenate(pred)','logits=np.concatenate(pred) if pred else np.empty((0,40),np.float32)')
    ast.parse(code);worker=BUNDLE/'generalization_imu_full18.py';assert not worker.exists();worker.write_text(code,encoding='utf-8')
    ip=copy.deepcopy(read(LOCAL/'imu_dropout_seed1_protocol.json'));ip.update(id='R3-IMU-D-FULL18-20260911-V1',frozen_at_local=now,status='FROZEN_BEFORE_ONE_FULL18_FIT',jobs={'IMU_D_ALL18_R3':dict(arm='D',fold=18,train_users=parent['training_users'],predict_users=[])},
        hypothesis=p['hypothesis'],controls='Existing replicatedD development evidence; production fit has no heldout metrics',budget=p['budget'],release_protocol=str(RUN/'protocol.json'))
    ip['code_sha256']={v.relative_to(BUNDLE).as_posix():sha(v) for v in BUNDLE.rglob('*.py')}
    atomic(LOCAL/'imu_full18_r3_protocol.json',ip);p['full18_protocol_sha256']=sha(LOCAL/'imu_full18_r3_protocol.json');atomic(RUN/'protocol.json',p)
    fact=dict(event='E-R3-IMU-FUSION-RELEASE-REGISTERED-20260911-V1',at_local=now,protocol=str((RUN/'protocol.json').relative_to(ROOT)),protocol_sha256=sha(RUN/'protocol.json'),status=p['status'],new_GPU_fits=1,new_CPU_fits=3,Test_inference_authorized=True,uploaded=False)
    with (ROOT/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(fact,ensure_ascii=False)+'\n')
    queue=read(LOCAL/'QUEUE.json');assert not any(j['id']=='IMU_D_ALL18_R3' for j in queue['jobs']);queue['concurrency']=1
    queue['jobs'].append(dict(id='IMU_D_ALL18_R3',worker=worker.name,protocol='imu_full18_r3_protocol.json',protocol_sha256=p['full18_protocol_sha256'],requires=['DEVICE_CACHE_READY.json']))
    atomic(LOCAL/'QUEUE_DONE_BEFORE_R3.json',done);(LOCAL/'QUEUE_DONE.json').unlink();atomic(LOCAL/'QUEUE.json',queue)
    with (LOCAL/'controller_r3.stdout.log').open('a') as out,(LOCAL/'controller_r3.stderr.log').open('a') as err:
        proc=subprocess.Popen([sys.executable,'-X','utf8','-B',str(BUNDLE/'native_queue_no_wall.py'),'--root',str(LOCAL)],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
    state['current_authorization']['r3_release']=dict(at_local=now,request='云端预计还有多少时间能够结束，以及本机的具体训练是否已经可以产生一个有提升的CSV文件，把它给我，让我来提交Kaggle来验证一下。',protocol=fact['protocol'],Test_inference_authorized=True,one_full18_IMU_fit_authorized=True,assistant_upload=False)
    state['release_preparation']['r3_imu_fusion']=dict(status='ONE_FULL18_IMU_FIT_STARTED_TRANSFER_CHECK_REGISTERED',protocol=fact['protocol'],probe='One convexV/S/I family only,2inner-fit development checks plus1prospective production fit')
    state['runtime']['local'].update(status='RUNNING_ONE_FULL18_IMU_PRODUCTION_FIT',current_gpu_training=True,execution_protocol='runs/generalization_training_20260910_v1/local/imu_full18_r3_protocol.json')
    state['updated_at_local']=now;atomic(ROOT/'PROJECT_STATE.json',state);print(json.dumps(dict(**fact,controller_spawn_pid=proc.pid),ensure_ascii=False))

if __name__=='__main__':main()
