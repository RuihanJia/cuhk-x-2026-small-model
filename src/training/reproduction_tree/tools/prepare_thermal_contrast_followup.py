"""Register two matched-host Thermal contrast invariance tests and place them in native queues."""
from pathlib import Path
from datetime import datetime
import ast,copy,importlib.util,json,subprocess,sys
import generalization_cloud as c
ROOT=c.ROOT;BASE=c.BASE;LOCAL=BASE/'local';CLOUD=BASE/'cloud';RUN=ROOT/'runs/thermal_contrast_20260911_v1'
sys.path.insert(0,str(LOCAL/'bundle'))
from dual_host_runtime.io import atomic,read
from dual_host_runtime.health import alive
from research_runtime.common import sha

def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main():
    import torch
    assert not RUN.exists();RUN.mkdir();now=datetime.now().astimezone().isoformat(timespec='seconds');source=CLOUD/'bundle/generalization_thermal_worker_io2.py';code=source.read_text(encoding='utf-8')
    old='def __init__(self,rows,root,mean,std,training=False,epoch=0):self.rows=rows;self.root=Path(root);self.mean=mean;self.std=std;self.training=training;self.epoch=epoch;self.seed=2026091003'
    assert code.count(old)==1;code=code.replace(old,old.replace('epoch=0):','epoch=0,photometric=False):')+';self.photometric=photometric')
    old="x=TF.center_crop(TF.resize(torch.stack(frames).float().div_(255),[256],antialias=True),[224,224]);x=TF.normalize(x,self.mean,self.std)"
    new="""x=TF.center_crop(TF.resize(torch.stack(frames).float().div_(255),[256],antialias=True),[224,224])
        if self.training and self.photometric and rng.random()<.5:
            contrast=rng.uniform(.8,1.2);offset=rng.uniform(-.05,.05);center=x.mean((0,2,3),keepdim=True)
            x=((x-center)*contrast+center+offset).clamp(0,1)
        x=TF.normalize(x,self.mean,self.std)"""
    assert code.count(old)==1;code=code.replace(old,new)
    old='ThermalClips(train,data_root,mean.tolist(),std.tolist(),True,epoch)';assert code.count(old)==1;code=code.replace(old,"ThermalClips(train,data_root,mean.tolist(),std.tolist(),True,epoch,photometric=spec['photometric'])")
    code=code.replace('from dual_host_runtime.health import Guard','from optional_time_budget import Guard');code=code.replace('if epoch<20 and time.time()','if args.deadline>0 and epoch<20 and time.time()');ast.parse(code)
    for host in [LOCAL,CLOUD]:
        path=host/'bundle/generalization_thermal_contrast.py';assert not path.exists();path.write_text(code,encoding='utf-8')
    # Verify the exact input delta on Training clips only, without touching a heldout metric.
    oldmod=module('thermal_contrast_control',source);newmod=module('thermal_contrast_candidate',LOCAL/'bundle/generalization_thermal_contrast.py');rows=read(LOCAL/'bundle/data/thermal_index.json')[:6];rawroot=read(LOCAL/'thermal_protocol.json')['data']['root'];mean=[.4]*3;std=[.2]*3;changed=0
    for epoch in [1,2]:
        a=oldmod.ThermalClips(rows,rawroot,mean,std,True,epoch);b=newmod.ThermalClips(rows,rawroot,mean,std,True,epoch,False);d=newmod.ThermalClips(rows,rawroot,mean,std,True,epoch,True)
        for i in range(len(rows)):
            x=a[i];y=b[i];z=d[i];assert torch.equal(x['x'],y['x']);assert x['clip_id']==z['clip_id'] and x['y']==z['y'];assert torch.isfinite(z['x']).all() and torch.equal(z['x'],d[i]['x']);changed+=int(not torch.equal(x['x'],z['x']))
    a=oldmod.ThermalClips(rows,rawroot,mean,std);b=newmod.ThermalClips(rows,rawroot,mean,std,photometric=True)
    assert all(torch.equal(a[i]['x'],b[i]['x']) for i in range(len(rows))) and 0<changed<12
    qa=dict(status='PASS_EXACT_CONTROL_AND_EVALUATION_INPUT_PARITY',training_clips=6,training_epoch_draws=12,changed_draws=changed,augmentation_disabled_bitwise_equal=True,evaluation_bitwise_equal=True,repeated_augmentation_bitwise_equal=True,Test_access=False);atomic(RUN/'INPUT_DELTA_CHECKS.json',qa)
    protocol=dict(id='THERMAL-CLIP-CONTRAST-INVARIANCE-20260911-V1',frozen_at_local=now,status='FROZEN_BEFORE_TWO_NEW_FITS',
        hypothesis='Thermal has positive marginal complementarity(+4in matched convex fusion) but may depend on per-person/scene brightness and contrast. A temporally constant affine intensity perturbation can encourage retention of spatial/motion patterns while reducing dependence on global appearance.',
        intervention='Training only: probability.5,contrast uniform[.8,1.2] around per-clip/per-channel mean,add one scalar offset uniform[-.05,.05],clamp[0,1]. Same transform for all16frames/channels apart from eachchannel mean. No hue/palette swap. Existing sample positions and flip draws unchanged.',
        controls=dict(F0='Original T_F0_outer trained on this same4070localhost',F3='Original T_F3_outer trained on this same5090cloudhost'),
        exact=dict(seed=2026091003,epochs=20,micro_batch=2,accumulation=4,optimizer='AdamW.0003/.0001',smoothing=.1,checkpoint='fixed final20',normalizer='same permitted training-user frames only'),
        assessment='Paired standaloneT metrics on the exact correspondingThermalcohort; report user differences and errors. ReplacingT in the frozen current fusion is only a transfer diagnostic because no augmented innerT model was fitted. Do not automatically replace the submitted package.',
        budget=dict(new_fits=2,one_local_F0=True,one_cloud_F3=True,epochs_each=20,expected_local_minutes=[20,40],expected_cloud_minutes=[18,30],local_wall_cap=False,cloud_deadline_unix=read(CLOUD/'QUEUE.json')['deadline_unix'],no_cloud_window_extension=True),
        boundaries=dict(F0_F3_development=True,Test_access=False,F2_F4_outer_evaluation=False,new_all18_augmentation_fit=False,Public_score_tuning=False,upload=False),
        input_check_sha256=sha(RUN/'INPUT_DELTA_CHECKS.json'),worker_sha256=sha(LOCAL/'bundle/generalization_thermal_contrast.py'),next_decision='Review bothregisteredfolds and finitefollowup cost; noautomaticdirectionban or arbitraryminimum-countgate.')
    localp=copy.deepcopy(read(LOCAL/'thermal_protocol.json'));cloudp=copy.deepcopy(read(CLOUD/'thermal_handoff_protocol.json'))
    for host,p,fold in [(LOCAL,localp,0),(CLOUD,cloudp,3)]:
        p.pop('scientific_parent_sha256',None);spec=copy.deepcopy(p['jobs'][f'T_F{fold}_outer']);spec['photometric']=True
        p.update(id=f'THERMAL-CONTRAST-F{fold}-20260911-V1',frozen_at_local=now,jobs={f'T_CONTRAST_F{fold}':spec},hypothesis=protocol['hypothesis'],budget=protocol['budget'])
        for name in ['generalization_thermal_contrast.py','optional_time_budget.py']:p['code_sha256'][name]=sha(host/'bundle'/name)
        atomic(host/'thermal_contrast_protocol.json',p)
    protocol['local_protocol_sha256']=sha(LOCAL/'thermal_contrast_protocol.json');protocol['cloud_protocol_sha256']=sha(CLOUD/'thermal_contrast_protocol.json');atomic(RUN/'protocol.json',protocol)
    fact=dict(event='E-THERMAL-CONTRAST-FOLLOWUP-FROZEN-20260911-V1',at_local=now,protocol=(RUN/'protocol.json').relative_to(ROOT).as_posix(),protocol_sha256=sha(RUN/'protocol.json'),new_fits=2,epochs=20,matched_same_host_controls=True)
    with (ROOT/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(fact,ensure_ascii=False)+'\n')
    # Cloud successor starts when the productionThermal job releases a slot; video may continue concurrently.
    m,client=c.access.connection()
    try:
        with client.open_sftp() as ftp:
            ftp.put(str(CLOUD/'bundle/generalization_thermal_contrast.py'),c.REMOTE+'/bundle/generalization_thermal_contrast.py');ftp.put(str(CLOUD/'thermal_contrast_protocol.json'),c.REMOTE+'/thermal_contrast_protocol.json')
        r=c.access.execute(m,client,f'''import json,hashlib,os,time
from pathlib import Path
root=Path({c.REMOTE!r});p=root/'thermal_contrast_protocol.json';assert hashlib.sha256(p.read_bytes()).hexdigest().upper()=={protocol['cloud_protocol_sha256']!r}
v=json.loads(p.read_text())
for name,expected in v['code_sha256'].items():assert hashlib.sha256((root/'bundle'/name).read_bytes()).hexdigest().upper()==expected,name
q=json.loads((root/'QUEUE.json').read_text());assert not any(j['id']=='T_CONTRAST_F3' for j in q['jobs']);q['jobs'].append(dict(id='T_CONTRAST_F3',worker='generalization_thermal_contrast.py',protocol='thermal_contrast_protocol.json',protocol_sha256={protocol['cloud_protocol_sha256']!r},requires=['output/T_ALL18_R3/DONE.json']))
tmp=root/'QUEUE.json.contrast.partial';tmp.write_text(json.dumps(q,indent=2));os.replace(tmp,root/'QUEUE.json');print(json.dumps(dict(status='F3_CONTRAST_QUEUED_AFTER_PRODUCTION_THERMAL',at_unix=time.time(),queue=q)))''',timeout=60)
    finally:client.close()
    atomic(CLOUD/'QUEUE.json',r.pop('queue'));atomic(RUN/'CLOUD_QUEUED.json',r)
    assert not alive(read(LOCAL/'CONTROLLER.json'));done=read(LOCAL/'QUEUE_DONE.json');assert done['status']=='COMPLETE';q=read(LOCAL/'QUEUE.json');q['concurrency']=1
    q['jobs'].append(dict(id='T_CONTRAST_F0',worker='generalization_thermal_contrast.py',protocol='thermal_contrast_protocol.json',protocol_sha256=protocol['local_protocol_sha256'],requires=[]));atomic(LOCAL/'QUEUE_DONE_BEFORE_CONTRAST.json',done);(LOCAL/'QUEUE_DONE.json').unlink();atomic(LOCAL/'QUEUE.json',q)
    with (LOCAL/'controller_contrast.stdout.log').open('a') as out,(LOCAL/'controller_contrast.stderr.log').open('a') as err:
        proc=subprocess.Popen([sys.executable,'-X','utf8','-B',str(LOCAL/'bundle/native_queue_no_wall.py'),'--root',str(LOCAL)],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
    state=read(ROOT/'PROJECT_STATE.json');state['updated_at_local']=now;state['successor_preparation'].update(status='TWO_MATCHED_HOST_THERMAL_CONTRAST_FITS_REGISTERED_AND_QUEUED',thermal_contrast_protocol=(RUN/'protocol.json').relative_to(ROOT).as_posix(),next='Finish the fixed joint release from unaugmentedfull18V/S/T/I; contrast tests are separate same-host controlled followups and do not change thisCSV. Audit contrastF0/F3 terminal outputs whenready.')
    state['runtime']['local'].update(status='RUNNING_THERMAL_CONTRAST_F0',current_gpu_training=True,execution_protocol='runs/generalization_training_20260910_v1/local/thermal_contrast_protocol.json');atomic(ROOT/'PROJECT_STATE.json',state);print(json.dumps(dict(**fact,checks=qa,cloud=r,local_controller_pid=proc.pid),ensure_ascii=False))

if __name__=='__main__':main()
