"""Move one not-yet-started Thermal outer fit to the cloud's existing queue."""
from pathlib import Path
from datetime import datetime
import copy,json,sys,time
import generalization_cloud as cloud
ROOT=cloud.ROOT;BASE=cloud.BASE;L=BASE/'local';C=BASE/'cloud';REMOTE=cloud.REMOTE
def read(p):return cloud.read(p)
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.partial');cloud.write(tmp,v);tmp.replace(p)
def main():
    job='T_F3_outer';ls=read(L/'QUEUE_STATUS.json');assert job not in ls['active'] and job not in ls['finished'] and not (L/'output'/job).exists();p=copy.deepcopy(read(L/'thermal_io2_protocol.json'));now=datetime.now().astimezone().isoformat(timespec='seconds');original=cloud.access.sha(L/'thermal_io2_protocol.json');p['id']='GENERALIZATION-THERMAL-F3-OUTER-CLOUD-HANDOFF-20260910-V1';p['at_local']=now;p['jobs']={job:p['jobs'][job]};p['data']['root']=REMOTE+'/thermal_data';p['dependencies']['root']='/home/featurize/work/cuhkx_multimodal_20260907/source_retry1/runs/_x3d_deps';p['budget']=dict(deadline_unix=read(C/'QUEUE.json')['deadline_unix'],existing_cloud_phase_cap_seconds=21600,global_cloud_concurrency=2,local_phase_cap_seconds=10800,budget_extension=False);p['code_sha256']=dict(read(C/'protocol.json')['code_sha256'])
    for name in ['generalization_thermal_worker_io2.py','threaded_data.py']:
        dst=C/'bundle'/name;dst.write_bytes((L/'bundle'/name).read_bytes());p['code_sha256'][name]=cloud.access.sha(dst)
    p['execution_delta']=dict(parent_local_execution_sha256=original,logical_job_id=job,training_already_started=False,reason='Measured local two-job throughput projects beyond the existing3hourcap if all8Thermal fits remain local. Transfer the unstartedF3outer fit; retain all other local jobs.',same_seed_batch_epochs_input=True,source_dataset_transfer='THERMAL_DATA_READY.json must exist before worker launch',limitation='Same scientific recipe on a different host/runtime; no bitwise retraining claim.')
    protocol=C/'thermal_handoff_protocol.json';assert not protocol.exists();write(protocol,p);ph=cloud.access.sha(protocol);m,c=cloud.access.connection()
    try:
        code=f'''import json,hashlib,shutil,sys,site
from pathlib import Path
root=Path({REMOTE!r});deps=Path({p['dependencies']['root']!r})
for rel,expected in {p['dependencies']['source_sha256']!r}.items():
 assert hashlib.sha256((deps/rel).read_bytes()).hexdigest().upper()==expected,rel
source=Path('/home/featurize/work/cuhkx_multimodal_20260907/source_retry1/runs/_torch_cache/hub/checkpoints/X3D_M.pyth');dest=root/'bundle/assets/X3D_M.pyth'
if source.exists():
 assert hashlib.sha256(source.read_bytes()).hexdigest().upper()=={p['weights']['sha256']!r};shutil.copy2(source,dest)
sys.path.insert(0,str(deps));site.addsitedir(str(deps));import torch;from torch import nn
from pytorchvideo.models.hub import x3d_m
model=x3d_m(pretrained=False);model.blocks[-1].proj=nn.Linear(model.blocks[-1].proj.in_features,40);model.blocks[-1].activation=None;assert sum(v.numel() for v in model.parameters())==3056634
print(json.dumps(dict(dependencies_verified=True,model_CPU_build=True,weight_reused=dest.exists())))'''
        preflight=cloud.access.execute(m,c,code,timeout=60)
        with c.open_sftp() as ftp:
            for name in ['generalization_thermal_worker_io2.py','threaded_data.py']:ftp.put(str(C/'bundle'/name),REMOTE+'/bundle/'+name,confirm=True)
            ftp.put(str(L/'bundle/data/thermal_index.json'),REMOTE+'/bundle/data/thermal_index.json',confirm=True);ftp.put(str(protocol),REMOTE+'/thermal_handoff_protocol.json',confirm=True)
            if not preflight['weight_reused']:ftp.put(str(L/'bundle/assets/X3D_M.pyth'),REMOTE+'/bundle/assets/X3D_M.pyth',confirm=True)
        # Remove the unstarted local job before adding its cloud counterpart; active work is untouched.
        ls=read(L/'QUEUE_STATUS.json');assert job not in ls['active'] and job not in ls['finished'];lq=read(L/'QUEUE.json');lq['jobs']=[v for v in lq['jobs'] if v['id']!=job];write(L/'QUEUE.json',lq)
        q=read(C/'QUEUE.json');assert job not in [v['id'] for v in q['jobs']];q['jobs'].append(dict(id=job,worker='generalization_thermal_worker_io2.py',protocol='thermal_handoff_protocol.json',protocol_sha256=ph,requires=['THERMAL_DATA_READY.json']));write(C/'QUEUE.json',q)
        with c.open_sftp() as ftp:ftp.put(str(C/'QUEUE.json'),REMOTE+'/QUEUE.updated.json',confirm=True)
        cloud.access.execute(m,c,f'''from pathlib import Path
import json,hashlib,os
p=Path({REMOTE!r});assert hashlib.sha256((p/'thermal_handoff_protocol.json').read_bytes()).hexdigest().upper()=={ph!r};assert hashlib.sha256((p/'bundle/assets/X3D_M.pyth').read_bytes()).hexdigest().upper()=={p['weights']['sha256']!r};os.replace(p/'QUEUE.updated.json',p/'QUEUE.json');print(json.dumps(dict(status='ONE_UNSTARTED_THERMAL_FIT_QUEUED_WITH_DATA_READY_GATE')))''')
    finally:c.close()
    fact=dict(event='E-GENERALIZATION-THERMAL-F3-OUTER-HANDOFF-20260910-V1',at_local=now,status='PREPARED_AND_QUEUED_WAITING_FOR_EXISTING_DATA_TRANSFER',logical_job_id=job,local_fits_remaining_total=7,cloud_additional_fits=1,protocol=protocol.relative_to(ROOT).as_posix(),protocol_sha256=ph,preflight=preflight,science_changes=False,budget_extension=False,already_started_jobs_moved=0)
    write(BASE/'THERMAL_HANDOFF.json',fact)
    with (ROOT/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(fact,ensure_ascii=False)+'\n')
    print(json.dumps(fact,ensure_ascii=False))
if __name__=='__main__':main()
