"""Budget-expired local recovery: preserve scientific state and all old artifacts."""
from pathlib import Path
import ast,copy,hashlib,json,os,subprocess,sys,time
E=Path(__file__).resolve().parent;C=E.parent;OLD=C/'v2_window_train';ROOT=C.parents[1]
sys.path.insert(0,str(C.parent/'error_resolution_20260913/bundle'))
from dual_host_runtime.io import atomic,read,save_torch
from dual_host_runtime.health import identity,alive,stop_owned
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not(E/'protocol.json').exists();old=read(OLD/'protocol.json');oldph=sha(OLD/'protocol.json');o=OLD/'output/local';own=read(o/'CONTROLLER.json');assert not alive(own['worker']) and not alive(own['controller']);assert read(o/'ERROR.json')['error']=='FROZEN_WALL_CAP_INCOMPLETE'
for name in ['worker.py','data.py','temporal_data.py','codec.py','resilient_io.py']:(E/name).write_bytes((OLD/name).read_bytes())
control=(OLD/'controller.py').read_text(encoding='utf-8');needle="   if (prior/'CONTROLLER_ERROR.json').exists():raise RuntimeError('PREDECESSOR_FAILED_REVIEW_REQUIRED')"
assert needle in control
control=control.replace(needle,"""   if (prior/'CONTROLLER_ERROR.json').exists():
    assert local and predecessor['allow_budget_failed'] and sha(prior/'CONTROLLER_ERROR.json')==predecessor['error_sha256']
    ct=read(prior/'CONTROLLER.json');assert not alive(ct['worker']) and not alive(ct['controller']);break""")
(E/'controller.py').write_text(control,encoding='utf-8')
launch=(OLD/'launch.py').read_text(encoding='utf-8').replace('v2_window_train','v2_window_resume').replace('WINDOW_MATCHED4_DUAL_HOST_RUNNING','WINDOW_RESUME_DUAL_HOST_RUNNING');launch=launch.replace(" v=NVML().sample();assert decision(v,False,True) in ('RUN','WARNING')"," # Local GPU thermal control belongs to its separate already-running controller.")
start=launch.index(" with(E/'local_controller.log').open('ab') as log:");end=launch.index("value=dict(status='DUAL_HOST_DISPATCHED'",start);launch=launch[:start]+" local=read(E/'LOCAL_DISPATCH.json')['owner'];"+launch[end:];(E/'launch.py').write_text(launch,encoding='utf-8')
watch=(OLD/'watch.py').read_text(encoding='utf-8').replace('v2_window_train','v2_window_resume');(E/'watch.py').write_text(watch,encoding='utf-8')
s=copy.deepcopy(old);s.update(id='V2_WINDOW_MATCHED4_BUDGET_RECOVERY',registered_at_unix=time.time(),deadline_unix=time.time()+2820,prior_scientific_protocol_sha256=oldph,engineering_reason='Original10minute local per-job budget reached during epoch4. Three complete epochs saved. Preserve old failure; migrate exact resume state for sole started job, continue original scientific4epoch endpoints under a separately registered finite incremental budget. No scientific hyperparameter change.',resource_note='New incremental local45min/cloud30min caps. Previous failed run remains cost and evidence; original deadline not edited. No extra scientific epochs.')
for host,h in s['hosts'].items():
 h['output_directory']=h['output_directory'].replace('v2_window_train','v2_window_resume');h['job_wall_cap_seconds']=1200
 if host=='local':h['predecessor']=dict(directory=str(o),protocol_sha256=oldph,allow_budget_failed=True,error_sha256=sha(o/'CONTROLLER_ERROR.json'),success_only=False)
src=o/'V2_BASE_F3_CONTROL/resume.pt';s['resume_migration']=dict(job='V2_BASE_F3_CONTROL',source_path=str(src),source_sha256=sha(src),old_protocol_sha256=oldph,completed_epochs=3,only_mutated_metadata_key='protocol_sha256',all_tensor_optimizer_rng_history_values_preserved=True)
s['code_sha256']={k:v for k,v in old['code_sha256'].items()if not k.startswith('v2_window_train/')}
for n in ['worker.py','data.py','temporal_data.py','codec.py','resilient_io.py','controller.py','launch.py','watch.py']:ast.parse((E/n).read_text(encoding='utf-8'));s['code_sha256']['v2_window_resume/'+n]=sha(E/n)
s['worker_sha256']=sha(E/'worker.py');s['controller_sha256']=sha(E/'controller.py');atomic(E/'protocol.json',s);ph=sha(E/'protocol.json')
import torch
torch.set_num_threads(2);saved=torch.load(src,map_location='cpu',weights_only=True);assert saved['protocol_sha256']==oldph and saved['epoch']==3 and saved['updates']==441 and len(saved['history'])==3
dst=E/'output/local/V2_BASE_F3_CONTROL';dst.mkdir(parents=True,exist_ok=True);saved['protocol_sha256']=ph;save_torch(dst/'resume.pt',saved);check=torch.load(dst/'resume.pt',map_location='cpu',weights_only=True)
def exact(a,b):
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor)and a.dtype==b.dtype and torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys()and all(exact(a[k],b[k])for k in a)
 if isinstance(a,(list,tuple)):return type(a)==type(b)and len(a)==len(b)and all(exact(x,y)for x,y in zip(a,b))
 return a==b
assert exact(saved,check);assert sha(src)==s['resume_migration']['source_sha256'];atomic(E/'RESUME_MIGRATION.json',dict(status='PASS_EXACT_STATE_MIGRATION',protocol_sha256=ph,old_resume_sha256=sha(src),new_resume_sha256=sha(dst/'resume.pt'),model_optimizer_rng_history_identical=True,original_unchanged=True,epochs=3,updates=441,at_unix=time.time()));del saved,check
names=['worker.py','data.py','temporal_data.py','codec.py','resilient_io.py','controller.py','launch.py','watch.py','protocol.json'];remote='/home/featurize/cuhkx_motion_transfer_20260911_v1/cloud_rethink_20260914/v2_window_resume';atomic(E/'TRANSFER_MANIFEST.json',dict(remote_directory=remote,source_authorization_sha256=s['source_authorization_sha256'],files=[dict(name=n,bytes=(E/n).stat().st_size,sha256=sha(E/n))for n in names]))
p=C/'post_aux_cloud_access.py';text=p.read_text(encoding='utf-8');assert "'v2_window_resume/protocol.json'"not in text;text=text.replace("'v2_window_train/protocol.json']","'v2_window_train/protocol.json','v2_window_resume/protocol.json']");p.write_text(text,encoding='utf-8')
for k,h in s['hosts']['local']['input_bindings'].items():assert sha(k)==h,k
with(E/'local_controller.log').open('ab')as log:child=subprocess.Popen([sys.executable,'-X','utf8','-B','-u',str(E/'controller.py'),'--protocol',str(E/'protocol.json'),'--host','local'],cwd=C,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
atomic(E/'LOCAL_DISPATCH.json',dict(protocol_sha256=ph,owner=identity(child.pid),at_unix=time.time()))
prior=C/'v2_window_fusion/wait/OWNER.json'
if prior.exists():
 owner=read(prior)['owner']
 if alive(owner):stop_owned(owner)
atomic(C/'v2_window_fusion/SUPERSEDED_ENGINEERING.json',dict(reason='Original local stage exhausted per-job budget. No fusion fit began; replacement watches explicit checkpoint continuation.',at_unix=time.time(),replacement='v2_window_resume_fusion'))
st=read(ROOT/'PROJECT_STATE.json');st['phase']='WINDOW_CHECKPOINT_RESUME_LOCAL_RUNNING_CLOUD_UNAVAILABLE';st['active_runs']=[dict(host='local',stage='v2_window_resume',protocol_sha256=ph,dispatch=read(E/'LOCAL_DISPATCH.json'))];st['current_authorization']['active_protocols']=[str((E/'protocol.json').relative_to(ROOT))];st['current_authorization']['deadline_unix']=s['deadline_unix'];atomic(ROOT/'PROJECT_STATE.json',st)
with(ROOT/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf-8')as f:f.write(json.dumps(dict(event='WINDOW_BUDGET_RECOVERY_STARTED',protocol_sha256=ph,prior_protocol_sha256=oldph,resume_epochs=3,no_new_scientific_epochs=True,local_incremental_cap=2700,cloud_incremental_cap=1800,at_unix=time.time()))+'\n')
print(json.dumps(dict(status='EXACT_RESUME_LOCAL_DISPATCHED',protocol_sha256=ph,owner=identity(child.pid),deadline_unix=s['deadline_unix'])))
