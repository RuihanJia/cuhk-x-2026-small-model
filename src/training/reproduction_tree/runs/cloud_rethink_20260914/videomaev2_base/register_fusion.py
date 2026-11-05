"""Freeze exactly the source-only global/classwise TI/Base comparison."""
from pathlib import Path
import ast,copy,hashlib,json,os,shutil,subprocess,sys,time
W=Path(__file__).resolve().parent;C=W.parent;R=C.parents[1];Q=C/'videomaev2_base_quant';S=C/'videomaev2_base_source';E=C/'videomaev2_base_fusion';E.mkdir(exist_ok=True)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert not(E/'protocol.json').exists();assert (Q/'LAUNCH.json').exists()
w=(C/'videomae_base_fusion/worker.py').read_text(encoding='utf-8').replace('videomae_base_quant','videomaev2_base_quant')
w=w.replace("['videomae_base_source4','videomae_base_source_rest','videomae_base_kinetics']","['videomaev2_base','videomaev2_base_source']")
w=w.replace("qroot=C/'videomaev2_base_quant/cloud'/j['id']", "qroot=C/'videomaev2_base_quant'/('cloud' if j['host']=='cloud' else 'output/local')/j['id']")
w=w.replace("reference['classwise7']", "reference['global7']").replace(".sum()==884", ".sum()==900")
w=w.replace('a.near(globalp,fm.predict(data,g))','a.near(globalp,convex_torch(data,g))')
w=w.replace('COMPLETE_BASE_TI_INT8_SOURCE_ONLY_FUSION','COMPLETE_V2_BASE_TI_INT8_SOURCE_ONLY_FUSION').replace("event='BASE_TI_INT8_SOURCE_ONLY_FUSION_COMPLETE'", "event='V2_BASE_TI_INT8_SOURCE_ONLY_FUSION_COMPLETE'")
old=(C/'videomae_base_fusion_final/replay_saved.py').read_text(encoding='utf-8');helper=old[old.index('def convex_torch'):old.index('def main():')];w=w.replace('def main():',helper+'def main():')
(E/'worker.py').write_text(w,encoding='utf-8');shutil.copyfile(C/'videomae_base_fusion/classwise3.py',E/'classwise3.py')
activate='''"""Activate once on complete actual quantization and parent downloads; bounded CPU."""
from pathlib import Path
import hashlib,os,subprocess,sys,time
E=Path(__file__).resolve().parent;C=E.parent
sys.path.insert(0,str(C.parent/'error_resolution_20260913/bundle'))
from dual_host_runtime.io import atomic,read
from dual_host_runtime.health import identity,stop_owned
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=read(E/'protocol.json');ph=sha(E/'protocol.json');out=E/'wait';out.mkdir(exist_ok=True);fd=os.open(str(out/'CLAIM'),os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
atomic(out/'OWNER.json',dict(owner=identity(os.getpid()),protocol_sha256=ph,deadline_unix=s['activation_deadline_unix']))
required=[C/'videomaev2_base/cloud/DOWNLOAD_DONE.json',C/'videomaev2_base/output/local/CONTROLLER_DONE.json',C/'videomaev2_base_source/cloud/DOWNLOAD_DONE.json',C/'videomaev2_base_quant/cloud/DOWNLOAD_DONE.json',C/'videomaev2_base_quant/output/local/CONTROLLER_DONE.json']
while time.time()<s['activation_deadline_unix']:
 if all(p.exists() for p in required):
  assert not(E/'output/DONE.json').exists()
  with(out/'worker.log').open('ab') as log:p=subprocess.Popen([sys.executable,'-X','utf8','-B','-u',str(E/'worker.py')],cwd=C,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
  owner=identity(p.pid);atomic(out/'ACTIVATED.json',dict(worker=owner,at_unix=time.time()));end=min(s['activation_deadline_unix'],time.time()+s['cpu_fit_cap_seconds'])
  while p.poll() is None and time.time()<end:time.sleep(2)
  if p.poll() is None:stop_owned(owner);raise TimeoutError('CPU_FUSION_CAP')
  if p.returncode!=0:atomic(out/'ERROR.json',dict(error='WORKER_EXIT:'+str(p.returncode),at_unix=time.time()));raise RuntimeError('SOURCE_FUSION_FAILED_REVIEW_REQUIRED')
  atomic(out/'DONE.json',dict(status='COMPLETE',terminal_sha256=sha(E/'output/DONE.json'),at_unix=time.time()));break
 time.sleep(min(30,max(0,s['activation_deadline_unix']-time.time())))
else:atomic(out/'EXPIRED.json',dict(no_relaunch=True,at_unix=time.time()))
'''
(E/'activate.py').write_text(activate,encoding='utf-8')
for n in ['worker.py','classwise3.py','activate.py']:ast.parse((E/n).read_text(encoding='utf-8'))
# Exercise the exact previously failing interface with retained source-fitted weights, no fit or data labels.
import importlib.util
sp=importlib.util.spec_from_file_location('v2base_fusion_interface_qa',E/'worker.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);m.torch.set_num_threads(1)
g=read(C/'videomae_base_fusion_final/output/F0_global.json');rng=m.np.random.RandomState(913);data=dict(logits=rng.normal(size=(9,3,40)).astype(m.np.float32),available=m.np.ones((9,3),bool));data['available'][0,0]=False
x=m.m.convex_replay(data,g);y=m.convex_torch(data,g);m.a.near(x,y)
qa=dict(status='PASS_SOURCE_ONLY_THREE_BRANCH_NUMPY_TORCH_INTERFACE',no_fit=True,no_data_labels=True,no_Test=True,no_GPU=True,rows=9,branches=3,max_abs=float(m.np.max(m.np.abs(x-y))),missing_unused_depth_ir_supported=True,worker_sha256=sha(E/'worker.py'),old_source_calibrator_sha256=sha(C/'videomae_base_fusion_final/output/F0_global.json'),at_unix=time.time());write(E/'INTERFACE_QA.json',qa)
s=read(C/'videomae_base_fusion/protocol.json');now=time.time();s.update(id='VIDEOMAEV2_BASE_TI_FIXED50_SOURCE_GLOBAL_CLASSWISE',registered_at_unix=now,activation_deadline_unix=read(Q/'protocol.json')['deadline_unix']+300,cpu_fit_cap_seconds=240,hypothesis='V2Base raw918 and positive paired errors justify fixed source-only TI/Base fusion after all eight actual50INT8 replays. Compare the same global convex500 and classwise penalty0.1 procedure against frozen V2global900. No new penalty, temperature, branch or class scans.',quant_protocol_sha256=sha(Q/'protocol.json'),parent_protocols={k:sha(C/k/'protocol.json') for k in ['videomaev2_base','videomaev2_base_source']},current_reference=str(C/'videomaev2_fusion/output/PAIRED.npz'),current_reference_sha256=sha(C/'videomaev2_fusion/output/PAIRED.npz'),all_four_calibrators_before_outer_arrays=True,readout_note='Current raw outer evidence already observed before this finite source-validation registration; all four calibration states frozen before new quantized outer arrays are read. Adaptive development, not new independent confirmation.',interface_qa_sha256=sha(E/'INTERFACE_QA.json'))
s['code_sha256']={k:v for k,v in s['code_sha256'].items() if not k.startswith('videomae_base_fusion/')}
for n in ['worker.py','classwise3.py','activate.py']:s['code_sha256'][(E/n).relative_to(C).as_posix()]=sha(E/n)
for k,v in s['code_sha256'].items():assert sha(C/k)==v,k
write(E/'protocol.json',s)
with(R/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(event='V2_BASE_TI_SOURCE_FUSION_REGISTERED_WITH_INTERFACE_QA',at_unix=now,protocol=str((E/'protocol.json').relative_to(R)),protocol_sha256=sha(E/'protocol.json'),cpu_cap_seconds=240,activation_deadline_unix=s['activation_deadline_unix'],no_Test=True))+'\n')
with(E/'launcher.log').open('ab') as log:p=subprocess.Popen([sys.executable,'-X','utf8','-B','-u',str(E/'activate.py')],cwd=C,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
from dual_host_runtime.health import identity
write(E/'LAUNCH.json',dict(status='WAITING_FOR_ALL_FROZEN_QUANT_AND_SOURCE_OUTPUTS',worker=identity(p.pid),protocol_sha256=sha(E/'protocol.json'),at_unix=time.time()))
print(json.dumps(dict(protocol_sha256=sha(E/'protocol.json'),activation_deadline_unix=s['activation_deadline_unix'],interface_QA=qa,wait_pid=p.pid)))
