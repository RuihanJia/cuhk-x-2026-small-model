"""Activate once on complete actual quantization and parent downloads; bounded CPU."""
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
