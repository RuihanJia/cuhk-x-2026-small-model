"""One bounded CPU audit on cloud pair completion; never relaunches training."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
E=Path(__file__).resolve().parent;C=E.parent;sys.path.insert(0,str(C.parent/'error_resolution_20260913/bundle'))
from dual_host_runtime.io import atomic,read
from dual_host_runtime.health import identity,stop_owned
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
out=E/'review_wait';out.mkdir(exist_ok=True);fd=os.open(str(out/'CLAIM'),os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
reg=read(E/'REVIEW_REGISTRATION.json');atomic(out/'OWNER.json',dict(owner=identity(os.getpid()),registration_sha256=sha(E/'REVIEW_REGISTRATION.json'),deadline_unix=reg['deadline_unix']))
while time.time()<reg['deadline_unix']:
 if (E/'cloud/DOWNLOAD_DONE.json').exists():
  assert not(E/'review/DONE.json').exists()
  with(out/'worker.log').open('ab') as log:
   p=subprocess.Popen([sys.executable,'-X','utf8','-B','-u',str(E/'review.py')],cwd=C,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
  owner=identity(p.pid);atomic(out/'ACTIVATED.json',dict(worker=owner,at_unix=time.time()));end=min(reg['deadline_unix'],time.time()+reg['wall_cap_seconds'])
  while p.poll() is None and time.time()<end:time.sleep(2)
  if p.poll() is None:stop_owned(owner);raise TimeoutError('CPU_AUDIT_CAP')
  assert p.returncode==0,'REVIEW_FAILED:'+str(p.returncode)
  atomic(out/'DONE.json',dict(status='COMPLETE',terminal_sha256=sha(E/'review/DONE.json'),at_unix=time.time()));break
 time.sleep(min(30,max(0,reg['deadline_unix']-time.time())))
else:atomic(out/'EXPIRED.json',dict(no_relaunch=True,at_unix=time.time()))
