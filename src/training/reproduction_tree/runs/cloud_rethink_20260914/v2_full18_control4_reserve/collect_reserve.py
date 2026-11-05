"""Finite artifact-only collector, 45s unchanged-state cadence, no GPU calls."""
from pathlib import Path
import os,sys,json,hashlib,time,traceback
E=Path(__file__).resolve().parent;C=E.parent;O=C.parent/'error_resolution_20260913'
sys.path[:0]=[str(C),str(E),str(O/'bundle')]
import resilient_io;resilient_io.install()
from dual_host_runtime.io import atomic,read,retry
from dual_host_runtime.health import identity
import post_aux_cloud_access as a
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 w=E/'collector';w.mkdir(exist_ok=True);fd=os.open(str(w/'CLAIM'),os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
 pp=E/'protocol.cloud.json';s=read(pp);ph=sha(pp);remote=s['hosts']['cloud']['output_directory'];deadline=s['deadline_unix']+180
 atomic(w/'OWNER.json',dict(owner=identity(os.getpid()),protocol_sha256=ph,deadline_unix=deadline))
 query='from pathlib import Path\nimport json,hashlib,psutil\np=Path('+repr(remote)+')\nsha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()\ndef alive(v):\n try:return abs(psutil.Process(v["pid"]).create_time()-v["create_time"])<.01\n except psutil.NoSuchProcess:return False\nnames=["CONTROLLER_DONE.json","CONTROLLER_ERROR.json","ERROR.json","TELEMETRY.json"]\nsnap={n:json.loads((p/n).read_text()) for n in names if(p/n).exists()}\nif "CONTROLLER_DONE.json" in snap:\n d=json.loads((p/"DONE.json").read_text());ct=json.loads((p/"CONTROLLER.json").read_text());snap["owned_absent"]=not alive(ct["worker"])and not alive(ct["controller"]);snap["DONE.json"]=d;snap["terminal_sha256"]=sha(p/"DONE.json");files=dict(d["artifacts"])\n for n in ["DONE.json","CONTROLLER_DONE.json","CONTROLLER.json","BUDGET.json","ACTIVATED.json"]:files[n]=sha(p/n)\n snap["files"]={n:dict(sha256=h,bytes=(p/n).stat().st_size)for n,h in files.items()}\nprint(json.dumps(snap))'
 try:
  while time.time()<deadline:
   try:snap=a.run(query,timeout=30)
   except Exception as error:
    atomic(w/'CONNECTION_TRANSIENT.json',dict(error=str(error),at_unix=time.time()));time.sleep(45);continue
   atomic(w/'SNAPSHOT.json',dict(snapshot=snap,at_unix=time.time()))
   assert 'CONTROLLER_ERROR.json'not in snap and 'ERROR.json'not in snap,snap
   if 'CONTROLLER_DONE.json'in snap and snap['owned_absent']:
    ct=snap['CONTROLLER_DONE.json'];d=snap['DONE.json']
    assert ct['protocol_sha256']==d['protocol_sha256']==ph and ct['status']=='COMPLETE' and ct['exit_code']==0
    assert d['status']=='COMPLETE_FULL18_CONTROL4_AND_FIXED_AVERAGE_RESERVE' and ct['terminal_sha256']==snap['terminal_sha256']
    mod,client=a.connect()
    try:
     ftp=client.open_sftp();local=E/'cloud';local.mkdir(exist_ok=True)
     # Publish controller terminal last, after every hashed per-job artifact.
     ordered=sorted([n for n in snap['files'] if not n.endswith('.pt') or n=='average_model_int8.pt'],key=lambda n:n=='CONTROLLER_DONE.json')
     for n in ordered:
      p=(local/n).resolve();assert p.is_relative_to(local.resolve());p.parent.mkdir(parents=True,exist_ok=True);expected=snap['files'][n]
      if p.exists():assert sha(p)==expected['sha256'];continue
      tmp=Path(str(p)+'.partial');ftp.get(remote+'/'+n,str(tmp));assert tmp.stat().st_size==expected['bytes'] and sha(tmp)==expected['sha256'];retry(lambda:os.replace(tmp,p))
    finally:client.close()
    atomic(local/'REMOTE_MODEL_BINDINGS.json',dict(models={n:dict(**v,remote=remote+'/'+n) for n,v in snap['files'].items() if n.endswith('.pt') and n!='average_model_int8.pt'},checkpoint_downloaded='average_model_int8.pt',remote_owned_absent=True,protocol_sha256=ph));atomic(w/'DONE.json',dict(status='COMPLETE_CLOUD_FULL18_RESERVE_COLLECTION',protocol_sha256=ph,files=snap['files'],owned_absent=True,remote_terminal_sha256=snap['terminal_sha256'],at_unix=time.time()));return
   time.sleep(45)
  raise TimeoutError('FINITE_COLLECTION_DEADLINE')
 except BaseException as error:
  atomic(w/'ERROR.json',dict(error=str(error),traceback=traceback.format_exc(),protocol_sha256=ph,at_unix=time.time()));raise
if __name__=='__main__':main()
