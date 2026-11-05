"""Upload the frozen offline release and run a bounded cloud replay."""
from pathlib import Path
import sys,json,time,hashlib
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R));import cloud_transport as t
def main():
 archive=R/'pose_dual_release_frozen.zip';h=hashlib.sha256(archive.read_bytes()).hexdigest();m,c=t.connect()
 try:
  with c.open_sftp() as ftp:ftp.put(str(archive),t.REMOTE+'/pose_dual_release_frozen.zip')
  code='from pathlib import Path\nimport json,hashlib,zipfile,time,subprocess\nr=Path('+repr(t.REMOTE)+');z=r/"pose_dual_release_frozen.zip";assert hashlib.sha256(z.read_bytes()).hexdigest()=='+repr(h)+'\n'
  code+='d=r/"pose_dual_release";assert not d.exists();d.mkdir()\nwith zipfile.ZipFile(z) as archive:\n for n in archive.namelist():\n  dest=(d/n).resolve();assert dest.is_relative_to(d.resolve());archive.extract(n,d)\n'
  code+='s=json.loads((d/"INFERENCE_MANIFEST.json").read_text())\nfor n,q in s["files"].items():assert hashlib.sha256((d/n).read_bytes()).hexdigest()==q["sha256"],n\n'
  worker="from pathlib import Path\nimport sys,json,time,traceback\nR=Path(__file__).resolve().parent;D=R/'pose_dual_release';sys.path.insert(0,str(D));sys.path.insert(1,str(R/'bundle'));sys.path.append(str(R/'pose_site'))\nfrom dual_host_runtime.health import Guard\nfrom dual_host_runtime.io import atomic\nimport infer\nout=R/'release_cloud_output';out.mkdir(exist_ok=True)\ntry:\n guard=Guard(out,1789323192.0883653,out/'TELEMETRY.json');a=infer.infer("+repr(t.DATA)+",D/'cloud_submission.csv',guard,workers=8,video_batch=8);atomic(out/'DONE.json',dict(status='COMPLETE',at_unix=time.time(),audit=a))\nexcept BaseException as e:atomic(out/'ERROR.json',dict(error=str(e),traceback=traceback.format_exc(),at_unix=time.time()));raise\n"
  controller=(R/'pose_adapter_controller.py').read_text().replace('POSE_ADAPTER_PROTOCOL.json','RELEASE_PROTOCOL.json').replace("'pose_adapter_output/cloud'","'release_cloud_output'").replace("str(R/'pose_adapter_worker.py'),'--host','cloud'","str(R/'release_cloud_worker.py')")
  code+='(r/"release_cloud_worker.py").write_text('+repr(worker)+');(r/"release_cloud_controller.py").write_text('+repr(controller)+')\n'
  code+='(r/"RELEASE_PROTOCOL.json").write_text((d/"RELEASE_PROTOCOL.json").read_text());out=r/"release_cloud_output";out.mkdir(exist_ok=True);p=subprocess.Popen(['+repr(t.PYTHON)+',"-B","-u",str(r/"release_cloud_controller.py")],cwd=r,stdout=(out/"launcher.log").open("a"),stderr=subprocess.STDOUT,start_new_session=True);print(json.dumps(dict(pid=p.pid,at_unix=time.time(),archive_sha256='+repr(h)+',files_verified=len(s["files"]))))'
  res=t.old.execute(m,c,code,50);(R/'RELEASE_CLOUD_LAUNCH.json').write_text(json.dumps(res,indent=2));print(json.dumps(res))
 finally:c.close()
if __name__=='__main__':main()
