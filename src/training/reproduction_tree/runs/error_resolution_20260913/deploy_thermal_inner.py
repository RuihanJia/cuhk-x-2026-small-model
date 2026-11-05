from pathlib import Path
import json,zipfile,hashlib
import cloud_transport as t
R=Path(__file__).resolve().parent;s=json.loads((R/'THERMAL_INNER_PROTOCOL.json').read_text());z=R/'thermal_inner_deploy.zip'
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as f:
 for n in ['THERMAL_INNER_PROTOCOL.json',*s['code_sha256'],*s['asset_sha256']]:f.write(R/n,n)
h=hashlib.sha256(z.read_bytes()).hexdigest();mod,c=t.connect()
try:
 with c.open_sftp() as sf:sf.put(str(z),t.REMOTE+'/'+z.name,confirm=True)
 code='''from pathlib import Path
import json,zipfile,hashlib,subprocess,time
r=Path(%r);z=r/'thermal_inner_deploy.zip';assert hashlib.sha256(z.read_bytes()).hexdigest()==%r
out=r/'thermal_inner_output/cloud';out.mkdir(parents=True,exist_ok=True);assert not(out/'WAITING_CONTROLLER.json').exists()
with zipfile.ZipFile(z) as f:
 for n in f.namelist():
  dest=(r/n).resolve();assert dest.is_relative_to(r.resolve());b=f.read(n)
  if dest.exists():assert dest.read_bytes()==b,n
  else:dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
s=json.loads((r/'THERMAL_INNER_PROTOCOL.json').read_text());assert time.time()<s['deadline_unix']
for n,h in {**s['code_sha256'],**s['asset_sha256']}.items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h
bridge=r/'thermal_inner_wait_bridge.py'
bc="""from pathlib import Path
import json,time,subprocess
r=Path(%r);s=json.loads((r/'THERMAL_INNER_PROTOCOL.json').read_text());prior=r/'crop_inner_output';out=r/'thermal_inner_output/cloud'
try:
 while not(prior/'CONTROLLER_DONE.json').exists() or not(r/'thermal_dataset/READY.json').exists():
  assert time.time()<s['deadline_unix']
  assert not(out/'STOP_REQUESTED').exists()
  assert not(prior/'ERROR.json').exists() and not(prior/'CONTROLLER_ERROR.json').exists()
  time.sleep(5)
 assert json.loads((prior/'CONTROLLER_DONE.json').read_text())['exit_code']==0
 assert json.loads((r/'thermal_dataset/READY.json').read_text())['Test_files']==0
 subprocess.run([%r,'-B','-u',str(r/'thermal_inner_cloud_controller.py')],cwd=r,check=True)
except BaseException as e:
 (out/'BRIDGE_ERROR.json').write_text(json.dumps(dict(error=str(e),at_unix=time.time())));raise
"""
assert not bridge.exists();bridge.write_text(bc)
p=subprocess.Popen([%r,'-B','-u',str(bridge)],cwd=r,stdout=(out/'bridge.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True)
result=dict(status='QUEUED_WAIT_CROP_AND_VERIFIED_THERMAL_CACHE',pid=p.pid,at_unix=time.time(),protocol_sha256=hashlib.sha256((r/'THERMAL_INNER_PROTOCOL.json').read_bytes()).hexdigest())
(out/'WAITING_CONTROLLER.json').write_text(json.dumps(result));print(json.dumps(result))
'''%(t.REMOTE,h,t.REMOTE,t.PYTHON,t.PYTHON)
 result=t.old.execute(mod,c,code,50);(R/'THERMAL_INNER_CLOUD_LAUNCH.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
finally:c.close()
