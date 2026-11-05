from pathlib import Path
import json,zipfile,hashlib
import cloud_transport as t
R=Path(__file__).resolve().parent;spec=json.loads((R/'IR_ADAPTER_PROTOCOL.json').read_text());files=['IR_ADAPTER_PROTOCOL.json',*spec['code_sha256'],*spec['asset_sha256']];z=R/'ir_adapter_deploy.zip'
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as f:
 for n in files:f.write(R/n,n)
h=hashlib.sha256(z.read_bytes()).hexdigest();mod,c=t.connect()
try:
 with c.open_sftp() as s:s.put(str(z),t.REMOTE+'/'+z.name,confirm=True)
 code='''from pathlib import Path
import json,zipfile,hashlib,subprocess,time
r=Path(%r);z=r/'ir_adapter_deploy.zip';assert hashlib.sha256(z.read_bytes()).hexdigest()==%r
out=r/'ir_adapter_output/cloud';assert not(out/'CONTROLLER.json').exists();out.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(z) as f:
 for n in f.namelist():
  dest=(r/n).resolve();assert dest.is_relative_to(r.resolve());payload=f.read(n)
  if dest.exists():assert dest.read_bytes()==payload,'EXISTING_DEPENDENCY_CHANGED:'+n
  else:dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(payload)
s=json.loads((r/'IR_ADAPTER_PROTOCOL.json').read_text());assert time.time()<s['deadline_unix']
for n,h in {**s['code_sha256'],**s['asset_sha256']}.items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h
for j in s['jobs']:
 assert hashlib.sha256(Path(j['source']['cloud']).read_bytes()).hexdigest()==j['source_sha256']
 assert hashlib.sha256(Path(j['phase_source']).read_bytes()).hexdigest()==j['phase_source_sha256']
bridge=r/'ir_adapter_waiting_bridge.py'
bridge_code="""from pathlib import Path
import json,time,subprocess
r=Path(%r)
spec=json.loads((r/'IR_ADAPTER_PROTOCOL.json').read_text())
old=r/'video_conflict_output';out=r/'ir_adapter_output/cloud'
while not(old/'CONTROLLER_DONE.json').exists():
 assert time.time()<spec['deadline_unix']
 assert not(old/'ERROR.json').exists() and not(old/'CONTROLLER_ERROR.json').exists()
 assert not(out/'STOP_REQUESTED').exists()
 time.sleep(5)
assert json.loads((old/'CONTROLLER_DONE.json').read_text())['exit_code']==0 and(old/'QUEUE_DONE.json').exists()
subprocess.run([%r,'-B','-u',str(r/'ir_adapter_controller.py')],cwd=r,check=True)
"""
assert not bridge.exists();bridge.write_text(bridge_code)
p=subprocess.Popen([%r,'-B','-u',str(bridge)],cwd=r,stdout=(out/'bridge.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(status='QUEUED_WAITING_FOR_ORIGINAL_VIDEO_QUEUE',pid=p.pid,at_unix=time.time(),protocol_sha256=hashlib.sha256((r/'IR_ADAPTER_PROTOCOL.json').read_bytes()).hexdigest())
(out/'WAITING_CONTROLLER.json').write_text(json.dumps(receipt))
print(json.dumps(receipt))'''%(t.REMOTE,h,t.REMOTE,t.PYTHON,t.PYTHON)
 result=t.old.execute(mod,c,code,50);(R/'IR_ADAPTER_LAUNCH.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
finally:c.close()
