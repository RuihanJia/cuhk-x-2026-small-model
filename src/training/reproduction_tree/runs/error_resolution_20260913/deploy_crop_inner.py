from pathlib import Path
import json,zipfile,hashlib
import cloud_transport as t
R=Path(__file__).resolve().parent;s=json.loads((R/'CROP_INNER_PROTOCOL.json').read_text());z=R/'crop_inner_deploy.zip'
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as f:
 for n in ['CROP_INNER_PROTOCOL.json',*s['code_sha256'],*s['asset_sha256']]:f.write(R/n,n)
h=hashlib.sha256(z.read_bytes()).hexdigest();mod,c=t.connect()
try:
 with c.open_sftp() as sf:sf.put(str(z),t.REMOTE+'/'+z.name,confirm=True)
 code='''from pathlib import Path
import json,zipfile,hashlib,subprocess,time
r=Path(%r);z=r/'crop_inner_deploy.zip';assert hashlib.sha256(z.read_bytes()).hexdigest()==%r
assert json.loads((r/'ir_adapter_output/cloud/CONTROLLER_DONE.json').read_text())['exit_code']==0
out=r/'crop_inner_output';out.mkdir(exist_ok=True);assert not(out/'CONTROLLER.json').exists()
with zipfile.ZipFile(z) as f:
 for n in f.namelist():
  dest=(r/n).resolve();assert dest.is_relative_to(r.resolve());b=f.read(n)
  if dest.exists():assert dest.read_bytes()==b,n
  else:dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
s=json.loads((r/'CROP_INNER_PROTOCOL.json').read_text());assert time.time()<s['deadline_unix']
for n,h in {**s['code_sha256'],**s['asset_sha256']}.items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h
p=subprocess.Popen([%r,'-B','-u',str(r/'crop_inner_controller.py')],cwd=r,stdout=(out/'controller.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(status='STARTED_SOURCE_EXCLUDED_CROP_INFERENCE',pid=p.pid,at_unix=time.time(),protocol_sha256=hashlib.sha256((r/'CROP_INNER_PROTOCOL.json').read_bytes()).hexdigest())))
'''%(t.REMOTE,h,t.PYTHON)
 result=t.old.execute(mod,c,code,50);(R/'CROP_INNER_LAUNCH.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
finally:c.close()
