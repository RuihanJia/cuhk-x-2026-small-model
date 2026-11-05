from pathlib import Path
import json,zipfile,hashlib,time
import cloud_transport as t
R=Path(__file__).resolve().parent
spec=json.loads((R/'DETAIL_PROTOCOL.json').read_text(encoding='utf-8'))
files=['DETAIL_PROTOCOL.json',*spec['code_sha256'],*spec['asset_sha256']]
z=R/'detail_deploy.zip'
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as f:
 for n in files:f.write(R/n,n)
checksum=hashlib.sha256(z.read_bytes()).hexdigest();mod,client=t.connect()
try:
 with client.open_sftp() as s:s.put(str(z),t.REMOTE+'/'+z.name,confirm=True)
 code='''from pathlib import Path
import json,zipfile,hashlib,subprocess,time
r=Path(%r);z=r/'detail_deploy.zip';assert hashlib.sha256(z.read_bytes()).hexdigest()==%r
out=r/'detail_output/cloud';assert not (out/'CONTROLLER.json').exists();out.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(z) as f:
 for name in f.namelist():
  dest=(r/name).resolve();assert dest.is_relative_to(r.resolve());dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(f.read(name))
p=json.loads((r/'DETAIL_PROTOCOL.json').read_text());assert time.time()<p['deadline_unix']
for n,h in {**p['code_sha256'],**p['asset_sha256']}.items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h,n
proc=subprocess.Popen([%r,'-B','-u',str(r/'detail_controller.py'),'--host','cloud'],cwd=r,stdout=(out/'controller.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(status='LAUNCHED',controller_pid=proc.pid,at_unix=time.time(),protocol_sha256=hashlib.sha256((r/'DETAIL_PROTOCOL.json').read_bytes()).hexdigest())))'''%(t.REMOTE,checksum,t.PYTHON)
 result=t.old.execute(mod,client,code,50);(R/'DETAIL_CLOUD_LAUNCH.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
finally:client.close()
