from pathlib import Path
import json,zipfile,hashlib
import cloud_transport as t
R=Path(__file__).resolve().parent;spec=json.loads((R/'VIDEO_CONFLICT_PROTOCOL.json').read_text());files=['VIDEO_CONFLICT_PROTOCOL.json',*spec['code_sha256'],*spec['asset_sha256']];z=R/'video_conflict_deploy.zip'
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as f:
 for n in files:f.write(R/n,n)
h=hashlib.sha256(z.read_bytes()).hexdigest();mod,c=t.connect()
try:
 with c.open_sftp() as s:s.put(str(z),t.REMOTE+'/'+z.name,confirm=True)
 code='''from pathlib import Path
import json,zipfile,hashlib,subprocess,time
r=Path(%r);z=r/'video_conflict_deploy.zip';assert hashlib.sha256(z.read_bytes()).hexdigest()==%r
assert (r/'detail_output/cloud/HEAD_QUEUE_DONE.json').exists()
out=r/'video_conflict_output';assert not (out/'CONTROLLER.json').exists();out.mkdir(exist_ok=True)
with zipfile.ZipFile(z) as f:
 for n in f.namelist():
  dest=(r/n).resolve();assert dest.is_relative_to(r.resolve());dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(f.read(n))
s=json.loads((r/'VIDEO_CONFLICT_PROTOCOL.json').read_text());assert time.time()<s['deadline_unix']
for n,h in {**s['code_sha256'],**s['asset_sha256']}.items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h
for j in s['jobs']:assert hashlib.sha256(Path(j['source']).read_bytes()).hexdigest()==j['source_sha256']
p=subprocess.Popen([%r,'-B','-u',str(r/'video_conflict_controller.py')],cwd=r,stdout=(out/'controller.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(status='LAUNCHED',pid=p.pid,at_unix=time.time(),protocol_sha256=hashlib.sha256((r/'VIDEO_CONFLICT_PROTOCOL.json').read_bytes()).hexdigest())))'''%(t.REMOTE,h,t.PYTHON)
 result=t.old.execute(mod,c,code,50);(R/'VIDEO_CONFLICT_LAUNCH.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
finally:c.close()
