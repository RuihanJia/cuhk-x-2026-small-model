from pathlib import Path
import json,hashlib
import cloud_transport as t
R=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
spec=json.loads((R/'CROP_PROBE_PROTOCOL.json').read_text(encoding='utf-8'))
files=['crop_probe.py','crop_controller.py','CROP_PROBE_PROTOCOL.json','CROP_INPUTS.json','base_F0.npz','base_F3.npz',*spec['bundle_hashes']]
manifest={n:sha(R/n) for n in files}
m,c=t.connect()
try:
    code="from pathlib import Path\nimport json\np=Path(%r)\nassert not (p/'CROP_CONTROLLER.json').exists()\nfor n in ['', 'bundle', 'bundle/dual_host_runtime','crop_probe_output']:(p/n).mkdir(parents=True,exist_ok=True)\nprint(json.dumps({'status':'READY'}))"%t.REMOTE
    t.old.execute(m,c,code)
    with c.open_sftp() as s:
        for n in files:s.put(str(R/n),t.REMOTE+'/'+n,confirm=True)
    code='''from pathlib import Path
import json,hashlib,subprocess,os,time
r=Path(%r);files=%r
for n,h in files.items():assert hashlib.sha256((r/n).read_bytes()).hexdigest().upper()==h
assert not (r/'CROP_CONTROLLER.json').exists()
p=subprocess.Popen([%r,'-u',str(r/'crop_controller.py')],cwd=r,stdout=(r/'controller.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(status='LAUNCHED',controller_pid=p.pid,at_unix=time.time(),protocol_sha256=files['CROP_PROBE_PROTOCOL.json'])))'''%(t.REMOTE,manifest,t.PYTHON)
    result=t.old.execute(m,c,code);(R/'CROP_LAUNCH.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
finally:c.close()
