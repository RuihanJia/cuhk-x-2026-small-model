"""Small, scoped transport for the current authorized cloud queue."""
from pathlib import Path
import argparse,json,time
import visual_coverage_cloud as access
ROOT=access.ROOT;BASE=ROOT/'runs/generalization_training_20260910_v1';LOCAL=BASE/'cloud';REMOTE='/home/featurize/cuhkx_generalization_20260910_v1';PYTHON=access.PYTHON
def read(p):return access.read(p)
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def prepare(m,c):
    files={p.relative_to(LOCAL).as_posix():access.sha(p) for p in LOCAL.rglob('*') if p.is_file() and '__pycache__' not in p.parts};candidates={'bundle/assets/mvit_k400.pth':['/home/featurize/work/cuhkx_mvit_v2_s_20260908_v1/mvit_v2_s-ae3be167.pth'],'bundle/data/frame_index.json':['/home/featurize/cuhkx_visual_coverage_20260910_v1/bundle/data/frame_index.json'],'bundle/data/measured_sensors.npz':['/home/featurize/cuhkx_visual_coverage_20260910_v1/bundle/data/measured_sensors.npz']}
    code=f'''import json,hashlib,shutil
from pathlib import Path
root=Path({REMOTE!r});root.mkdir(parents=True,exist_ok=True);assert not (root/'LAUNCH.json').exists();copied=[]
for rel,sources in {candidates!r}.items():
 for source in sources:
  p=Path(source)
  if p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest().upper()=={files!r}[rel]:
   dst=root/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst);copied.append(rel);break
print(json.dumps(dict(copied=copied)))'''
    result=access.execute(m,c,code)
    with c.open_sftp() as ftp:
        for rel in files:
            if rel in result['copied']:continue
            current=REMOTE
            for part in Path(rel).parent.as_posix().split('/'):
                if part=='.':continue
                current+='/'+part
                try:ftp.stat(current)
                except OSError:ftp.mkdir(current)
            ftp.put(str(LOCAL/rel),REMOTE+'/'+rel,confirm=True)
    code=f'''import json,hashlib
from pathlib import Path
root=Path({REMOTE!r})
for rel,expected in {files!r}.items():
 with (root/rel).open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest().upper()
 assert actual==expected,rel
print(json.dumps(dict(status='PREPARED_HASH_VERIFIED',files={len(files)},at_unix=__import__('time').time())))'''
    r=access.execute(m,c,code);write(BASE/'CLOUD_PREPARED.json',r);return r
def launch(m,c):
    ph=access.sha(LOCAL/'protocol.json')
    code=f'''import json,subprocess,time,psutil,hashlib
from pathlib import Path
root=Path({REMOTE!r});assert not (root/'LAUNCH.json').exists();assert hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest().upper()=={ph!r}
with (root/'controller.stdout.log').open('w') as so,(root/'controller.stderr.log').open('w') as se:
 p=subprocess.Popen([{PYTHON!r},'-B',str(root/'bundle/native_queue.py'),'--root',str(root)],stdin=subprocess.DEVNULL,stdout=so,stderr=se,cwd=root,start_new_session=True)
v=dict(pid=p.pid,create_time=psutil.Process(p.pid).create_time(),started_unix=time.time(),remote_root=str(root),protocol_sha256={ph!r});(root/'LAUNCH.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))'''
    r=access.execute(m,c,code);write(BASE/'CLOUD_LAUNCH.json',r);return r
def snapshot(m,c):
    code=f'''import json,time,psutil
from pathlib import Path
root=Path({REMOTE!r});r=dict(at_unix=time.time())
for name in ['QUEUE_STATUS.json','QUEUE_DONE.json','QUEUE_ERROR.json','TELEMETRY.json','LAUNCH.json','CONTROLLER.json']:
 p=root/name
 if p.exists():r[name]=json.loads(p.read_text())
v=r.get('CONTROLLER.json',r.get('LAUNCH.json'))
if v:
 try:p=psutil.Process(v['pid']);r['controller_alive']=p.is_running() and p.status()!=psutil.STATUS_ZOMBIE and abs(p.create_time()-v['create_time'])<.01
 except psutil.NoSuchProcess:r['controller_alive']=False
r['workers']={{}}
for key in r.get('QUEUE_STATUS.json',{{}}).get('active',{{}}):
 d=root/'output'/key;v={{}}
 for name in ['STATUS.json','LP.json','ERROR.json','metrics.json']:
  p=d/name
  if p.exists():
   value=json.loads(p.read_text());v[name]=value[-3:] if name=='metrics.json' else value
 p=d/'worker.log'
 if p.exists():v['log_tail']=p.read_text()[-1200:]
 r['workers'][key]=v
p=root/'controller.stderr.log'
if p.exists():r['controller_stderr']=p.read_text()[-1800:]
print(json.dumps(r))'''
    r=access.execute(m,c,code);write(BASE/'CLOUD_SNAPSHOT.json',r);return r
def main():
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','launch','snapshot']);args=a.parse_args();m,c=access.connection()
    try:r=globals()[args.action](m,c);print(json.dumps(r,ensure_ascii=False))
    finally:c.close()
if __name__=='__main__':main()
