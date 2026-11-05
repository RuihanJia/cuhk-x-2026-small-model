"""Move only hash-manifested Training Thermal frames to the owned cloud cache."""
from pathlib import Path, PurePosixPath
import json,time,hashlib,zipfile,sys
import cloud_transport as t
R=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(n,v):
 p=R/n;q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');q.replace(p)
def main():
 started=time.time();spec=json.loads((R/'THERMAL_ADAPTER_PROTOCOL.json').read_text());root=Path(spec['hosts']['local']['data_root']).resolve();rows=json.loads((R/'THERMAL_INPUTS.json').read_text())['rows'];z=R/'training_thermal_frames.zip';manifest=R/'THERMAL_TRANSFER_MANIFEST.json'
 assert time.time()<spec['deadline_unix']
 if not manifest.exists():
  records=[];seen=set();total=0;tmp=z.with_suffix('.zip.partial')
  with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as f:
   for row in rows:
    for stream in row['streams']:
     prefix=PurePosixPath(stream['directory']);assert prefix.parts[:6]==('Training','data','HAR','data','Thermal',row['clip_id'].split('/')[0])
     for name in stream['files']:
      rel=str(prefix/name);assert rel not in seen;seen.add(rel);path=(root/rel).resolve();assert path.is_relative_to(root)
      b=path.read_bytes();f.writestr(rel,b);records.append([rel,len(b),hashlib.sha256(b).hexdigest()]);total+=len(b)
    if len(records)%1000<100:write('THERMAL_TRANSFER_PROGRESS.json',dict(phase='ARCHIVING',frames=len(records),bytes=total,at_unix=time.time()))
  tmp.replace(z);write(manifest.name,dict(status='ARCHIVE_VERIFIED_LOCAL',frames=len(records),clips=len(rows),raw_bytes=total,archive_bytes=z.stat().st_size,archive_sha256=sha(z),inputs_sha256=sha(R/'THERMAL_INPUTS.json'),records=records,Training_only=True))
 m=json.loads(manifest.read_text());assert sha(z)==m['archive_sha256'];remote=t.REMOTE+'/thermal_dataset';mod,c=t.connect()
 try:
  with c.open_sftp() as s:
   part=t.REMOTE+'/'+z.name+'.partial';last=[0]
   def progress(done,allbytes):
    if time.time()-last[0]>10:last[0]=time.time();write('THERMAL_TRANSFER_PROGRESS.json',dict(phase='UPLOAD',bytes=done,total_bytes=allbytes,at_unix=time.time()))
   s.put(str(z),part,callback=progress,confirm=True);s.put(str(manifest),t.REMOTE+'/'+manifest.name,confirm=True)
  code='''from pathlib import Path,PurePosixPath
import json,zipfile,hashlib,time,os
r=Path(%r);root=r/'thermal_dataset';root.mkdir(exist_ok=True);m=json.loads((r/'THERMAL_TRANSFER_MANIFEST.json').read_text());z=r/'training_thermal_frames.zip.partial'
h=hashlib.sha256()
with z.open('rb') as f:
 for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
assert h.hexdigest()==m['archive_sha256']
records={n:(size,h) for n,size,h in m['records']};assert len(records)==m['frames']
with zipfile.ZipFile(z) as f:
 assert set(f.namelist())==set(records)
 for n in f.namelist():
  p=(root/n).resolve();assert p.is_relative_to(root.resolve());assert PurePosixPath(n).parts[:5]==('Training','data','HAR','data','Thermal')
  size,h=records[n];b=f.read(n);assert len(b)==size and hashlib.sha256(b).hexdigest()==h
  if p.exists():assert p.read_bytes()==b
  else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
z.rename(r/'training_thermal_frames.zip')
result=dict(status='VERIFIED_ALL_TRAINING_THERMAL_FILES',frames=m['frames'],clips=m['clips'],raw_bytes=m['raw_bytes'],archive_sha256=m['archive_sha256'],root=str(root),at_unix=time.time(),Test_files=0)
(root/'READY.json').write_text(json.dumps(result));print(json.dumps(result))
'''%t.REMOTE
  result=t.old.execute(mod,c,code,180);result['elapsed_seconds']=time.time()-started;write('THERMAL_TRANSFER_DONE.json',result);print(json.dumps(result),flush=True)
 finally:c.close()
if __name__=='__main__':
 try:main()
 except BaseException as e:
  import traceback
  write('THERMAL_TRANSFER_ERROR.json',dict(at_unix=time.time(),error=str(e),traceback=traceback.format_exc()));raise
