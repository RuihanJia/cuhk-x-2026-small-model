from pathlib import Path,PurePosixPath
import json,hashlib,zipfile,time
r=Path(__file__).resolve().parent;m=json.loads((r/'THERMAL_TRANSFER_MANIFEST.json').read_text());parts=json.loads((r/'THERMAL_PARTS.json').read_text());dest=r/'training_thermal_frames_joined.zip';h=hashlib.sha256()
with dest.open('wb') as out:
 for p in parts['parts']:
  b=(r/('thermal_part_%04d'%p['i'])).read_bytes();assert len(b)==p['bytes'] and hashlib.sha256(b).hexdigest()==p['sha256'];out.write(b);h.update(b)
assert h.hexdigest()==m['archive_sha256']==parts['archive_sha256'];root=r/'thermal_dataset';root.mkdir(exist_ok=True);records={p:(n,h) for p,n,h in m['records']}
with zipfile.ZipFile(dest) as f:
 assert set(f.namelist())==set(records) and len(records)==m['frames']
 for name in f.namelist():
  p=(root/name).resolve();assert p.is_relative_to(root.resolve());assert PurePosixPath(name).parts[:5]==('Training','data','HAR','data','Thermal');n,h=records[name];b=f.read(name);assert len(b)==n and hashlib.sha256(b).hexdigest()==h
  if p.exists():assert p.read_bytes()==b
  else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
result=dict(status='VERIFIED_ALL_TRAINING_THERMAL_FILES',frames=m['frames'],clips=m['clips'],raw_bytes=m['raw_bytes'],archive_sha256=m['archive_sha256'],root=str(root),at_unix=time.time(),Test_files=0)
(root/'READY.json').write_text(json.dumps(result));print(json.dumps(result))
