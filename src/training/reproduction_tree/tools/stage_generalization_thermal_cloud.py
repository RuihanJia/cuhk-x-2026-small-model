"""Prepare existing private training data for a possible unstarted-job handoff; starts no GPU work."""
from pathlib import Path
import hashlib,json,time
import generalization_cloud as cloud
ROOT=cloud.ROOT;BASE=cloud.BASE;REMOTE=cloud.REMOTE
def write(value):
    p=BASE/'THERMAL_TRANSFER.json';temp=p.with_suffix('.partial.json');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temp.replace(p)
def main():
    source=ROOT/'archive/20260907_handoff/project/runs/_upload/20260907_multimodal/cuhkx_train_thermal_20260907.zip';index=BASE/'local/bundle/data/thermal_index.json';started=time.time();deadline=cloud.read(BASE/'cloud/QUEUE.json')['deadline_unix'];total=source.stat().st_size;value=dict(status='HASHING_LOCAL_ADMITTED_ARCHIVE',started_unix=started,total_bytes=total,sent_bytes=0,deadline_unix=deadline,source=source.relative_to(ROOT).as_posix(),GPU_training_started=False);write(value)
    with source.open('rb') as f:expected=hashlib.file_digest(f,'sha256').hexdigest().upper()
    value.update(status='UPLOADING',sha256=expected,index_sha256=cloud.access.sha(index));write(value);last=0;m,c=cloud.access.connection()
    try:
        def progress(sent,all_bytes):
            nonlocal last
            if time.time()>=deadline:raise TimeoutError('TRANSFER_PHASE_DEADLINE')
            now=time.time()
            if now-last>=15 or sent==all_bytes:value.update(sent_bytes=sent,at_unix=now,elapsed_seconds=now-started);write(value);last=now
        with c.open_sftp() as ftp:
            ftp.put(str(index),REMOTE+'/thermal_index.json',confirm=True)
            ftp.put(str(source),REMOTE+'/thermal_train.zip.partial',callback=progress,confirm=True)
        value.update(status='REMOTE_HASH_AND_SAFE_EXTRACTION',at_unix=time.time());write(value)
        code=f'''from pathlib import Path
import hashlib,json,zipfile,time,os
root=Path({REMOTE!r});archive=root/'thermal_train.zip.partial'
with archive.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest().upper()
assert actual=={expected!r};os.replace(archive,root/'thermal_train.zip');archive=root/'thermal_train.zip';dest=(root/'thermal_data').resolve();dest.mkdir(exist_ok=True)
with zipfile.ZipFile(archive) as z:
 assert len(z.infolist())==201469 and sum(i.file_size for i in z.infolist())==3561179862
 for item in z.infolist():
  assert item.filename.startswith(('Training/data/HAR/data/Thermal/','metadata/'))
  assert (dest/item.filename).resolve().is_relative_to(dest)
 z.extractall(dest)
rows=json.loads((root/'thermal_index.json').read_text());assert len(rows)==2891
for row in rows:
 d=dest/row['directory'];actual=sorted(p.name for p in d.iterdir() if p.suffix.lower() in ['.jpg','.jpeg','.png']);assert actual==row['files'],row['clip_id']
v=dict(status='READY',zip_sha256={expected!r},data_root=str(dest),index=str(root/'thermal_index.json'),index_sha256={cloud.access.sha(index)!r},training_clips=len(rows),files=201466,source_manifest_sha256='82E73CDD3DC8E72382826C0E1320B80C16BC94148B86765EB62E3734957852A3',at_unix=time.time(),GPU_training_started=False);(root/'THERMAL_DATA_READY.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))'''
        # Native transfer process owns this bounded operation; no assistant polling loop.
        result=cloud.access.execute(m,c,code,timeout=600);value.update(status='READY',remote=result,at_unix=time.time(),elapsed_seconds=time.time()-started);write(value);print(json.dumps(dict(status='READY',elapsed_seconds=value['elapsed_seconds'],remote=result),ensure_ascii=False))
    except BaseException as e:value.update(status='FAILED',error=str(e)[:1200],at_unix=time.time());write(value);raise
    finally:c.close()
if __name__=='__main__':main()
