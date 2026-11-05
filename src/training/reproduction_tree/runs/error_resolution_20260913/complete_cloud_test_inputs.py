"""Stage exact, label-free official Test inputs; reuse verified existing raw bytes."""
from pathlib import Path
import sys,json,time,hashlib,zipfile
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R));import cloud_transport as t
DEADLINE=1789323192.0883653
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def progress(**kw):
    (R/'CLOUD_TEST_STAGING_STATUS.json').write_text(json.dumps(dict(at_unix=time.time(),**kw),indent=2))
def main():
    m,c=t.connect()
    try:
        code="from pathlib import Path\nimport json,hashlib\nr=Path("+repr(t.REMOTE)+");s=Path('/home/featurize/work/cuhkx_sensor_release_20260908_v1/inputs_selected/Testing/data/small_model_track_test');d={}\nfor p in s.rglob('*'):\n if p.is_file() and p.suffix=='.png':d[hashlib.sha256(p.read_bytes()).hexdigest()]=str(p)\n(r/'existing_Test_raw_hashes.json').write_text(json.dumps(d));print(json.dumps(dict(unique_hashes=len(d))))"
        progress(phase='HASH_EXISTING_CLOUD_INPUTS');print(t.old.execute(m,c,code,50),flush=True)
        with c.open_sftp() as ftp:ftp.get(t.REMOTE+'/existing_Test_raw_hashes.json',str(R/'existing_Test_raw_hashes.json'))
        existing=json.loads((R/'existing_Test_raw_hashes.json').read_text());root=Path(r'D:\Datasets\CUHK-X_Small_Model_Track\Small-Model-Track\Testing\data\small_model_track_test');rows=[];missing=[]
        allowed={'Depth_Color','IR','IMU','Skeleton','Thermal'}
        for sample in sorted(root.glob('SM_test_*')):
            for mod in sorted(allowed):
                d=sample/mod
                if not d.is_dir():continue
                for p in sorted(d.rglob('*')):
                    if not p.is_file() or any(n.startswith('.') for n in p.relative_to(d).parts):continue
                    h=sha(p);q=dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=h,reuse_path=existing.get(h));rows.append(q)
                    if not q['reuse_path']:missing.append((p,q['path']))
        manifest=dict(source='official raw label-free Testing only',rows=rows,samples=405,labels_included=False,deadline_unix=DEADLINE)
        mp=R/'CLOUD_TEST_INPUT_MANIFEST.json';mp.write_text(json.dumps(manifest));archive=R/'cloud_test_missing.zip'
        progress(phase='COMPRESS_MISSING_INPUTS',files=len(rows),missing_files=len(missing),missing_bytes=sum(p.stat().st_size for p,_ in missing))
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            for p,n in missing:z.write(p,n)
        h=sha(archive);total=archive.stat().st_size;last=0
        def callback(done,allbytes):
            nonlocal last
            assert time.time()<DEADLINE-600,'INSUFFICIENT_REPLAY_TIME'
            if time.time()-last>5:progress(phase='UPLOAD_MISSING_INPUTS',uploaded_bytes=done,total_bytes=allbytes);last=time.time()
        progress(phase='UPLOAD_MISSING_INPUTS',uploaded_bytes=0,total_bytes=total)
        with c.open_sftp() as ftp:
            ftp.put(str(mp),t.REMOTE+'/CLOUD_TEST_INPUT_MANIFEST.json')
            ftp.put(str(archive),t.REMOTE+'/cloud_test_missing.zip',callback=callback)
        code="from pathlib import Path\nimport json,hashlib,os,zipfile,subprocess,time\nr=Path("+repr(t.REMOTE)+");z=r/'cloud_test_missing.zip';assert hashlib.sha256(z.read_bytes()).hexdigest()=="+repr(h)+"\nd=r/'release_Test_inputs/small_model_track_test';assert not d.exists();d.mkdir(parents=True)\nwith zipfile.ZipFile(z) as a:\n for n in a.namelist():\n  p=(d/n).resolve();assert p.is_relative_to(d.resolve());a.extract(n,d)\ns=json.loads((r/'CLOUD_TEST_INPUT_MANIFEST.json').read_text())\nfor q in s['rows']:\n p=(d/q['path']).resolve();assert p.is_relative_to(d.resolve());p.parent.mkdir(parents=True,exist_ok=True)\n if q['reuse_path']:\n  src=Path(q['reuse_path']);assert src.resolve().is_relative_to(Path('/home/featurize/work/cuhkx_sensor_release_20260908_v1/inputs_selected').resolve());assert hashlib.sha256(src.read_bytes()).hexdigest()==q['sha256'];os.link(src,p)\n assert p.stat().st_size==q['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==q['sha256'],q['path']\n"
        worker="from pathlib import Path\nimport sys,json,time,traceback\nR=Path(__file__).resolve().parent;D=R/'pose_dual_release';sys.path.insert(0,str(D));sys.path.insert(1,str(R/'bundle'));sys.path.append(str(R/'pose_site'))\nfrom dual_host_runtime.health import Guard\nfrom dual_host_runtime.io import atomic\nimport infer\ndef main():\n out=R/'release_cloud_output_v2';out.mkdir(exist_ok=True)\n try:\n  guard=Guard(out,1789323192.0883653,out/'TELEMETRY.json');a=infer.infer(R/'release_Test_inputs',D/'cloud_submission.csv',guard,workers=8,video_batch=8);atomic(out/'DONE.json',dict(status='COMPLETE',at_unix=time.time(),audit=a))\n except BaseException as e:atomic(out/'ERROR.json',dict(error=str(e),traceback=traceback.format_exc(),at_unix=time.time()));raise\nif __name__=='__main__':main()\n"
        code+="(r/'release_cloud_worker_v2.py').write_text("+repr(worker)+");controller=(r/'release_cloud_controller.py').read_text().replace('release_cloud_output','release_cloud_output_v2').replace('release_cloud_worker.py','release_cloud_worker_v2.py');(r/'release_cloud_controller_v2.py').write_text(controller)\nout=r/'release_cloud_output_v2';out.mkdir();p=subprocess.Popen(["+repr(t.PYTHON)+",'-B','-u',str(r/'release_cloud_controller_v2.py')],cwd=r,stdout=(out/'launcher.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True);print(json.dumps(dict(pid=p.pid,at_unix=time.time(),input_files_verified=len(s['rows']),archive_bytes="+str(total)+",archive_sha256="+repr(h)+",labels_included=False)))"
        progress(phase='VERIFY_AND_START_REPLAY');res=t.old.execute(m,c,code,50);(R/'RELEASE_CLOUD_LAUNCH_V2.json').write_text(json.dumps(res,indent=2));progress(phase='COMPLETE',launch=res);print(json.dumps(res),flush=True)
    except BaseException as e:
        import traceback
        progress(phase='ERROR',error=str(e),traceback=traceback.format_exc());raise
    finally:c.close()
if __name__=='__main__':main()
