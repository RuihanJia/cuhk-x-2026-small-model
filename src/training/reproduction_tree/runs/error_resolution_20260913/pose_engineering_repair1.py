from pathlib import Path
import sys,json,time,subprocess,os,traceback,psutil
r=Path('/home/featurize/cuhkx_motion_transfer_20260911_v1/error_resolution_20260913');deadline=1789323192.0883653
out=r/'pose_engineering_repair1';out.mkdir(exist_ok=True)
def dump(n,d): (out/n).write_text(json.dumps(d,indent=2))
try:
 assert time.time()<deadline
 p=subprocess.run([sys.executable,'-m','pip','install','--no-deps','--disable-pip-version-check','--no-input','--target',str(r/'pose_site')]+['matplotlib==3.9.2', 'pandas==2.2.3', 'seaborn==0.13.2', 'requests==2.32.3', 'PyYAML==6.0.2', 'packaging==24.1', 'contourpy==1.3.0', 'cycler==0.12.1', 'fonttools==4.54.1', 'kiwisolver==1.4.7', 'pyparsing==3.1.4', 'python-dateutil==2.9.0.post0', 'six==1.16.0', 'pytz==2024.2', 'tzdata==2024.2', 'certifi==2024.8.30', 'charset-normalizer==3.4.0', 'idna==3.10', 'urllib3==2.2.3'],stdout=(out/'pip.log').open('w'),stderr=subprocess.STDOUT,timeout=600)
 assert p.returncode==0,'DEPENDENCY_INSTALL_FAILED'
 env=os.environ.copy();env['PYTHONPATH']=str(r);probe="import extract_pose_regions_gpu as w; w.SETTINGS.update({'sync':False}); w.torch_utils.NUM_THREADS=2; w.torch.set_num_threads(2); w.cv2.setNumThreads(1); m=w.YOLO(str(w.A/'yolo11n-pose.pt')); print(m.predict(w.np.zeros((480,640,3),dtype=w.np.uint8),imgsz=640,device='cuda:0',verbose=False,save=False)[0].boxes.shape)"
 p=subprocess.run([sys.executable,'-B','-u','-c',probe],cwd=r,env=env,stdout=(out/'smoke.log').open('w'),stderr=subprocess.STDOUT,timeout=120)
 assert p.returncode==0,'POSE_IMPORT_INFERENCE_SMOKE_FAILED'
 for name in ['pose_regions_gpu','pose_crop_output']:
  src=(r/name).resolve();dst=(r/(name+'_failed_import_20260913')).resolve();assert src.is_relative_to(r.resolve()) and dst.is_relative_to(r.resolve());assert src.exists() and not dst.exists()
  if(src/'CONTROLLER.json').exists():
   d=json.loads((src/'CONTROLLER.json').read_text())
   for k in ['owner','controller']:
    ident=d[k];pid=ident['pid']
    if psutil.pid_exists(pid):
     proc=psutil.Process(pid);assert proc.status()==psutil.STATUS_ZOMBIE or abs(proc.create_time()-ident.get('create_time',-1))>1,'BOUND_PROCESS_STILL_LIVE'
  assert (src/'CONTROLLER_DONE.json').exists() or (src/'BRIDGE_ERROR.json').exists()
  src.rename(dst);src.mkdir()
 dump('REPAIR.json',dict(at_unix=time.time(),type='ENGINEERING_DEPENDENCIES_ONLY',packages=['matplotlib==3.9.2', 'pandas==2.2.3', 'seaborn==0.13.2', 'requests==2.32.3', 'PyYAML==6.0.2', 'packaging==24.1', 'contourpy==1.3.0', 'cycler==0.12.1', 'fonttools==4.54.1', 'kiwisolver==1.4.7', 'pyparsing==3.1.4', 'python-dateutil==2.9.0.post0', 'six==1.16.0', 'pytz==2024.2', 'tzdata==2024.2', 'certifi==2024.8.30', 'charset-normalizer==3.4.0', 'idna==3.10', 'urllib3==2.2.3'],frozen_protocols_unchanged=True,worker_weights_geometry_unchanged=True,prior_attempts_preserved=True,deadline_unix=deadline))
 for script,folder in [('pose_regions_gpu_controller.py','pose_regions_gpu'),('pose_crop_controller.py','pose_crop_output')]:
  assert time.time()<deadline
  p=subprocess.run([sys.executable,'-B','-u',str(r/script)],cwd=r,stdout=(out/(folder+'.log')).open('w'),stderr=subprocess.STDOUT,timeout=max(1,deadline-time.time()))
  assert p.returncode==0,(script,p.returncode)
  d=json.loads((r/folder/'CONTROLLER_DONE.json').read_text());assert d['exit_code']==0,(folder,d)
 dump('DONE.json',dict(status='COMPLETE',at_unix=time.time()))
except BaseException as e:
 dump('ERROR.json',dict(error=str(e),traceback=traceback.format_exc(),at_unix=time.time()));raise
