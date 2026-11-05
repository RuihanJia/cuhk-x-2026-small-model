from pathlib import Path
import json,time,zipfile,hashlib
import cloud_transport as t
R=Path(__file__).resolve().parent;ROOT=R.parents[1];z=R/'pose_geometry_gpu_deploy.zip';files={n:R/n for n in ['POSE_REGIONS_GPU_PROTOCOL.json','extract_pose_regions_gpu.py','pose_regions_gpu_controller.py','pose_roi.py']}
for n in ['pose_state_fp16.pt','yolo11n-pose.pt','ADMISSION.json','LICENSE']:files['assets/pose_gpu/'+n]=ROOT/'assets/pose'/n
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as f:
 for n,p in files.items():f.write(p,n)
h=hashlib.sha256(z.read_bytes()).hexdigest();mod,c=t.connect()
try:
 with c.open_sftp() as sf:sf.put(str(z),t.REMOTE+'/'+z.name,confirm=True)
 code='''from pathlib import Path
import zipfile,json,hashlib,subprocess,time
r=Path(%r);z=r/'pose_geometry_gpu_deploy.zip';assert hashlib.sha256(z.read_bytes()).hexdigest()==%r
with zipfile.ZipFile(z) as f:
 for n in f.namelist():
  p=(r/n).resolve();assert p.is_relative_to(r.resolve());b=f.read(n)
  if p.exists():assert p.read_bytes()==b,n
  else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
out=r/'pose_regions_gpu';out.mkdir(exist_ok=True);assert not(out/'WAITING_CONTROLLER.json').exists();s=json.loads((r/'POSE_REGIONS_GPU_PROTOCOL.json').read_text());assert hashlib.sha256((r/'pose_regions_gpu_controller.py').read_bytes()).hexdigest()==s['controller_sha256']
bridge=r/'pose_geometry_bridge.py';code="""from pathlib import Path
import json,time,subprocess
r=Path(%r);s=json.loads((r/'POSE_REGIONS_GPU_PROTOCOL.json').read_text());out=r/'pose_regions_gpu';prior=r/'thermal_inner_output/cloud'
try:
 while not(prior/'CONTROLLER_DONE.json').exists():
  assert time.time()<s['deadline_unix'];assert not(out/'STOP_REQUESTED').exists();assert not(prior/'ERROR.json').exists() and not(prior/'CONTROLLER_ERROR.json').exists();time.sleep(5)
 assert json.loads((prior/'CONTROLLER_DONE.json').read_text())['exit_code']==0
 assert 'Successfully installed' in (r/'pose_dependencies.log').read_text()
 subprocess.run([%r,'-B','-u',str(r/'pose_regions_gpu_controller.py')],cwd=r,check=True)
except BaseException as e:
 (out/'BRIDGE_ERROR.json').write_text(json.dumps(dict(error=str(e),at_unix=time.time())));raise
""";assert not bridge.exists();bridge.write_text(code);p=subprocess.Popen([%r,'-B','-u',str(bridge)],cwd=r,stdout=(out/'bridge.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True)
result=dict(status='QUEUED_AFTER_THERMAL_SOURCE_FOUR_FITS',pid=p.pid,at_unix=time.time(),protocol_sha256=hashlib.sha256((r/'POSE_REGIONS_GPU_PROTOCOL.json').read_bytes()).hexdigest());(out/'WAITING_CONTROLLER.json').write_text(json.dumps(result));print(json.dumps(result))
'''%(t.REMOTE,h,t.REMOTE,t.PYTHON,t.PYTHON)
 result=t.old.execute(mod,c,code,50);(R/'POSE_GPU_LAUNCH.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
finally:c.close()
