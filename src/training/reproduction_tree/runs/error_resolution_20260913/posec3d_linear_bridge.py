from pathlib import Path
import json,time,subprocess,sys
r=Path(__file__).resolve().parent;s=json.loads((r/'POSEC3D_LINEAR_PROTOCOL.json').read_text());out=r/'posec3d_linear_output';out.mkdir(exist_ok=True);prior=r/'posec3d_feature_output'
try:
 while not(prior/'CONTROLLER_DONE.json').exists():
  assert time.time()<s['deadline_unix'];assert not(prior/'ERROR.json').exists() and not(prior/'CONTROLLER_ERROR.json').exists();time.sleep(5)
 assert json.loads((prior/'CONTROLLER_DONE.json').read_text())['exit_code']==0
 subprocess.run([sys.executable,'-X','utf8','-B','-u',str(r/'posec3d_linear.py')],cwd=r,check=True)
except BaseException as e:
 (out/'BRIDGE_ERROR.json').write_text(json.dumps(dict(error=str(e),at_unix=time.time())));raise
