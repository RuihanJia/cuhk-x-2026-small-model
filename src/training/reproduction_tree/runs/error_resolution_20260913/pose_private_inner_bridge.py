from pathlib import Path
import time,json,subprocess,sys
r=Path(__file__).resolve().parent;s=json.loads((r/'POSE_PRIVATE_INNER_PROTOCOL.json').read_text());out=r/'pose_private_inner_output/cloud';out.mkdir(parents=True,exist_ok=True);prior=r/'pose_dense_output'
try:
 while not(prior/'CONTROLLER_DONE.json').exists():
  assert time.time()<s['deadline_unix'];assert not(prior/'ERROR.json').exists() and not(prior/'CONTROLLER_ERROR.json').exists();time.sleep(5)
 assert json.loads((prior/'CONTROLLER_DONE.json').read_text())['exit_code']==0
 subprocess.run([sys.executable,'-B','-u',str(r/'pose_private_inner_controller.py')],cwd=r,check=True)
except BaseException as e:
 (out/'BRIDGE_ERROR.json').write_text(json.dumps(dict(error=str(e),at_unix=time.time())));raise
