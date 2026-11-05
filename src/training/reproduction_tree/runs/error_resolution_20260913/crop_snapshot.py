from pathlib import Path
import json,time,subprocess
r=Path('/home/featurize/cuhkx_motion_transfer_20260911_v1/error_resolution_20260913');d=dict(at_unix=time.time())
for name in ['CROP_CONTROLLER.json','CROP_CONTROLLER_DONE.json','CROP_CONTROLLER_ERROR.json','TELEMETRY.json','crop_probe_output/STATUS.json','crop_probe_output/DONE.json','crop_probe_output/ERROR.json','crop_probe_output/F0_REPLAY_QA.json','crop_probe_output/F3_REPLAY_QA.json','crop_probe_output/F0_RESULT.json','crop_probe_output/F3_RESULT.json']:
 p=r/name
 if p.exists():d[name]=json.loads(p.read_text())
for name in ['crop_probe.log','controller.log']:
 p=r/name
 if p.exists():d[name]=p.read_text()[-2500:]
print(json.dumps(d))
