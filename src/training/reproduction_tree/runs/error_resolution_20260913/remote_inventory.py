import json, subprocess, time
from pathlib import Path
p=Path('/home/featurize/cuhkx_motion_transfer_20260911_v1')
r=dict(at_unix=time.time(),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,utilization.gpu,temperature.gpu,memory.used','--format=csv,noheader'],text=True).strip(),sources={})
for f in [0,3]:
    matches=list(p.glob(f'activity_token_stopgrad_20260913/output/C_TOKEN_STOPGRAD_F{f}_AUX025/model_fp16.pt'))
    r['sources'][str(f)]=[str(x) for x in matches]
r['project_processes']=subprocess.check_output(['pgrep','-af','python'],text=True).splitlines()
print(json.dumps(r))
