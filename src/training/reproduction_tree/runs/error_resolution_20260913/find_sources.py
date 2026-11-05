import json
from pathlib import Path
p=Path('/home/featurize/cuhkx_motion_transfer_20260911_v1')
m=[str(x) for x in p.rglob('model_fp16.pt') if 'STOPGRAD_F' in str(x) or 'r6_f' in str(x)]
print(json.dumps(m))
