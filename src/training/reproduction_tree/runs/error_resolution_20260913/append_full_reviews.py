import json,sys,time,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
for row in json.load(sys.stdin):
    sheets=[]
    for name in row.pop('sheets'):
        p=R/'audit/full_frames'/f'{name}.jpg'
        m=json.loads(p.with_suffix('.json').read_text(encoding='utf-8'))
        h=hashlib.sha256(p.read_bytes()).hexdigest()
        assert h==m['image_sha256']
        sheets.append(dict(path=str(p),sha256=h,modality=m['modality'],indices=[x['index'] for x in m['frames']],total=m['total_frames']))
    row.update(sheets=sheets,viewed=True,at_unix=time.time(),used_as_training_label_or_feature=False)
    with (R/'audit/FULL_FRAME_OBSERVATIONS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
print('Full-frame review records appended')
