"""Byte checks of all cross-class Training overlaps, including containment and Depth."""
import json,hashlib,time
from pathlib import Path
R=Path(__file__).resolve().parent
p=R/'audit/ALL_TRAINING_TEMPORAL_OVERLAP.json'
d=json.loads(p.read_text(encoding='utf-8'));pairs=[]
for r in d['cross_class_overlaps']:
    n=r['identical_IR_bytes'];a,b=r['a'],r['b']
    if n!=min(a['frames'],b['frames']):continue
    q={k:r[k] for k in ['a','b','shared_timestamp_and_index','identical_IR_bytes']}
    depth=[]
    for x in r['frames']:
        fs=[]
        for y in [a,b]:
            ip=Path(y['path']); dp=Path(str(ip).replace('\\IR\\','\\Depth_Color\\'))/('Depth_'+x['frame']+'_Color.png')
            fs.append(dict(path=str(dp),sha256=hashlib.sha256(dp.read_bytes()).hexdigest() if dp.exists() else None))
        depth.append(dict(frame=x['frame'],files=fs,equal=bool(fs[0]['sha256']) and fs[0]['sha256']==fs[1]['sha256']))
    q['Depth_frames']=depth;q['identical_Depth_bytes']=sum(x['equal'] for x in depth);pairs.append(q)
target=[]
for r in d['cross_class_overlaps']:
    if (r['a']['user']=='user7' and any(v['activity'].startswith(('24_','27_')) for v in [r['a'],r['b']])) or (r['a']['user']=='user21' and any(v['activity'].startswith('38_') and v['clip']=='7-2-2' for v in [r['a'],r['b']])):
        target.append({k:r[k] for k in ['a','b','shared_timestamp_and_index','identical_IR_bytes']})
out=dict(at_unix=time.time(),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),all_cross_class_full_IR_containment_pairs=pairs,targeted_phone_selfie_massage_overlaps=target,Test_read=False,training_fits=0,automatic_relabel=False)
(R/'audit/FULL_CONTAINMENT_DEPTH_PROOF.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(full_containment_pairs=len(pairs),summary=[{k:v for k,v in x.items() if k!='Depth_frames'} for x in pairs],targeted=target),ensure_ascii=False))
