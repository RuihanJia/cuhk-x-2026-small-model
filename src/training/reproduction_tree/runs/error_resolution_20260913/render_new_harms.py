"""Register and render newly harmed cases without altering the285-case inventory."""
from pathlib import Path
import json,csv,time,sys
import numpy as np
R=Path(__file__).resolve().parent;ROOT=R.parents[1];sys.path.insert(0,str(R))
import audit_cases as a
def rows(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig')))
old={json.loads(x)['clip_id'] for x in (R/'audit/OBSERVATIONS.jsonl').read_text(encoding='utf-8').splitlines()}
index=R/'audit/NEW_ENDPOINT_CASES.json';known=json.loads(index.read_text(encoding='utf-8')) if index.exists() else []
covered={x['clip_id'] for x in known}|old;base={x['clip_id']:x for x in rows(ROOT/'runs/joint_relation_20260913/analysis/ALL_1094_COMPARISONS.csv')};joint={x['ids']:x for x in rows(ROOT/'runs/joint_relation_20260913/independent_fusion/PAIRED_CASES.csv')}
z=dict(np.load(ROOT/'runs/joint_relation_20260913/data/sensors.npz'));ix={k:i for i,k in enumerate(z['clip_ids'])};created=[]
for x in rows(R/'training_analysis/ALL_PAIRED_CASES.csv'):
 if x['clip_id'] in covered or x['R6']!=x['target']:continue
 changed={k:int(v) for k,v in x.items() if k not in ['clip_id','fold','user','target','R6','CROP_ZERO'] and v and v!=x['target']}
 if not changed:continue
 r=dict(base[x['clip_id']],case_id=f'N{len(known)+1:03d}',predictions={'R6':int(x['R6']),**changed},joint_independent_prediction=int(joint[x['clip_id']]['head']))
 i=ix[r['clip_id']];r['sensor_numeric']=dict(acc_observed_fraction=z['acc_mask'][i].mean(-1).tolist(),gyro_observed_fraction=z['gyro_mask'][i].mean(-1).tolist(),skeleton_temporal_mean_abs_delta=float(np.abs(np.diff(z['skeleton'][i],axis=0)).mean()),all_finite=bool(np.isfinite(z['x'][i]).all() and np.isfinite(z['skeleton'][i]).all()))
 a.render_one(r);known.append(dict(case_id=r['case_id'],clip_id=r['clip_id'],predictions=r['predictions'],true_class=int(x['target']),rendered_at_unix=time.time(),viewed=False));covered.add(r['clip_id']);created.append(r['case_id'])
index.write_text(json.dumps(known,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(new=created,registered=len(known))))
