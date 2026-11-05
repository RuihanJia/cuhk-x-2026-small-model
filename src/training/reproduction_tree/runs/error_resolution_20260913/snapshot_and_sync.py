"""One bounded existing-instance snapshot; mirror only completed small artifacts."""
from pathlib import Path
import json,time,hashlib
import cloud_transport as t
R=Path(__file__).resolve().parent
def main():
 mod,c=t.connect()
 try:
  code='''from pathlib import Path
import json,time,hashlib
r=Path(%r);out=dict(at_unix=time.time(),components={})
for name in ['video_conflict_output','video_successor_output','ir_adapter_output/cloud','crop_inner_output','thermal_inner_output/cloud','pose_regions_gpu','pose_crop_output','pose_adapter_output/cloud','pose_private_inner_output/cloud','posec3d_finetune_output/cloud']:
 p=r/name;data=dict(done={},running=[],errors=[],telemetry=None)
 if not p.exists():continue
 for f in p.glob('*/DONE.json'):
  d=json.loads(f.read_text());assert d['status']=='COMPLETE';assert hashlib.sha256((f.parent/'predictions.npz').read_bytes()).hexdigest()==d['predictions_sha256'];data['done'][f.parent.name]=d
 for f in p.glob('*/STATUS.json'):
  if not(f.parent/'DONE.json').exists():data['running'].append(json.loads(f.read_text()))
 if (p/'STATUS.json').exists() and not(p/'CONTROLLER_DONE.json').exists():data['running'].append(json.loads((p/'STATUS.json').read_text()))
 data['errors']=[dict(path=str(f),content=f.read_text()) for f in p.glob('*ERROR.json')]
 if(p/'TELEMETRY.json').exists():data['telemetry']=json.loads((p/'TELEMETRY.json').read_text())
 if(p/'CONTROLLER_DONE.json').exists():data['controller_done']=json.loads((p/'CONTROLLER_DONE.json').read_text())
 out['components'][name]=data
print(json.dumps(out))
'''%t.REMOTE
  z=t.old.execute(mod,c,code,50)
  with c.open_sftp() as s:
   for component,d in z['components'].items():
    for job,done in d['done'].items():
     out=R/component/job;out.mkdir(parents=True,exist_ok=True)
     for n in (['predictions.npz'] if component in ['crop_inner_output','pose_crop_output'] else ['HISTORY.json','SPLIT.json','predictions.npz','JOB_BUDGET.json'])+(['adapter_fp16.pt'] if component.startswith(('ir_adapter','thermal_inner','pose_adapter','pose_private_inner')) else [])+(['model_fp16.pt'] if component.startswith('posec3d_finetune') else [])+['DONE.json']:
      if not(out/n).exists():
       part=out/(n+'.download.partial');s.get(t.REMOTE+'/'+component+'/'+job+'/'+n,str(part));part.replace(out/n)
     assert hashlib.sha256((out/'predictions.npz').read_bytes()).hexdigest()==done['predictions_sha256']
 except BaseException as e:
  z=dict(at_unix=time.time(),transport_error=repr(e));raise
 finally:c.close()
 z['local']={}
 for component in ['detail_finetune_output','thermal_adapter_output/local','thermal_inner_output/local','pose_inner_output','posec3d_feature_output','posec3d_finetune_output/local']:
  p=R/component;items={}
  for n in ['TELEMETRY.json','ERROR.json','CONTROLLER_ERROR.json','CONTROLLER_DONE.json','QUEUE_DONE.json']:
   if(p/n).exists():items[n]=json.loads((p/n).read_text(encoding='utf-8'))
  items['done']={f.parent.name:json.loads(f.read_text(encoding='utf-8')) for f in p.glob('*/DONE.json')}
  items['running']=[json.loads(f.read_text(encoding='utf-8')) for f in p.glob('*/STATUS.json') if not(f.parent/'DONE.json').exists()]
  if (p/'STATUS.json').exists() and not(p/'CONTROLLER_DONE.json').exists():items['running'].append(json.loads((p/'STATUS.json').read_text(encoding='utf-8')))
  z['local'][component]=items
 stamp=time.strftime('%Y%m%d_%H%M%S');dest=R/'snapshots';dest.mkdir(exist_ok=True);(dest/(stamp+'.json')).write_text(json.dumps(z,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 for v in z.get('components',{}).values():v['done']={k:dict(correct=d.get('correct'),n=d['n'],seconds=d['seconds']) for k,d in v['done'].items()}
 for v in z['local'].values():v['done']={k:dict(correct=d.get('correct'),n=d['n'],seconds=d['seconds']) for k,d in v['done'].items()}
 print(json.dumps(z,ensure_ascii=False))
if __name__=='__main__':main()
