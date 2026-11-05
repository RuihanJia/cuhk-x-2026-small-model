from pathlib import Path
import json,time,hashlib
import numpy as np
from PIL import Image,ImageDraw
from pose_roi import box_from_detections
R=Path(__file__).resolve().parent;p=R/'pose_geometry_pilot/RESULT.json';source=json.loads(p.read_text());inp=json.loads((R/'IR_ADAPTER_INPUTS.json').read_text());out=R/'pose_geometry_review';out.mkdir(exist_ok=True);records=[]
for r in source['records']:
 frames=[np.asarray(Image.open(x['path']).convert('RGB')) for x in r['frames']];h,w=frames[0].shape[:2];box,reason=box_from_detections(r['detections'],w,h);old=inp['regions'][r['clip_id']][0]['box'];canvas=Image.new('RGB',(960,410),'white');draw=ImageDraw.Draw(canvas);draw.text((8,5),f"{r['case_id']} {r['clip_id']} | GREEN pose; RED old region",fill='black')
 for j,(a,d) in enumerate(zip(frames,r['detections'])):
  im=Image.fromarray(a);v=ImageDraw.Draw(im)
  if old:v.rectangle([old[0],old[1],old[0]+old[2],old[1]+old[3]],outline='red',width=3)
  if box:v.rectangle([box[0],box[1],box[0]+box[2],box[1]+box[3]],outline='#00ff00',width=3)
  if d['boxes']:
   for x,y,c in d['keypoints'][0]:
    if c>=.5 and x>0 and y>0:v.ellipse((x-3,y-3,x+3,y+3),fill='cyan')
  canvas.paste(im.resize((320,240)),(j*320,30))
  if box:canvas.paste(Image.fromarray(a).crop((box[0],box[1],box[0]+box[2],box[1]+box[3])).resize((128,128)),(j*320,276))
 image=out/(r['case_id']+'.jpg');canvas.save(image,quality=92);records.append({**r,'box':box,'reason':reason,'image_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),'actually_viewed':False})
result=dict(status='RENDERED_GEOMETRY_API_REPAIR_NO_ACTION_FIT',at_unix=time.time(),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),reason='Official Keypoints constructor masks x/y below0.5 tozero; skip missing keypoints. This repairs parsing of detector output, no action scores were queried.',source_code='assets/pose/site/ultralytics/engine/results.py:1309',geometry_code_sha256=hashlib.sha256((R/'pose_roi.py').read_bytes()).hexdigest(),records=records,action_predictions=0)
(out/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(cases=len(records),reliable=sum(x['reason']=='POSE_HEAD_ARMS' for x in records),fallback=sum(x['reason']=='PERSON_UPPER_BODY_FALLBACK' for x in records),no_person=sum(x['box'] is None for x in records))))
