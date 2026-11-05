"""Every changed correctness case: original frames plus exact two crop regions."""
from pathlib import Path
import json,csv,time,hashlib
import numpy as np
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parent;out=R/'audit/crop_contrasts';out.mkdir(exist_ok=True)
src=R/'pose_complement_calibration/ALL_PAIRED_CASES.csv';rows=[q for q in csv.DictReader(src.open(encoding='utf-8-sig')) if q['recipe']=='POSE_DUAL6' and (q['R6']==q['target'])!=(q['prediction']==q['target'])];rows.sort(key=lambda q:(q['R6']!=q['target'],int(q['target']),int(q['user']),q['clip_id']))
inp=json.loads((R/'IR_ADAPTER_INPUTS.json').read_text());meta={x['clip_id']:x for x in inp['rows']};pose=json.loads((R/'POSE_ADAPTER_INPUTS.json').read_text())['regions'];root=Path(r'D:\Datasets\CUHK-X_Small_Model_Track\Small-Model-Track');records=[]
for i,q in enumerate(rows):
 k=q['clip_id'];st=meta[k]['streams'][0];old=inp['regions'][k][0]['box'];new=pose[k][0]['box'];im=Image.new('RGB',(800,520),'white');d=ImageDraw.Draw(im);d.text((5,3),f"X{i:03d} {'HARM' if q['R6']==q['target'] else 'FIX'} {k}",fill='black');d.text((5,19),f"R6 {q['R6']} -> dual {q['prediction']} truth {q['target']} | red old; green new",fill='black');d.text((5,34),f"Old: {old}  New: {new}  Native frames {len(st['files'])}",fill='black');hashes=[]
 for j,index in enumerate(np.linspace(0,len(st['files'])-1,4).round().astype(int)):
  path=root/st['directory']/st['files'][index];a=Image.open(path).convert('RGB');b=a.copy();dr=ImageDraw.Draw(b)
  for box,color in [(old,'red'),(new,'lime')]:
   if box:x,y,w,h=box;dr.rectangle((x,y,x+w,y+h),outline=color,width=3)
  im.paste(b.resize((200,150)),(j*200,54));d.text((j*200+3,204),f'frame {index}',fill='black')
  for row,box in enumerate([old,new]):
   if box:x,y,w,h=box;c=a.crop((x,y,x+w,y+h)).resize((150,150))
   else:
    # Match global center geometry for unlocalized diagnostic view.
    scale=256/min(a.size);bsize=tuple(round(x*scale) for x in a.size);b=a.resize(bsize);left=round((b.width-224)/2);top=round((b.height-224)/2);c=b.crop((left,top,left+224,top+224)).resize((150,150))
   im.paste(c,(j*200,222+row*150))
  hashes.append(dict(index=int(index),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
 dst=out/f'X{i:03d}.jpg';im.save(dst,quality=90);records.append(dict(case_id=f'X{i:03d}',**q,image=dst.name,image_sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),old_box=old,new_box=new,source_frames=hashes,actually_viewed=False))
for start in range(0,len(records),4):
 canvas=Image.new('RGB',(1600,1040),'#dddddd')
 for j,rec in enumerate(records[start:start+4]):canvas.paste(Image.open(out/rec['image']),(800*(j%2),520*(j//2)))
 canvas.save(out/f'SHEET_{start//4:03d}.jpg',quality=88)
(out/'INDEX.json').write_text(json.dumps(dict(status='RENDERED_NOT_YET_REVIEWED',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),at_unix=time.time(),records=records,diagnostic_only=True,model_input_pixels_unchanged=True),indent=2)+'\n');print(json.dumps(dict(cases=len(records),sheets=(len(records)+3)//4,harm=sum(q['R6']==q['target'] for q in records))))
