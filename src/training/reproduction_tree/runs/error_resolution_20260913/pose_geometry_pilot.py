"""CPU visual QA of an admitted person pose locator; no action predictions or fitting."""
from pathlib import Path
import os,sys,json,hashlib,time
import numpy as np
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parent;ROOT=R.parents[1];A=ROOT/'assets/pose'
os.environ['YOLO_CONFIG_DIR']=str(A/'settings');os.environ['YOLO_AUTOINSTALL']='false';os.environ['PYTEST_CURRENT_TEST']='pose_geometry_audit';sys.path.insert(0,str(A/'site'))
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import PoseModel
from ultralytics.utils import SETTINGS
SETTINGS.update({'sync':False})
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def proposed_box(preds,w,h):
 points=[];valid=0;fallback=[]
 for p in preds:
  if not len(p.boxes):continue
  b=p.boxes.xyxy[0].cpu().numpy();fallback.extend([[b[0],b[1]],[b[2],b[1]+.75*(b[3]-b[1])]])
  k=p.keypoints.data[0].cpu().numpy()
  if (k[[5,6],2]>.3).all() and (k[[9,10],2]>.3).any():
   valid+=1;points.extend(k[:11,:2][k[:11,2]>.3].tolist())
 if valid<2:points=fallback
 if len(points)<4:return None,'NO_RELIABLE_PERSON'
 a=np.asarray(points);lo=a.min(0);hi=a.max(0);side=min(min(w,h),max(.25*min(w,h),1.35*float((hi-lo).max())));cx,cy=(lo+hi)/2;x=max(0,min(w-side,cx-side/2));y=max(0,min(h-side,cy-side/2))
 return [round(x),round(y),round(side),round(side)],'POSE_HEAD_ARMS' if valid>=2 else 'PERSON_UPPER_BODY_FALLBACK'
def main():
 torch.set_num_threads(2);admit=json.loads((A/'ADMISSION.json').read_text());assert admit['status']=='ADMITTED_PUBLIC_NONLANGUAGE_COCO_POSE' and sha(A/'pose_state_fp16.pt')==admit['inference_weight_sha256'];out=R/'pose_geometry_pilot';out.mkdir(exist_ok=True);assert not(out/'RESULT.json').exists()
 inp=json.loads((R/'IR_ADAPTER_INPUTS.json').read_text());rows=inp['rows'];byid={r['clip_id']:r for r in rows};ids=[]
 for u in sorted({r['user'] for r in rows}):ids.extend([r['clip_id'] for r in sorted([r for r in rows if r['user']==u],key=lambda r:hashlib.sha256(r['clip_id'].encode()).hexdigest())[:2]])
 cases=json.loads((R/'audit/NEW_ENDPOINT_CASES.json').read_text());ids=list(dict.fromkeys(ids+[x['clip_id'] for x in cases]));protocol=dict(id='POSE_GEOMETRY_CPU_VISUAL_QA_V1',at_unix=time.time(),action_training=0,action_labels_used=False,mechanism='Gamma0.5 IR forpersonlocalizer only; classifierpixels unchanged. Head/shoulder/elbow/wrist hull expanded1.35, upper-personfallback, originalcropwhen no reliableperson.',sample_rule='2 clips perTraininguser bySHA256 ordering plus14alreadyreviewederrorcases',clip_ids=ids,phases=[.1,.5,.9],confidence=.25,kpt_confidence=.3,imgsz=640,model_sha256=admit['inference_weight_sha256'],code_sha256=sha(__file__),CPU_seconds_cap=600,no_Test=True)
 (out/'PROTOCOL.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 cp=torch.load(A/'pose_state_fp16.pt',map_location='cpu',weights_only=True);net=PoseModel(cp['yaml'],ch=3,nc=1,data_kpt_shape=(17,3),verbose=False);net.load_state_dict(cp['state_dict']);net.kpt_shape=[17,3];net.names={0:'person'};net.eval().requires_grad_(False);model=YOLO(str(A/'yolo11n-pose.pt'));model.model=net
 root=Path(json.loads((R/'THERMAL_ADAPTER_PROTOCOL.json').read_text())['hosts']['local']['data_root']);records=[];start=time.time()
 for i,k in enumerate(ids):
  assert time.time()-start<600;row=byid[k];st=row['streams'][0];frames=[];paths=[]
  for phase in protocol['phases']:
   p=root/st['directory']/st['files'][min(len(st['files'])-1,int(phase*len(st['files'])))];paths.append(p);frames.append(np.asarray(Image.open(p).convert('RGB')))
  # Ultralytics expects BGR arrays; IR is grayscale but channel order is explicit.
  gamma=[np.round(np.sqrt(a.astype(np.float32)/255)*255).astype(np.uint8)[:,:,::-1].copy() for a in frames];preds=model.predict(gamma,imgsz=640,conf=.25,iou=.7,max_det=1,device='cpu',verbose=False,save=False);w,h=frames[0].shape[1],frames[0].shape[0];box,reason=proposed_box(preds,w,h);old=inp['regions'][k][0]['box'];canvas=Image.new('RGB',(960,410),'white');draw=ImageDraw.Draw(canvas);draw.text((8,5),f'P{i:03d} {k} | GREEN pose; RED Depth-guided/current crop',fill='black')
  for j,(a,pred) in enumerate(zip(frames,preds)):
   im=Image.fromarray(a);d=ImageDraw.Draw(im)
   if old:d.rectangle([old[0],old[1],old[0]+old[2],old[1]+old[3]],outline='red',width=3)
   if box:d.rectangle([box[0],box[1],box[0]+box[2],box[1]+box[3]],outline='#00ff00',width=3)
   if len(pred.boxes):
    for x,y,c in pred.keypoints.data[0].cpu().numpy():
     if c>.3:d.ellipse((x-3,y-3,x+3,y+3),fill='cyan')
   canvas.paste(im.resize((320,240)),(j*320,30))
   if box:canvas.paste(Image.fromarray(a).crop((box[0],box[1],box[0]+box[2],box[1]+box[3])).resize((128,128)),(j*320,276))
  image=out/f'P{i:03d}.jpg';canvas.save(image,quality=92);records.append(dict(case_id=f'P{i:03d}',clip_id=k,box=box,reason=reason,frames=[dict(path=str(p),sha256=sha(p)) for p in paths],detections=[dict(boxes=p.boxes.xyxy.tolist(),confidence=p.boxes.conf.tolist(),keypoints=p.keypoints.data.tolist()) for p in preds],image_sha256=sha(image),actually_viewed=False));print(json.dumps(dict(completed=i+1,total=len(ids),box=box)),flush=True)
 (out/'RESULT.json').write_text(json.dumps(dict(status='RENDERED_NOT_YET_REVIEWED',records=records,seconds=time.time()-start,protocol_sha256=sha(out/'PROTOCOL.json'),action_predictions=0),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()
