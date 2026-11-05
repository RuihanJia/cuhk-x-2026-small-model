"""Frozen contextual, Depth-guided IR, and COCO-guided IR inference inputs."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image,UnidentifiedImageError
from torchvision.transforms import functional as TF
import video_io as io
from activity_inputs import load_activity_five,normalized,global_tensor
from pose_roi import box_from_detections

def decode_nearest(files,target,cache):
 bad=0
 for i in sorted(range(len(files)),key=lambda i:(abs(i-target),i)):
  if i not in cache:
   try:
    with Image.open(files[i]) as im:cache[i]=im.convert('RGB').copy()
   except (OSError,ValueError,UnidentifiedImageError):cache[i]=None
  if cache[i] is not None:return cache[i],int(i!=target),bad
  bad+=1
 return None,0,bad

def detector_frames(sample):
 files=io.sorted_pngs(sample.sample_dir/'IR');cache={};frames=[];sub=bad=0
 for index in sorted(set(min(len(files)-1,int((k+.5)*len(files)/8)) for k in range(8))) if files else []:
  im,s,b=decode_nearest(files,index,cache);sub+=s;bad+=b
  if im is None:continue
  a=np.asarray(im);frames.append(np.round(np.sqrt(a.astype(np.float32)/255)*255).astype(np.uint8)[:,:,::-1].copy())
 return frames,sub,bad

def detector_result(predictions,frames):
 if not frames:return dict(box=None,reason='NO_IR',detections=[])
 ds=[dict(boxes=p.boxes.xyxy.tolist(),confidence=p.boxes.conf.tolist(),keypoints=p.keypoints.data.tolist()) for p in predictions];h,w=frames[0].shape[:2];box,reason=box_from_detections(ds,w,h)
 return dict(box=box,reason=reason,native_size=[w,h],detections=ds)

def local_five(path,box):
 files=io.sorted_pngs(path);cache={};vs=[];sub=bad=0
 if not files:return None,sub,bad
 for phase,crop in [(.5,'center'),(.5,'low'),(.5,'high'),(.25,'center'),(.75,'center')]:
  if phase==.5 and vs and box is not None:vs.append(vs[0]);continue
  frames=[]
  for k in range(16):
   im,s,b=decode_nearest(files,min(len(files)-1,int((k+phase)*len(files)/16)),cache);sub+=s;bad+=b
   if im is None:return None,sub,bad
   if box:
    x,y,w,h=box;assert min(x,y)>=0 and x+w<=im.width and y+h<=im.height;im=im.crop((x,y,x+w,y+h))
   frames.append(TF.pil_to_tensor(im))
  value=torch.stack(frames).float().div_(255);value=TF.resize(value,[224,224],antialias=True) if box is not None else global_tensor(value,crop);vs.append(normalized(value))
 return torch.stack(vs),sub,bad

class ContextViews:
 def __init__(self,samples,regions):self.samples=samples;self.regions=regions
 def __len__(self):return len(self.samples)
 def __getitem__(self,i):
  sample=self.samples[i];values=[];names=[];regions={};sub=bad=0
  for name in io.MODALITIES:
   x,s,b,region=load_activity_five(sample.sample_dir/name);sub+=s;bad+=b;regions[name]=region
   if x is not None:values.append(x);names.append(name)
  if not values:return dict(index=i,x=torch.empty(0),pose=torch.empty(0),names=names,sub=sub,bad=bad)
  x=torch.stack(values,2);pose=torch.empty(0);fallback='NO_IR'
  if 'IR' in names:
   ir=names.index('IR');irfiles=io.sorted_pngs(sample.sample_dir/'IR');box=regions['IR']['box'];dep=regions.get('Depth_Color',{});db=dep.get('box');fallback='IR_ACTIVITY'
   if db is not None:
    with Image.open(irfiles[0]) as im:size=list(im.size)
    aligned=len(irfiles)==len(io.sorted_pngs(sample.sample_dir/'Depth_Color')) and size==dep.get('native_size')
    if aligned:box=db;fallback='DEPTH_GUIDED';x[:,1,ir],s,b=local_five(sample.sample_dir/'IR',box);sub+=s;bad+=b
   pb=self.regions[i]['box'];pb=pb if pb is not None else box
   pose,s,b=local_five(sample.sample_dir/'IR',pb);sub+=s;bad+=b
   if pose is None:pose=torch.empty(0)
  return dict(index=i,x=x,pose=pose,names=names,sub=sub,bad=bad,fallback=fallback)

def predict_views(model,item,device,batch_size):
 if not item['x'].numel():return dict(video=np.zeros(40,np.float32),depth=np.zeros(40,np.float32),ir=np.zeros(40,np.float32),pose_zero=np.zeros(40,np.float32),pose_private=np.zeros(40,np.float32),video_available=False,pose_available=False)
 def call(x,private):
  vals=[]
  for start in range(0,len(x),batch_size):vals.append(model(x[start:start+batch_size].to(device,non_blocking=True),private).float().cpu().numpy())
  return np.concatenate(vals)
 with torch.inference_mode():
  n=len(item['names']);z=call(item['x'].flatten(0,2),False).reshape(5,2,n,40);video=.5*(z[:,0].mean((0,1))+z[:,1].mean((0,1)));by={name:.5*(z[:,0,j].mean(0)+z[:,1,j].mean(0)) for j,name in enumerate(item['names'])};pz=pp=np.zeros(40,np.float32)
  if item['pose'].numel():pz=call(item['pose'],False).mean(0);pp=call(item['pose'],True).mean(0)
 return dict(video=video,depth=by.get('Depth_Color',video),ir=by.get('IR',video),pose_zero=pz,pose_private=pp,video_available=True,pose_available=bool(item['pose'].numel()),phase_logits=z)
