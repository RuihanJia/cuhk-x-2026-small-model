from pathlib import Path
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from pose_view_data import PoseClips
from activity_view_data import global_tensor,normalized
from motion_video_runtime.data import safe_path

def frame_indices(n,start,end):
 assert n>0 and 0<=start<end<=1
 return [min(n-1,int((start+(k+.5)*(end-start)/16)*n)) for k in range(16)]

class TemporalHalves(PoseClips):
 def __init__(self,rows,root,regions):super().__init__(rows,root,0,regions)
 def tensor_span(self,row,start,end,cache):
  st=row['streams'][0];box=self.roi[row['clip_id']][0]['box'];frames=[];ix=frame_indices(len(st['files']),start,end)
  for i in ix:
   if i not in cache:
    with Image.open(safe_path(self.root,st['directory'])/st['files'][i]) as im:
     im=im.convert('RGB')
     if box is not None:
      x,y,w,h=box;assert min(x,y)>=0 and x+w<=im.width and y+h<=im.height;im=im.crop((x,y,x+w,y+h))
     cache[i]=TF.pil_to_tensor(im)
   frames.append(cache[i])
  x=torch.stack(frames).float().div_(255);x=TF.resize(x,[224,224],antialias=True) if box is not None else global_tensor(x,'center')
  return normalized(x,False),torch.tensor(ix,dtype=torch.int64)
 def __getitem__(self,i):
  row=self.rows[i];cache={};a,ai=self.tensor_span(row,0.,.5,cache);b,bi=self.tensor_span(row,.5,1.,cache)
  return dict(x=torch.stack([a,b]),frame_indices=torch.stack([ai,bi]),clip_id=row['clip_id'],y=row['label'],user=row['user'])
