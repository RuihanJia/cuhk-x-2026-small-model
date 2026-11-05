"""Single IR local view, identical sampling/resize but no unused global tensors."""
import hashlib,random
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from activity_view_data import global_tensor,normalized
from motion_video_runtime.data import safe_path

class PoseClips:
 def __init__(self,rows,data_root,seed_value,roi,local_view=True,training=False,epoch=0):
  self.rows,self.root,self.seed,self.roi,self.training,self.epoch=rows,data_root,seed_value,roi,training,epoch
 def __len__(self):return len(self.rows)
 def tensor(self,row,relative,flip,cache,crop='center'):
  st=row['streams'][0];box=self.roi[row['clip_id']][0]['box'];frames=[]
  for k in range(16):
   i=min(len(st['files'])-1,int((k+relative[k])*len(st['files'])/16))
   if i not in cache:
    with Image.open(safe_path(self.root,st['directory'])/st['files'][i]) as im:
     im=im.convert('RGB')
     if box is not None:
      x,y,w,h=box;assert min(x,y)>=0 and x+w<=im.width and y+h<=im.height;im=im.crop((x,y,x+w,y+h))
     cache[i]=TF.pil_to_tensor(im)
   frames.append(cache[i])
  v=torch.stack(frames).float().div_(255)
  v=TF.resize(v,[224,224],antialias=True) if box is not None else global_tensor(v,crop)
  return normalized(v,flip)
 def __getitem__(self,index):
  row=self.rows[index];rng=random.Random(int.from_bytes(hashlib.sha256(f'{self.seed}:{self.epoch}:{row["clip_id"]}'.encode()).digest()[:8],'big'));relative=[rng.random() if self.training else .5 for _ in range(16)];flip=self.training and rng.random()<.5
  return dict(x=self.tensor(row,relative,flip,{}),y=row['label'],user=row['user'],clip_id=row['clip_id'])

class PoseFiveViews(PoseClips):
 def __init__(self,rows,data_root,roi,local_view=True):super().__init__(rows,data_root,0,roi)
 def __getitem__(self,index):
  row=self.rows[index];cache={};vs=[];box=self.roi[row['clip_id']][0]['box']
  for phase,crop in [(.5,'center'),(.5,'low'),(.5,'high'),(.25,'center'),(.75,'center')]:
   if phase==.5 and vs and box is not None:vs.append(vs[0])
   else:vs.append(self.tensor(row,[phase]*16,False,cache,crop))
  return dict(x=torch.stack(vs),y=row['label'],user=row['user'],clip_id=row['clip_id'])
