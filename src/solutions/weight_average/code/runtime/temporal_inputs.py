"""Fixed first/last half views using the original decoder and ROI transform."""
import torch
from torchvision.transforms import functional as TF
import video_io as io
from detail_inputs import decode_nearest
from activity_inputs import normalized,global_tensor

def local_halves(path,box):
 files=io.sorted_pngs(path);cache={};vs=[];sub=bad=0
 if not files:return None,sub,bad
 for start,end in [(0.,.5),(.5,1.)]:
  frames=[]
  for k in range(16):
   ix=min(len(files)-1,int((start+(k+.5)*(end-start)/16)*len(files)))
   im,s,b=decode_nearest(files,ix,cache);sub+=s;bad+=b
   if im is None:return None,sub,bad
   if box is not None:
    x,y,w,h=box;assert min(x,y)>=0 and x+w<=im.width and y+h<=im.height;im=im.crop((x,y,x+w,y+h))
   frames.append(TF.pil_to_tensor(im))
  v=torch.stack(frames).float().div_(255);v=TF.resize(v,[224,224],antialias=True) if box is not None else global_tensor(v,'center');vs.append(normalized(v))
 return torch.stack(vs),sub,bad
