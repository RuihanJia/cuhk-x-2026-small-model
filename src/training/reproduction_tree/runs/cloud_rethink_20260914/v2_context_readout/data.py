"""Fixed 1.5 ROI context, three original whole-clip temporal phases; no fit."""
import math
import torch
from pose_view_data import PoseClips

PHASES=(.5,.25,.75)
METHOD='CENTERED_SQUARE_EXPAND_1P5_CLAMP_TO_NATIVE_V1'

def expand_box(box,native_size,factor=1.5):
 if box is None:return None
 width,height=map(int,native_size);x,y,w,h=map(int,box)
 assert min(x,y)>=0 and min(w,h)>0 and x+w<=width and y+h<=height
 assert w==h,'ORIGINAL_ROI_NOT_SQUARE'
 assert factor in (1.,1.5),'NO_SCALE_SCAN'
 side=min(math.ceil(factor*w),min(width,height));cx=x+w/2.;cy=y+h/2.
 left=min(width-side,max(0,math.floor(cx-side/2.)))
 top=min(height-side,max(0,math.floor(cy-side/2.)))
 return [left,top,side,side]

class ContextThreeViews(PoseClips):
 def __init__(self,rows,data_root,regions,factor=1.5):
  self.original_regions=regions
  mapped={}
  for row in rows:
   cid=row['clip_id'];region=dict(regions[cid][0])
   region['box']=expand_box(region['box'],region['native_size'],factor)
   mapped[cid]=[region]
  super().__init__(rows,data_root,0,mapped,training=False)
 def __getitem__(self,index):
  row=self.rows[index];cache={};box=self.roi[row['clip_id']][0]['box']
  values=[self.tensor(row,[phase]*16,False,cache,'center') for phase in PHASES]
  n=len(row['streams'][0]['files'])
  indices=[[min(n-1,int((k+phase)*n/16)) for k in range(16)] for phase in PHASES]
  return dict(x=torch.stack(values),frame_indices=torch.tensor(indices,dtype=torch.int64),
   context_box=torch.tensor(box if box is not None else [-1]*4,dtype=torch.int64),
   native_size=torch.tensor(self.roi[row['clip_id']][0]['native_size']),
   fallback=box is None,clip_id=row['clip_id'],y=row['label'],user=row['user'])
