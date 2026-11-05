"""Exact previous seven views, stable module name for Windows spawn."""
import torch
from photometric_data_numpy import PhotometricFiveViews
from temporal_data import TemporalHalves

class SevenViews:
 def __init__(self,rows,root,regions):
  self.whole=PhotometricFiveViews(rows,root,regions);self.halves=TemporalHalves(rows,root,regions)
 def __len__(self):return len(self.whole)
 def __getitem__(self,i):
  a=self.whole[i];b=self.halves[i];assert a['clip_id']==b['clip_id']
  return dict(a,x=torch.cat([a['x'],b['x']],0),frame_indices=torch.cat([a['frame_indices'],b['frame_indices']],0))
