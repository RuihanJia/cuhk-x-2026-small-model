"""Deterministic half-window augmentation; original short clips stay intact."""
import hashlib,random
import torch
from photometric_data_numpy import PhotometricClips,PhotometricFiveViews,train_metadata

def window(seed,epoch,cid,n,arm):
 assert n>0 and arm in ('CONTROL','WINDOW')
 rng=random.Random(int.from_bytes(hashlib.sha256(f'window:{seed}:{epoch}:{cid}'.encode()).digest()[:8],'big'))
 if arm=='CONTROL' or n<=16 or rng.random()>=.5:return 0,n
 length=min(n,max(16,(n+1)//2));start=rng.randrange(n-length+1)
 return start,start+length

class WindowClips(PhotometricClips):
 def __init__(self,*args,epoch_value,arm,**kwargs):
  super().__init__(*args,arm='RAW_CONTROL',**kwargs);self.epoch_value=epoch_value;self.window_arm=arm
 def __getitem__(self,i):
  self.epoch=self.epoch_value.value+16;row=self.rows[i];st=row['streams'][0]
  lo,hi=window(self.seed,self.epoch,row['clip_id'],len(st['files']),self.window_arm)
  if (lo,hi)==(0,len(st['files'])):
   b=super().__getitem__(i)
  else:
   copy=dict(row,streams=[dict(st,files=st['files'][lo:hi])])
   one=PhotometricClips([copy],self.root,self.seed,self.roi,training=True,epoch=self.epoch)
   b=one[0];b['frame_indices']=b['frame_indices']+lo
  return dict(b,window=torch.tensor([lo,hi],dtype=torch.int64))

class SevenViews:
 def __init__(self,rows,root,regions):
  from temporal_data import TemporalHalves
  self.whole=PhotometricFiveViews(rows,root,regions);self.halves=TemporalHalves(rows,root,regions)
 def __len__(self):return len(self.whole)
 def __getitem__(self,i):
  a=self.whole[i];b=self.halves[i];assert a['clip_id']==b['clip_id']
  return dict(a,x=torch.cat([a['x'],b['x']],0),frame_indices=torch.cat([a['frame_indices'],b['frame_indices']],0))
