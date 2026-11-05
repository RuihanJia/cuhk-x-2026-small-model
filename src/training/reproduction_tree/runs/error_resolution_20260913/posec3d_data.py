"""COCO17 heatmaps with official PoseCompact and UniformSample conventions."""
import numpy as np
import torch
SWAP=[0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15]

def sample_indices(n,rng=None,length=48):
 # One official deterministic test clip (seed255); training uses per-clip RNG.
 training=rng is not None;rng=np.random.RandomState(255) if rng is None else rng
 if n<length:
  start=int(rng.randint(n)) if training else 0;idx=np.arange(start,start+length)
 elif n<2*length:
  offset=np.zeros(length+1,dtype=np.int64);offset[rng.choice(length+1,n-length,replace=False)]=1;idx=np.arange(length)+offset.cumsum()[:-1]
 else:
  bins=np.array([i*n//length for i in range(length+1)]);idx=bins[:-1]+rng.randint(np.diff(bins))
 return idx%n

def compact_and_resize(kps,native_size,size=64):
 kps=kps.copy();w,h=native_size;valid=(kps[...,2]>0)&(kps[...,:2]>0).all(-1);kps[~valid]=0
 x=kps[...,0];y=kps[...,1]
 if valid.any():
  x0,x1=float(x[valid].min()),float(x[valid].max());y0,y1=float(y[valid].min()),float(y[valid].max())
  if x1-x0>=10 and y1-y0>=10:
   cx,cy=(x0+x1)/2,(y0+y1)/2;half=max((x1-x0)*.625,(y1-y0)*.625);x0,x1,y0,y1=int(cx-half),int(cx+half),int(cy-half),int(cy+half);x[valid]-=x0;y[valid]-=y0;w,h=x1-x0,y1-y0
 ratio=size/min(w,h);neww,newh=int(w*ratio+.5),int(h*ratio+.5);x[valid]=x[valid]*(neww/w)-(neww-size)//2;y[valid]=y[valid]*(newh/h)-(newh-size)//2
 return kps.astype(np.float32)

def heatmaps(kps,size=64,sigma=.6,flip=False):
 # kps B,T,J,3. Match official integer support truncation at 3 sigma.
 if flip:return heatmaps(kps,size,sigma,False)[:,SWAP].flip(-1)
 xy=kps[...,:2];score=kps[...,2];mx=xy[...,0,None,None];my=xy[...,1,None,None];grid=torch.arange(size,dtype=kps.dtype,device=kps.device);xx=grid[None,None,None,None,:];yy=grid[None,None,None,:,None]
 x0=(mx-3*sigma).to(torch.int32).clamp_min(0);x1=(mx+3*sigma).to(torch.int32).add(1).clamp_max(size);y0=(my-3*sigma).to(torch.int32).clamp_min(0);y1=(my+3*sigma).to(torch.int32).add(1).clamp_max(size)
 z=torch.exp(-((xx-mx).square()+(yy-my).square())/(2*sigma*sigma))*score[...,None,None];z*=((xx>=x0)&(xx<x1)&(yy>=y0)&(yy<y1)&(score[...,None,None]>=1e-4));return z.permute(0,2,1,3,4).contiguous()
