"""Fixed clip-level contrast stretch on original P8 headarm inputs.

RAW_CONTROL returns original tensors without arithmetic. CLIP_STRETCH applies
one float32 transform to each sampled16-frame view independently. No geometry,
label routing, temporal gain fitting, or external model is introduced.
"""
import hashlib,random
import torch
from pose_view_data import PoseClips,PoseFiveViews
ARMS=('RAW_CONTROL','CLIP_STRETCH')
PHASES=(.5,.5,.5,.25,.75)


def stretch_normalized(x):
    assert x.dtype==torch.float32 and x.ndim==4 and x.shape[:2]==(3,16)
    value=(x*.225+.45).clamp(0,1)
    low,high=torch.quantile(value.reshape(-1),torch.tensor([.01,.99],dtype=torch.float32,device=value.device),interpolation='linear')
    denominator=torch.maximum(high-low,value.new_tensor(1/255))
    transformed=(((value-low)/denominator).clamp(0,1)-.45)/.225
    assert transformed.dtype==torch.float32 and torch.isfinite(transformed).all()
    return transformed,float(low),float(high),float(denominator)


def train_metadata(seed,epoch,clip_id,n,training):
    rng=random.Random(int.from_bytes(hashlib.sha256(f'{seed}:{epoch}:{clip_id}'.encode()).digest()[:8],'big'))
    relative=[rng.random() if training else .5 for _ in range(16)]
    flip=bool(training and rng.random()<.5)
    indices=[min(n-1,int((k+relative[k])*n/16)) for k in range(16)]
    return torch.tensor(indices,dtype=torch.int64),flip


class PhotometricClips(PoseClips):
    def __init__(self,rows,root,seed,roi,training=False,epoch=0,arm='RAW_CONTROL'):
        super().__init__(rows,root,seed,roi,True,training,epoch)
        assert arm in ARMS;self.arm=arm
    def __getitem__(self,index):
        b=super().__getitem__(index);row=self.rows[index]
        indices,flip=train_metadata(self.seed,self.epoch,row['clip_id'],len(row['streams'][0]['files']),self.training)
        low,high,denominator=0.,1.,1.
        if self.arm=='CLIP_STRETCH':b['x'],low,high,denominator=stretch_normalized(b['x'])
        return dict(b,frame_indices=indices,flip=flip,stretch_low=low,stretch_high=high,denominator=denominator)


class PhotometricFiveViews(PoseFiveViews):
    def __init__(self,rows,root,roi,arm='RAW_CONTROL'):
        super().__init__(rows,root,roi);assert arm in ARMS;self.arm=arm
    def __getitem__(self,index):
        b=super().__getitem__(index);row=self.rows[index];n=len(row['streams'][0]['files'])
        indices=torch.tensor([[min(n-1,int((k+p)*n/16)) for k in range(16)] for p in PHASES],dtype=torch.int64)
        low,high,denominator=[0.]*5,[1.]*5,[1.]*5
        if self.arm=='CLIP_STRETCH':
            transformed=[]
            for i in range(5):
                if i in (1,2) and self.roi[row['clip_id']][0]['box'] is not None:
                    value=transformed[0];lo,hi,den=low[0],high[0],denominator[0]
                else:value,lo,hi,den=stretch_normalized(b['x'][i])
                transformed.append(value);low[i],high[i],denominator[i]=lo,hi,den
            b['x']=torch.stack(transformed)
        return dict(b,frame_indices=indices,flip=False,stretch_low=torch.tensor(low,dtype=torch.float32),stretch_high=torch.tensor(high,dtype=torch.float32),denominator=torch.tensor(denominator,dtype=torch.float32))
