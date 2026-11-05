"""Label-independent clip activity crop, paired with the unchanged global view."""
import hashlib
from pathlib import Path
import random
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from motion_video_runtime.data import safe_path

METHOD='ACTIVITY_MAXMASS_SQUARE060_SHORTSIDE_V1'


def box_from_thumbnails(frames, native_size):
    """No labels, user metadata, classes, skeleton projection or fitted model."""
    a=np.asarray(frames,dtype=np.float32)/255.
    assert a.ndim==4 and a.shape[-1]==3 and np.isfinite(a).all()
    width,height=map(int,native_size);h,w=a.shape[1:3]
    if len(a)<2:return dict(box=None,reason='INSUFFICIENT_TEMPORAL_OBSERVATIONS',mass_fraction=0.)
    changes=np.diff(a,axis=0)
    changes-=np.median(changes,axis=(1,2),keepdims=True)
    activity=np.abs(changes).mean(axis=(0,3))
    mean_activity=float(activity.mean())
    # Remove the per-clip background noise floor; the same formula is used on
    # every clip independently, including inference. No population statistics.
    energy=np.maximum(activity-float(np.median(activity)),0.)
    side=max(1,min(h,w,round(.6*min(h,w))))
    ii=np.pad(energy,((1,0),(1,0))).cumsum(0,dtype=np.float64).cumsum(1,dtype=np.float64)
    windows=ii[side:,side:]-ii[:-side,side:]-ii[side:,:-side]+ii[:-side,:-side]
    top,left=map(int,np.unravel_index(np.argmax(windows),windows.shape))
    total=float(energy.sum(dtype=np.float64));fraction=float(windows[top,left]/total) if total else 0.
    if mean_activity<.002 or total<=0:
        return dict(box=None,reason='LOW_ACTIVITY_GLOBAL_FALLBACK',mass_fraction=fraction,mean_activity=mean_activity)
    if fraction<.6:
        return dict(box=None,reason='DIFFUSE_ACTIVITY_GLOBAL_FALLBACK',mass_fraction=fraction,mean_activity=mean_activity)
    native_side=round(.6*min(width,height))
    x=min(width-native_side,max(0,round(left*width/w)))
    y=min(height-native_side,max(0,round(top*height/h)))
    return dict(box=[x,y,native_side,native_side],reason='LOCAL_ACTIVITY_CROP',mass_fraction=fraction,
                mean_activity=mean_activity,native_size=[width,height],thumbnail_size=[w,h])


def compute_stream_box(data_root,stream):
    directory=safe_path(data_root,stream['directory']);files=stream['files']
    indices=sorted(set(min(len(files)-1,int((k+.5)*len(files)/16)) for k in range(16)))
    frames=[];hashes=[];native_size=None
    for index in indices:
        path=directory/files[index]
        with Image.open(path) as image:
            image=image.convert('RGB')
            if native_size is None:native_size=image.size
            assert image.size==native_size,'FRAME_SIZE_DRIFT'
            scale=120/min(image.size);size=tuple(round(v*scale) for v in image.size)
            frames.append(np.asarray(image.resize(size,Image.Resampling.BILINEAR),dtype=np.uint8))
        hashes.append(dict(file=files[index],sha256=hashlib.sha256(path.read_bytes()).hexdigest().upper()))
    result=box_from_thumbnails(np.stack(frames),native_size)
    result.update(method=METHOD,sampled_frame_hashes=hashes,unique_frames=len(indices),native_size=list(native_size))
    return result


def global_tensor(value,crop='center'):
    x=TF.resize(value,[256],antialias=True);h,w=x.shape[-2:]
    if crop=='center':return TF.center_crop(x,[224,224])
    if w>=h:return TF.crop(x,int(round((h-224)/2.)),0 if crop=='low' else w-224,224,224)
    return TF.crop(x,0 if crop=='low' else h-224,int(round((w-224)/2.)),224,224)


def local_tensor(value,box,fallback):
    if box is None:return fallback.clone()
    x,y,w,h=box
    assert min(x,y)>=0 and min(w,h)>0 and x+w<=value.shape[-1] and y+h<=value.shape[-2]
    return TF.resize(TF.crop(value,y,x,h,w),[224,224],antialias=True)


def normalized(value,flip=False):
    if flip:value=value.flip(-1)
    return TF.normalize(value,[.45]*3,[.225]*3).permute(1,0,2,3).contiguous()


class ActivityClips:
    def __init__(self,rows,data_root,seed_value,roi,local_view,training=False,epoch=0):
        self.rows,self.root,self.seed,self.roi,self.local_view,self.training,self.epoch=rows,str(data_root),seed_value,roi,local_view,training,epoch
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]
        rng=random.Random(int.from_bytes(hashlib.sha256(f'{self.seed}:{self.epoch}:{row["clip_id"]}'.encode()).digest()[:8],'big'))
        relative=[rng.random() if self.training else .5 for _ in range(16)]
        flip=self.training and rng.random()<.5
        global_views=[];second_views=[]
        for j,stream in enumerate(row['streams']):
            directory=safe_path(self.root,stream['directory']);files=stream['files'];frames=[]
            for k in range(16):
                with Image.open(directory/files[min(len(files)-1,int((k+relative[k])*len(files)/16))]) as im:
                    frames.append(TF.pil_to_tensor(im.convert('RGB')))
            value=torch.stack(frames).float().div_(255)
            g=global_tensor(value)
            loc=local_tensor(value,self.roi[row['clip_id']][j]['box'],g) if self.local_view else g.clone()
            global_views.append(normalized(g,flip));second_views.append(normalized(loc,flip))
        return dict(x=torch.stack([torch.stack(global_views),torch.stack(second_views)]),
                    y=row['label'],user=row['user'],clip_id=row['clip_id'])


class ActivityFiveViews:
    def __init__(self,rows,data_root,roi,local_view):
        self.rows,self.root,self.roi,self.local_view=rows,str(data_root),roi,local_view;self.seed=0;self.epoch=0
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index];views=[]
        for phase,crop in [(.5,'center'),(.5,'low'),(.5,'high'),(.25,'center'),(.75,'center')]:
            globals_=[];locals_=[]
            for j,stream in enumerate(row['streams']):
                directory=safe_path(self.root,stream['directory']);files=stream['files'];frames=[]
                for k in range(16):
                    with Image.open(directory/files[min(len(files)-1,int((k+phase)*len(files)/16))]) as im:
                        frames.append(TF.pil_to_tensor(im.convert('RGB')))
                value=torch.stack(frames).float().div_(255);g=global_tensor(value,crop)
                loc=local_tensor(value,self.roi[row['clip_id']][j]['box'],g) if self.local_view else g.clone()
                globals_.append(normalized(g));locals_.append(normalized(loc))
            views.append(torch.stack([torch.stack(globals_),torch.stack(locals_)]))
        return dict(x=torch.stack(views),y=row['label'],user=row['user'],clip_id=row['clip_id'])
