"""Standalone label-independent activity views with the established decode fallback."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image,UnidentifiedImageError
from torchvision.transforms import functional as TF
import video_io as io

METHOD='ACTIVITY_MAXMASS_SQUARE060_SHORTSIDE_V1'


def box_from_thumbnails(frames,native_size):
    a=np.asarray(frames,dtype=np.float32)/255.;assert a.ndim==4 and a.shape[-1]==3 and np.isfinite(a).all()
    width,height=map(int,native_size);h,w=a.shape[1:3]
    if len(a)<2:return dict(box=None,reason='INSUFFICIENT_TEMPORAL_OBSERVATIONS',mass_fraction=0.)
    changes=np.diff(a,axis=0);changes-=np.median(changes,axis=(1,2),keepdims=True);activity=np.abs(changes).mean(axis=(0,3))
    mean_activity=float(activity.mean());energy=np.maximum(activity-float(np.median(activity)),0.)
    side=max(1,min(h,w,round(.6*min(h,w))));ii=np.pad(energy,((1,0),(1,0))).cumsum(0,dtype=np.float64).cumsum(1,dtype=np.float64)
    windows=ii[side:,side:]-ii[:-side,side:]-ii[side:,:-side]+ii[:-side,:-side]
    top,left=map(int,np.unravel_index(np.argmax(windows),windows.shape));total=float(energy.sum(dtype=np.float64));fraction=float(windows[top,left]/total) if total else 0.
    if mean_activity<.002 or total<=0:return dict(box=None,reason='LOW_ACTIVITY_GLOBAL_FALLBACK',mass_fraction=fraction,mean_activity=mean_activity)
    if fraction<.6:return dict(box=None,reason='DIFFUSE_ACTIVITY_GLOBAL_FALLBACK',mass_fraction=fraction,mean_activity=mean_activity)
    side=round(.6*min(width,height));x=min(width-side,max(0,round(left*width/w)));y=min(height-side,max(0,round(top*height/h)))
    return dict(box=[x,y,side,side],reason='LOCAL_ACTIVITY_CROP',mass_fraction=fraction,mean_activity=mean_activity,native_size=[width,height],thumbnail_size=[w,h])


def global_tensor(value,crop):
    x=TF.resize(value,[256],antialias=True);h,w=x.shape[-2:]
    if crop=='center':return TF.center_crop(x,[224,224])
    if w>=h:return TF.crop(x,int(round((h-224)/2.)),0 if crop=='low' else w-224,224,224)
    return TF.crop(x,0 if crop=='low' else h-224,int(round((w-224)/2.)),224,224)


def normalized(value):return TF.normalize(value,[.45]*3,[.225]*3).permute(1,0,2,3).contiguous()


def load_activity_five(path):
    files=io.sorted_pngs(Path(path))
    if not files:return None,0,0,dict(box=None,reason='MISSING_MODALITY')
    decoded={};invalid=set();substitutions=0
    def decode(target):
        nonlocal substitutions
        for index in sorted(range(len(files)),key=lambda i:(abs(i-target),i)):
            if index not in decoded:
                try:
                    with Image.open(files[index]) as im:decoded[index]=im.convert('RGB').copy()
                except (OSError,ValueError,UnidentifiedImageError):decoded[index]=None;invalid.add(index)
            if decoded[index] is not None:
                substitutions+=int(index!=target);return decoded[index]
        return None
    indices=sorted(set(min(len(files)-1,int((k+.5)*len(files)/16)) for k in range(16)))
    frames=[];native=None
    for index in indices:
        im=decode(index)
        if im is None:return None,substitutions,len(invalid),dict(box=None,reason='ALL_FRAMES_UNDECODABLE')
        if native is None:native=im.size
        assert im.size==native,'FRAME_SIZE_DRIFT'
        scale=120/min(im.size);size=tuple(round(v*scale) for v in im.size)
        frames.append(np.asarray(im.resize(size,Image.Resampling.BILINEAR),dtype=np.uint8))
    region=box_from_thumbnails(np.stack(frames),native);region['method']=METHOD
    views=[]
    for phase,crop in [(.5,'center'),(.5,'low'),(.5,'high'),(.25,'center'),(.75,'center')]:
        images=[]
        for k in range(16):
            im=decode(min(len(files)-1,int((k+phase)*len(files)/16)))
            if im is None:return None,substitutions,len(invalid),region
            assert im.size==native,'FRAME_SIZE_DRIFT';images.append(TF.pil_to_tensor(im))
        value=torch.stack(images).float().div_(255);g=global_tensor(value,crop)
        if region['box'] is None:local=g.clone()
        else:
            x,y,w,h=region['box'];local=TF.resize(TF.crop(value,y,x,h,w),[224,224],antialias=True)
        views.append(torch.stack([normalized(g),normalized(local)]))
    value=torch.stack(views);assert value.shape==(5,2,3,16,224,224)
    return value,substitutions,len(invalid),region


def video_prediction(model,sample,device,return_phase_logits=False):
    values=[];names=[];substitutions=invalid=0;regions={}
    for name in io.MODALITIES:
        x,sub,bad,region=load_activity_five(sample.sample_dir/name);substitutions+=sub;invalid+=bad;regions[name]=region
        if x is not None:values.append(x);names.append(name)
    if not values:
        assert not return_phase_logits,'QA_REQUIRES_AVAILABLE_VIDEO'
        return np.zeros(40,np.float32),np.zeros(40,np.float32),np.zeros(40,np.float32),False,substitutions,invalid
    inputs=torch.stack(values,2);logits=[]
    with torch.inference_mode():
        for phase in range(5):
            pair=[]
            for spatial in range(2):pair.append(model(inputs[phase,spatial].to(device)).float().cpu().numpy())
            logits.append(np.stack(pair))
    z=np.stack(logits).astype(np.float32);assert np.isfinite(z).all()
    global_z=z[:,0].mean((0,1));local_z=z[:,1].mean((0,1));v=(global_z+local_z)*.5
    by_mod={name:.5*(z[:,0,i].mean(0)+z[:,1,i].mean(0)) for i,name in enumerate(names)}
    if return_phase_logits:return dict(logits=v,depth=by_mod.get('Depth_Color',v),ir=by_mod.get('IR',v),phase_logits=z,regions=regions,substitutions=substitutions,invalid=invalid)
    return v,by_mod.get('Depth_Color',v),by_mod.get('IR',v),True,substitutions,invalid
