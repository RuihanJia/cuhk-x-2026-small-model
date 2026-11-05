"""Three prospective IMU-only arms with equal B/C capacity and identical training schedules."""
from pathlib import Path
import argparse, hashlib, json, sys, time, traceback
import numpy as np
import torch
from torch import nn

class ArmIMU(nn.Module):
    def __init__(self):
        super().__init__();self.features=nn.Sequential(nn.Conv1d(6,64,7,2,3),nn.GroupNorm(8,64),nn.GELU(),nn.Conv1d(64,128,5,2,2),nn.GroupNorm(8,128),nn.GELU(),nn.Conv1d(128,128,3,2,1),nn.GroupNorm(8,128),nn.GELU(),nn.AdaptiveAvgPool1d(1));self.head=nn.Sequential(nn.Flatten(),nn.Dropout(.2),nn.Linear(128,40))
    def forward(self,x,valid):return self.head(self.features(x[:,:2].flatten(1,2)))

class DeviceIMU(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.Sequential(nn.Conv1d(3,64,7,2,3),nn.GroupNorm(8,64),nn.GELU(),nn.Conv1d(64,128,5,2,2),nn.GroupNorm(8,128),nn.GELU(),nn.Conv1d(128,128,3,2,1),nn.GroupNorm(8,128),nn.GELU(),nn.AdaptiveAvgPool1d(1));self.head=nn.Sequential(nn.Linear(645,128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,40))
    def forward(self,x,valid):
        x=torch.where(valid[:,:,None,None],x,0.);z=self.encoder(x.flatten(0,1)).reshape(len(x),5,128);z=z*valid[:,:,None]
        return self.head(torch.cat([z.flatten(1),torch.zeros_like(valid,dtype=torch.float32)],1))

def state_sha(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest().upper()

def rotations(n,fold,epoch):
    key=f'imu-device-20260910:2026091003:{fold}:{epoch}'.encode();rng=np.random.Generator(np.random.PCG64(int.from_bytes(hashlib.sha256(key).digest()[:8],'little')))
    active=rng.random(n)<.5;q=rng.standard_normal((n,5,4));q/=np.linalg.norm(q,axis=-1,keepdims=True);w,x,y,z=np.moveaxis(q,-1,0)
    mat=np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],axis=-1).reshape(n,5,3,3).astype(np.float32);mat[~active]=np.eye(3,dtype=np.float32)
    return mat

def main():
    a=argparse.ArgumentParser();a.add_argument('--root',required=True);a.add_argument('--job',required=True);a.add_argument('--protocol',required=True);a.add_argument('--deadline',type=float,required=True);args=a.parse_args();root=Path(args.root);sys.path.insert(0,str(root/'bundle'))
    from dual_host_runtime.io import atomic,read,save_torch
    from optional_time_budget import Guard
    from research_runtime.common import seed,configure,sha,digest
    from research_runtime.train import load_epoch,save_epoch
    configure();p=read(args.protocol);ph=sha(args.protocol);spec=p['jobs'][args.job];out=root/'output'/args.job;out.mkdir(parents=True,exist_ok=True);guard=Guard(out,args.deadline,root/'TELEMETRY.json')
    try:
        source=root/'bundle/data/arm_acceleration_t256.npz';assert sha(source)==p['data']['arm_sha256'];z=dict(np.load(source,allow_pickle=False));keep=z['valid'];ids=z['clip_ids'][keep];users=z['users'][keep];target=z['targets'][keep]
        raw=np.zeros((len(ids),5,3,256),np.float32);raw[:,:2]=z['x'][keep].reshape(-1,2,3,256);valid=np.zeros((len(ids),5),bool);valid[:,:2]=True;device_sha=None
        if spec['arm'] in ['C','D']:
            admission=read(root/'DEVICE_CACHE_READY.json');source=root/admission['path'];assert sha(source)==admission['sha256'];device_sha=admission['sha256'];d=dict(np.load(source,allow_pickle=False));assert np.array_equal(d['clip_ids'],z['clip_ids']) and np.array_equal(d['x'][:,:2].reshape(-1,6,256),z['x']);raw=d['x'][keep];valid=d['valid'][keep]
        train=np.flatnonzero(np.isin(users,spec['train_users']));test=np.flatnonzero(np.isin(users,spec['predict_users']));assert len(train)>0 and len(test)>0 and not set(users[train])&set(users[test]);assert set(target[train])==set(range(40))
        mean=np.zeros((1,5,3,1),np.float32);std=np.ones_like(mean)
        for dev in range(5):
            ix=train[valid[train,dev]]
            if len(ix):a64=raw[ix,dev].astype(np.float64);mean[:,dev]=a64.mean((0,2))[None,:,None];std[:,dev]=a64.std((0,2)).clip(1e-4)[None,:,None]
        seed(p['training']['seed']);model=(ArmIMU() if spec['arm']=='A' else DeviceIMU()).cuda();initial_sha=state_sha(model.state_dict());params=sum(v.numel() for v in model.parameters())
        # GroupNorm has no cross-example statistics: arbitrary values at missing slots cannot influence the output.
        if spec['arm']!='A':
            model.eval();v=torch.tensor([[True,True,False,False,False]],device='cuda');xx=torch.randn(1,5,3,256,device='cuda');xx2=xx.clone();xx2[:,2:]=1e6
            with torch.no_grad():assert torch.equal(model(xx,v),model(xx2,v))
        seed(p['training']['seed']);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001,foreach=False,fused=False);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,60,eta_min=1e-5)
        identity=dict(protocol_sha256=ph,job=spec,initial_state_sha256=initial_sha,parameters=params,train_users=spec['train_users'],normalizer_users=sorted(set(users[train].tolist())),train_ids_sha256=digest(ids[train].tolist()),arm_cache_sha256=p['data']['arm_sha256'],device_cache_sha256=device_sha)
        atomic(out/'PROVENANCE.json',identity);atomic(out/'SPLIT.json',dict(train_ids=ids[train].tolist(),predict_ids=ids[test].tolist(),train_users=spec['train_users'],predict_users=spec['predict_users']))
        np.savez_compressed(out/'normalizer.npz',mean=mean,std=std)
        raw=torch.from_numpy(raw).cuda();valid=torch.from_numpy(valid).cuda();y=torch.from_numpy(target).long().cuda();mu=torch.from_numpy(mean).cuda();sd=torch.from_numpy(std).cuda()
        recovered=load_epoch(out,model,opt,None,identity,scheduler);history=[] if recovered is None else recovered['history'];updates=0 if recovered is None else recovered['updates'];del recovered
        for epoch in range(len(history)+1,61):
            model.train();started=time.time();order=np.random.permutation(train);r=torch.from_numpy(rotations(len(ids),spec['fold'],epoch)).cuda();total=0.
            train_valid=valid
            if spec['arm']=='D':
                rng=np.random.default_rng(p['training']['seed']+spec['fold']*1000003+epoch);kept=torch.from_numpy(rng.random((len(ids),3))>=.25).cuda();train_valid=valid.clone();train_valid[:,2:]=train_valid[:,2:] & kept
            for k in range(0,len(order),64):
                guard.check(phase='IMU_TRAIN',arm=spec['arm'],epoch=epoch,epochs=60,updates=updates);ix=order[k:k+64];x=(torch.matmul(r[ix],raw[ix])-mu)/sd;opt.zero_grad(set_to_none=True);loss=nn.functional.cross_entropy(model(x,train_valid[ix]),y[ix],label_smoothing=.1)
                if not torch.isfinite(loss):raise ValueError('NONFINITE_LOSS')
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();updates+=1;total+=float(loss.detach())*len(ix)
            scheduler.step();history.append(dict(epoch=epoch,loss=total/len(train),seconds=time.time()-started,updates=updates));atomic(out/'metrics.json',history);save_epoch(out,model,opt,None,identity,history,updates,0,scheduler)
            if epoch==1 or epoch%10==0:print(json.dumps(dict(event='EPOCH_COMPLETE',job=args.job,**history[-1])),flush=True)
        checkpoint=out/'model_fp16.pt';save_torch(checkpoint,dict(state_dict={k:v.detach().cpu().half() for k,v in model.state_dict().items()},mean=torch.from_numpy(mean),std=torch.from_numpy(std),protocol_sha256=ph,train_users=spec['train_users'],arm=spec['arm']))
        cp=torch.load(checkpoint,map_location='cpu',weights_only=True);model=(ArmIMU() if spec['arm']=='A' else DeviceIMU()).cuda();model.load_state_dict(cp['state_dict'],strict=True);model.eval();pred=[]
        with torch.no_grad():
            for k in range(0,len(test),64):
                guard.check(phase='IMU_FP32_RELOAD_PREDICTION');ix=test[k:k+64];logits=model((raw[ix]-mu)/sd,valid[ix]);assert logits.dtype==torch.float32 and torch.isfinite(logits).all();pred.append(logits.cpu().numpy())
        np.savez_compressed(out/'predictions.npz',clip_ids=ids[test],users=users[test],targets=target[test],logits=np.concatenate(pred))
        atomic(out/'DONE.json',dict(status='COMPLETE',protocol_sha256=ph,epochs=60,updates=updates,checkpoint_sha256=sha(checkpoint),checkpoint_bytes=checkpoint.stat().st_size,predictions_sha256=sha(out/'predictions.npz'),initial_state_sha256=initial_sha,parameters=params,heldout_metric_selection=False))
    except BaseException as e:
        atomic(out/'ERROR.json',dict(error=repr(e),traceback=traceback.format_exc(),at_unix=time.time()));raise

if __name__=='__main__':main()
