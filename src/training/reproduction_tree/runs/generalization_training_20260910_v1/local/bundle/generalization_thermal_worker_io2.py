"""Matched inner/outer native Thermal X3D fits, independent of the video host."""
from pathlib import Path
import argparse,hashlib,json,math,random,site,sys,time,traceback
import numpy as np
import torch
from torch import nn
from PIL import Image
from torchvision.transforms import functional as TF

class ThermalClips(torch.utils.data.Dataset):
    def __init__(self,rows,root,mean,std,training=False,epoch=0):self.rows=rows;self.root=Path(root);self.mean=mean;self.std=std;self.training=training;self.epoch=epoch;self.seed=2026091003
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        r=self.rows[index];rng=random.Random(self.seed+self.epoch*1000003+index*9176);files=r['files'];positions=[min(len(files)-1,int((k+(rng.random() if self.training else .5))*len(files)/16)) for k in range(16)];flip=self.training and rng.random()<.5;frames=[]
        for k in positions:
            with Image.open(self.root/r['directory']/files[k]) as im:frames.append(TF.pil_to_tensor(im.convert('RGB')))
        x=TF.center_crop(TF.resize(torch.stack(frames).float().div_(255),[256],antialias=True),[224,224]);x=TF.normalize(x,self.mean,self.std)
        if flip:x=x.flip(-1)
        return dict(x=x.permute(1,0,2,3).contiguous(),y=r['label'],user=r['user'],clip_id=r['clip_id'])

def main():
    a=argparse.ArgumentParser();a.add_argument('--root',required=True);a.add_argument('--job',required=True);a.add_argument('--protocol',required=True);a.add_argument('--deadline',type=float,required=True);args=a.parse_args();root=Path(args.root);sys.path.insert(0,str(root/'bundle'))
    from dual_host_runtime.io import atomic,read,save_torch
    from dual_host_runtime.health import Guard
    from research_runtime.common import configure,seed,sha,digest
    from research_runtime.train import load_epoch,save_epoch
    from research_runtime.data import safe_path
    from threaded_data import loader
    p=read(args.protocol);ph=p.get('scientific_parent_sha256',sha(args.protocol));spec=p['jobs'][args.job];out=root/'output'/args.job;out.mkdir(parents=True,exist_ok=True);guard=Guard(out,args.deadline,root/'TELEMETRY.json');configure()
    atomic(out/'EXECUTION.json',dict(execution_protocol_sha256=sha(args.protocol),scientific_protocol_sha256=ph,loader='bounded CPU threads, exact sampled input parity verified'))
    try:
        sys.path.insert(0,p['dependencies']['root']);site.addsitedir(p['dependencies']['root'])
        for rel,expected in p['dependencies']['source_sha256'].items():assert sha(Path(p['dependencies']['root'])/rel)==expected,'DEPENDENCY_DRIFT'
        from pytorchvideo.models.hub import x3d_m
        assert sha(root/'bundle/data/thermal_index.json')==p['data']['index_sha256'];rows=read(root/'bundle/data/thermal_index.json');train=[r for r in rows if r['user'] in spec['train_users']];pred=[r for r in rows if r['user'] in spec['predict_users']]
        assert train and pred and not set(spec['train_users'])&set(spec['predict_users']);assert not set(spec['train_users'])&set(p['outer_folds'][str(spec['fold'])]);assert set(r['label'] for r in train)==set(range(40))
        data_root=Path(p['data']['root']);sums=np.zeros(3,np.float64);squares=np.zeros(3,np.float64);pixels=0;ledger=[]
        for i,r in enumerate(train):
            d=safe_path(data_root,r['directory']);file=d/r['files'][len(r['files'])//2]
            with Image.open(file) as im:x=np.asarray(im.convert('RGB'),dtype=np.float64).reshape(-1,3)/255.
            sums+=x.sum(0);squares+=np.square(x).sum(0);pixels+=len(x);ledger.append([r['clip_id'],file.name]);guard.check(phase='TRAIN_ONLY_THERMAL_NORMALIZATION',records=i+1,total=len(train))
        mean=sums/pixels;std=np.sqrt(np.maximum(squares/pixels-mean**2,1e-8));normalizer=dict(train_users=spec['train_users'],train_ids=[r['clip_id'] for r in train],mean=mean.tolist(),std=std.tolist(),pixels=pixels,selected_clip_frame_sha256=digest(ledger));atomic(out/'NORMALIZER.json',normalizer)
        weights=root/p['weights']['path'];assert sha(weights)==p['weights']['sha256'];seed(p['training']['seed']);model=x3d_m(pretrained=False);model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True)['model_state'],strict=True);model.blocks[-1].proj=nn.Linear(model.blocks[-1].proj.in_features,40);model.blocks[-1].activation=None;assert sum(v.numel() for v in model.parameters())==3056634;model.cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,20,eta_min=1e-6);scaler=torch.amp.GradScaler('cuda',init_scale=128)
        identity=dict(protocol_sha256=ph,job=spec,initialization_sha256=p['weights']['sha256'],train_users=spec['train_users'],train_ids_sha256=digest([r['clip_id'] for r in train]),normalizer_sha256=sha(out/'NORMALIZER.json'),target_pretrained_users=[])
        atomic(out/'PROVENANCE.json',identity);atomic(out/'SPLIT.json',dict(train_users=spec['train_users'],predict_users=spec['predict_users'],train_ids=[r['clip_id'] for r in train],predict_ids=[r['clip_id'] for r in pred]))
        recovered=load_epoch(out,model,opt,scaler,identity,scheduler);history=[] if recovered is None else recovered['history'];updates=0 if recovered is None else recovered['updates'];skips=0 if recovered is None else recovered['skips'];del recovered
        for epoch in range(len(history)+1,21):
            dl=loader(ThermalClips(train,data_root,mean.tolist(),std.tolist(),True,epoch),batch=2,workers=4,shuffle=True);model.train();opt.zero_grad(set_to_none=True);started=time.time();total=0.;count=0
            for i,batch in enumerate(dl):
                guard.check(phase='THERMAL_TRAIN',epoch=epoch,epochs=20,records=count,total=len(train),updates=updates);x=batch['x'].cuda(non_blocking=True);y=batch['y'].cuda(non_blocking=True)
                with torch.autocast('cuda',dtype=torch.float16):z=model(x);loss=nn.functional.cross_entropy(z,y,label_smoothing=.1)
                if not torch.isfinite(loss):raise ValueError('NONFINITE_THERMAL_LOSS')
                size=len(y);denom=min(8,len(train)-(i//4)*8);scaler.scale(loss*(size/denom)).backward()
                if (i+1)%4==0 or i+1==len(dl):
                    old=scaler.get_scale();scaler.unscale_(opt);grad=nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=False);scaler.step(opt);scaler.update();skip=scaler.get_scale()<old
                    if skip:raise ValueError('AMP_SKIP_INVALID_TRAJECTORY')
                    assert torch.isfinite(grad);updates+=1;opt.zero_grad(set_to_none=True)
                total+=float(loss.detach())*size;count+=size
            scheduler.step();history.append(dict(epoch=epoch,loss=total/count,seconds=time.time()-started,updates=updates,amp_skips=skips));atomic(out/'metrics.json',history);save_epoch(out,model,opt,scaler,identity,history,updates,skips,scheduler);print(json.dumps(dict(event='EPOCH_COMPLETE',job=args.job,**history[-1])),flush=True)
            if epoch<20 and time.time()+(20-epoch)*np.median([r['seconds'] for r in history[-3:]])+60>=args.deadline:raise TimeoutError('THERMAL_FIXED_EPOCHS_EXCEED_REMAINING_CAP')
        assert updates==20*math.ceil(len(train)/8);path=out/'model_fp16.pt';save_torch(path,dict(state_dict={k:v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu() for k,v in model.state_dict().items()},protocol_sha256=ph,train_users=spec['train_users'],mean=mean.tolist(),std=std.tolist()));assert path.stat().st_size<10000000
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['state_dict'],strict=True);model.eval();ids=[];targets=[];users=[];values=[]
        with torch.no_grad():
            for batch in loader(ThermalClips(pred,data_root,mean.tolist(),std.tolist()),batch=2,workers=4):
                guard.check(phase='THERMAL_FINAL_FP32',records=len(ids),total=len(pred));z=model(batch['x'].cuda(non_blocking=True));assert z.dtype==torch.float32 and torch.isfinite(z).all();values.append(z.cpu().numpy());ids.extend(batch['clip_id']);targets.extend(batch['y'].tolist());users.extend(batch['user'].tolist())
        np.savez_compressed(out/'predictions.npz',clip_ids=np.asarray(ids),targets=np.asarray(targets),users=np.asarray(users),logits=np.concatenate(values));atomic(out/'DONE.json',dict(status='COMPLETE',protocol_sha256=ph,epochs=20,updates=updates,checkpoint_sha256=sha(path),checkpoint_bytes=path.stat().st_size,predictions_sha256=sha(out/'predictions.npz'),heldout_metric_selection=False,execution_protocol_sha256=sha(args.protocol)))
    except BaseException as e:atomic(out/'ERROR.json',dict(error=repr(e),traceback=traceback.format_exc(),at_unix=time.time()));raise
if __name__=='__main__':main()
