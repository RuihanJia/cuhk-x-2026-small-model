from __future__ import annotations
import gc, math, os, time, shutil
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .common import atomic, capture_rng, digest, read, restore_rng, runtime, save_torch, seed, sha, prior_floor
from .data import Clips, loader, rows_from_bundle, sensor_arrays, split_rows
from .models import accumulation_weight, build_video, objective, set_train_mode, view_logits

def job_spec(p, fold=None, seed_value=20260909, arm='individual', calibration=False):
    all_users=p['training_users']
    if calibration:
        c=p['calibration']; train=c['encoder_fit_users']; predict=c['calibration_users']+c['assessment_users']
        source='mvit_public'; epochs=20
    else:
        predict=p['folds'][str(fold)]; train=[u for u in all_users if u not in predict]
        source='mvit_f'+str(fold); epochs=12
    settings=p.get('throughput',{});micro=settings.get('micro_batch',2)
    assert micro in [2,4,8] and 8%micro==0
    return dict(train_users=train,predict_users=predict,source=source,epochs=epochs,seed=seed_value,arm=arm,
                fold=fold,calibration=calibration,micro_batch=micro,accumulation=8//micro,effective_batch=8,
                loader_workers=settings.get('workers',4),non_blocking=settings.get('non_blocking',False),
                backbone_lr=.0001 if calibration else .00003,head_lr=.001 if calibration else .0003)

def provenance(p,spec,train):
    return dict(protocol_sha256=p['_sha'],job=spec,train_users=sorted(spec['train_users']),
                train_ids_sha256=digest([r['clip_id'] for r in train]),target_pretrained_users=[],
                initialization_sha256=p['assets'][spec['source']]['sha256'],runtime=runtime())

def load_epoch(out,model,opt,scaler,identity,scheduler=None):
    path=out/'resume.pt'
    if not path.exists(): return None
    meta=read(out/'RESUME.json')
    if sha(path)!=meta['sha256']: raise ValueError('RESUME_HASH_MISMATCH')
    value=torch.load(path,map_location='cpu',weights_only=False)
    if value['identity']!=identity or value['mid_epoch'] is not False: raise ValueError('RESUME_IDENTITY_MISMATCH')
    if value['epoch']!=len(value['history']): raise ValueError('RESUME_EPOCH_INCONSISTENT')
    model.load_state_dict(value['model'],strict=True);opt.load_state_dict(value['optimizer'])
    if scaler is not None: scaler.load_state_dict(value['scaler'])
    if scheduler is not None: scheduler.load_state_dict(value['scheduler'])
    restore_rng(value['rng']); return value

def save_epoch(out,model,opt,scaler,identity,history,updates,skips,scheduler=None):
    save_torch(out/'resume.pt',dict(identity=identity,epoch=len(history),history=history,updates=updates,skips=skips,
                                   model=model.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict() if scaler else None,
                                   scheduler=scheduler.state_dict() if scheduler else None,rng=capture_rng(),mid_epoch=False))
    atomic(out/'RESUME.json',dict(sha256=sha(out/'resume.pt'),completed_epoch=len(history),identity_sha256=digest(identity)))


def save_tail_epoch(out,model,identity,epoch,selected):
    if epoch not in selected:return
    path=out/f'tail_epoch_{epoch:02d}.pt';meta=path.with_suffix('.json')
    if path.exists() and meta.exists():
        saved=read(meta);assert saved['identity_sha256']==digest(identity) and saved['epoch']==epoch and saved['sha256']==sha(path)
        return
    state={k:v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu() for k,v in model.state_dict().items()}
    save_torch(path,dict(state_dict=state,epoch=epoch,identity_sha256=digest(identity)))
    atomic(meta,dict(epoch=epoch,identity_sha256=digest(identity),sha256=sha(path)))


def average_tail(out,selected,identity):
    states=[];sources=[]
    for epoch in selected:
        path=out/f'tail_epoch_{epoch:02d}.pt';meta=read(path.with_suffix('.json'));assert sha(path)==meta['sha256'] and meta['identity_sha256']==digest(identity)
        cp=torch.load(path,map_location='cpu',weights_only=True);assert cp['epoch']==epoch and cp['identity_sha256']==digest(identity)
        states.append(cp['state_dict']);sources.append(dict(epoch=epoch,path=path.name,sha256=meta['sha256']))
    assert all(set(s)==set(states[0]) for s in states)
    values={}
    for k,v in states[0].items():
        if v.is_floating_point():values[k]=torch.stack([s[k].float() for s in states]).mean(0).half()
        else:
            assert all(torch.equal(s[k],v) for s in states);values[k]=v.clone()
        assert not values[k].is_floating_point() or torch.isfinite(values[k]).all()
    return values,sources

def features(model,train,p,data_root,cache,guard,workers,device):
    key=digest(dict(initialization=p['assets'][p['_source']]['sha256'],ids=[r['clip_id'] for r in train],
                    preprocessing=p['input'],runtime=runtime()))
    path=cache/(key+'.npz'); meta=cache/(key+'.json')
    if path.exists():
        if sha(path)!=read(meta)['sha256']: raise ValueError('FEATURE_CACHE_HASH_MISMATCH')
        z=np.load(path,allow_pickle=False)
        if z['clip_ids'].tolist()!=[r['clip_id'] for r in train]: raise ValueError('FEATURE_CACHE_ALIGNMENT')
        return torch.from_numpy(z['features']),torch.from_numpy(z['targets']).long(),key
    model.eval(); saved=[]; xs=[]; ys=[]; ids=[]
    hook=model.head.register_forward_pre_hook(lambda _module,args:saved.append(args[0].detach().cpu()))
    try:
        with torch.no_grad():
            for batch in loader(Clips(train,data_root,0),batch=2,workers=workers):
                saved.clear(); view_logits(model,batch,device)
                xs.append(saved[0].reshape(len(batch['y']),2,-1)); ys.extend(batch['y'].tolist());ids.extend(batch['clip_id'])
                guard.check(phase='TRAIN_ONLY_FEATURES',records=len(ids),total=len(train))
    finally: hook.remove()
    x=torch.cat(xs); y=torch.tensor(ys); cache.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.partial.npz');np.savez_compressed(tmp,features=x.numpy(),targets=y.numpy(),clip_ids=np.asarray(ids));os.replace(tmp,path)
    atomic(meta,dict(sha256=sha(path),train_ids_sha256=digest(ids),heldout_access=False))
    return x,y,key

def linear_probe(model,train,p,spec,data_root,out,cache,guard,workers,device):
    p=dict(p,_source=spec['source']);x,y,key=features(model,train,p,data_root,cache,guard,workers,device)
    head=nn.Linear(model.head.in_features,40);head.load_state_dict({k:v.cpu() for k,v in model.head.state_dict().items()})
    opt=torch.optim.AdamW(head.parameters(),lr=.003,weight_decay=.0001);losses=[]
    for _ in range(200):
        guard.bounds();opt.zero_grad();loss=objective(head(x),y,spec['arm'])
        if not torch.isfinite(loss): raise ValueError('NONFINITE_LINEAR_PROBE')
        loss.backward();nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()))
    if losses[-1]>=losses[0]: raise ValueError('LINEAR_PROBE_NOT_LEARNING')
    model.head.load_state_dict(head.state_dict())
    atomic(out/'LP.json',dict(initial_loss=losses[0],final_loss=losses[-1],steps=200,feature_cache_key=key,
                             objective=spec['arm'],train_ids_sha256=digest([r['clip_id'] for r in train]),heldout_access=False))

@torch.no_grad()
def video_predictions(model,rows,data_root,spec,guard,workers,device):
    model.eval();ids=[];ys=[];users=[];zs=[]
    for batch in loader(Clips(rows,data_root,spec['seed']),batch=2,workers=workers):
        z=view_logits(model,batch,device).mean(1)
        if z.dtype!=torch.float32 or not torch.isfinite(z).all(): raise ValueError('NONFINITE_OR_NON_FP32_PREDICTIONS')
        zs.append(z.cpu().numpy());ids.extend(batch['clip_id']);ys.extend(batch['y'].tolist());users.extend(batch['user'].tolist())
        guard.check(phase='FP32_FINAL_PREDICTIONS',records=len(ids),total=len(rows))
    return dict(clip_ids=np.asarray(ids),targets=np.asarray(ys),users=np.asarray(users),logits=np.concatenate(zs))

def train_video(p,spec,data_root,out,cache,guard,workers=4,device='cuda'):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);seed(spec['seed'])
    workers=spec['loader_workers'];micro=spec['micro_batch'];accum=spec['accumulation']
    assert micro*accum==spec['effective_batch']==8
    train,pred=split_rows(rows_from_bundle(p['_bundle']),spec['train_users'],spec['predict_users'])
    identity=provenance(p,spec,train);atomic(out/'PROVENANCE.json',identity)
    atomic(out/'SPLIT.json',dict(train_ids=[r['clip_id'] for r in train],predict_ids=[r['clip_id'] for r in pred],
                               train_users=spec['train_users'],predict_users=spec['predict_users']))
    model=build_video(p,spec['source'],held_users=spec['predict_users']).to(device)
    if not (out/'resume.pt').exists(): linear_probe(model,train,p,spec,data_root,out,Path(cache),guard,workers,device)
    seed(spec['seed']);body=[v for n,v in model.named_parameters() if not n.startswith('head.')]
    opt=torch.optim.AdamW([dict(params=body,lr=spec['backbone_lr']),dict(params=model.head.parameters(),lr=spec['head_lr'])],weight_decay=.05)
    scaler=torch.amp.GradScaler('cuda',init_scale=128,enabled=device=='cuda')
    recovered=load_epoch(out,model,opt,scaler,identity);history=[] if recovered is None else recovered['history']
    updates=0 if recovered is None else recovered['updates'];skips=0 if recovered is None else recovered['skips']
    del recovered;epochs=spec['epochs'];floor=prior_floor([r['label'] for r in train]);start=time.time()
    selected=spec.get('tail_weight_average_epochs',[])
    if selected:
        assert selected==[9,10,11,12]
        # If an epoch checkpoint survived a crash just before its tail copy,
        # regenerate that copy from the restored model before advancing.
        if history:save_tail_epoch(out,model,identity,len(history),selected)
        for epoch in selected:
            if epoch<len(history):assert (out/f'tail_epoch_{epoch:02d}.json').exists(),'MISSING_OLDER_TAIL_SNAPSHOT'
    for epoch in range(len(history)+1,epochs+1):
        ratio=epoch/2 if epoch<=2 else .5*(1+math.cos(math.pi*(epoch-2)/(epochs-2)))
        for group,base in zip(opt.param_groups,[spec['backbone_lr'],spec['head_lr']]): group['lr']=1e-6+(base-1e-6)*ratio
        ds=Clips(train,data_root,spec['seed'],True,epoch);dl=loader(ds,micro,workers,True)
        set_train_mode(model,'mvit');opt.zero_grad(set_to_none=True);loss_sum=0.;n=0;started=time.time();pred_counts=np.zeros(40,np.int64)
        for index,batch in enumerate(dl):
            guard.check(phase='VIDEO_TRAIN',epoch=epoch,epochs=epochs,records=n,total=len(train),updates=updates,amp_skips=skips)
            with torch.autocast(device_type=device,dtype=torch.float16,enabled=device=='cuda'):
                z=view_logits(model,batch,device,non_blocking=spec['non_blocking']);loss=objective(z,batch['y'].to(device,non_blocking=spec['non_blocking']),spec['arm'])
            if not torch.isfinite(loss): raise ValueError('NONFINITE_TRAIN_LOSS')
            batch_n=len(batch['y']);weight=accumulation_weight(batch_n,index,len(train),effective=spec['effective_batch'],accumulation=accum)
            scaler.scale(loss*weight).backward()
            if (index+1)%accum==0 or index+1==len(dl):
                previous=scaler.get_scale();scaler.unscale_(opt)
                grad=nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=False)
                scaler.step(opt);scaler.update();skipped=scaler.get_scale()<previous
                if not bool(torch.isfinite(grad)) and not skipped: raise ValueError('NONFINITE_UNSKIPPED_GRADIENT')
                # Any skipped update invalidates a matched trajectory; fail immediately, never call it a performance loss.
                if skipped: raise ValueError('AMP_SKIP_MATCHED_TRAJECTORY_INVALID')
                updates+=1;opt.zero_grad(set_to_none=True)
            n+=batch_n;loss_sum+=float(loss.detach())*batch_n
            pred_counts+=np.bincount(z.detach().mean(1).argmax(1).cpu().numpy(),minlength=40)
        row=dict(epoch=epoch,loss=loss_sum/n,seconds=time.time()-started,updates=updates,amp_skips=skips,
                 constant_prior_floor=floor,prediction_concentration=float(pred_counts.max()/n))
        history.append(row);atomic(out/'metrics.json',history)
        save_epoch(out,model,opt,scaler,identity,history,updates,skips)
        save_tail_epoch(out,model,identity,epoch,selected)
        print(__import__('json').dumps(dict(event='EPOCH_COMPLETE',job=out.name,**row)),flush=True)
        if len(history)>=2 and all(abs(h['loss']-floor)<.005 and h['prediction_concentration']>.98 for h in history[-2:]):
            raise ValueError('PRIOR_COLLAPSE_ENGINEERING_INVALID')
        remaining=epochs-epoch
        if remaining and time.time()+remaining*np.median([h['seconds'] for h in history[-3:]])+60>=guard.deadline:
            raise TimeoutError('FIXED_EPOCHS_CANNOT_FIT_REMAINING_BUDGET')
    if updates!=epochs*math.ceil(len(train)/8): raise ValueError('OPTIMIZER_UPDATE_COUNT_MISMATCH')
    path=out/'model_fp16.pt'
    save_torch(path,dict(state_dict={k:v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu() for k,v in model.state_dict().items()},
                         protocol_sha256=p['_sha'],train_users=spec['train_users'],initialization_sha256=identity['initialization_sha256']))
    if path.stat().st_size+700000>=90_000_000: raise ValueError('SERIALIZED_SIZE_GATE')
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['state_dict'],strict=True)
    values=video_predictions(model,pred,data_root,spec,guard,workers,device)
    np.savez_compressed(out/'predictions.npz',**values)
    done=dict(status='COMPLETE',protocol_sha256=p['_sha'],epochs=epochs,updates=updates,amp_skips=0,
              checkpoint_sha256=sha(path),checkpoint_bytes=path.stat().st_size,predictions_sha256=sha(out/'predictions.npz'),
              elapsed_fit_seconds=time.time()-start,provenance_sha256=sha(out/'PROVENANCE.json'),
              spec=spec,metric_selection_during_training=False)
    if selected:
        # Preserve the actual last-epoch control from this same trajectory.
        control=out.parent/out.name.replace('_candidate','_control');control.mkdir()
        for name in ['model_fp16.pt','predictions.npz','PROVENANCE.json','SPLIT.json','LP.json','metrics.json']:shutil.copy2(out/name,control/name)
        control_done=dict(done,checkpoint_selection='last_epoch_12',same_training_trajectory_output=out.name)
        atomic(control/'DONE.json',control_done)
        averaged,sources=average_tail(out,selected,identity)
        save_torch(path,dict(state_dict=averaged,protocol_sha256=p['_sha'],train_users=spec['train_users'],initialization_sha256=identity['initialization_sha256'],tail_average_sources=sources))
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['state_dict'],strict=True)
        values=video_predictions(model,pred,data_root,spec,guard,workers,device);np.savez_compressed(out/'predictions.npz',**values)
        atomic(out/'TAIL_AVERAGE.json',dict(status='COMPLETE',sources=sources,formula='fp16(equal FP32 mean of the four FP16 epoch states)',same_trajectory_control=control.name,control_checkpoint_sha256=control_done['checkpoint_sha256'],new_training_updates=0))
        done.update(checkpoint_selection='equal_mean_epochs_9_10_11_12',checkpoint_sha256=sha(path),checkpoint_bytes=path.stat().st_size,predictions_sha256=sha(out/'predictions.npz'),tail_average_sha256=sha(out/'TAIL_AVERAGE.json'))
        assert path.stat().st_size+700000<100000000
    atomic(out/'DONE.json',done);del model,opt,scaler;gc.collect()
    if device=='cuda':torch.cuda.empty_cache()
    return done

def train_sensor(p,out,guard,device='cuda'):
    SensorTemporal = SensorConstructor
    out=Path(out);out.mkdir(parents=True,exist_ok=True);c=p['calibration'];seed(p['_seed'])
    raw,train,fit,mean,std,s,imu=sensor_arrays(p['_bundle'],c['encoder_fit_users'])
    # Only measured tensors and real labels enter this branch. No video teacher or held-user fitted normalizer.
    s=s.to(device);imu=imu.to(device);y=torch.from_numpy(raw['targets']).long().to(device)
    model=SensorTemporal().to(device);opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,60,eta_min=1e-5)
    identity=dict(protocol_sha256=p['_sha'],train_users=c['encoder_fit_users'],normalizer_users=sorted(set(raw['users'][fit].tolist())),
                  train_ids_sha256=digest(raw['clip_ids'][train].tolist()),normalizer_fit_ids_sha256=digest(raw['clip_ids'][fit].tolist()),
                  target_pretrained_users=[],runtime=runtime(),teacher_used=False)
    atomic(out/'PROVENANCE.json',identity)
    atomic(out/'SPLIT.json',dict(train_ids=raw['clip_ids'][train].tolist(),normalizer_fit_ids=raw['clip_ids'][fit].tolist(),
                               train_users=c['encoder_fit_users'],predict_users=c['calibration_users']+c['assessment_users']))
    recovered=load_epoch(out,model,opt,None,identity,scheduler);history=[] if recovered is None else recovered['history'];updates=0 if recovered is None else recovered['updates'];del recovered
    initial_loss=None
    for epoch in range(len(history)+1,61):
        model.train();order=np.random.permutation(train);total=0.;started=time.time()
        for k in range(0,len(order),64):
            guard.check(phase='SENSOR_TRAIN',epoch=epoch,epochs=60,updates=updates)
            ix=order[k:k+64];opt.zero_grad(set_to_none=True);loss=F.cross_entropy(model(s[ix],imu[ix]),y[ix],label_smoothing=.1)
            if not torch.isfinite(loss):raise ValueError('NONFINITE_SENSOR_LOSS')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();updates+=1;total+=float(loss.detach())*len(ix)
        scheduler.step();history.append(dict(epoch=epoch,loss=total/len(train),seconds=time.time()-started,updates=updates))
        atomic(out/'metrics.json',history);save_epoch(out,model,opt,None,identity,history,updates,0,scheduler)
        if epoch%10==0:print(__import__('json').dumps(dict(event='SENSOR_EPOCH',**history[-1])),flush=True)
    path=out/'model_fp16.pt';save_torch(path,dict(state_dict={k:v.detach().cpu().half() for k,v in model.state_dict().items()},
                                               mean=torch.from_numpy(mean),std=torch.from_numpy(std),protocol_sha256=p['_sha'],train_users=c['encoder_fit_users']))
    if path.stat().st_size>=2000000: raise ValueError('SENSOR_SIZE_GATE')
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['state_dict'],strict=True);model.eval()
    indices=np.flatnonzero(np.isin(raw['users'],c['calibration_users']+c['assessment_users']));pred=[]
    with torch.no_grad():
        for k in range(0,len(indices),64):
            guard.check(phase='SENSOR_PREDICTIONS');ix=indices[k:k+64];z=model(s[ix],imu[ix])
            if z.dtype!=torch.float32 or not torch.isfinite(z).all():raise ValueError('SENSOR_FP32_PREDICTIONS')
            pred.append(z.cpu().numpy())
    np.savez_compressed(out/'predictions.npz',clip_ids=raw['clip_ids'][indices],targets=raw['targets'][indices],users=raw['users'][indices],logits=np.concatenate(pred))
    done=dict(status='COMPLETE',protocol_sha256=p['_sha'],epochs=60,updates=updates,checkpoint_sha256=sha(path),checkpoint_bytes=path.stat().st_size,
              predictions_sha256=sha(out/'predictions.npz'),provenance_sha256=sha(out/'PROVENANCE.json'),initial_train_loss=history[0]['loss'],
              final_train_loss=history[-1]['loss'],heldout_metrics_computed=False,teacher_used=False)
    atomic(out/'DONE.json',done);del model,opt,s,imu,y;gc.collect()
    if device=='cuda':torch.cuda.empty_cache()
    return done
