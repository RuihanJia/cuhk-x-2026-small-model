"""Four-epoch photometric screening: original P8 warm starts, no dynamic geometry."""
from pathlib import Path
import argparse,hashlib,json,math,os,sys,time,traceback
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
P=Path(__file__).resolve().parent;D=P.parent;C=D.parents[1];O=C.parent/'error_resolution_20260913'
sys.path[:0]=[str(P),str(D),str(C),str(O),str(O/'bundle')]
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from cloud_worker import read,sha,checked,arrays,save_npz,new_model,private_state,state_digest,private_digest
from photometric_data_numpy import PhotometricClips,PhotometricFiveViews
from dual_host_runtime.io import atomic,save_torch
from dual_host_runtime.health import Guard,identity,owned_tree
ARMS=('RAW_CONTROL','CLIP_STRETCH')

def fit_job(job,s,ph,outroot,inputs,stage_deadline,telemetry):
    out=outroot/job['id'];out.mkdir(exist_ok=True)
    if (out/'TRAIN_DONE.json').exists():
        d=read(out/'TRAIN_DONE.json');assert d['protocol_sha256']==ph
        for n,h in d['artifacts'].items():checked(out/n,h)
        return
    bp=out/'JOB_BUDGET.json'
    if not bp.exists():
        now=time.time();atomic(bp,dict(protocol_sha256=ph,first_started_unix=now,deadline_unix=min(stage_deadline,now+s['job_wall_cap_seconds']),scope='Setup+fit+save only; deferred held evaluation shares fixed stage clock'))
    bd=read(bp);assert bd['protocol_sha256']==ph
    guard=Guard(out,min(stage_deadline,bd['deadline_unix']),telemetry);guard.check(force=True,phase='PHOTOMETRIC_SETUP',job=job['id'])
    tr=[r for r in inputs['rows'] if r['user'] in job['train_users']]
    held_ids=[r['clip_id'] for r in inputs['rows'] if r['user'] in job['held_users']]
    assert len(tr)+len(held_ids)==len(inputs['rows']) and set(r['user'] for r in tr)==set(job['train_users'])
    assert all(len(r['streams'])==1 and r['streams'][0]['modality']=='IR' for r in tr)
    quality=inputs['conflict_weights'];assert all(np.isfinite(quality[r['clip_id']]) and quality[r['clip_id']]>0 for r in tr)
    atomic(out/'SPLIT.json',dict(protocol_sha256=ph,train_users=job['train_users'],held_users=job['held_users'],train_ids=[r['clip_id'] for r in tr],held_ids=held_ids,held_labels_used_in_fit=False))
    model=new_model(job,s);initial=private_state(model);initial_digest=state_digest(initial);base_digest=model.frozen_digest()
    assert base_digest==job['original_frozen_base_digest'],'SOURCE_FROZEN_BASE_MISMATCH'
    record=dict(protocol_sha256=ph,fold=job['fold'],seed=s['seed'],initial_state_digest=initial_digest,source_sha256=job['source_sha256'],original_private_sha256=job['original_private_sha256'])
    shared=outroot/f"F{job['fold']}_INITIAL.json"
    if shared.exists():assert read(shared)==record,'PAIRED_INITIAL_STATES_DIFFER'
    else:atomic(shared,record)
    atomic(out/'INITIAL.json',dict(**record,arm=job['arm'],frozen_base_digest=base_digest))
    params=[p for p in model.parameters() if p.requires_grad];assert sum(p.numel() for p in params)==s['private_parameters']
    opt=torch.optim.AdamW(params,lr=s['lr'],weight_decay=s['weight_decay']);scaler=torch.amp.GradScaler('cuda',init_scale=128)
    micro=s['micro_clips'];accum=s['effective_clip_batch']//micro;total=s['epochs']*math.ceil(len(tr)/s['effective_clip_batch'])
    hist=[];updates=0;first=1
    if (out/'resume.pt').exists():
        q=torch.load(out/'resume.pt',map_location='cpu',weights_only=True)
        assert q['protocol_sha256']==ph and q['initial_state_digest']==initial_digest and q['arm']==job['arm']
        model.load_private(q['private']);opt.load_state_dict(q['optimizer']);scaler.load_state_dict(q['scaler']);hist=q['history'];updates=q['updates'];first=q['epoch']+1
        torch.set_rng_state(q['rng']);torch.cuda.set_rng_state_all(q['cuda_rng']);del q
    for epoch in range(first,s['epochs']+1):
        model.train();ds=PhotometricClips(tr,s['data_root'],s['seed'],inputs['regions'],training=True,epoch=epoch,arm=job['arm'])
        generator=torch.Generator().manual_seed(s['seed']+epoch)
        loader=DataLoader(ds,batch_size=micro,shuffle=True,generator=generator,num_workers=s['workers'],pin_memory=True,prefetch_factor=1)
        seen=0;ls=0.;began=time.time();wait=0.;previous=time.perf_counter();stretch_stats=[];sampling_digest=hashlib.sha256();opt.zero_grad(set_to_none=True)
        for bi,b in enumerate(loader):
            wait+=time.perf_counter()-previous;n=len(b['y']);x=b['x'].cuda(non_blocking=True);y=b['y'].cuda(non_blocking=True)
            q=torch.tensor([quality[k] for k in b['clip_id']],device='cuda')
            for i,cid in enumerate(b['clip_id']):
                sampling_digest.update(json.dumps([cid,b['frame_indices'][i].tolist(),bool(b['flip'][i])],separators=(',',':')).encode())
                stretch_stats.append([float(b['stretch_low'][i]),float(b['stretch_high'][i]),float(b['denominator'][i])])
            assert np.isfinite(stretch_stats[-n:]).all()
            if bi%accum==0:
                lr=s['min_lr']+(s['lr']-s['min_lr'])*.5*(1+math.cos(math.pi*updates/total))
                for g in opt.param_groups:g['lr']=lr
            with torch.autocast('cuda',dtype=torch.float16):
                z=model(x,True);loss=(q*F.cross_entropy(z,y,label_smoothing=s['label_smoothing'],reduction='none')).mean()
            assert torch.isfinite(loss),'NONFINITE_LOSS'
            window=min(s['effective_clip_batch'],len(tr)-(bi//accum)*s['effective_clip_batch'])
            scaler.scale(loss*n/window).backward();seen+=n;ls+=float(loss.detach())*n
            if bi%accum==accum-1 or bi+1==len(loader):
                scale=scaler.get_scale();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
                scaler.step(opt);scaler.update();assert scaler.get_scale()>=scale,'AMP_SKIP'
                opt.zero_grad(set_to_none=True);updates+=1
            if epoch==1 and bi==0:
                atomic(out/'FIRST_BATCH.json',dict(protocol_sha256=ph,arm=job['arm'],clip_ids=b['clip_id'],frame_indices=b['frame_indices'].tolist(),flip=b['flip'].tolist(),stretch_low=b['stretch_low'].tolist(),stretch_high=b['stretch_high'].tolist(),denominator=b['denominator'].tolist(),sampling_seed=s['seed'],sampling_epoch=epoch,loss=float(loss.detach()),updates=updates,AMP_scale=scaler.get_scale()))
            guard.check(phase='PHOTOMETRIC_CE_TRAIN',job=job['id'],arm=job['arm'],epoch=epoch,epochs=s['epochs'],records=seen,total=len(tr),updates=updates)
            previous=time.perf_counter()
        assert seen==len(tr) and len(stretch_stats)==seen
        values=np.asarray(stretch_stats)
        hist.append(dict(epoch=epoch,loss=ls/seen,records=seen,seconds=time.time()-began,input_wait_seconds=wait,updates=updates,stretch_low_mean=float(values[:,0].mean()),stretch_high_mean=float(values[:,1].mean()),denominator_min=float(values[:,2].min()),sampling_sha256=sampling_digest.hexdigest()))
        atomic(out/'HISTORY.json',hist)
        save_torch(out/'resume.pt',dict(protocol_sha256=ph,arm=job['arm'],initial_state_digest=initial_digest,private=private_state(model),optimizer=opt.state_dict(),scaler=scaler.state_dict(),history=hist,updates=updates,epoch=epoch,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
        print(json.dumps(dict(job=job['id'],arm=job['arm'],**hist[-1])),flush=True)
    assert updates==total and len(hist)==s['epochs'] and model.frozen_digest()==base_digest
    trained=private_state(model);changed=[k for k in trained if not torch.equal(trained[k],initial[k])];assert changed,'NO_PARAMETER_LEARNING'
    cp=out/'adapter_fp16.pt';payload=dict(private_state=private_state(model,True),source_sha256=job['source_sha256'],original_private_sha256=job['original_private_sha256'],original_protocol_sha256=s['original_protocol_sha256'],train_users=job['train_users'],held_users=job['held_users'],protocol_sha256=ph,rank=s['rank'],arm=job['arm'],epochs=s['epochs'],continued_from_epochs=8,initial_state_digest=initial_digest)
    save_torch(cp,payload);assert cp.stat().st_size<=s['private_cap_bytes']
    restored=torch.load(cp,map_location='cpu',weights_only=True);model.load_private(restored['private_state'])
    current=private_state(model);assert all(v.dtype==torch.float16 and torch.equal(v.float(),current[k]) for k,v in restored['private_state'].items())
    assert model.frozen_digest()==base_digest
    atomic(out/'FROZEN.json',dict(protocol_sha256=ph,arm=job['arm'],checkpoint_sha256=sha(cp),checkpoint_bytes=cp.stat().st_size,initial_state_digest=initial_digest,serialized_private_digest=private_digest(model),frozen_base_digest=base_digest,source_sha256=job['source_sha256'],original_private_sha256=job['original_private_sha256'],train_users=job['train_users'],held_users=job['held_users'],epochs=s['epochs'],updates=updates,changed_private_parameters=changed,FP16_reload_exact=True,held_evaluation_started=False,at_unix=time.time()))
    guard.check(force=True,phase='PHOTOMETRIC_FIT_FROZEN',job=job['id'])
    names=['adapter_fp16.pt','FROZEN.json','INITIAL.json','SPLIT.json','HISTORY.json','FIRST_BATCH.json']
    atomic(out/'TRAIN_DONE.json',dict(protocol_sha256=ph,status='COMPLETE_TRAIN_ONLY',job=job['id'],seconds=time.time()-bd['first_started_unix'],artifacts={n:sha(out/n) for n in names},held_evaluation_started=False))
    del model,opt,scaler,params,initial,trained,payload,restored,current
    torch.cuda.empty_cache()


def verify_host_training(s,ph,out,jobs):
    hd=read(out/'HOST_TRAIN_DONE.json');ct=read(out/'train/CONTROLLER_DONE.json');done=read(out/'train/DONE.json')
    assert hd['status']=='COMPLETE_HOST_TRAINING' and hd['protocol_sha256']==ct['protocol_sha256']==done['protocol_sha256']==ph and hd['held_evaluation_started'] is False
    assert ct['exit_code']==0 and ct['terminal_sha256']==sha(out/'train/DONE.json') and done['status']=='COMPLETE_TRAIN_PHASE'
    assert sha(out/'HOST_TRAIN_DONE.json')==sha(out/'train/HOST_TRAIN_DONE.json')==done['artifacts']['HOST_TRAIN_DONE.json']
    assert set(hd['jobs'])=={j['id'] for j in jobs} and len(jobs)==2
    for n,h in hd['artifacts'].items():checked(out/n,h)
    for job in jobs:
        v=hd['jobs'][job['id']];d=out/job['id'];td=read(d/'TRAIN_DONE.json');fr=read(d/'FROZEN.json');sp=read(d/'SPLIT.json');hist=read(d/'HISTORY.json')
        assert td['status']=='COMPLETE_TRAIN_ONLY' and td['protocol_sha256']==fr['protocol_sha256']==sp['protocol_sha256']==ph
        for n,h in v['metadata'].items():checked(d/n,h)
        assert sha(d/'TRAIN_DONE.json')==v['train_done_sha256'] and fr['checkpoint_sha256']==td['artifacts']['adapter_fp16.pt']==v['checkpoint_sha256']
        assert td['held_evaluation_started'] is False and fr['held_evaluation_started'] is False and fr['epochs']==len(hist)==4
        assert fr['train_users']==sp['train_users']==job['train_users'] and fr['held_users']==sp['held_users']==job['held_users'] and sp['held_labels_used_in_fit'] is False
        assert fr['updates']==4*math.ceil(len(sp['train_ids'])/8) and fr['FP16_reload_exact'] and fr['changed_private_parameters']
        assert fr['frozen_base_digest']==job['original_frozen_base_digest'] and fr['source_sha256']==job['source_sha256'] and fr['original_private_sha256']==job['original_private_sha256']
        for e,x in enumerate(hist,1):assert x['epoch']==e and x['records']==len(sp['train_ids']) and x['updates']==e*math.ceil(len(sp['train_ids'])/8) and __import__('math').isfinite(x['loss'])
    return sha(out/'HOST_TRAIN_DONE.json')


def evaluate_job(job,s,ph,out,phaseout,inputs,guard,barrier_sha):
    from trajectory_fixed_fusion import align
    trained=out/job['id'];dest=phaseout/job['id'];dest.mkdir(exist_ok=True)
    fr=read(trained/'FROZEN.json');assert fr['protocol_sha256']==ph and fr['epochs']==4
    model=new_model(job,s);cp=torch.load(checked(trained/'adapter_fp16.pt',fr['checkpoint_sha256']),map_location='cpu',weights_only=True)
    assert cp['protocol_sha256']==ph and cp['arm']==job['arm'] and cp['epochs']==4 and cp['continued_from_epochs']==8
    model.load_private(cp['private_state']);assert private_digest(model)==fr['serialized_private_digest'];model.eval()
    va=[r for r in inputs['rows'] if r['user'] in job['held_users']]
    loader=DataLoader(PhotometricFiveViews(va,s['data_root'],inputs['regions'],arm=job['arm']),batch_size=1,num_workers=s['workers'],pin_memory=True,prefetch_factor=1)
    ids=[];ys=[];us=[];values=[];photo=[]
    with torch.inference_mode():
        for b in loader:
            x=b['x'].flatten(0,1);z=[]
            for k in range(0,len(x),4):
                guard.check(phase='PHOTOMETRIC_HELD',job=job['id'],records=len(ids),total=len(va));v=model(x[k:k+4].cuda(non_blocking=True),True)
                assert v.dtype==torch.float32 and torch.isfinite(v).all();z.append(v.cpu().numpy())
            values.append(np.concatenate(z).reshape(1,5,40));ids+=b['clip_id'];ys+=b['y'].tolist();us+=b['user'].tolist()
            photo.append(dict(clip_id=b['clip_id'][0],stretch_low=b['stretch_low'][0].tolist(),stretch_high=b['stretch_high'][0].tolist(),denominator=b['denominator'][0].tolist()))
    prior=align(arrays(checked(job['original_predictions'],job['original_predictions_sha256'])),np.asarray(ids));y=np.asarray(ys);u=np.asarray(us)
    assert np.array_equal(y,prior['targets']) and np.array_equal(u,prior['users']) and model.frozen_digest()==fr['frozen_base_digest'] and private_digest(model)==fr['serialized_private_digest']
    five=np.concatenate(values);z=five.mean(1);assert z.shape==(len(ids),40)
    save_npz(dest/'predictions.npz',clip_ids=np.asarray(ids),targets=y,users=u,pose_logits=z,logits=z,headarm_five_view_logits=five,old_private_logits=prior['pose_logits'])
    atomic(dest/'PHOTOMETRIC.json',dict(protocol_sha256=ph,arm=job['arm'],per_view16_shared_scalar=True,records=photo))
    done=dict(status='COMPLETE_PHOTOMETRIC_HELD',protocol_sha256=ph,job=job['id'],fold=job['fold'],arm=job['arm'],predictions_sha256=sha(dest/'predictions.npz'),checkpoint_sha256=fr['checkpoint_sha256'],host_training_barrier_sha256=barrier_sha,records=len(ids),global_metrics_not_computed=True,no_fit=True,no_Test=True,at_unix=time.time())
    atomic(dest/'DONE.json',done);del model,cp,loader;torch.cuda.empty_cache();return done



def verify_parent_host(job,s,original,host):
    parent=next(j for j in original['jobs'] if j['id']==job['parent_job'])
    assert parent['fold']==job['fold'] and parent['train_users']==job['train_users'] and parent['held_users']==job['held_users'] and parent['source_sha256']==job['source_sha256']
    assert not set(job['train_users'])&set(job['held_users'])
    canonical=parent['source']['cloud'];assert canonical==job['canonical_source_cloud']
    if host=='cloud':expected=Path(canonical)
    else:
        marker='/activity_token_stopgrad_20260913/cloud/output/';assert marker in canonical
        expected=C.parent/'activity_token_stopgrad_20260913/cloud_mirror/output'/canonical.split(marker,1)[1]
    assert expected.resolve()==Path(job['source_checkpoint']).resolve(),'EXACT_SOURCE_HOST_ALIAS_MISMATCH'
    done=read(checked(job['original_done'],job['original_done_sha256']))
    assert done['status']=='COMPLETE' and done['epochs']==8 and done['job']==job['parent_job'] and done['protocol_sha256']==s['original_protocol_sha256']
    assert done['checkpoint_sha256']==job['original_private_sha256'] and done['predictions_sha256']==job['original_predictions_sha256'] and done['source_sha256']==job['source_sha256']
    assert done['frozen_base_hash_before']==job['original_frozen_base_digest'] and done['frozen_base_unchanged']


def main(protocol,host,phase):
    pp=Path(protocol).resolve();s=read(pp);ph=sha(pp);assert pp.parent==P and s['status']=='FROZEN_FOR_EXECUTION'
    assert sha(__file__)==s['worker_sha256'] and host in ('cloud','local') and phase in ('train','evaluate') and (os.name=='nt')==(host=='local')
    assert s['epochs']==4 and s['original_pose_epochs']==8 and s['seed']==20260914 and s['rank']==8 and s['micro_clips']==s['effective_clip_batch']==8
    assert s['lr']==.0003 and s['min_lr']==.00001 and s['weight_decay']==.01 and s['label_smoothing']==.1 and s['private_parameters']==207954
    assert [(j['fold'],j['arm'],j['host']) for j in s['jobs']]==[(f,a,h) for f,h in ((0,'cloud'),(3,'local')) for a in ARMS]
    h=s['hosts'][host];assert h['wall_cap_seconds']==1800 and h['eval_videos']==4 and h['workers']==(2 if host=='local' else 8)
    out=Path(h['output_directory']).resolve();assert out.is_relative_to(P) and out!=P;stage=out/phase;stage.mkdir(parents=True,exist_ok=True)
    bd=read(out/'HOST_BUDGET.json');pb=read(stage/'CONTROLLER_BUDGET.json');ct=read(stage/'CONTROLLER.json')
    assert bd['protocol_sha256']==pb['protocol_sha256']==ct['protocol_sha256']==ph and bd['deadline_unix']<=min(s['deadline_unix'],bd['first_started_unix']+1800)
    assert pb['deadline_unix']<=bd['deadline_unix'];actual=identity(os.getpid());assert any(v['pid']==actual['pid'] and abs(v['create_time']-actual['create_time'])<.01 for v in owned_tree(ct['owner']))
    guard=Guard(stage,min(bd['deadline_unix'],pb['deadline_unix']),stage/'TELEMETRY.json');guard.check(force=True,phase='PHOTOMETRIC_PREFLIGHT')
    for rel,v in s['code_sha256'].items():checked(C/rel,v)
    for p,v in h['input_bindings'].items():checked(p,v)
    auth=read(checked(h['authorization_file'],s['source_authorization_sha256']));assert auth[host+'_training_authorized'] and auth[host+'_inference_authorized']
    runtime=dict(s,**{k:v for k,v in h.items() if k not in s});runtime.update(data_root=h['data_root'],workers=h['workers'],pose_inputs=h['pose_inputs'],original_protocol=h['original_protocol'],job_wall_cap_seconds=h['job_wall_cap_seconds'])
    original=read(checked(h['original_protocol'],s['original_protocol_sha256']));jobs=[]
    for j in s['jobs']:
        if j['host']!=host:continue
        value=dict(j,**j['paths'][host]);verify_parent_host(value,runtime,original,host);jobs.append(value)
    inputs=read(checked(h['pose_inputs'],s['pose_inputs_sha256']));assert len(inputs['rows'])==len({r['clip_id'] for r in inputs['rows']})==2931
    torch.set_num_threads(2 if host=='local' else 1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available() and ('4070' if host=='local' else '5090') in torch.cuda.get_device_name(0)
    if phase=='train':
        for j in jobs:fit_job(j,runtime,ph,out,inputs,bd['deadline_unix'],stage/'TELEMETRY.json')
        descriptions={};artifacts={}
        names=['TRAIN_DONE.json','FROZEN.json','INITIAL.json','SPLIT.json','HISTORY.json','FIRST_BATCH.json']
        for j in jobs:
            d=out/j['id'];td=read(d/'TRAIN_DONE.json');fr=read(d/'FROZEN.json');assert td['protocol_sha256']==fr['protocol_sha256']==ph
            for n,v in td['artifacts'].items():checked(d/n,v);artifacts[j['id']+'/'+n]=v
            metadata={n:sha(d/n) for n in names};artifacts[j['id']+'/TRAIN_DONE.json']=metadata['TRAIN_DONE.json']
            descriptions[j['id']]=dict(host=host,checkpoint_sha256=fr['checkpoint_sha256'],train_done_sha256=metadata['TRAIN_DONE.json'],metadata=metadata)
        value=dict(status='COMPLETE_HOST_TRAINING',protocol_sha256=ph,host=host,jobs=descriptions,artifacts=artifacts,held_evaluation_started=False)
        atomic(out/'HOST_TRAIN_DONE.json',value);atomic(stage/'HOST_TRAIN_DONE.json',value);assert sha(out/'HOST_TRAIN_DONE.json')==sha(stage/'HOST_TRAIN_DONE.json')
        guard.check(force=True,phase='HOST_TRAINING_COMPLETE');atomic(stage/'DONE.json',dict(status='COMPLETE_TRAIN_PHASE',protocol_sha256=ph,host=host,artifacts={'HOST_TRAIN_DONE.json':sha(stage/'HOST_TRAIN_DONE.json')}));return
    barrier=verify_host_training(s,ph,out,jobs)
    for j in jobs:evaluate_job(j,runtime,ph,out,stage,inputs,guard,barrier)
    atomic(stage/'RESULT.json',dict(status='COMPLETE_PHOTOMETRIC_RAW_READOUT',protocol_sha256=ph,host=host,host_training_barrier_sha256=barrier,jobs=[j['id'] for j in jobs],all_four_jobs_protocol_frozen_before_training=True,global_metrics_not_computed=True,no_Test=True,no_production=True))
    files={'RESULT.json':sha(stage/'RESULT.json')}
    for j in jobs:
        for n in ('predictions.npz','PHOTOMETRIC.json','DONE.json'):files[j['id']+'/'+n]=sha(stage/j['id']/n)
    atomic(stage/'DONE.json',dict(status='COMPLETE_EVALUATE_PHASE',protocol_sha256=ph,host=host,artifacts=files,calibration_fit_steps=0,screening_only=True,global_metrics_not_computed=True,all_four_jobs_protocol_frozen_before_training=True));guard.check(force=True,phase='COMPLETE_EVALUATE_PHASE')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--host',choices=['cloud','local'],required=True);p.add_argument('--phase',choices=['train','evaluate'],required=True);a=p.parse_args()
    try:main(a.protocol,a.host,a.phase)
    except BaseException as e:
        s=read(a.protocol);out=Path(s['hosts'][a.host]['output_directory']).resolve()/a.phase;assert out.is_relative_to(P);out.mkdir(parents=True,exist_ok=True)
        atomic(out/'ERROR.json',dict(protocol_sha256=sha(a.protocol),error=str(e),traceback=traceback.format_exc(),at_unix=time.time()));raise