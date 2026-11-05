"""Matched R6 continuations: depth-guided IR views, with/without conflict-weighted CE."""
from pathlib import Path
import json,time,sys,os,argparse,hashlib,traceback,random,math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.models.video import mvit_v2_s
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'bundle'))
from dual_host_runtime.io import atomic,save_torch
from dual_host_runtime.health import Guard
from activity_token_readout import ActivityTokenModel
from activity_view_data import ActivityClips,ActivityFiveViews
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    spec=json.loads((R/'VIDEO_CONFLICT_PROTOCOL.json').read_text());checksum=sha(R/'VIDEO_CONFLICT_PROTOCOL.json')
    for n,h in spec['code_sha256'].items():assert sha(R/n)==h,n
    for n,h in spec['asset_sha256'].items():assert sha(R/n)==h,n
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    inputs=json.loads((R/'VIDEO_INPUTS.json').read_text());rows=inputs['rows'];regions=inputs['regions'];quality=inputs['conflict_weights'];data=spec['data_root'];outroot=R/'video_conflict_output';outroot.mkdir(exist_ok=True)
    for job in spec['jobs']:
        out=outroot/job['id'];out.mkdir(exist_ok=True)
        if (out/'DONE.json').exists():continue
        start=time.time();budget=out/'JOB_BUDGET.json'
        if not budget.exists():atomic(budget,dict(protocol_sha256=checksum,first_started_unix=start,deadline_unix=min(spec['deadline_unix'],start+spec['job_wall_cap_seconds'])))
        bound=json.loads(budget.read_text());assert bound['protocol_sha256']==checksum
        guard=Guard(out,bound['deadline_unix'],outroot/'TELEMETRY.json')
        source=Path(job['source']);assert sha(source)==job['source_sha256']
        cp=torch.load(source,map_location='cpu',weights_only=True);assert sorted(cp['train_users'])==job['train_users'];assert not set(cp['train_users'])&set(job['held_users'])
        torch.manual_seed(spec['seed']);np.random.seed(spec['seed']);random.seed(spec['seed'])
        v=mvit_v2_s(weights=None);v.head=nn.Linear(768,40);m=ActivityTokenModel(v,'TOKEN_DIFF');m.load_state_dict(cp['state_dict'],strict=True);del cp;m.cuda().train()
        groups=m.optimizer_groups(spec['backbone_lr'],spec['head_lr'],spec['head_lr']);initial=[g['lr'] for g in groups];opt=torch.optim.AdamW(groups,weight_decay=.05);scaler=torch.amp.GradScaler('cuda',init_scale=128)
        tr=[r for r in rows if r['user'] in job['train_users']];va=[r for r in rows if r['user'] in job['held_users']];assert len(tr)+len(va)==len(rows)
        atomic(out/'SPLIT.json',dict(train_users=job['train_users'],held_users=job['held_users'],train_ids=[r['clip_id'] for r in tr],held_ids=[r['clip_id'] for r in va],protocol_sha256=checksum))
        hist=[];updates=0;stepsepoch=math.ceil(len(tr)/8);total=spec['epochs']*stepsepoch;first_epoch=1
        if (out/'resume.pt').exists():
            resume=torch.load(out/'resume.pt',map_location='cpu',weights_only=True);assert resume['protocol_sha256']==checksum
            m.load_state_dict(resume['model']);opt.load_state_dict(resume['optimizer']);scaler.load_state_dict(resume['scaler']);hist=resume['history'];updates=resume['updates'];first_epoch=resume['epoch']+1;torch.set_rng_state(resume['torch_rng']);torch.cuda.set_rng_state_all(resume['cuda_rng'])
        for epoch in range(first_epoch,spec['epochs']+1):
            ds=ActivityClips(tr,data,spec['seed'],regions,True,True,epoch)
            gen=torch.Generator().manual_seed(spec['seed']+epoch)
            loader=DataLoader(ds,batch_size=4,shuffle=True,generator=gen,num_workers=8,pin_memory=True,prefetch_factor=1)
            began=time.time();loss_sum=0;seen=0;wait=0;previous=time.perf_counter();opt.zero_grad(set_to_none=True)
            for bi,b in enumerate(loader):
                wait+=time.perf_counter()-previous
                if bi%2==0:
                    factor=.5*(1+math.cos(math.pi*updates/max(1,total)))
                    for g,lr in zip(opt.param_groups,initial):g['lr']=1e-6+(lr-1e-6)*factor
                n=len(b['y']);x=b['x'].flatten(0,2).cuda(non_blocking=True);y=b['y'].cuda(non_blocking=True).repeat_interleave(4)
                q=torch.tensor([quality[k] if job['weighted'] else 1. for k in b['clip_id']],device='cuda').repeat_interleave(4)
                with torch.autocast('cuda',dtype=torch.float16):
                    z,_,aux=m(x,return_components=True,auxiliary_backbone_stopgrad=True)
                    ce=F.cross_entropy(z,y,label_smoothing=.1,reduction='none');ac=F.cross_entropy(aux,y,label_smoothing=.1,reduction='none');loss=((ce+.25*ac)*q).mean()
                assert torch.isfinite(loss),'NONFINITE_LOSS';window=min(8,len(tr)-(bi//2)*8);scaler.scale(loss*n/window).backward();loss_sum+=float(loss.detach())*n;seen+=n
                if bi%2==1 or bi+1==len(loader):
                    oldscale=scaler.get_scale();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);scaler.step(opt);scaler.update();assert scaler.get_scale()>=oldscale,'AMP_SKIP';opt.zero_grad(set_to_none=True);updates+=1
                guard.check(phase='MATCHED_VIDEO_CONFLICT_TRAIN',job=job['id'],epoch=epoch,epochs=spec['epochs'],records=seen,total=len(tr),updates=updates)
                previous=time.perf_counter()
            assert seen==len(tr);hist.append(dict(epoch=epoch,loss=loss_sum/seen,records=seen,seconds=time.time()-began,input_wait_seconds=wait,updates=updates))
            atomic(out/'HISTORY.json',hist);save_torch(out/'resume.pt',dict(protocol_sha256=checksum,model=m.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),history=hist,updates=updates,epoch=epoch,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
            print(json.dumps(dict(job=job['id'],**hist[-1])),flush=True)
        checkpoint=out/'model_fp16.pt';save_torch(checkpoint,dict(state_dict={k:v.detach().cpu().half() for k,v in m.state_dict().items()},train_users=job['train_users'],protocol_sha256=checksum,source_sha256=job['source_sha256'],architecture='ACTIVITY_TOKEN_DIFF_R6_DEPTH_GUIDED_CONTINUATION'))
        p=torch.load(checkpoint,map_location='cpu',weights_only=True);m.load_state_dict(p['state_dict']);del p;m.eval().requires_grad_(False)
        ds=ActivityFiveViews(va,data,regions,True);loader=DataLoader(ds,batch_size=1,num_workers=6,pin_memory=True,prefetch_factor=1);ids=[];ys=[];users=[];values=[]
        with torch.inference_mode():
            for b in loader:
                xx=b['x'].flatten(0,3);zz=[]
                for st in range(0,len(xx),16):zz.append(m(xx[st:st+16].cuda(non_blocking=True)).float().cpu().numpy())
                z=np.concatenate(zz).reshape(1,5,2,2,40);values.append(z);ids+=b['clip_id'];ys+=b['y'].tolist();users+=b['user'].tolist();guard.check(phase='FINAL_FP32_VIDEO',job=job['id'],records=len(ids),total=len(va))
        z=np.concatenate(values);logits=(z[:,:,0].mean((1,2))+z[:,:,1].mean((1,2)))*.5
        np.savez_compressed(out/'predictions.npz',clip_ids=np.array(ids),targets=np.array(ys),users=np.array(users),phase_spatial_modality_logits=z,logits=logits)
        done=dict(status='COMPLETE',job=job['id'],spec=job,epochs=spec['epochs'],protocol_sha256=checksum,checkpoint_sha256=sha(checkpoint),predictions_sha256=sha(out/'predictions.npz'),seconds=time.time()-start,n=len(ids),correct=int((logits.argmax(1)==np.array(ys)).sum()),no_Test=True)
        atomic(out/'DONE.json',done);print(json.dumps(done),flush=True);del m,opt,scaler;torch.cuda.empty_cache()
    atomic(outroot/'QUEUE_DONE.json',dict(status='COMPLETE',protocol_sha256=checksum,at_unix=time.time()))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        atomic(R/'video_conflict_output/ERROR.json',dict(error=str(e),traceback=traceback.format_exc(),at_unix=time.time()));raise
