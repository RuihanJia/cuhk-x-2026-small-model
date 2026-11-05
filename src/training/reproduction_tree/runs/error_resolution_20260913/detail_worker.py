"""Frozen DINO visual-cache and controlled Training-user-only head fitting."""
from pathlib import Path
import sys,os,json,time,hashlib,argparse,random,traceback,math
os.environ['XFORMERS_DISABLED']='1'
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from PIL import Image,ImageOps
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'bundle'))
from dual_host_runtime.health import Guard
from dual_host_runtime.io import atomic,save_torch
from detail_model import DetailHead
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class Frames(Dataset):
    def __init__(self,rows,parts,data):self.rows,self.parts,self.data=rows,parts,Path(data)
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i];views=[];available=[]
        for part in self.parts:
            s=row['views'][part];files=s['files'];available.append(bool(files));frames=[];decoded={}
            if not files:
                views.append(np.zeros((16,3,224,224),np.uint8));continue
            directory=self.data/s['directory'];assert 'Training' in directory.parts and 'Test' not in directory.parts
            for k in range(16):
                index=min(len(files)-1,int((k+.5)*len(files)/16))
                if index not in decoded:
                    with Image.open(directory/files[index]) as im:
                        im=im.convert('RGB')
                        if s['box'] is not None:
                            x,y,w,h=s['box'];im=im.crop((x,y,x+w,y+h))
                        im=ImageOps.pad(im,(224,224),method=Image.Resampling.BICUBIC,color=(0,0,0))
                        decoded[index]=np.array(im,dtype=np.uint8).transpose(2,0,1).copy()
                frames.append(decoded[index])
            views.append(np.stack(frames))
        return i,np.stack(views),np.asarray(available,bool)

def load_dino():
    sys.path.insert(0,str(R/'assets/dino/source'))
    from dinov2.hub.backbones import dinov2_vits14
    model=dinov2_vits14(pretrained=False)
    cp=torch.load(R/'assets/dino/dino_int8.pt',map_location='cpu',weights_only=True)
    state={k:(v['q'].float()*v['scale'].float() if 'q' in v else v['value'].float()) for k,v in cp['state_dict'].items()}
    model.load_state_dict(state,strict=True);return model.eval().requires_grad_(False).cuda().half()

def cache(spec,host,rows,guard):
    s=spec['hosts'][host];parts=s['cache_parts'];out=R/'detail_output'/host;out.mkdir(parents=True,exist_ok=True)
    path=out/'features.npy';done=out/'CACHE_DONE.json'
    if done.exists():
        d=json.loads(done.read_text());assert d['protocol_sha256']==sha(R/'DETAIL_PROTOCOL.json') and d['features_sha256']==sha(path);return
    assert not path.exists(),'PARTIAL_CACHE_REQUIRES_REGISTERED_RECOVERY'
    model=load_dino();start=time.time();count=0;images=0;wait_seconds=0
    dest=np.lib.format.open_memmap(path,mode='w+',dtype=np.float16,shape=(len(rows),len(parts),16,5,384))
    av=np.zeros((len(rows),len(parts)),bool)
    loader=DataLoader(Frames(rows,parts,s['data_root']),batch_size=s['cache_clip_batch'],shuffle=False,num_workers=s['cache_workers'],prefetch_factor=2,pin_memory=True)
    mean=torch.tensor([.485,.456,.406],device='cuda').reshape(1,3,1,1);std=torch.tensor([.229,.224,.225],device='cuda').reshape(1,3,1,1)
    begin=time.perf_counter()
    with torch.inference_mode():
        for indices,x,available in loader:
            wait_seconds+=time.perf_counter()-begin
            shape=x.shape;x=x.flatten(0,2);valid=available.unsqueeze(-1).expand(-1,-1,16).flatten();features=torch.zeros((len(x),5,384),dtype=torch.float16)
            valid_indices=valid.nonzero().flatten()
            for j in range(0,len(valid_indices),s['image_batch']):
                ii=valid_indices[j:j+s['image_batch']]
                xx=x[ii].cuda(non_blocking=True).float().div_(255);xx=((xx-mean)/std).half()
                f=model.forward_features(xx);patch=f['x_norm_patchtokens'];assert patch.shape[1:]==(256,384)
                pooled=F.adaptive_avg_pool2d(patch.transpose(1,2).reshape(-1,384,16,16),(2,2)).flatten(2).transpose(1,2)
                zz=torch.cat([f['x_norm_clstoken'].unsqueeze(1),pooled],1)
                assert torch.isfinite(zz).all();features[ii]=zz.cpu();images+=len(ii)
            dest[indices.numpy()]=features.reshape(shape[0],len(parts),16,5,384).numpy();av[indices.numpy()]=available.numpy()
            count+=len(indices);guard.check(phase='FROZEN_DINO_CACHE',records=count,total=len(rows),images=images,images_per_second=images/max(1,time.time()-start))
            if count%100==0:print(json.dumps(dict(phase='CACHE',host=host,records=count,total=len(rows),images=images,seconds=time.time()-start)),flush=True)
            begin=time.perf_counter()
    dest.flush();del dest
    np.save(out/'available.npy',av,allow_pickle=False)
    d=dict(status='COMPLETE',host=host,protocol_sha256=sha(R/'DETAIL_PROTOCOL.json'),features_sha256=sha(path),available_sha256=sha(out/'available.npy'),parts=parts,records=count,images=images,seconds=time.time()-start,input_wait_seconds=wait_seconds,training_fits=0,Test_read=False)
    atomic(done,d);print(json.dumps(d),flush=True)

def train(spec,host,rows,guard):
    s=spec['hosts'][host];fold=s['fold'];q=spec['folds'][str(fold)];out=R/'detail_output'/host
    receipt=R/'detail_output/MERGED.json'
    while not receipt.exists():
        guard.check(phase='WAIT_FOR_OTHER_CACHE_SHARD');time.sleep(5)
    merged=json.loads(receipt.read_text());assert merged['protocol_sha256']==sha(R/'DETAIL_PROTOCOL.json')
    arrays=[];masks=[]
    for name in ['cloud','local']:
        p=R/'detail_output'/name/'features.npy';v=R/'detail_output'/name/'available.npy'
        assert sha(p)==merged[name]['features_sha256'] and sha(v)==merged[name]['available_sha256']
        arrays.append(np.load(p,mmap_mode='r'));masks.append(np.load(v))
    x=torch.from_numpy(np.concatenate(arrays,axis=1)).cuda();available=torch.from_numpy(np.concatenate(masks,axis=1)).cuda()
    y=torch.tensor([r['label'] for r in rows],device='cuda');users=np.array([r['user'] for r in rows])
    tr=np.where(np.isin(users,q['train_users']))[0];va=np.where(np.isin(users,q['held_users']))[0]
    assert not set(users[tr])&set(users[va]);assert len(tr)+len(va)==len(rows)
    weights=torch.tensor([max(.25,1-.75*r['conflict_fraction']) for r in rows],device='cuda')
    # Conflict fraction derives solely from original same-user Training frame bytes and official labels.
    # For each training record every compared clip belongs to that same training user.
    for arm in spec['head_arms']:
        job=f'DETAIL_F{fold}_{arm}';jobout=out/job;jobout.mkdir(exist_ok=True)
        if (jobout/'DONE.json').exists():continue
        start=time.time();guard.deadline=min(spec['deadline_unix'],start+spec['head_wall_cap_seconds']);seed=spec['head_training']['seed'];torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
        m=DetailHead(ordered=arm!='STATIC').cuda();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01)
        assert sum(p.numel() for p in m.parameters())*2<800000
        hist=[];config=spec['head_training'];steps=math.ceil(len(tr)/config['batch']);total=steps*config['epochs'];updates=0
        atomic(jobout/'SPLIT.json',dict(train_users=q['train_users'],held_users=q['held_users'],train_ids=[rows[i]['clip_id'] for i in tr],held_ids=[rows[i]['clip_id'] for i in va],outer_labels_used_for_fit=False))
        for epoch in range(1,config['epochs']+1):
            m.train();order=np.random.permutation(tr);loss_sum=0;correct=0
            for st in range(0,len(order),config['batch']):
                ii=torch.tensor(order[st:st+config['batch']],device='cuda');lr=.00005+.00095*.5*(1+math.cos(math.pi*updates/total))
                for g in opt.param_groups:g['lr']=lr
                opt.zero_grad(set_to_none=True);z,views,gate=m(x[ii].float(),available[ii],drop_modalities=True)
                loss=F.cross_entropy(z,y[ii],label_smoothing=.1,reduction='none')
                aux=F.cross_entropy(views.flatten(0,1),y[ii].repeat_interleave(4),label_smoothing=.1,reduction='none').reshape(-1,4)
                aw=available[ii].float()
                if arm=='TEMPORAL_CONFLICT':
                    aw=aw.clone();aw[:,:3]*=weights[ii,None]
                    # Fused supervision is retained; modality-specific auxiliary gradients are reduced.
                aux=(aux*aw).sum(1)/aw.sum(1).clamp_min(1)
                objective=(loss+.25*aux).mean();assert torch.isfinite(objective);objective.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
                loss_sum+=float(objective.detach())*len(ii);correct+=int((z.argmax(1)==y[ii]).sum());updates+=1
                guard.check(phase='DETAIL_HEAD_TRAIN',job=job,epoch=epoch,epochs=config['epochs'],updates=updates)
            hist.append(dict(epoch=epoch,loss=loss_sum/len(tr),train_correct=correct,n=len(tr),seconds=time.time()-start));atomic(jobout/'HISTORY.json',hist)
            if epoch%10==0:print(json.dumps(dict(job=job,**hist[-1])),flush=True)
        cp=jobout/'head_fp16.pt';save_torch(cp,dict(state_dict={k:v.detach().cpu().half() for k,v in m.state_dict().items()},ordered=m.ordered,protocol_sha256=sha(R/'DETAIL_PROTOCOL.json'),train_users=q['train_users'],arm=arm,feature_checkpoint_sha256=spec['dino_int8_sha256']))
        restored=torch.load(cp,map_location='cpu',weights_only=True);m.load_state_dict(restored['state_dict']);m.eval();zs=[];vs=[];gs=[]
        with torch.inference_mode():
            for st in range(0,len(va),config['batch']):
                ii=torch.tensor(va[st:st+config['batch']],device='cuda');z,v,g=m(x[ii].float(),available[ii]);zs.append(z.cpu().numpy());vs.append(v.cpu().numpy());gs.append(g.cpu().numpy())
                guard.check(phase='FINAL_HELD_DETAIL',job=job,records=min(len(va),st+config['batch']))
        zz=np.concatenate(zs);yy=y[torch.tensor(va,device='cuda')].cpu().numpy();np.savez_compressed(jobout/'predictions.npz',clip_ids=np.array([rows[i]['clip_id'] for i in va]),targets=yy,users=users[va],logits=zz,view_logits=np.concatenate(vs),view_weights=np.concatenate(gs))
        done=dict(status='COMPLETE',job=job,fold=fold,epochs=config['epochs'],head_bytes=cp.stat().st_size,head_sha256=sha(cp),predictions_sha256=sha(jobout/'predictions.npz'),protocol_sha256=sha(R/'DETAIL_PROTOCOL.json'),seconds=time.time()-start,n=len(va),correct=int((zz.argmax(1)==yy).sum()),users={str(u):dict(n=int((users[va]==u).sum()),correct=int(((zz.argmax(1)==yy)&(users[va]==u)).sum())) for u in q['held_users']},Test_read=False)
        atomic(jobout/'DONE.json',done);print(json.dumps(done),flush=True);del m,opt
    atomic(out/'HEAD_QUEUE_DONE.json',dict(status='COMPLETE',at_unix=time.time(),fold=fold,arms=spec['head_arms'],protocol_sha256=sha(R/'DETAIL_PROTOCOL.json')))

def main():
    a=argparse.ArgumentParser();a.add_argument('--host',required=True);args=a.parse_args();spec=json.loads((R/'DETAIL_PROTOCOL.json').read_text(encoding='utf-8'))
    for name,h in spec['code_sha256'].items():assert sha(R/name)==h,name
    for name,h in spec['asset_sha256'].items():assert sha(R/name)==h,name
    rows=json.loads((R/'DETAIL_INPUTS.json').read_text(encoding='utf-8'));torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    out=R/'detail_output'/args.host;out.mkdir(parents=True,exist_ok=True);guard=Guard(out,spec['deadline_unix'],out/'TELEMETRY.json')
    guard.deadline=min(spec['deadline_unix'],time.time()+spec['cache_wall_cap_seconds'])
    cache(spec,args.host,rows,guard);torch.cuda.empty_cache();guard.deadline=spec['deadline_unix'];train(spec,args.host,rows,guard)
if __name__=='__main__':
    try:main()
    except BaseException as e:
        dest=R/'detail_output'/('local' if os.name=='nt' else 'cloud');dest.mkdir(parents=True,exist_ok=True);atomic(dest/'ERROR.json',dict(error=str(e),at_unix=time.time(),traceback=traceback.format_exc()));raise
