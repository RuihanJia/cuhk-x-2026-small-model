"""Registered zero-fit, held-user IR crop interventions. No Test inference."""
from pathlib import Path
import argparse, json, hashlib, time, sys, os
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import mvit_v2_s
from torchvision.transforms import functional as TF
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'bundle'))
from activity_token_readout import ActivityTokenModel
from activity_inputs import load_activity_five, normalized
from dual_host_runtime.health import Guard
from dual_host_runtime.io import atomic
torch.set_num_threads(2)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()

class Views(Dataset):
    def __init__(self,rows):self.rows=rows
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i];box=row['depth_box'];p=Path(row['IR_directory'])
        assert 'Training' in p.parts
        if box is None:return i,torch.empty(0)
        files=sorted(p.glob('*.png'));assert len(files)==row['frames']
        decoded={};views=[];x,y,w,h=box
        for phase in [.5,.25,.75]:
            frames=[]
            for k in range(16):
                index=min(len(files)-1,int((k+phase)*len(files)/16))
                if index not in decoded:
                    with Image.open(files[index]) as im:
                        im=im.convert('RGB');assert list(im.size)==row['native_size'];decoded[index]=TF.pil_to_tensor(im.crop((x,y,x+w,y+h)))
                frames.append(decoded[index])
            a=TF.resize(torch.stack(frames).float().div_(255),[224,224],antialias=True)
            views.extend([normalized(a),normalized(a.sqrt())])
        return i,torch.stack(views)

def main():
    spec=json.loads((R/'CROP_PROBE_PROTOCOL.json').read_text());assert sha(R/'crop_probe.py')==spec['code_sha256']
    for rel,h in spec['bundle_hashes'].items():assert sha(R/rel)==h
    source=R/'CROP_INPUTS.json';assert sha(source)==spec['inputs_sha256']
    rows=json.loads(source.read_text());out=R/'crop_probe_output';out.mkdir(exist_ok=True)
    guard=Guard(out,min(time.time()+spec['wall_cap_seconds'],spec['deadline_unix']),R/'TELEMETRY.json')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    start=time.time();all_results=[]
    for fold in [0,3]:
        q=spec['folds'][str(fold)];cp_path=Path(q['checkpoint']);assert sha(cp_path)==q['checkpoint_sha256']
        cp=torch.load(cp_path,map_location='cpu',weights_only=True)
        assert sorted(cp['train_users'])==q['train_users'] and not set(cp['train_users'])&set(q['held_users'])
        assert cp['protocol_sha256']==q['source_protocol_sha256']
        video=mvit_v2_s(weights=None);video.head=nn.Linear(768,40);model=ActivityTokenModel(video,'TOKEN_DIFF')
        model.load_state_dict(cp['state_dict'],strict=True);del cp;model.eval().requires_grad_(False).cuda()
        npz=R/f'base_F{fold}.npz';assert sha(npz)==q['prediction_sha256'];base=dict(np.load(npz,allow_pickle=False))
        ids=[str(x) for x in base['clip_ids']];mapping={x:i for i,x in enumerate(ids)}
        rr=[r for r in rows if r['fold']==fold];assert set(ids)=={r['clip_id'] for r in rr}
        original=base['phase_spatial_modality_logits'];assert original.shape==(len(ids),5,2,2,40)
        qa=[]
        with torch.inference_mode():
            for row in sorted(rr,key=lambda r:r['clip_id'])[:2]:
                rowidx=mapping[row['clip_id']];parts=[]
                for mod in ['Depth_Color','IR']:
                    z,sub,bad,region=load_activity_five(row[mod+'_directory']);assert not sub and not bad;parts.append(z)
                xx=torch.stack(parts,2);zz=[]
                for phase in range(5):
                    zz.append(np.stack([model(xx[phase,s].cuda()).float().cpu().numpy() for s in range(2)]))
                zz=np.stack(zz);err=float(np.abs(zz-original[rowidx]).max());assert err<1e-4,(fold,row['clip_id'],err)
                qa.append(dict(clip_id=row['clip_id'],max_abs_error=err))
            atomic(out/f'F{fold}_REPLAY_QA.json',dict(protocol_sha256=sha(R/'CROP_PROBE_PROTOCOL.json'),cases=qa))
            z1=original.copy();z2=original.copy();completed=0
            loader=DataLoader(Views(rr),batch_size=None,num_workers=4,prefetch_factor=1,pin_memory=True)
            for i,xx in loader:
                row=rr[i];k=mapping[row['clip_id']];assert int(base['targets'][k])==row['target']
                if xx.numel():
                    zz=model(xx.cuda(non_blocking=True)).float().cpu().numpy().reshape(3,2,40)[[0,0,0,1,2]]
                    z1[k,:,1,1]=zz[:,0];z2[k,:,1,1]=zz[:,1]
                completed+=1;guard.check(fold=fold,completed=completed,total=len(rr),phase='ZERO_FIT_IR_CROP_INTERVENTIONS')
                if completed%50==0:print(json.dumps(dict(fold=fold,completed=completed,total=len(rr))),flush=True)
            aggregate=lambda z:(z[:,:,0].mean((1,2))+z[:,:,1].mean((1,2)))*.5
            target=base['targets'];v0=aggregate(original);v1=aggregate(z1);v2=aggregate(z2)
            np.savez_compressed(out/f'F{fold}_INTERVENTIONS.npz',clip_ids=base['clip_ids'],targets=target,users=base['users'],baseline_phase=original,depth_box_phase=z1,depth_box_gamma_phase=z2)
            scores={key:dict(correct=int((v.argmax(1)==target).sum()),fix=int(((v.argmax(1)==target)&(v0.argmax(1)!=target)).sum()),harm=int(((v.argmax(1)!=target)&(v0.argmax(1)==target)).sum())) for key,v in [('original',v0),('depth_box',v1),('depth_box_gamma',v2)]}
            result=dict(fold=fold,n=len(rr),scores=scores,changed_crops=sum(r['depth_box'] is not None for r in rr),seconds=time.time()-start)
            atomic(out/f'F{fold}_RESULT.json',result);all_results.append(result);print(json.dumps(result),flush=True)
        del model;torch.cuda.empty_cache()
    atomic(out/'DONE.json',dict(status='COMPLETE_ZERO_FIT_DIAGNOSTIC',training_fits=0,Test_read=False,protocol_sha256=sha(R/'CROP_PROBE_PROTOCOL.json'),seconds=time.time()-start,results=all_results))

if __name__=='__main__':
    try:main()
    except BaseException as e:
        import traceback
        atomic(R/'crop_probe_output'/'ERROR.json',dict(at_unix=time.time(),error=str(e),traceback=traceback.format_exc()));raise
