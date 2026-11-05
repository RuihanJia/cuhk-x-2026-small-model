"""Fresh, user-disjoint video/sensor fits. Target-trained outer weights never initialize inner fits."""
from pathlib import Path
import argparse, json, sys, time, traceback

class FiveViews:
    def __init__(self,rows,root):self.rows=rows;self.root=root;self.seed=0;self.epoch=0
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        import torch
        from PIL import Image
        from torchvision.transforms import functional as TF
        from research_runtime.data import safe_path
        r=self.rows[index];values=[]
        for phase,crop in [(.5,'center'),(.5,'low'),(.5,'high'),(.25,'center'),(.75,'center')]:
            streams=[]
            for stream in r['streams']:
                d=safe_path(self.root,stream['directory']);files=stream['files'];frames=[]
                for k in range(16):
                    with Image.open(d/files[min(len(files)-1,int((k+phase)*len(files)/16))]) as im:frames.append(TF.pil_to_tensor(im.convert('RGB')))
                x=TF.resize(torch.stack(frames).float().div_(255),[256],antialias=True);h,w=x.shape[-2:]
                if crop=='center':x=TF.center_crop(x,[224,224])
                elif w>=h:x=TF.crop(x,int(round((h-224)/2.)),0 if crop=='low' else w-224,224,224)
                else:x=TF.crop(x,0 if crop=='low' else h-224,int(round((w-224)/2.)),224,224)
                streams.append(TF.normalize(x,[.45]*3,[.225]*3).permute(1,0,2,3).contiguous())
            values.append(torch.stack(streams))
        return dict(x=torch.stack(values),y=r['label'],user=r['user'],clip_id=r['clip_id'])

def five_predictions(model,rows,data_root,spec,guard,workers,device):
    import numpy as np, torch
    from research_runtime.data import loader
    from research_runtime.models import view_logits
    model.eval();ids=[];ys=[];users=[];values=[]
    with torch.no_grad():
        for batch in loader(FiveViews(rows,data_root),batch=1,workers=workers):
            z=torch.stack([view_logits(model,dict(x=batch['x'][:,j]),device) for j in range(5)],1)
            assert z.dtype==torch.float32 and torch.isfinite(z).all();values.append(z.cpu().numpy());ids.extend(batch['clip_id']);ys.extend(batch['y'].tolist());users.extend(batch['user'].tolist());guard.check(phase='FP32_FIXED_FIVEVIEW_EXPORT',records=len(ids),total=len(rows))
    z=np.concatenate(values);return dict(clip_ids=np.asarray(ids),targets=np.asarray(ys),users=np.asarray(users),view_modality_logits=z,depth_logits=z[:,:,0].mean(1),ir_logits=z[:,:,1].mean(1),center_logits=z[:,0].mean(1),logits=z.mean((1,2)))

def main():
    a=argparse.ArgumentParser();a.add_argument('--root',required=True);a.add_argument('--job',required=True);a.add_argument('--protocol',required=True);a.add_argument('--deadline',type=float,required=True);args=a.parse_args()
    root=Path(args.root);sys.path.insert(0,str(root/'bundle'))
    import numpy as np, torch
    from dual_host_runtime.io import atomic,read
    from optional_time_budget import Guard
    from research_runtime.common import configure,sha
    from research_runtime.train import train_video,train_sensor
    p=read(args.protocol);p.update(_sha=p.get('scientific_parent_sha256',sha(args.protocol)),_bundle=str(root/'bundle'))
    spec=p['jobs'][args.job];out=root/'output'/args.job;out.mkdir(parents=True,exist_ok=True)
    configure();guard=Guard(out,args.deadline,root/'TELEMETRY.json')
    assert not set(spec['train_users'])&set(spec['predict_users'])
    assert not set(spec['train_users'])&set(p['folds'][str(spec['fold'])])
    for name in ['mvit_public']:
        if spec['kind']=='video':assert sha(root/'bundle'/p['assets'][name]['path'])==p['assets'][name]['sha256']
    try:
        guard.check(force=True,phase='INITIALIZING',job=args.job)
        if spec['kind']=='video':
            import research_runtime.train as train_runtime
            train_runtime.video_predictions=five_predictions
            train_video(p,spec,p['remote_data'],out,root/'feature_cache',guard)
        elif spec['kind']=='sensor':
            p['calibration']=dict(encoder_fit_users=spec['train_users'],calibration_users=spec['predict_users'],assessment_users=[])
            train_sensor(p,out,guard)
        else:raise ValueError('UNKNOWN_JOB')
    except BaseException as e:
        atomic(out/'ERROR.json',dict(status='ENGINEERING_OR_BUDGET_INTERRUPTION',error=repr(e),traceback=traceback.format_exc(),at_unix=time.time()));raise

if __name__=='__main__':main()
