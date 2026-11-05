"""Offline inference for Activity token-stopgrad with frozen R5 S/T/I calibration."""
from pathlib import Path
import argparse,json,os,sys,time
import numpy as np
import torch
from torch import nn
from PIL import Image,UnidentifiedImageError
from torchvision.transforms import functional as TF
import video_io as io
from activity_inputs import video_prediction
from template_contract import verify_paths
from raw_sensor import skeleton_feature,imu_feature
from device_inputs import device_feature
from imu_model import DeviceIMU
from sensor_model import SensorTemporal
from fusion_math import predict as fusion_predict
from torchvision.models.video import mvit_v2_s
from activity_token_readout import ActivityTokenModel, validate_token_payload

ROOT=Path(__file__).resolve().parent
DLL_HANDLES=[]

def setup():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False

def video_constructor(payload, config):
    video=mvit_v2_s(weights=None);video.head=nn.Linear(768,40)
    model=ActivityTokenModel(video,'TOKEN_DIFF')
    spec=dict(arm='TOKEN_DIFF',train_users=config['training_users'],source_sha256=config['video_source_sha256'])
    validate_token_payload(payload,model,config['checkpoints']['V']['protocol_sha256'],spec)
    model.load_state_dict(payload['state_dict'],strict=True)
    return model.eval().requires_grad_(False)


def thermal_constructor():
    deps=ROOT/'third_party';sys.path.insert(0,str(deps))
    if os.name=='nt':
        sys.path[:0]=[str(deps/'win32'),str(deps/'win32/lib')]
        if hasattr(os,'add_dll_directory'):DLL_HANDLES.append(os.add_dll_directory(str(deps/'pywin32_system32')))
    from pytorchvideo.models.hub import x3d_m
    model=x3d_m(pretrained=False);model.blocks[-1].proj=nn.Linear(model.blocks[-1].proj.in_features,40);model.blocks[-1].activation=None
    return model

def thermal_input(path,mean,std):
    path=Path(path);files=sorted((p for p in path.glob('*') if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png'}),key=lambda p:p.name)
    if not files:return None,0,0
    decoded={};frames=[];bad=set();substitutions=0
    for k in range(16):
        target=min(len(files)-1,int((k+.5)*len(files)/16));chosen=None
        for index in sorted(range(len(files)),key=lambda i:(abs(i-target),i)):
            if index not in decoded:
                try:
                    with Image.open(files[index]) as im:decoded[index]=TF.pil_to_tensor(im.convert('RGB'))
                except (OSError,ValueError,UnidentifiedImageError):decoded[index]=None;bad.add(index)
            if decoded[index] is not None:chosen=index;break
        if chosen is None:return None,substitutions,len(bad)
        frames.append(decoded[chosen]);substitutions+=int(chosen!=target)
    x=TF.center_crop(TF.resize(torch.stack(frames).float().div_(255),[256],antialias=True),[224,224]);x=TF.normalize(x,mean,std)
    return x.permute(1,0,2,3).contiguous(),substitutions,len(bad)

def measured_arrays(samples,progress=None):
    sk=[];arms=[];arm_valid=[];sk_present=[];devices=[];device_masks=[];fallback=0
    for i,sample in enumerate(samples):
        path=sample.sample_dir;present=bool(list((path/'Skeleton/predictions').glob('*.json')))
        sk.append(skeleton_feature(path/'Skeleton') if present else np.zeros((64,120),np.float32));a,v=imu_feature(path/'IMU');x,mask,recovered=device_feature(path/'IMU',a,v)
        arms.append(a);arm_valid.append(v);sk_present.append(present);devices.append(x);device_masks.append(mask);fallback+=int(recovered)
        if progress and (i+1)%50==0:progress('RAW_SENSOR_FEATURES',i+1,len(samples))
    return dict(skeleton=np.stack(sk),imu=np.stack(arms),imu_valid=np.asarray(arm_valid,bool),skeleton_present=np.asarray(sk_present,bool),devices=np.stack(devices),device_valid=np.stack(device_masks),device_parse_fallback=fallback)

def sensor_predictions(data,payload,device):
    x=((data['imu']-payload['mean'].numpy())/payload['std'].numpy()).astype(np.float32);valid=data['imu_valid'];x[~valid]=0;x=np.concatenate([x,np.broadcast_to(valid[:,None,None],(len(valid),1,256)).astype(np.float32)],1)
    sk=torch.from_numpy(data['skeleton'].transpose(0,2,1).copy()).to(device);im=torch.from_numpy(x).to(device);model=SensorTemporal().to(device);model.load_state_dict(payload['state_dict'],strict=True);model.eval();values=[]
    with torch.inference_mode():
        for k in range(0,len(valid),64):values.append(model(sk[k:k+64],im[k:k+64]).float().cpu().numpy())
    del model,sk,im;return np.concatenate(values)

def imu_predictions(data,payload,device):
    model=DeviceIMU().to(device);model.load_state_dict(payload['state_dict'],strict=True);model.eval();raw=data['devices'];x=(raw-payload['mean'].numpy())/payload['std'].numpy();values=[]
    with torch.inference_mode():
        for k in range(0,len(x),64):
            values.append(model(torch.from_numpy(x[k:k+64]).to(device),torch.from_numpy(data['device_valid'][k:k+64]).to(device)).float().cpu().numpy())
    del model;return np.concatenate(values)


def infer(data_dir,output,guard=None):
    setup();output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);config=json.loads((ROOT/'release.json').read_text(encoding='utf-8'));device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');start=time.time()
    def progress(phase,records,total):
        if guard:guard.check(force=True,phase=phase,records=records,total=total)
        io.atomic_json(Path(str(output)+'.status.json'),dict(phase=phase,records=records,total=total,at_unix=time.time()))
        print(json.dumps(dict(phase=phase,records=records,total=total)),flush=True)
    size=0;payloads={}
    for key,entry in config['checkpoints'].items():
        path=ROOT/entry['path'];assert io.sha256(path)==entry['sha256'];size+=path.stat().st_size;cp=torch.load(path,map_location='cpu',weights_only=True);assert cp['protocol_sha256']==entry['protocol_sha256'];assert sorted(cp['train_users'])==config['training_users'];payloads[key]=cp
    fusion_path=ROOT/'calibrator.json';assert io.sha256(fusion_path)==config['calibrator_sha256'];calibrator=json.loads(fusion_path.read_text());size+=fusion_path.stat().st_size;assert size<=100000000
    samples,discovery=io.discover_samples(Path(data_dir));assert len(samples)==405;verify_paths([s.output_path for s in samples]);ids=np.asarray([s.sample_dir.name for s in samples]);data=measured_arrays(samples,progress)
    sensor=sensor_predictions(data,payloads['S'],device);imu=imu_predictions(data,payloads['I'],device);assert sensor.dtype==imu.dtype==np.float32
    thermal=thermal_constructor().to(device);thermal.load_state_dict(payloads['T']['state_dict'],strict=True);thermal.eval();tz=[];tvalid=[];recoveries=dict(video_substitutions=0,video_invalid_files=0,thermal_substitutions=0,thermal_invalid_files=0,device_parse_fallback=data['device_parse_fallback'])
    pending=[];pending_indices=[];tz=[np.zeros(40,np.float32) for _ in samples]
    with torch.inference_mode():
        for i,sample in enumerate(samples):
            x,sub,bad=thermal_input(sample.sample_dir/'Thermal',payloads['T']['mean'],payloads['T']['std']);recoveries['thermal_substitutions']+=sub;recoveries['thermal_invalid_files']+=bad;tvalid.append(x is not None)
            if x is not None:pending.append(x);pending_indices.append(i)
            if len(pending)==2 or (i+1==len(samples) and pending):
                values=thermal(torch.stack(pending).to(device)).float().cpu().numpy()
                for index,value in zip(pending_indices,values):tz[index]=value
                pending=[];pending_indices=[]
            if (i+1)%100==0:progress('THERMAL_FP32_PREDICTION',i+1,405)
    del thermal;torch.cuda.empty_cache();video=video_constructor(payloads['V'],config).to(device).eval();vz=[];depth=[];ir=[];vvalid=[]
    for i,sample in enumerate(samples):
        if guard:guard.bounds()
        v,d,r,valid,sub,bad=video_prediction(video,sample,device);vz.append(v);depth.append(d);ir.append(r);vvalid.append(valid);recoveries['video_substitutions']+=sub;recoveries['video_invalid_files']+=bad
        if (i+1)%50==0:progress('VIDEO_GLOBAL_AND_ACTIVITY_FIVE_VIEW_FP32',i+1,405)
    del video;torch.cuda.empty_cache();logits=np.stack([np.stack(vz),sensor,np.stack(tz),imu],1).astype(np.float32);available=np.stack([vvalid,data['skeleton_present'],tvalid,data['imu_valid']],1).astype(bool)
    assert available.any(1).all() and np.isfinite(logits).all();fusion_data=dict(logits=logits,available=available,depth=np.stack(depth),ir=np.stack(ir));fused,weights=fusion_predict(fusion_data,calibrator,True)
    assert fused.shape==(405,40) and fused.dtype==np.float32 and np.isfinite(fused).all();assert np.array_equal(weights[~available],np.zeros((~available).sum()))
    rows=[(sample.output_path,int(z.argmax())) for sample,z in zip(samples,fused)];io.write_submission(output,rows);np.savez_compressed(str(output)+'.logits.npz',clip_ids=ids,output_paths=np.asarray([s.output_path for s in samples]),branch_logits=logits,available=available,fused=fused,weights=weights.astype(np.float32),depth=fusion_data['depth'],ir=fusion_data['ir'])
    audit=dict(status='FIXED_ACTIVITY_TOKEN_STOPGRAD_R5_TRANSFER_CSV_CREATED',rows=405,classes=40,csv_sha256=io.sha256(output),calibrator_sha256=config['calibrator_sha256'],all_serialized_inference_weight_bytes=size,
        checkpoints=config['checkpoints'],availability={name:int(available[:,i].sum()) for i,name in enumerate(['V','S','T','I'])},valid_IMU_devices=data['device_valid'].sum(0).tolist(),recoveries=recoveries,
        elapsed_seconds=time.time()-start,Test_labels_read=False,uploaded=False,code_environment='Existing projectPython environment; independent cleanenvironment not claimed',runtime=dict(torch=torch.__version__,device=str(device)))
    io.atomic_json(Path(str(output)+'.audit.json'),audit);progress('COMPLETE',405,405);print(json.dumps(audit,ensure_ascii=False));return audit

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--data-dir',required=True);a.add_argument('--output',required=True);args=a.parse_args();infer(args.data_dir,args.output)
