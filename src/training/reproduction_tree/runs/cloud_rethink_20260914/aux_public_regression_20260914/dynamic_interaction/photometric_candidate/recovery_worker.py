"""Inference-only recovery of a stale cloud-only utility import; no training code is called."""
from worker import *
RECOVERY_PH=None

def align(a,ids):
    lookup={str(k):i for i,k in enumerate(a['clip_ids'])};assert len(lookup)==len(a['clip_ids'])
    assert set(lookup)==set(map(str,ids))
    ix=[lookup[str(k)] for k in ids]
    return {k:v[ix] for k,v in a.items() if isinstance(v,np.ndarray) and v.ndim and len(v)==len(ix)}

def evaluate_job(job,s,ph,out,phaseout,inputs,guard,barrier_sha):
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
    atomic(dest/'PHOTOMETRIC.json',dict(protocol_sha256=ph,recovery_protocol_sha256=RECOVERY_PH,arm=job['arm'],per_view16_shared_scalar=True,records=photo))
    done=dict(status='COMPLETE_PHOTOMETRIC_HELD',protocol_sha256=ph,recovery_protocol_sha256=RECOVERY_PH,job=job['id'],fold=job['fold'],arm=job['arm'],predictions_sha256=sha(dest/'predictions.npz'),checkpoint_sha256=fr['checkpoint_sha256'],host_training_barrier_sha256=barrier_sha,records=len(ids),global_metrics_not_computed=True,no_fit=True,no_Test=True,at_unix=time.time())
    atomic(dest/'DONE.json',done);del model,cp,loader;torch.cuda.empty_cache();return done

def main(protocol,host,phase):
 global RECOVERY_PH
 assert os.name=='nt' and host=='local' and phase=='evaluate'
 rp=Path(protocol).resolve();r=read(rp);rh=sha(rp);RECOVERY_PH=rh
 assert rp.parent==P and r['status']=='FROZEN_FOR_EXECUTION' and r['wall_cap_seconds']==900
 assert sha(__file__)==r['worker_sha256'];sp=checked(r['original_protocol'],r['original_protocol_sha256']);s=read(sp);ph=sha(sp)
 assert s['status']=='FROZEN_FOR_EXECUTION' and s['epochs']==4 and s['micro_clips']==s['effective_clip_batch']==8
 h=s['hosts']['local'];trainroot=Path(h['output_directory']);root=Path(r['hosts']['local']['output_directory']).resolve();stage=root/'evaluate'
 assert root.is_relative_to(P/'recovery_20260915') and not any((stage/j['id']).exists() for j in s['jobs'] if j['host']=='local')
 bd=read(root/'HOST_BUDGET.json');pb=read(stage/'CONTROLLER_BUDGET.json');ct=read(stage/'CONTROLLER.json')
 assert bd['protocol_sha256']==pb['protocol_sha256']==ct['protocol_sha256']==rh and bd['deadline_unix']<=min(r['deadline_unix'],bd['first_started_unix']+900)
 actual=identity(os.getpid());assert any(v['pid']==actual['pid'] and abs(v['create_time']-actual['create_time'])<.01 for v in owned_tree(ct['owner']))
 guard=Guard(stage,min(bd['deadline_unix'],pb['deadline_unix']),stage/'TELEMETRY.json');guard.check(force=True,phase='RECOVERY_INFERENCE_ONLY_PREFLIGHT')
 for rel,v in s['code_sha256'].items():checked(C/rel,v)
 for rel,v in r['code_sha256'].items():checked(C/rel,v)
 for path,v in h['input_bindings'].items():checked(path,v)
 auth=read(checked(h['authorization_file'],s['source_authorization_sha256']));assert auth['local_inference_authorized']
 for kind in ('train','evaluate'):
  oldct=read(trainroot/kind/'CONTROLLER.json')
  from dual_host_runtime.health import alive
  assert not alive(oldct['owner']) and not alive(oldct['controller'])
 runtime=dict(s,**{k:v for k,v in h.items() if k not in s});runtime.update(data_root=h['data_root'],workers=h['workers'],pose_inputs=h['pose_inputs'],original_protocol=h['original_protocol'],job_wall_cap_seconds=h['job_wall_cap_seconds'])
 original=read(checked(h['original_protocol'],s['original_protocol_sha256']));jobs=[]
 for j in s['jobs']:
  if j['host']!='local':continue
  value=dict(j,**j['paths']['local']);verify_parent_host(value,runtime,original,'local');jobs.append(value)
 inputs=read(checked(h['pose_inputs'],s['pose_inputs_sha256']));assert len(inputs['rows'])==len({x['clip_id'] for x in inputs['rows']})==2931
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert torch.cuda.is_available() and '4070' in torch.cuda.get_device_name(0)
 barrier=verify_host_training(s,ph,trainroot,jobs)
 for job in jobs:evaluate_job(job,runtime,ph,trainroot,stage,inputs,guard,barrier)
 atomic(stage/'RESULT.json',dict(status='COMPLETE_PHOTOMETRIC_RAW_READOUT',protocol_sha256=rh,training_protocol_sha256=ph,host='local',host_training_barrier_sha256=barrier,jobs=[j['id'] for j in jobs],all_four_jobs_protocol_frozen_before_training=True,global_metrics_not_computed=True,no_Test=True,no_production=True,training_updates=0))
 files={'RESULT.json':sha(stage/'RESULT.json')}
 for j in jobs:
  for name in ('predictions.npz','PHOTOMETRIC.json','DONE.json'):files[j['id']+'/'+name]=sha(stage/j['id']/name)
 atomic(stage/'DONE.json',dict(status='COMPLETE_EVALUATE_PHASE',protocol_sha256=rh,training_protocol_sha256=ph,host='local',artifacts=files,calibration_fit_steps=0,training_updates=0,screening_only=True,global_metrics_not_computed=True,all_four_jobs_protocol_frozen_before_training=True))
 guard.check(force=True,phase='COMPLETE_EVALUATE_PHASE')

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);parser.add_argument('--host',required=True);parser.add_argument('--phase',required=True);args=parser.parse_args()
 try:main(args.protocol,args.host,args.phase)
 except BaseException as exc:
  r=read(args.protocol);out=Path(r['hosts']['local']['output_directory'])/'evaluate';out.mkdir(parents=True,exist_ok=True)
  atomic(out/'ERROR.json',dict(protocol_sha256=sha(args.protocol),error=str(exc),traceback=traceback.format_exc(),at_unix=time.time()));raise
