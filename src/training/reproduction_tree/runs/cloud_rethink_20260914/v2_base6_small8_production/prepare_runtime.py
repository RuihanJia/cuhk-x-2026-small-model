"""Prepare only the paired inference source; no model/calibration/Test actions."""
from pathlib import Path
import hashlib,json,time
E=Path(__file__).resolve().parent;R=E.parents[2];P=R/'deliverables/videomaev2_base_temporal_halves_candidate_20260915'
t=(P/'infer.py').read_text(encoding='utf-8')
t=t.replace('from mae_model import VideoMAEBase\nfrom mae_codec import unpack','from dual_model import PairVideo\nfrom loader import load_state')
a=t.index('def constructor(');b=t.index('\nclass BaseViews:',a)
t=t[:a]+'''def constructor(container,kind,s):
 state,origin=load_state(container,kind)
 assert origin['train_users']==s['training_users'] and origin['held_users']==[] and origin['epochs']==16
 assert origin['base_fp16_sha256']==s['pair_provenance']['base_fp16_sha256'] and origin['small_fp16_sha256']==s['pair_provenance']['small_fp16_sha256']
 model=PairVideo(kind);model.load_state_dict(state,strict=True);return model.eval().requires_grad_(False)
''' +t[b:]
a=t.index('@torch.inference_mode()\ndef predict(');b=t.index('\ndef fusion(',a)
t=t[:a]+'''@torch.inference_mode()
def predict(model,item,device,micro=4,kind='Base'):
 x=item['mae']
 if not x.numel():return np.zeros(40,np.float32),False
 assert x.shape==(7,3,16,224,224)
 x=x if kind=='Base' else x[:5];dup=torch.equal(x[0],x[1]) and torch.equal(x[0],x[2]);sel=[0,3,4,5,6] if kind=='Base' else [0,3,4];selected=x[sel] if dup else x;values=[]
 for i in range(0,len(selected),micro):
  z=model(selected[i:i+micro].to(device,non_blocking=True));assert z.dtype==torch.float32 and torch.isfinite(z).all();values.append(z.cpu().numpy())
 z=np.concatenate(values)
 if dup:z=z[[0,0,0,1,2,3,4] if kind=='Base' else [0,0,0,1,2]]
 return ((.5*z[:5].mean(0)+.25*z[5]+.25*z[6]) if kind=='Base' else z.mean(0)).astype(np.float32),True
''' +t[b:]
t=t.replace("if k!='POSE':","if k not in ('POSE','PAIR'):")
a=t.index(" model=constructor(cp['B'],s).to(device)");b=t.index('\n io.write_submission',a)
t=t[:a]+''' models={kind:constructor(ROOT/s['checkpoints']['PAIR']['path'],kind,s).to(device) for kind in ('Base','Small')};values={kind:[]for kind in models};available={kind:[]for kind in models};loader=DataLoader(BaseViews(samples,regions),batch_size=None,num_workers=workers,pin_memory=True,prefetch_factor=1)
 for i,item in enumerate(loader):
  assert item['index']==i
  for kind,model in models.items():
   z,valid=predict(model,item,device,micro,kind);values[kind].append(z);available[kind].append(valid)
  recoveries['video_substitutions']+=item['sub'];recoveries['video_invalid_files']+=item['bad']
  if (i+1)%25==0:progress('PAIR_BASE6_SMALL8_FP32',i+1,405)
 del models,model;torch.cuda.empty_cache();logits=np.stack([tz,imu,values['Base'],values['Small']],1).astype(np.float32);mask=np.stack([tvalid,ivalid,available['Base'],available['Small']],1).astype(bool);fused,weights=fusion(logits,mask,cal);assert fused.shape==(405,40)
''' +t[b:]
t=t.replace("enumerate(['T','I','B'])","enumerate(['T','I','Base','Small'])").replace('BASE_TI_FIXED50_TEMPORAL_HALVES_CSV_CREATED','BASE6_SMALL8_TI_CLASSWISE_CSV_CREATED')
# New artifact hashes are canonicalized; old original-template caches are explicit.
t=t.replace("io.atomic_json=atomic_with_retry", "io.atomic_json=atomic_with_retry\n\ndef sha_matches(path,expected):return io.sha256(path).lower()==str(expected).lower()")
t=t.replace("assert io.sha256(p)==v['sha256']", "assert sha_matches(p,v['sha256'])")
t=t.replace("assert io.sha256(calpath)==s['calibrator_sha256']", "assert sha_matches(calpath,s['calibrator_sha256'])")
t=t.replace("assert io.sha256(sensor_cache)==s['optional_sensor_cache_sha256']", "assert strict_original_template;assert sha_matches(sensor_cache,s['optional_sensor_cache_sha256'])")
t=t.replace("assert io.sha256(ROOT/cache['path'])==cache['sha256']", "assert sha_matches(ROOT/cache['path'],cache['sha256'])")
t=t.replace("value['model_sha256'].upper()==s['checkpoints']['POSE']['sha256']", "value['model_sha256'].lower()==s['checkpoints']['POSE']['sha256'].lower()")
t=t.replace("def infer(data_dir,output,guard=None,workers=4,micro=4,sensor_cache=None):", "def infer(data_dir,output,guard=None,workers=4,micro=4,sensor_cache=None,strict_original_template=False):")
t=t.replace("assert len(samples)==405;b.verify_paths([x.output_path for x in samples]);ids=", "n=len(samples);assert n>0\n if strict_original_template:\n  assert n==405;b.verify_paths([x.output_path for x in samples])\n ids=")
t=t.replace("if cache and (ROOT/cache['path']).exists():", "if strict_original_template and cache and (ROOT/cache['path']).exists():")
t=t.replace(",405,405)",",n,n)").replace(",i+1,405)",",i+1,n)").replace("fused.shape==(405,40)","fused.shape==(n,40)").replace("rows=405,classes=40","rows=n,classes=40")
t=t.replace("p.add_argument('--sensor-cache');", "p.add_argument('--sensor-cache');p.add_argument('--strict-original-template',action='store_true');")

compile(t,'paired_infer.py','exec');(E/'paired_infer.py').write_text(t,encoding='utf-8')
r=dict(status='PREPARED_RUNTIME_NOT_REGISTERED_NOT_EXECUTED',at_unix=time.time(),template_sha256=hashlib.sha256((P/'infer.py').read_bytes()).hexdigest(),prepared_infer_sha256=hashlib.sha256((E/'paired_infer.py').read_bytes()).hexdigest(),new_optimizer_updates=0,new_Test_forwards=0,new_final_calibrators=0,requires=['Complete correct8model source/outer validation before production selection','Exact independent container loader SHA and provenance','Source4768 final4branch cal fit+normalization+independent replay','Full18 containerpack originalFP16 parents','Actualsource interfaceQA then405readout+independentCSV+100MB audit'],no_existing_release_modified=True)
(E/'PREPARATION.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(json.dumps(r))
