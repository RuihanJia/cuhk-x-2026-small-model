from pathlib import Path
import os,hashlib,json,time
os.environ['CUDA_VISIBLE_DEVICES']=''
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
import torch
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def audit(E):
 torch.set_num_threads(1);p=read(E/'protocol.cloud.json');ph=sha(E/'protocol.cloud.json');out=E/'output/cloud';ct=read(out/'CONTROLLER_DONE.json');done=read(out/'DONE.json')
 assert ct['status']=='COMPLETE' and ct['exit_code']==0 and ct['protocol_sha256']==done['protocol_sha256']==ph and ct['terminal_sha256']==sha(out/'DONE.json')
 for f,h in done['artifacts'].items():assert sha(out/f)==h,f
 raw=torch.load(E/'parent/model_fp16.pt',map_location='cpu',weights_only=True);control=torch.load(out/'model_fp16.pt',map_location='cpu',weights_only=True);avg=torch.load(out/'average_model_fp16.pt',map_location='cpu',weights_only=True);q=torch.load(out/'average_model_int8.pt',map_location='cpu',weights_only=True)
 assert raw['epochs']==16 and control['epochs']==4 and avg['epochs']==q['epochs']==4
 assert avg['raw_epochs']==q['raw_epochs']==16 and avg['control_epochs']==q['control_epochs']==4
 assert avg['epoch_semantics']==q['epoch_semantics']=='freshCONTROL4_branch_not_continuous_epoch20'
 assert avg['control_optimizer']==q['control_optimizer']=='freshAdamW' and avg['alpha']==q['alpha']==.5
 assert avg['raw_sha256']==q['raw_sha256']==p['parent_checkpoint_sha256']==sha(E/'parent/model_fp16.pt')
 assert avg['control_sha256']==q['control_sha256']==sha(out/'model_fp16.pt')
 assert avg['raw_protocol_sha256']==q['raw_protocol_sha256']==raw['protocol_sha256']==p['parent_protocol_sha256']
 assert avg['control_protocol_sha256']==q['control_protocol_sha256']==control['protocol_sha256']==ph
 for cp in (raw,control,avg,q):assert cp['train_users']==p['jobs'][0]['train_users'] and cp['held_users']==[]
 assert raw['model'].keys()==control['model'].keys()==avg['model'].keys()==q['model'].keys()
 dig=hashlib.sha256();quant=0;values=0
 for name,a in raw['model'].items():
  b=control['model'][name];v=avg['model'][name];assert a.dtype==b.dtype==v.dtype==torch.float16
  expected=((a.float()+b.float())/2).to(torch.float16);assert torch.equal(expected,v),name
  field=q['model'][name]
  if name.endswith('.weight')and v.ndim>=2:
   flat=v.float().reshape(v.shape[0],-1);scale=flat.abs().amax(1).clamp_min(1e-12)/127
   qs=(flat/scale[:,None]).round().clamp(-127,127).to(torch.int8)
   assert torch.equal(field['q'].reshape_as(qs),qs)and torch.equal(field['scale'].reshape(-1),scale),name
   quant+=1
  else:assert torch.equal(field['value'],expected);values+=1
  dig.update(name.encode());dig.update(a.float().contiguous().numpy().tobytes())
 fr=read(out/'FROZEN.json');assert fr['initial_digest']==dig.hexdigest() and fr['updates']==736 and fr['epochs']==4
 h=read(out/'HISTORY.json');qa=read(E/'CPU_QA.json')
 assert len(h)==len(qa['expected_history'])==4
 for actual,expected in zip(h,qa['expected_history']):
  assert all(actual[k]==expected[k]for k in ('epoch','records','updates','sampling_sha256'))
 assert quant==50 and values==112
 size=(out/'average_model_int8.pt').stat().st_size
 assert size+p['retained_aux_bytes']+p['calibration_reserved_bytes']<=100000000
 report=dict(status='PASS_INDEPENDENT_FULL18_CONTROL4_AVERAGE_STORAGE',protocol_sha256=ph,model_sha256=sha(out/'average_model_int8.pt'),model_bytes=size,model_plus_aux_and_reserved_cal_bytes=size+p['retained_aux_bytes']+p['calibration_reserved_bytes'],raw_parent_sha256=sha(E/'parent/model_fp16.pt'),control_sha256=sha(out/'model_fp16.pt'),actual_average_fp16_sha256=sha(out/'average_model_fp16.pt'),raw_initial_digest_exact=True,actual_average_all_tensors_equal=quant+values,quantized_weights_direct_from_average_FP16_exact=quant,other_FP16_exact=values,four_sampling_histories_match_frozen_preregistered=True,updates=736,metadata_preserves_fresh_control4=True,no_GPU=True,no_Test=True,at_unix=time.time())
 (out/'INDEPENDENT_STORAGE_AUDIT.json').write_text(json.dumps(report,indent=2));return dict(**report,audit_sha256=sha(out/'INDEPENDENT_STORAGE_AUDIT.json'))
