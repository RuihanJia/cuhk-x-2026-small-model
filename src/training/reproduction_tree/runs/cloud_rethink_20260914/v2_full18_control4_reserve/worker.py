"""Authorized fixed full18 CONTROL4 reserve and one 50/50 weight average; not selected."""
from pathlib import Path
import argparse,hashlib,importlib.util,os,sys,time,traceback
E=Path(__file__).resolve().parent;C=E.parent;K=C/'videomaev2_base';O=C.parent/'error_resolution_20260913'
sys.path[:0]=[str(K),str(K.parent/'v2_window_cloud52910'),str(O/'bundle')]
sp=importlib.util.spec_from_file_location('k400_training_worker',K/'worker.py');k=importlib.util.module_from_spec(sp);sys.modules[sp.name]=k;sp.loader.exec_module(k)
import torch
from codec import pack,unpack,FORMAT
from dual_host_runtime.io import atomic,read,save_torch
from dual_host_runtime.health import Guard,identity,owned_tree
sha=k.sha
BASE_MODEL=k.VideoMAEBase
WARM_PARENT=None
WARM_ARM=None
sp=importlib.util.spec_from_file_location('full18_window_inputs',C/'v2_window_cloud52910/data.py');wd=importlib.util.module_from_spec(sp);sp.loader.exec_module(wd)
class WarmClips(wd.WindowClips):
 def __init__(self,*args,arm='RAW_CONTROL',**kwargs):super().__init__(*args,arm=WARM_ARM,**kwargs)
def warm_model(*args,**kwargs):
 model=BASE_MODEL(*args,**kwargs);cp=torch.load(WARM_PARENT,map_location='cpu',weights_only=True);model.load_state_dict(cp['model'],strict=True);return model

def main(protocol,host):
 global WARM_PARENT,WARM_ARM
 assert host=='cloud' and os.name!='nt';p=Path(protocol).resolve();s=read(p);ph=sha(p);assert p.parent==E and s['status']=='FROZEN_FOR_EXECUTION'
 h=s['hosts'][host];out=Path(h['output_directory']);assert out.is_relative_to(E)
 for n,v in s['code_sha256'].items():k.checked(C/n,v)
 for n,v in h['input_bindings'].items():k.checked(n,v)
 k.checked(K/'vit_b_k710_dl_from_giant.pth',s['pretrained_sha256']);k.checked(K/'ADMISSION.json',s['admission_sha256'])
 bd=read(out/'BUDGET.json');ct=read(out/'CONTROLLER.json');assert bd['protocol_sha256']==ct['protocol_sha256']==ph
 owner=identity(os.getpid());assert any(v['pid']==owner['pid'] and abs(v['create_time']-owner['create_time'])<.01 for v in owned_tree(ct['worker']))
 guard=Guard(out,bd['deadline_unix'],out/'TELEMETRY.json');guard.check(force=True,phase='FULL18_PREFLIGHT')
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;assert '5090' in torch.cuda.get_device_name(0)
 inp=read(k.checked(h['pose_inputs'],s['pose_inputs_sha256']));job=s['jobs'][0];assert len(inp['rows'])==len({r['clip_id'] for r in inp['rows']})==2931
 assert job['train_users']==sorted({r['user'] for r in inp['rows']}) and len(job['train_users'])==18 and job['held_users']==[] and s['epochs']==4
 WARM_PARENT=k.checked(h['parent_checkpoint'],s['parent_checkpoint_sha256']);WARM_ARM=s['arm'];assert WARM_ARM=='CONTROL' and s['average_alpha']==.5
 cp=torch.load(WARM_PARENT,map_location='cpu',weights_only=True);assert cp['protocol_sha256']==s['parent_protocol_sha256'] and cp['epochs']==16 and cp['held_users']==[] and cp['train_users']==job['train_users'];del cp
 atomic(out/'PARENT_BINDING.json',dict(protocol_sha256=ph,parent_checkpoint_sha256=s['parent_checkpoint_sha256'],parent_protocol_sha256=s['parent_protocol_sha256'],arm=WARM_ARM,at_unix=time.time()))
 k.VideoMAEBase=warm_model;k.EpochClips=WarmClips
 assert not(out/'resume.pt').exists(),'NEW_STAGE_MUST_START_FRESH_ADAMW'
 runtime=dict(s);runtime.update(h);k.fit(job,runtime,ph,out,inp,guard)
 history=read(out/'HISTORY.json');expected_history=read(E/'CPU_QA.json')['expected_history'];assert all(all(a[key]==b[key]for key in ('epoch','records','updates','sampling_sha256'))for a,b in zip(history,expected_history)) and len(history)==len(expected_history)==4
 fr=read(out/'FROZEN.json');cp=torch.load(k.checked(out/'model_fp16.pt',fr['checkpoint_sha256']),map_location='cpu',weights_only=True);assert cp['held_users']==[] and cp['train_users']==job['train_users']
 q=pack(cp);save_torch(out/'model_int8.pt',q);loaded=torch.load(out/'model_int8.pt',map_location='cpu',weights_only=True);r1=unpack(q);r2=unpack(loaded);assert all(torch.equal(r1[n],r2[n]) for n in r1)
 assert len(q['quantized_names'])==50 and (out/'model_int8.pt').stat().st_size<87000000
 atomic(out/'QUANTIZED.json',dict(protocol_sha256=ph,parent_checkpoint_sha256=fr['checkpoint_sha256'],checkpoint_sha256=sha(out/'model_int8.pt'),checkpoint_bytes=(out/'model_int8.pt').stat().st_size,storage_format=FORMAT,quantized_matrices=50,reload_exact=True,no_Test=True,at_unix=time.time()))
 # Fixed average, retaining both alternative endpoints outside any submitted package.
 raw=torch.load(WARM_PARENT,map_location='cpu',weights_only=True);averaged={};changed=0
 for name,value in raw['model'].items():
  other=cp['model'][name];assert value.dtype==other.dtype==torch.float16 and value.shape==other.shape
  averaged[name]=((value.float()+other.float())*.5).half();changed+=int(not torch.equal(value,other))
 assert changed>0
 avg=dict(model=averaged,protocol_sha256=ph,pretrained_sha256=s['pretrained_sha256'],train_users=job['train_users'],held_users=[],epochs=4,epoch_semantics='freshCONTROL4_branch_not_continuous_epoch20',raw_epochs=16,control_epochs=4,control_optimizer='freshAdamW',raw_protocol_sha256=s['parent_protocol_sha256'],control_protocol_sha256=ph,seed=s['seed'],alpha=.5,raw_sha256=s['parent_checkpoint_sha256'],control_sha256=fr['checkpoint_sha256'],arithmetic='FP32 midpoint then FP16')
 save_torch(out/'average_model_fp16.pt',avg);packed=pack(avg);save_torch(out/'average_model_int8.pt',packed)
 reloaded=torch.load(out/'average_model_int8.pt',map_location='cpu',weights_only=True);expected=unpack(packed);replayed=unpack(reloaded)
 assert set(expected)==set(replayed) and all(torch.equal(expected[n],replayed[n]) for n in expected)
 assert len(reloaded['quantized_names'])==50 and (out/'average_model_int8.pt').stat().st_size<87000000
 # All finite source-interface checks use known training examples and a single seven-view readout.
 qa_row=next(r for r in inp['rows'] if r['user']==1);source=wd.SevenViews([qa_row],h['data_root'],inp['regions'])[0]
 model=BASE_MODEL(classes=40,fast=True,seed=s['seed']).cuda();model.load_state_dict(replayed,strict=True);model.eval()
 with torch.inference_mode():
  logits=torch.cat([model(source['x'][i:i+4].cuda()).cpu()for i in range(0,7,4)])
 assert logits.shape==(7,40) and logits.dtype==torch.float32 and torch.isfinite(logits).all()
 qa=dict(protocol_sha256=ph,source_clip_id=qa_row['clip_id'],source_user=1,views=7,FP32_logits_finite=True,logits_sha256=hashlib.sha256(logits.numpy().tobytes()).hexdigest(),no_Test=True)
 atomic(out/'SOURCE_INTERFACE_QA.json',qa)
 atomic(out/'AVERAGE_FROZEN.json',dict(protocol_sha256=ph,alpha=.5,arithmetic='FP32 midpoint of saved raw16/control4 FP16 then FP16 then originalALL50INT8',checkpoint_sha256=sha(out/'average_model_int8.pt'),checkpoint_bytes=(out/'average_model_int8.pt').stat().st_size,raw_checkpoint_sha256=s['parent_checkpoint_sha256'],control_checkpoint_sha256=fr['checkpoint_sha256'],average_fp16_sha256=sha(out/'average_model_fp16.pt'),tensors=len(expected),changed_tensors=changed,quantized_matrices=50,actual_FP32_storage_replay_exact=True,model_plus_aux_bytes=(out/'average_model_int8.pt').stat().st_size+s['retained_aux_bytes'],calibration_reserved_bytes=s['calibration_reserved_bytes'],no_selection_claim=True,at_unix=time.time()))
 assert (out/'average_model_int8.pt').stat().st_size+s['retained_aux_bytes']+s['calibration_reserved_bytes']<=100000000
 names=['PARENT_BINDING.json','TRAIN_DONE.json','model_fp16.pt','model_int8.pt','QUANTIZED.json','INITIAL.json','SPLIT.json','HISTORY.json','FIRST_UPDATE.json','FROZEN.json','average_model_fp16.pt','average_model_int8.pt','AVERAGE_FROZEN.json','SOURCE_INTERFACE_QA.json'];guard.check(force=True,phase='FULL18_CONTROL_AND_AVERAGE_RESERVE_COMPLETE')
 atomic(out/'DONE.json',dict(status='COMPLETE_FULL18_CONTROL4_AND_FIXED_AVERAGE_RESERVE',protocol_sha256=ph,job=job['id'],artifacts={n:sha(out/n) for n in names},no_Test=True,no_selection_claim=True,held_evaluation_performed=False,at_unix=time.time()))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--host',required=True);args=p.parse_args()
 try:main(**vars(args))
 except BaseException as e:
  s=read(args.protocol);atomic(Path(s['hosts'][args.host]['output_directory'])/'ERROR.json',dict(error=str(e),traceback=traceback.format_exc(),protocol_sha256=sha(args.protocol),at_unix=time.time()));raise
