"""Fresh-optimizer CONTROL4 stage, then fixed FP32 50/50 midpoint and INT8 export."""
from pathlib import Path
import argparse,functools,hashlib,importlib.util,json,sys,time
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent
if sys.platform=='win32':HERE=Path('\\\\?\\'+str(HERE))
TREE=HERE/'reproduction_tree';C=TREE/'runs/cloud_rethink_20260914'
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--pretrained',required=True);p.add_argument('--parent',required=True);p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--micro',type=int,default=4);p.add_argument('--max-hours',type=float,default=4);p.add_argument('--check-only',action='store_true');p.add_argument('--allow-retrained-parent',action='store_true');p.add_argument('--generated-inputs',type=Path,required=True);a=p.parse_args()
 import torch
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 pp=C/'v2_full18_control4_reserve/protocol.cloud.json';s=json.loads(pp.read_text(encoding='utf-8'));ph=hashlib.sha256(pp.read_bytes()).hexdigest();parent=Path(a.parent).resolve();pre=Path(a.pretrained).resolve()
 assert hashlib.sha256(pre.read_bytes()).hexdigest()==s['pretrained_sha256'];raw=torch.load(parent,map_location='cpu',weights_only=True);assert raw['epochs']==16 and raw['held_users']==[] and raw['train_users']==s['jobs'][0]['train_users']
 parent_matches=hashlib.sha256(parent.read_bytes()).hexdigest()==s['parent_checkpoint_sha256'];assert parent_matches or a.allow_retrained_parent,'For a newly retrained parent, explicitly pass --allow-retrained-parent; it is not the original selected endpoint'
 W=C/'videomaev2_base';sys.path[:0]=[str(W),str(C/'videomaev2_base_quant')];spec=importlib.util.spec_from_file_location('selected_control_training',W/'worker.py');w=importlib.util.module_from_spec(spec);sys.modules[spec.name]=w;spec.loader.exec_module(w)
 from control_data import WarmClips
 original=w.VideoMAEBase
 def model(*args,**kw):
  m=original(*args,checkpoint=pre,**kw);m.load_state_dict(raw['model'],strict=True);return m
 w.VideoMAEBase=model;w.EpochClips=WarmClips
 ip=a.generated_inputs.resolve()/'POSE_ADAPTER_INPUTS.json';inp=json.loads(ip.read_text(encoding='utf-8'));out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True);assert not(out/'resume.pt').exists(),'Fresh optimizer is required for a new CONTROL4 stage'
 assert len(inp['rows'])==2931 and {r['user']for r in inp['rows']}==set(raw['train_users'])
 assert all(r['clip_id']in inp['regions']and r['clip_id']in inp['conflict_weights']for r in inp['rows'])
 runtime=dict(s);runtime.update(data_root=str(Path(a.data_dir).resolve()),workers=a.workers,micro_clips=a.micro,eval_micro=4)
 if a.check_only:
  m=model(classes=40,fast=True,seed=s['seed']);assert sum(p.numel()for p in m.parameters())==s['trainable_parameters']
  ds=WarmClips(inp['rows'][:1],runtime['data_root'],s['seed'],inp['regions'],training=True,epoch=1,epoch_value=SimpleNamespace(value=1));sample=ds[0]
  assert list(sample['x'].shape)==[3,16,224,224]and torch.isfinite(sample['x']).all()
  (out/'CHECK.json').write_text(json.dumps(dict(status='PASS_PARENT_MODEL_AND_GENERATED_INPUT_SAMPLE_ZERO_UPDATES',scientific_protocol_sha256=ph,epochs=4,fresh_optimizer=True,parent_matches_selected=parent_matches,training_inputs_sha256=hashlib.sha256(ip.read_bytes()).hexdigest(),input_records=len(inp['rows']),sample_shape=list(sample['x'].shape),optimizer_updates=0),indent=2)+'\n',encoding='utf-8');return
 deadline=time.time()+a.max_hours*3600
 class Guard:
  def check(self,**kw):
   if time.time()>deadline or (out/'STOP_REQUESTED').exists():raise InterruptedError('STOP_OR_WALL_LIMIT')
   (out/'STATUS.json').write_text(json.dumps(dict(at_unix=time.time(),**kw)))
 w.fit(s['jobs'][0],runtime,ph,out,inp,Guard());cp=torch.load(out/'model_fp16.pt',map_location='cpu',weights_only=True)
 averaged={n:((v.float()+cp['model'][n].float())*.5).half()for n,v in raw['model'].items()}
 avg=dict(model=averaged,protocol_sha256=ph,pretrained_sha256=s['pretrained_sha256'],train_users=raw['train_users'],held_users=[],epochs=4,epoch_semantics='freshCONTROL4_branch_not_continuous_epoch20',raw_epochs=16,control_epochs=4,control_optimizer='freshAdamW',raw_protocol_sha256=raw['protocol_sha256'],control_protocol_sha256=ph,seed=s['seed'],alpha=.5,raw_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),control_sha256=hashlib.sha256((out/'model_fp16.pt').read_bytes()).hexdigest(),arithmetic='FP32 midpoint then FP16')
 from codec import pack
 torch.save(avg,out/'average_model_fp16.pt');torch.save(pack(avg),out/'average_model_int8.pt')
if __name__=='__main__':main()
