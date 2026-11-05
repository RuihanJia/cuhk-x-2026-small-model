"""Portable adapter around unchanged historical Base/Small fit functions.
No final submission is overwritten. Full re-training is not claimed bitwise deterministic.
"""
from pathlib import Path
import argparse,functools,hashlib,importlib.util,json,sys,time
HERE=Path(__file__).resolve().parent
if sys.platform=='win32':HERE=Path('\\\\?\\'+str(HERE))
TREE=HERE/'reproduction_tree';C=TREE/'runs/cloud_rethink_20260914'
def main():
 p=argparse.ArgumentParser();p.add_argument('--kind',choices=['base','small'],required=True);p.add_argument('--data-dir',required=True);p.add_argument('--pretrained',required=True);p.add_argument('--output',required=True);p.add_argument('--protocol');p.add_argument('--job');p.add_argument('--workers',type=int,default=4);p.add_argument('--micro',type=int,default=4);p.add_argument('--max-hours',type=float,default=6);p.add_argument('--check-only',action='store_true');p.add_argument('--generated-inputs',type=Path);a=p.parse_args()
 import torch
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 W=C/('videomaev2_'+a.kind);default=C/('videomaev2_base_production' if a.kind=='base' else 'videomaev2_production')/'protocol.json'
 pp=Path(a.protocol) if a.protocol else default;s=json.loads(pp.read_text(encoding='utf-8-sig'));ph=hashlib.sha256(pp.read_bytes()).hexdigest()
 pretrained=Path(a.pretrained).resolve();assert hashlib.sha256(pretrained.read_bytes()).hexdigest()==s['pretrained_sha256']
 jobs=s['jobs'];jobs=list(jobs.values()) if isinstance(jobs,dict) else jobs
 job=next((x for x in jobs if x.get('id')==a.job),None) if a.job else jobs[0]
 assert job is not None;assert not set(job['train_users'])&set(job['held_users'])
 ip=(a.generated_inputs/'POSE_ADAPTER_INPUTS.json') if a.generated_inputs else TREE/'runs/error_resolution_20260913/POSE_ADAPTER_INPUTS.json';inp=json.loads(ip.read_text(encoding='utf-8'));excluded=job.get('outer_excluded_users',[]);inp['rows']=[r for r in inp['rows'] if r['user'] not in excluded]
 assert set(r['user'] for r in inp['rows'])==set(job['train_users'])|set(job['held_users'])
 sys.path.insert(0,str(W));spec=importlib.util.spec_from_file_location('selected_video_training',W/'worker.py');w=importlib.util.module_from_spec(spec);sys.modules[spec.name]=w;spec.loader.exec_module(w)
 cname='VideoMAEBase' if a.kind=='base' else 'VideoMAESmall';original=getattr(w,cname);setattr(w,cname,functools.partial(original,checkpoint=pretrained))
 runtime=dict(s);runtime.update(data_root=str(Path(a.data_dir).resolve()),workers=a.workers,micro_clips=a.micro,eval_micro=4)
 assert a.workers>=1 and runtime['effective_batch']%a.micro==0
 out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
 record=dict(kind=a.kind,scientific_protocol_sha256=ph,job=job,pretrained_sha256=s['pretrained_sha256'],training_inputs_sha256=hashlib.sha256(ip.read_bytes()).hexdigest(),data_root=runtime['data_root'],micro=a.micro,workers=a.workers,check_only=a.check_only,full_retraining_bitwise_claim=False)
 if a.check_only:
  model=getattr(w,cname)(classes=40,fast=True,seed=s['seed']);assert sum(v.numel() for v in model.parameters())==s['trainable_parameters']
  ds=w.PhotometricClips([next(r for r in inp['rows'] if r['user'] in job['train_users'])],runtime['data_root'],s['seed'],inp['regions'],training=True,epoch=1,arm='RAW_CONTROL');sample=ds[0]
  record.update(status='PASS_MODEL_AND_ONE_TRAINING_SAMPLE_ZERO_UPDATES',shape=list(sample['x'].shape),parameters=sum(v.numel() for v in model.parameters()))
  (out/'CHECK.json').write_text(json.dumps(record,indent=2));print(json.dumps(record));return
 assert torch.cuda.is_available();deadline=time.time()+a.max_hours*3600
 class Guard:
  def check(self,**kw):
   if time.time()>deadline or (out/'STOP_REQUESTED').exists():raise InterruptedError('STOP_OR_WALL_LIMIT')
   (out/'STATUS.json').write_text(json.dumps(dict(at_unix=time.time(),**kw)))
 (out/'EXECUTION.json').write_text(json.dumps(record,indent=2));w.fit(job,runtime,ph,out,inp,Guard())
if __name__=='__main__':main()
