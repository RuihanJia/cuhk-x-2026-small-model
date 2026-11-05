"""Six fixed source-inner fits across both hosts; same full16 recipe."""
from pathlib import Path
import argparse,hashlib,importlib.util,inspect,json,os,sys,time,traceback
E=Path(__file__).resolve().parent;C=E.parent;W=C/'videomaev2_base';O=C.parent/'error_resolution_20260913'
sys.path[:0]=[str(W),str(O),str(O/'bundle')]
sp=importlib.util.spec_from_file_location('videomae_training_runtime',W/'worker.py');base=importlib.util.module_from_spec(sp);sys.modules[sp.name]=base;sp.loader.exec_module(base)
from dual_host_runtime.io import read,atomic
from dual_host_runtime.health import Guard,identity,owned_tree
import torch

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(p,h):assert sha(p)==h,str(p);return Path(p)

def main(protocol,host):
 assert host in ('cloud','local') and (os.name=='nt')==(host=='local');s=read(protocol);ph=sha(protocol);assert s['status']=='FROZEN_FOR_EXECUTION';h=s['hosts'][host];out=Path(h['output_directory']);assert out.resolve().is_relative_to(E)
 for name,value in s['code_sha256'].items():check(C/name,value)
 for name,value in h['input_bindings'].items():check(name,value)
 source_protocol=read(check(W/'protocol.json',s['video_protocol_sha256']));original=read(check(h['original_protocol'],s['original_protocol_sha256']));inputs=read(check(h['pose_inputs'],s['pose_inputs_sha256']))
 assert len(inputs['rows'])==2931
 bd=read(out/'BUDGET.json');ct=read(out/'CONTROLLER.json');actual=identity(os.getpid());assert bd['protocol_sha256']==ct['protocol_sha256']==ph and any(v['pid']==actual['pid'] and abs(v['create_time']-actual['create_time'])<.01 for v in owned_tree(ct['worker']))
 guard=Guard(out,bd['deadline_unix'],out/'TELEMETRY.json');guard.check(force=True,phase='VIDEOMAE_SOURCE_PAIR_PREFLIGHT');torch.set_num_threads(2 if host=='local' else 1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert ('4070' if host=='local' else '5090') in torch.cuda.get_device_name(0)
 # Reuse the exact fit algorithm; only the source-subset cardinality assertion changes.
 fit=base.fit
 runtime=dict(source_protocol);runtime.update(epochs=16,micro_clips=h['micro_clips'],workers=h['workers'],data_root=h['data_root'],eval_micro=4)
 metadata=[]
 for job in [j for j in s['jobs'] if j['host']==host]:
  par=next(j for j in original['jobs'] if j['id']==job['split_parent'])
  for key in ['train_users','held_users','outer_excluded_users','fold','inner']:assert par[key]==job[key]
  assert not set(job['train_users'])&(set(job['held_users'])|set(job['outer_excluded_users'])) and not set(job['held_users'])&set(job['outer_excluded_users'])
  subset=dict(inputs);subset['rows']=[r for r in inputs['rows'] if r['user'] not in job['outer_excluded_users']]
  assert {r['user'] for r in subset['rows']}==set(job['train_users'])|set(job['held_users'])
  path=out/job['id'];path.mkdir(exist_ok=True);bp=path/'JOB_BUDGET.json'
  if not bp.exists():atomic(bp,dict(protocol_sha256=ph,first_started_unix=time.time(),deadline_unix=min(bd['deadline_unix'],time.time()+h['job_wall_cap_seconds'])))
  jb=read(bp);assert jb['protocol_sha256']==ph;fitguard=Guard(path,jb['deadline_unix'],out/'TELEMETRY.json')
  if not(path/'TRAIN_DONE.json').exists():fit(job,runtime,ph,path,subset,fitguard)
  else:
   done=read(path/'TRAIN_DONE.json');assert done['protocol_sha256']==ph
   for name,hsh in done['artifacts'].items():check(path/name,hsh)
  metadata.append(dict(job=job['id'],fold=job['fold'],inner=job['inner'],outer_excluded_users=job['outer_excluded_users'],checkpoint_sha256=sha(path/'model_fp16.pt'),train_done_sha256=sha(path/'TRAIN_DONE.json')))
 atomic(out/'HOST_SOURCE_FITS_FROZEN.json',dict(protocol_sha256=ph,jobs=metadata,no_source_held_evaluation_started=True,at_unix=time.time()))
 for job in [j for j in s['jobs'] if j['host']==host]:
  subset=dict(inputs);subset['rows']=[r for r in inputs['rows'] if r['user'] not in job['outer_excluded_users']];base.evaluate(job,runtime,ph,out/job['id'],subset,guard)
 files={'HOST_SOURCE_FITS_FROZEN.json':sha(out/'HOST_SOURCE_FITS_FROZEN.json')}
 for job in [j for j in s['jobs'] if j['host']==host]:
  for name in ['TRAIN_DONE.json','model_fp16.pt','INITIAL.json','SPLIT.json','HISTORY.json','FIRST_UPDATE.json','FROZEN.json','predictions.npz','HELD_DONE.json']:files[job['id']+'/'+name]=sha(out/job['id']/name)
 atomic(out/'DONE.json',dict(status='COMPLETE_VIDEOMAE_SOURCE_REST',protocol_sha256=ph,artifacts=files,no_outer_or_Test=True,at_unix=time.time()))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--host',required=True);a=p.parse_args()
 try:main(a.protocol,a.host)
 except BaseException as exc:
  s=read(a.protocol);atomic(Path(s['hosts'][a.host]['output_directory'])/'ERROR.json',dict(protocol_sha256=sha(a.protocol),error=str(exc),traceback=traceback.format_exc(),at_unix=time.time()));raise
