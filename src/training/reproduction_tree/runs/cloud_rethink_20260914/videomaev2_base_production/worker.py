"""One full18 fit of the already validated K400 raw16 recipe; no held/Test evaluation."""
from pathlib import Path
import argparse,hashlib,importlib.util,os,sys,time,traceback
E=Path(__file__).resolve().parent;C=E.parent;K=C/'videomaev2_base';O=C.parent/'error_resolution_20260913'
sys.path[:0]=[str(K),str(K.parent/'videomaev2_base_quant'),str(O/'bundle')]
sp=importlib.util.spec_from_file_location('k400_training_worker',K/'worker.py');k=importlib.util.module_from_spec(sp);sys.modules[sp.name]=k;sp.loader.exec_module(k)
import torch
from codec import pack,unpack,FORMAT
from dual_host_runtime.io import atomic,read,save_torch
from dual_host_runtime.health import Guard,identity,owned_tree
sha=k.sha

def main(protocol,host):
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
 assert job['train_users']==sorted({r['user'] for r in inp['rows']}) and len(job['train_users'])==18 and job['held_users']==[] and s['epochs']==16
 runtime=dict(s);runtime.update(h);k.fit(job,runtime,ph,out,inp,guard)
 fr=read(out/'FROZEN.json');cp=torch.load(k.checked(out/'model_fp16.pt',fr['checkpoint_sha256']),map_location='cpu',weights_only=True);assert cp['held_users']==[] and cp['train_users']==job['train_users']
 q=pack(cp);save_torch(out/'model_int8.pt',q);loaded=torch.load(out/'model_int8.pt',map_location='cpu',weights_only=True);r1=unpack(q);r2=unpack(loaded);assert all(torch.equal(r1[n],r2[n]) for n in r1)
 assert len(q['quantized_names'])==50 and (out/'model_int8.pt').stat().st_size<87000000
 atomic(out/'QUANTIZED.json',dict(protocol_sha256=ph,parent_checkpoint_sha256=fr['checkpoint_sha256'],checkpoint_sha256=sha(out/'model_int8.pt'),checkpoint_bytes=(out/'model_int8.pt').stat().st_size,storage_format=FORMAT,quantized_matrices=50,reload_exact=True,no_Test=True,at_unix=time.time()))
 names=['TRAIN_DONE.json','model_fp16.pt','model_int8.pt','QUANTIZED.json','INITIAL.json','SPLIT.json','HISTORY.json','FIRST_UPDATE.json','FROZEN.json'];guard.check(force=True,phase='FULL18_COMPLETE')
 atomic(out/'DONE.json',dict(status='COMPLETE_V2_BASE_FULL18_RAW16',protocol_sha256=ph,job=job['id'],artifacts={n:sha(out/n) for n in names},no_Test=True,held_evaluation_performed=False,at_unix=time.time()))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--host',required=True);args=p.parse_args()
 try:main(**vars(args))
 except BaseException as e:
  s=read(args.protocol);atomic(Path(s['hosts'][args.host]['output_directory'])/'ERROR.json',dict(error=str(e),traceback=traceback.format_exc(),protocol_sha256=sha(args.protocol),at_unix=time.time()));raise
