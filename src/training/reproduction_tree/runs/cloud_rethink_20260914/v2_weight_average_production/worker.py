from pathlib import Path
import os,sys,json,hashlib,time
E=Path(__file__).resolve().parent;C=E.parent;R=C.parents[1];D=Path(json.loads((E/'protocol.json').read_text(encoding='utf-8'))['package'])
sys.path[:0]=[str(E),str(D),str(C.parent/'error_resolution_20260913/bundle')]
import resilient_io;resilient_io.install()
from dual_host_runtime.io import atomic,read
from dual_host_runtime.health import Guard,identity,owned_tree
import infer
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(protocol,host):
 s=read(protocol);ph=sha(protocol);assert host=='local' and os.name=='nt';out=Path(s['hosts'][host]['output_directory']);b=read(out/'BUDGET.json');ct=read(out/'CONTROLLER.json');me=identity(os.getpid());assert b['protocol_sha256']==ct['protocol_sha256']==ph and any(v['pid']==me['pid'] and abs(v['create_time']-me['create_time'])<.01 for v in owned_tree(ct['worker']))
 for p,h in s['bindings'].items():assert sha(R/p)==h,p
 m=read(D/'inference_manifest.json')
 for p,v in m['files'].items():assert sha(D/p)==v['sha256'] and (D/p).stat().st_size==v['bytes']
 guard=Guard(out,b['deadline_unix'],out/'TELEMETRY.json');guard.check(force=True,phase='V2_INFERENCE_PREFLIGHT');a=infer.infer(s['data_root'],str(D/'submission.csv'),guard=guard,workers=4,micro=4,sensor_cache=Path(s['cache_path']),strict_original_template=True)
 for p,v in m['files'].items():assert sha(D/p)==v['sha256'],p
 atomic(out/'RESULT.json',dict(**a,protocol_sha256=ph,logits_npz_sha256=sha(D/'submission.csv.logits.npz'),all_frozen_package_files_unchanged=True))
 atomic(out/'DONE.json',dict(status='COMPLETE_WEIGHT_AVERAGE_INFERENCE',protocol_sha256=ph,artifacts={'RESULT.json':sha(out/'RESULT.json')},at_unix=time.time()))
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--host',required=True);a=p.parse_args();main(**vars(a))
