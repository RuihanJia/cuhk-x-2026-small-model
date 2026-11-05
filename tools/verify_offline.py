"""Reproduce both selected predictions in a fresh execution directory on Linux.

The launcher creates a network namespace with unshare, records the launch and
namespace state, and runs unchanged inference sources with the frozen models.
"""
from pathlib import Path
import argparse,hashlib,json,os,platform,shutil,socket,subprocess,sys,tempfile,time

SOLUTIONS=[('dual_mirror','56258094','0eda9483701b92a25ccc8cce284acf4b12cf2af4df974ede4f425461091210fc'),
           ('weight_average','56258110','8eff0fd318b966dfa420e92040a7351715f504a46c5cf0c5cec40b55b65f1c2d')]
def digest(path):
 with Path(path).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.partial')
 tmp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');os.replace(tmp,path)
def inventory(root):
 return {p.relative_to(root).as_posix():digest(p)for p in sorted(root.rglob('*'))if p.is_file()and'__pycache__'not in p.parts and'.git'not in p.parts and p.suffix!='.pyc'}
def namespace():
 return dict(pid=os.getpid(),network_namespace=os.readlink('/proc/self/ns/net'),
             interfaces=[name for _,name in socket.if_nameindex()],
             ipv4_routes=Path('/proc/net/route').read_text(),ipv6_routes=Path('/proc/net/ipv6_route').read_text())
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--repo',type=Path,required=True);p.add_argument('--models',type=Path,required=True)
 p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--max-minutes',type=float,default=30);p.add_argument('--check-only',action='store_true')
 p.add_argument('--inside',action='store_true',help=argparse.SUPPRESS)
 a=p.parse_args();repo=a.repo.resolve();models=a.models.resolve();data=a.data.resolve();out=a.output.resolve()
 assert sys.platform=='linux','Run this verifier on Linux or WSL2.'
 assert(repo/'src/predict.py').is_file()and(data/'Testing').is_dir()
 out.mkdir(parents=True,exist_ok=True)
 if not a.inside:
  assert not(out/'LAUNCH.json').exists(),'Use a new output directory for each verification.'
  outside=namespace();args=sys.argv[1:]
  command=['unshare','--net',sys.executable,'-B',str(Path(__file__).resolve()),*args,'--inside']
  save(out/'LAUNCH.json',dict(command=command,cwd=str(Path.cwd()),outside_namespace=outside,verifier_sha256=digest(__file__),started_unix=time.time(),check_only=a.check_only))
  subprocess.run(command,check=True);return
 launch=json.loads((out/'LAUNCH.json').read_text());inside=namespace()
 assert inside['network_namespace']!=launch['outside_namespace']['network_namespace'],'NETWORK_NAMESPACE_UNCHANGED'
 assert set(inside['interfaces'])=={'lo'},inside
 assert len(inside['ipv4_routes'].strip().splitlines())<=1,'EXTERNAL_IPV4_ROUTE_PRESENT'
 save(out/'NETWORK_NAMESPACE.json',dict(outside=launch['outside_namespace'],inside=inside,checks='Different namespace; loopback interface only; no external IPv4 route.'))
 import torch
 save(out/'ENVIRONMENT.json',dict(python=sys.version,executable=sys.executable,platform=platform.platform(),torch=torch.__version__,cuda_runtime=torch.version.cuda,gpu=torch.cuda.get_device_name(0)if torch.cuda.is_available()else None))
 packages=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True)
 (out/'pip-freeze.txt').write_text(packages.stdout,encoding='utf-8')
 sources=inventory(repo);save(out/'SOURCE_INPUT_MANIFEST.json',sources)
 if a.check_only:
  save(out/'RESULTS.json',dict(status='PASS_LAUNCH_AND_NAMESPACE_CHECK',inference_executed=False,source_files=len(sources)));return
 sys.path.insert(0,str(repo/'src/training/reproduction_tree/runs/generalization_training_20260910_v1/local/bundle'))
 from dual_host_runtime.health import NVML,Pause,identity,stop_owned
 probe=NVML();results=[];started=time.time()
 with tempfile.TemporaryDirectory(prefix='cuhkx_verify_')as temporary:
  work=Path(temporary)/'repository'
  shutil.copytree(repo,work,copy_function=shutil.copyfile,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git','model.pth','*.npz','*.csv'))
  executed=inventory(work);assert executed==sources,'EXECUTION_SOURCE_COPY_MISMATCH'
  save(out/'EXECUTED_SOURCE_MANIFEST.json',executed)
  assert not list(work.rglob('*.npz'))and not list(work.rglob('*.csv'))
  for kind,submission,expected in SOLUTIONS:
   destination=work/f'artifacts/{kind}/model.pth';shutil.copyfile(models/(kind+'-model.pth'),destination)
   expected_model=json.loads((work/f'artifacts/{kind}/CONTAINER.json').read_text())
   assert digest(destination)==expected_model['sha256']and destination.stat().st_size==expected_model['bytes']
  save(out/'EXECUTION_SETUP.json',dict(empty_prediction_and_feature_cache=True,source_copy_matches=True,execution_directory=str(work),models={kind:digest(work/f'artifacts/{kind}/model.pth')for kind,_,_ in SOLUTIONS}))
  for kind,submission,expected in SOLUTIONS:
   output=out/(kind+'.csv');begin=time.time()
   command=[sys.executable,'-B',str(work/'src/predict.py'),'--data-dir',str(data),'--output-csv',str(output),'--submission-id',submission,'--config',str(work/'configs/final.yaml')]
   with(out/(kind+'.log')).open('w',encoding='utf-8')as log:
    child=subprocess.Popen(command,cwd=work,stdout=log,stderr=subprocess.STDOUT)
    owner=identity(child.pid);pause=Pause(owner)
    child_ns=os.readlink(f'/proc/{child.pid}/ns/net');assert child_ns==inside['network_namespace']
    save(out/(kind+'-launch.json'),dict(command=command,cwd=str(work),pid=child.pid,network_namespace=child_ns,started_unix=begin))
    peak=0
    try:
     with(out/(kind+'-telemetry.jsonl')).open('w',encoding='utf-8')as telemetry:
      while child.poll()is None:
       if time.time()-started>a.max_minutes*60 or(out/'STOP_REQUESTED').exists():
        stop_owned(owner);raise InterruptedError('VERIFICATION_BUDGET_OR_STOP')
       sample=probe.sample();sample['control']=pause.tick(sample,local=True);peak=max(peak,sample['temperature_c'])
       telemetry.write(json.dumps(sample)+'\n');telemetry.flush();time.sleep(2)
    finally:pause.close()
   assert child.returncode==0,f'INFERENCE_EXIT:{kind}:{child.returncode}'
   assert digest(output)==expected,f'PREDICTION_MISMATCH:{kind}'
   result=dict(solution=kind,status='PASS_BYTE_IDENTICAL',rows=405,sha256=expected,elapsed_seconds=time.time()-begin,max_temperature_c=peak,network_namespace=child_ns)
   results.append(result);save(out/'RESULTS.json',dict(status='PASS'if len(results)==2 else'RUNNING',inference_executed=True,results=results));print(json.dumps(result),flush=True)
  final_sources={k:v for k,v in inventory(work).items()if not k.endswith('/model.pth')}
  assert final_sources==executed,'EXECUTION_SOURCE_CHANGED'
  save(out/'SOURCE_FINAL_CHECK.json',dict(source_unchanged=True,files=len(executed),input_manifest_sha256=digest(out/'SOURCE_INPUT_MANIFEST.json'),executed_manifest_sha256=digest(out/'EXECUTED_SOURCE_MANIFEST.json')))
if __name__=='__main__':main()
