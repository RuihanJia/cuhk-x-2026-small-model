"""Portable filesystem adapter for unchanged full18 Thermal/IMU workers.
This creates a fresh training work directory and never changes selected weights.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,shutil,sys,time
HERE=Path(__file__).resolve().parent;T=HERE/'reproduction_tree/runs/generalization_training_20260910_v1'
def main():
 p=argparse.ArgumentParser();p.add_argument('--kind',choices=['thermal','imu'],required=True);p.add_argument('--data-dir',required=True);p.add_argument('--output',required=True);p.add_argument('--runtime-dir',required=True);p.add_argument('--pretrained');p.add_argument('--max-hours',type=float,default=6);p.add_argument('--check-only',action='store_true');p.add_argument('--generated-inputs',type=Path);a=p.parse_args()
 thermal=a.kind=='thermal';source=T/('cloud' if thermal else 'local');protocol=source/('thermal_full18_r3_protocol.json' if thermal else 'imu_full18_r3_protocol.json');s=json.loads(protocol.read_text(encoding='utf-8'));ph=hashlib.sha256(protocol.read_bytes()).hexdigest();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
 shutil.copytree(source/'bundle',out/'bundle',dirs_exist_ok=True)
 if a.generated_inputs:
  (out/'bundle/data').mkdir(parents=True,exist_ok=True)
  for name in (['thermal_index.json'] if thermal else ['arm_acceleration_t256.npz','device_acceleration_t256.npz']):shutil.copyfile(a.generated_inputs/name,out/'bundle/data'/name)
  if thermal:s['data']['index_sha256']=hashlib.sha256((out/'bundle/data/thermal_index.json').read_bytes()).hexdigest().upper()
  else:
   shutil.copyfile(a.generated_inputs/'DEVICE_CACHE_READY.json',out/'DEVICE_CACHE_READY.json')
   s['data']['arm_sha256']=hashlib.sha256((out/'bundle/data/arm_acceleration_t256.npz').read_bytes()).hexdigest().upper()
   s['data']['device_cache_sha256']=hashlib.sha256((out/'bundle/data/device_acceleration_t256.npz').read_bytes()).hexdigest().upper()
  s['regenerated_input_provenance']=dict(directory=str(a.generated_inputs.resolve()),original_scientific_protocol_sha256=ph,byte_identical_retraining_claim=False)
 elif not thermal:shutil.copyfile(source/'DEVICE_CACHE_READY.json',out/'DEVICE_CACHE_READY.json')
 runtime=Path(a.runtime_dir).resolve();sys.path.insert(0,str(out/'bundle'));sys.path.insert(0,str(runtime/'third_party'))
 if thermal:
  assert a.pretrained;pre=Path(a.pretrained).resolve();assert hashlib.sha256(pre.read_bytes()).hexdigest().upper()==s['weights']['sha256']
  s['weights']['path']=str(pre);s['data']['root']=str(Path(a.data_dir).resolve());s['dependencies']['root']=str(runtime/'third_party');s['scientific_parent_sha256']=ph
  for rel,h in s['dependencies']['source_sha256'].items():assert hashlib.sha256((runtime/'third_party'/rel).read_bytes()).hexdigest().upper()==h,rel
  rows=json.loads((out/'bundle/data/thermal_index.json').read_text(encoding='utf-8'));assert (Path(a.data_dir)/rows[0]['directory']/rows[0]['files'][0]).is_file()
 else:
  import numpy as np
  ready=json.loads((out/'DEVICE_CACHE_READY.json').read_text(encoding='utf-8'));assert hashlib.sha256((out/ready['path']).read_bytes()).hexdigest().upper()==ready['sha256']
  assert hashlib.sha256((out/'bundle/data/arm_acceleration_t256.npz').read_bytes()).hexdigest().upper()==s['data']['arm_sha256']
 config=out/'execution_protocol.json';config.write_text(json.dumps(s,indent=2));worker='generalization_thermal_full18_r3.py' if thermal else 'generalization_imu_full18.py'
 record=dict(kind=a.kind,scientific_protocol_sha256=ph,worker=worker,paths_relocated=True,full_retraining_bitwise_claim=False,check_only=a.check_only)
 if a.check_only:
  (out/'CHECK.json').write_text(json.dumps(dict(status='PASS_REQUIRED_INPUTS_ZERO_UPDATES',**record),indent=2));print(json.dumps(record));return
 from dual_host_runtime import health
 deadline=time.time()+a.max_hours*3600
 class PortableGuard:
  def __init__(self,*args):pass
  def check(self,**kw):
   if time.time()>deadline or (out/'STOP_REQUESTED').exists():raise InterruptedError('STOP_OR_WALL_LIMIT')
   (out/'STATUS.json').write_text(json.dumps(dict(at_unix=time.time(),**kw)))
 health.Guard=PortableGuard
 spec=importlib.util.spec_from_file_location('selected_aux_worker',out/'bundle'/worker);w=importlib.util.module_from_spec(spec);sys.modules[spec.name]=w;spec.loader.exec_module(w)
 sys.argv=[worker,'--root',str(out),'--job','T_ALL18_R3' if thermal else 'IMU_D_ALL18_R3','--protocol',str(config),'--deadline',str(deadline)]
 (out/'EXECUTION.json').write_text(json.dumps(record,indent=2));w.main()
if __name__=='__main__':main()
