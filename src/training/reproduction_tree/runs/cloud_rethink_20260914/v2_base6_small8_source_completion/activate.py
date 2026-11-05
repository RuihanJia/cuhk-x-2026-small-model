"""Freeze once and queue the finite completion behind actual F3 success."""
from pathlib import Path
import os,json,hashlib,time,subprocess,sys,ast,importlib.util
os.environ['CUDA_VISIBLE_DEVICES']=''
E=Path(__file__).resolve().parent;C=E.parent;D=C/'v2_base6_small_design';R=C.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def main():
 assert not(E/'protocol.json').exists();pilot=read(D/'protocol.json');old=read(C/'v2_temporal_halves_fusion/protocol.json');inp=read(E/'INPUTS.json');bindings=dict(pilot['bindings'])
 import numpy as np,torch
 torch.set_num_threads(1)
 sp=importlib.util.spec_from_file_location('preflight_four_way',E/'classwise4.py');cw=importlib.util.module_from_spec(sp);sp.loader.exec_module(cw)
 gen=np.random.default_rng(20260915);z=gen.normal(size=(17,4,40));mask=gen.random((17,4))>.4;mask[:,0]=True
 g=dict(description=dict(mode='convex'),temperature=[.6,1.,2.,3.],state=dict(bias=[.1,-.2,.3,-.4]));m=cw.Classwise(g)
 with torch.no_grad():m.delta.copy_(torch.tensor(gen.normal(size=(4,40)),dtype=torch.float64));expected=m(torch.tensor(z),torch.tensor(mask)).numpy()
 cal=dict(temperature=g['temperature'],global_bias=g['state']['bias'],delta=m.delta.detach().tolist());actual=cw.predict(z,mask,cal);err=float(abs(expected-actual).max());assert err<1e-6 and np.array_equal(actual.argmax(1),expected.argmax(1))
 qa=dict(status='PASS_FOUR_BRANCH_CPU_MATH_PREFLIGHT',adaptation='Only branch dimension3->4 and descriptive parameter count120->160; same penalty0.1, optimizer500 and original math',max_abs_numpy_torch_error=err,source_inventory_sha256=sha(E/'INPUTS.json'),all_containers_within_100MB_with_TIpose_and_12000cal_reserve=True,no_real_label_fit=True,no_GPU=True,no_Test=True,at_unix=time.time())
 (E/'CPU_PREFLIGHT.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
 for p in [D/'protocol.json',E/'INPUTS.json',E/'CPU_PREPARATION_REGISTRATION_V2.json',E/'CPU_PREFLIGHT.json',C/'frozen_fusion_math.py',C/'trajectory_fixed_fusion_protocol.json',Path(old['source_oof']),C/'v2_temporal_halves_fusion/output/PAIRED.npz',R/'CORE_EVIDENCE_LOCK.json',R/'CORE_RULES_EVIDENCE.md']:
  bindings[str(p)]=sha(p)
 for j in inp['jobs']:
  if not j['reuse_pilot']:bindings[j['container']['path']]=j['container']['sha256']
  for parent in j['parents'].values():
   for v in parent.values():bindings[v['path']]=v['sha256']
 for p in E.glob('*.py'):
  ast.parse(p.read_text(encoding='utf-8'));bindings[str(p)]=sha(p)
 for p,h in bindings.items():assert sha(p)==h,p
 now=time.time();queue_deadline=now+3600;out=E/'output/local';auth=C/'AUTHORIZATION_DUAL_RESOURCE_20260914.json'
 s=dict(status='FROZEN_FOR_EXECUTION',id='K710_BASE6_SMALL8_REMAINING7_SOURCE4WAY_FIXED_COMPLETION',registered_at_unix=now,queue_deadline_unix=queue_deadline,deadline_unix=queue_deadline+5400,wall_cap_seconds=5400,source_authorization_sha256=sha(auth),hypothesis='A complementary K710Small49 model can coexist with K710Base in100MB by fixedBase6andSmallpatch8 storage. Complete allsix user-isolated source readouts andtwo outer, reusingF3. Fit only sourceT/I/Base6/Small8 global/classwise0.1 once; no precision/weight scan. Allfour calibrators freeze before newouter combinations.',seed=20260915,optimizer_updates=0,encoder_epochs=16,new_readout_jobs=7,reused_pilot_jobs=1,no_Test=True,no_CSV=True,no_Public_fit=True,branches=['T','I','Base6','Small8'],readout=dict(Base='0.5mean(original5)+0.25earlyhalf+0.25latehalf',Small='mean(original5)',dtype='FP32',TF32=False,loader_workers=2,eval_micro=3),storage_format=pilot['storage_format'],inputs_file=str(E/'INPUTS.json'),pilot_directory=str(D/'output/local'),pilot_protocol_sha256=sha(D/'protocol.json'),source_oof=old['source_oof'],source_oof_sha256=sha(old['source_oof']),fixed_fusion_protocol=str(C/'trajectory_fixed_fusion_protocol.json'),reference=str(C/'v2_temporal_halves_fusion/output/PAIRED.npz'),reference_sha256=sha(C/'v2_temporal_halves_fusion/output/PAIRED.npz'),outer_users={'0':[2,16,20],'3':[5,7,8,21]},models=pilot['models'],pose_inputs=pilot['pose_inputs'],data_root=pilot['data_root'],bindings=bindings,worker_sha256=sha(E/'worker.py'),controller_sha256=sha(E/'controller.py'),all_four_calibrators_before_new_outer_combinations=True,development_limitation='F3pilot rawprecision is already exposed. This is adaptive existing development, not new independent validation.',resource_note='Local only.60minbounded queue,90minactivated completion includes sevenactualpaired readouts, calibration andCPU audit. Existing successful F3never rerun. Source six first, otherouter last. No newtraining.',hosts={'local':dict(output_directory=str(out),wall_cap_seconds=5400,authorization_file=str(auth),predecessor=dict(directory=str(D/'output/local'),protocol_sha256=sha(D/'protocol.json'),success_status='COMPLETE_BASE6_SMALL8_PILOT',success_only=True))})
 (E/'protocol.json').write_text(json.dumps(s,indent=2),encoding='utf-8');ph=sha(E/'protocol.json')
 # Only controller/worker may use GPU; parent CPU preflight hides CUDA locally.
 env=dict(os.environ);env.pop('CUDA_VISIBLE_DEVICES',None)
 log=(E/'controller.log').open('ab');cmd=[sys.executable,'-X','utf8','-B','-u',str(E/'controller.py'),'--protocol',str(E/'protocol.json'),'--host','local'];proc=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
 import psutil
 launch=dict(status='QUEUED_HIDDEN',protocol_sha256=ph,launcher_pid=proc.pid,launcher_create_time=psutil.Process(proc.pid).create_time(),predecessor=s['hosts']['local']['predecessor'],queue_deadline_unix=queue_deadline,execution_cap_seconds=5400,command=cmd,at_unix=time.time());(E/'LAUNCH.json').write_text(json.dumps(launch,indent=2),encoding='utf-8');print(json.dumps(launch))
if __name__=='__main__':main()
