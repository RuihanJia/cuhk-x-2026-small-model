"""Sparse finite two-host train-to-readout transitions; download then one CPU audit."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time,traceback
P=Path(__file__).resolve().parent;C=P.parents[2];ROOT=C.parents[1]
sys.path[:0]=[str(C),str(C.parent/'noon_push_20260914/bundle')]
import post_aux_cloud_access as a
from dual_host_runtime.io import atomic,read
from dual_host_runtime.health import alive,identity

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(protocol,expected_protocol_sha):
 pp=Path(protocol).resolve();ph=sha(pp);assert ph==expected_protocol_sha;assert pp.parent==P
 s=read(pp);assert s['status']=='FROZEN_FOR_EXECUTION' and s['wall_cap_seconds']==1800
 out=P/'supervisor';out.mkdir(exist_ok=True);fd=os.open(str(out/'CLAIM'),os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
 deadline=s['deadline_unix']+180;remote_root=s['hosts']['cloud']['output_directory'];remote_P=remote_root.rsplit('/output/',1)[0];remote_pp=remote_P+'/protocol.json'
 def remote(code):
  assert time.time()<deadline,'SUPERVISOR_FINITE_WINDOW_ENDED'
  return a.run(code,30)
 def launch(host,phase):
  receipt=out/(host+'_'+phase+'_LAUNCH.json')
  if receipt.exists():return
  if host=='local':
   dest=Path(s['hosts'][host]['output_directory'])/phase;dest.mkdir(parents=True,exist_ok=True)
   assert not(dest/'CONTROLLER_CLAIM').exists()
   cmd=[sys.executable,'-X','utf8','-B','-u',str(P/'phase_controller.py'),'--protocol',str(pp),'--host',host,'--phase',phase]
   with(dest/'launch.log').open('ab') as log:z=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
   result=dict(**identity(z.pid),host=host,phase=phase,at_unix=time.time(),protocol_sha256=ph)
  else:
   code='PP='+repr(remote_pp)+'\nPH='+repr(ph)+'\nP='+repr(remote_P)+'\nOUT='+repr(remote_root)+'\nPHASE='+repr(phase)+'\nPY='+repr(a.PYTHON)+'\n'+"""from pathlib import Path
import hashlib,subprocess,json,time,sys
assert hashlib.sha256(Path(PP).read_bytes()).hexdigest()==PH
p=Path(P);d=Path(OUT)/PHASE;d.mkdir(parents=True,exist_ok=True);assert not(d/'CONTROLLER_CLAIM').exists()
with(d/'launch.log').open('ab') as log:z=subprocess.Popen([PY,'-X','utf8','-B','-u',str(p/'phase_controller.py'),'--protocol',PP,'--host','cloud','--phase',PHASE],cwd=p,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
sys.path.insert(0,str(p.parents[2].parent/'noon_push_20260914/bundle'));from dual_host_runtime.health import identity
print(json.dumps(dict(**identity(z.pid),host='cloud',phase=PHASE,at_unix=time.time(),protocol_sha256=PH)))
"""
   result=remote(code)
  atomic(receipt,result)
 def snapshot(host,phase):
  names=['CONTROLLER.json','CONTROLLER_DONE.json','CONTROLLER_ERROR.json','ERROR.json','DONE.json','STATUS.json','TELEMETRY.json']
  if host=='local':
   d=Path(s['hosts'][host]['output_directory'])/phase;records={n:read(d/n) for n in names if(d/n).exists()};dead=None
   if 'CONTROLLER_DONE.json' in records:
    c=records['CONTROLLER.json'];dead=not alive(c['owner']) and not alive(c['controller'])
   receipt=read(out/(host+'_'+phase+'_LAUNCH.json'))
   if 'CONTROLLER.json' not in records and not alive(receipt):records['ERROR.json']={'error':'PHASE_EARLY_EXIT_SEE_LAUNCH_LOG'}
   return dict(records=records,all_owned_dead=dead,at_unix=time.time())
  owner=read(out/(host+'_'+phase+'_LAUNCH.json'))
  code='OWNER='+repr(owner)+'\nD='+repr(remote_root+'/'+phase)+'\nNAMES='+repr(names)+'\nC='+repr(a.REMOTE+'/cloud_rethink_20260914')+'\n'+"""from pathlib import Path
import json,sys,time
p=Path(D);r={n:json.loads((p/n).read_text()) for n in NAMES if(p/n).exists()};dead=None
if 'CONTROLLER_DONE.json' in r:
 sys.path.insert(0,str(Path(C).parent/'noon_push_20260914/bundle'));from dual_host_runtime.health import alive
 c=r['CONTROLLER.json'];dead=not alive(c['owner']) and not alive(c['controller'])
if 'CONTROLLER.json' not in r:
 sys.path.insert(0,str(Path(C).parent/'noon_push_20260914/bundle'));from dual_host_runtime.health import alive
 if not alive(OWNER):r['ERROR.json']={'error':'PHASE_EARLY_EXIT_SEE_LAUNCH_LOG'}
print(json.dumps(dict(records=r,all_owned_dead=dead,at_unix=time.time())))
"""
  return remote(code)
 try:
  states={h:'train' for h in ['cloud','local']};errors={}
  launch('cloud','train');launch('local','train')
  while any(v!='done' for v in states.values()):
   assert time.time()<deadline,'SUPERVISOR_FINITE_WINDOW_ENDED'
   for host,phase in list(states.items()):
    if phase=='done':continue
    snap=snapshot(host,phase);atomic(out/(host+'_'+phase+'_SNAPSHOT.json'),snap);r=snap['records']
    if any(k in r for k in ['CONTROLLER_ERROR.json','ERROR.json']):
     errors[host]=dict(phase=phase,snapshot=snap);states[host]='done';continue
    if 'CONTROLLER_DONE.json' not in r or snap['all_owned_dead'] is not True:continue
    assert r['CONTROLLER_DONE.json']['exit_code']==0 and r['CONTROLLER_DONE.json']['protocol_sha256']==ph
    if phase=='train':launch(host,'evaluate');states[host]='evaluate'
    else:states[host]='done'
   atomic(out/'STATE.json',dict(protocol_sha256=ph,states=states,errors=errors,at_unix=time.time()))
   if any(v!='done' for v in states.values()):time.sleep(min(60,max(0,deadline-time.time())))
  code='ROOT='+repr(remote_root)+'\n'+"""from pathlib import Path
import hashlib,json
root=Path(ROOT);files={}
for p in sorted(root.rglob('*')):
 if not p.is_file() or p.suffix not in ['.json','.npz','.pt','.log']:continue
 assert p.resolve().is_relative_to(root.resolve()) and p.stat().st_size<=12000000
 files[p.relative_to(root).as_posix()]=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
assert sum(v['bytes'] for v in files.values())<=100000000
print(json.dumps(files))
"""
  files=remote(code);dest=out/'cloud_mirror';dest.mkdir(exist_ok=True);mod,client=a.connect()
  try:
   sf=client.open_sftp()
   for name,v in files.items():
    assert time.time()<deadline;p=(dest/name).resolve();assert p.is_relative_to(dest);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.partial');sf.get(remote_root+'/'+name,str(tmp));assert tmp.stat().st_size==v['bytes'] and sha(tmp)==v['sha256'];os.replace(tmp,p)
   sf.close()
  finally:client.close()
  atomic(out/'DOWNLOAD_MANIFEST.json',files)
  if errors:raise RuntimeError('HOST_PHASE_FAILURE_SEE_STATE')
  atomic(out/'ALL_FOUR_READOUTS_READY.json',dict(protocol_sha256=ph,host_states=states,all_owned_dead=True,at_unix=time.time(),metadata_only_before_this_barrier=True))
  audit=P/'independent_audit.py';assert sha(audit)==s['independent_audit_sha256']
  with(out/'audit.log').open('ab') as log:z=subprocess.run([sys.executable,'-X','utf8','-B',str(audit),'--protocol',str(pp),'--expected-protocol-sha',ph],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=180,creationflags=subprocess.CREATE_NO_WINDOW)
  assert z.returncode==0,'AUDIT_EXIT:'+str(z.returncode);ad=P/'audit';d=read(ad/'DONE.json');assert d['status']=='COMPLETE' and d['protocol_sha256']==ph
  atomic(out/'SUPERVISOR_DONE.json',dict(status='COMPLETE_DUAL_HOST_AND_AUDIT',protocol_sha256=ph,at_unix=time.time(),audit_result_sha256=sha(ad/'RESULT.json'),downloaded_bytes=sum(v['bytes'] for v in files.values()),all_owned_dead=True))
 except BaseException as e:
  atomic(out/'SUPERVISOR_ERROR.json',dict(protocol_sha256=ph,error=str(e),traceback=traceback.format_exc(),at_unix=time.time()));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--expected-protocol-sha',required=True);main(**vars(p.parse_args()))
