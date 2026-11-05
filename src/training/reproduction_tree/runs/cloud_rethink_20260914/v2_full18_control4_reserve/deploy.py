from pathlib import Path
import argparse,sys,json,hashlib,subprocess,time,os
E=Path(__file__).resolve().parent;C=E.parent;sys.path.insert(0,str(C))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(host):
 assert not(E/f'{host.upper()}_DISPATCH.json').exists();s=json.loads((E/f'protocol.{host}.candidate.json').read_text());assert s['status']=='CPU_PREPARED_AWAITING_ROOT_LAUNCH_DECISION'
 if host=='local':
  run=subprocess.run([sys.executable,'-B',str(E/'freeze.py'),'--host',host],capture_output=True,text=True);assert run.returncode==0,(run.stdout,run.stderr)
  import psutil
  pp=E/'protocol.local.json';log=(E/'local_controller.log').open('ab');p=subprocess.Popen([sys.executable,'-X','utf8','-B','-u',str(E/'controller.py'),'--protocol',str(pp),'--host',host],cwd=C,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
  receipt=dict(status='QUEUED',pid=p.pid,create_time=psutil.Process(p.pid).create_time(),protocol_sha256=sha(pp),output_directory=s['hosts'][host]['output_directory'],at_unix=time.time())
 else:
  import post_aux_cloud_access as a
  manifest=json.loads((E/'TRANSFER_MANIFEST.json').read_text());assert manifest['candidate_sha256']==sha(E/'protocol.cloud.candidate.json');remote=a.REMOTE+'/cloud_rethink_20260914/'+E.name
  mod,client=a.connect()
  try:
   code='from pathlib import Path\nimport json\np=Path('+repr(remote)+');p.mkdir(parents=True,exist_ok=True);(p/"parent").mkdir(exist_ok=True);assert not(p/"protocol.cloud.json").exists();print(json.dumps({"status":"NEW_STAGE_READY"}))'
   a.old.execute(mod,client,code,30);ftp=client.open_sftp()
   for f in manifest['files']:
    assert sha(f['local'])==f['sha256'];ftp.put(f['local'],f['remote']+'.uploading');ftp.rename(f['remote']+'.uploading',f['remote'])
   code='from pathlib import Path\nimport hashlib,json,subprocess\nfiles='+repr(manifest['files'])+'\nfor f in files:\n p=Path(f["remote"]);assert p.stat().st_size==f["bytes"]and hashlib.sha256(p.read_bytes()).hexdigest()==f["sha256"]\np=Path('+repr(remote)+')\nr=subprocess.run(['+repr(a.PYTHON)+',"-B",str(p/"freeze.py"),"--host","cloud"],cwd=p.parent,capture_output=True,text=True,timeout=60)\nassert r.returncode==0,(r.stdout,r.stderr)\nprint(r.stdout)'
   frozen=a.old.execute(mod,client,code,70);ftp.get(remote+'/protocol.cloud.json',str(E/'protocol.cloud.json'));assert sha(E/'protocol.cloud.json')==frozen['protocol_sha256']
   code='from pathlib import Path\nimport json,subprocess,psutil,hashlib,time\np=Path('+repr(remote)+');pp=p/"protocol.cloud.json";s=json.loads(pp.read_text());out=Path(s["hosts"]["cloud"]["output_directory"]);assert not(out/"CLAIM").exists();log=(p/"cloud_controller.log").open("ab");c=subprocess.Popen(['+repr(a.PYTHON)+',"-B","-u",str(p/"controller.py"),"--protocol",str(pp),"--host","cloud"],cwd=p.parent,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(status="QUEUED",pid=c.pid,create_time=psutil.Process(c.pid).create_time(),protocol_sha256=hashlib.sha256(pp.read_bytes()).hexdigest(),output_directory=str(out),at_unix=time.time())))'
   receipt=a.old.execute(mod,client,code,30)
  finally:client.close()
 (E/f'{host.upper()}_DISPATCH.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--host',required=True,choices=['local','cloud']);main(**vars(p.parse_args()))
