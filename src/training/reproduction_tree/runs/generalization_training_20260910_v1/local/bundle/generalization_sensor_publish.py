"""Publish only the eight completed local matched-sensor artifacts to the existing cloud fusion queue."""
from pathlib import Path
import argparse,json,sys,time,traceback

def main():
    a=argparse.ArgumentParser();a.add_argument('--root',required=True);a.add_argument('--job',required=True);a.add_argument('--protocol',required=True);a.add_argument('--deadline',type=float,required=True);args=a.parse_args();root=Path(args.root)
    sys.path.insert(0,str(root/'bundle'));sys.path.insert(0,str(root.parents[2]/'tools'))
    from dual_host_runtime.io import atomic,read
    from optional_time_budget import Guard
    from research_runtime.common import sha
    import generalization_cloud as cloud
    p=read(args.protocol);out=root/'output'/args.job;out.mkdir(parents=True,exist_ok=True);guard=Guard(out,args.deadline,root/'TELEMETRY.json')
    try:
        files={};dones={}
        for key in p['source_jobs']:
            path=root/'output'/key;done=read(path/'DONE.json');assert done['status']=='COMPLETE' and done['protocol_sha256']==p['source_scientific_protocol_sha256']
            assert sha(path/'model_fp16.pt')==done['checkpoint_sha256'] and sha(path/'predictions.npz')==done['predictions_sha256']
            dones[key]=done
            for name in ['PROVENANCE.json','SPLIT.json','model_fp16.pt','predictions.npz']:files[f'output/{key}/{name}']=dict(local=str(path/name),sha256=sha(path/name))
        m,c=cloud.access.connection()
        try:
            with c.open_sftp() as ftp:
                for rel,spec in files.items():
                    current=cloud.REMOTE
                    for part in Path(rel).parent.as_posix().split('/'):
                        current+='/'+part
                        try:ftp.stat(current)
                        except OSError:ftp.mkdir(current)
                    guard.check(phase='PUBLISH_MATCHED_SENSOR_ARTIFACTS',file=rel);ftp.put(spec['local'],cloud.REMOTE+'/'+rel+'.import.partial',confirm=True)
            expected={k:v['sha256'] for k,v in files.items()}
            code=f'''from pathlib import Path
import json,hashlib,os,time
root=Path({cloud.REMOTE!r})
for rel,expected in {expected!r}.items():
 path=root/(rel+'.import.partial')
 with path.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest().upper()
 assert actual==expected,rel
for rel in {list(expected)!r}:os.replace(root/(rel+'.import.partial'),root/rel)
for key,done in {dones!r}.items():
 out=root/'output'/key;assert not (out/'LAUNCH.json').exists();assert not (out/'DONE.json').exists()
 (out/'IMPORT.json').write_text(json.dumps(dict(source_host='local',scientific_protocol_sha256=done['protocol_sha256'],at_unix=time.time())))
 tmp=out/'DONE.json.import.partial';tmp.write_text(json.dumps(done,indent=2));os.replace(tmp,out/'DONE.json')
print(json.dumps(dict(status='EIGHT_LOCAL_SENSOR_RESULTS_HASH_VERIFIED_AND_PUBLISHED',jobs={list(dones)!r},at_unix=time.time())))'''
            result=cloud.access.execute(m,c,code,timeout=60)
        finally:c.close()
        atomic(out/'DONE.json',dict(status='COMPLETE',protocol_sha256=sha(args.protocol),**{k:v for k,v in result.items() if k!='status'}))
        atomic(root.parent/'SENSOR_RESULTS_PUBLISHED.json',result)
    except BaseException as e:atomic(out/'ERROR.json',dict(error=repr(e),traceback=traceback.format_exc(),at_unix=time.time()));raise

if __name__=='__main__':main()
