"""Transfer each completed frozen cache shard once and release both registered head queues."""
from pathlib import Path
import json,time,sys,hashlib,traceback
import cloud_transport as t
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'bundle'))
from dual_host_runtime.io import atomic
spec=json.loads((R/'DETAIL_PROTOCOL.json').read_text(encoding='utf-8'));checksum=hashlib.sha256((R/'DETAIL_PROTOCOL.json').read_bytes()).hexdigest()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
try:
    while time.time()<min(spec['deadline_unix'],spec['registered_at_unix']+5400):
        local=R/'detail_output/local'
        if (local/'ERROR.json').exists():raise RuntimeError('LOCAL_CACHE_FAILED')
        if not (local/'CACHE_DONE.json').exists():time.sleep(30);continue
        mod,c=t.connect()
        try:
            status=t.old.execute(mod,c,"from pathlib import Path\nimport json\nr=Path(%r)/'detail_output/cloud'\nprint(json.dumps({'done':json.loads((r/'CACHE_DONE.json').read_text()) if (r/'CACHE_DONE.json').exists() else None,'error':(r/'ERROR.json').read_text() if (r/'ERROR.json').exists() else None}))"%t.REMOTE)
            if status['error']:raise RuntimeError(status['error'])
            if not status['done']:continue
            ld=json.loads((local/'CACHE_DONE.json').read_text());cd=status['done'];assert ld['protocol_sha256']==cd['protocol_sha256']==checksum
            assert ld['features_sha256']==sha(local/'features.npy') and ld['available_sha256']==sha(local/'available.npy')
            t.old.execute(mod,c,"from pathlib import Path\nr=Path(%r)/'detail_output/local'\nr.mkdir(parents=True,exist_ok=True)\nprint('{}')"%t.REMOTE)
            with c.open_sftp() as s:
                for n in ['features.npy','available.npy','CACHE_DONE.json']:
                    dest=t.REMOTE+'/detail_output/local/'+n;s.put(str(local/n),dest+'.transfer',confirm=True);s.rename(dest+'.transfer',dest)
            merged=dict(protocol_sha256=checksum,at_unix=time.time(),cloud=cd,local=ld)
            code="from pathlib import Path\nimport json,hashlib\nr=Path(%r)/'detail_output';d=%r\nfor host in ['local','cloud']:\n for n,key in [('features.npy','features_sha256'),('available.npy','available_sha256')]:assert hashlib.sha256((r/host/n).read_bytes()).hexdigest()==d[host][key]\np=r/'MERGED.json.partial';p.write_text(json.dumps(d));p.replace(r/'MERGED.json')\nprint(json.dumps({'status':'CLOUD_HEAD_QUEUE_RELEASED'}))"%(t.REMOTE,merged)
            print(json.dumps(t.old.execute(mod,c,code,50)),flush=True)
            cloud=R/'detail_output/cloud';cloud.mkdir(parents=True,exist_ok=True)
            with c.open_sftp() as s:
                for n in ['features.npy','available.npy','CACHE_DONE.json']:
                    dst=cloud/n;s.get(t.REMOTE+'/detail_output/cloud/'+n,str(dst.with_suffix(dst.suffix+'.transfer')));dst.with_suffix(dst.suffix+'.transfer').replace(dst)
            assert sha(cloud/'features.npy')==cd['features_sha256'] and sha(cloud/'available.npy')==cd['available_sha256']
            atomic(R/'detail_output/MERGED.json',merged);atomic(R/'DETAIL_MERGE_DONE.json',dict(status='BOTH_REGISTERED_HEAD_QUEUES_RELEASED',at_unix=time.time(),protocol_sha256=checksum));print('BOTH_QUEUES_RELEASED',flush=True);break
        finally:c.close()
        time.sleep(60)
    else:raise TimeoutError('MERGE_BOUND_EXPIRED')
except BaseException as e:
    atomic(R/'DETAIL_MERGE_ERROR.json',dict(error=str(e),at_unix=time.time(),traceback=traceback.format_exc()));raise
