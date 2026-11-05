from pathlib import Path
import sys,json,hashlib,time
E=Path(__file__).resolve().parent;C=E.parent
sys.path.insert(0,str(C));import post_aux_cloud_access as a
R='/home/featurize/cuhkx_motion_transfer_20260911_v1/cloud_rethink_20260914/'+E.name
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 q=Path(str(p)+'.tmp');q.write_text(json.dumps(v,indent=2),encoding='utf-8');q.replace(p)
s=json.loads((C/'v2_base6_small8_source_completion/INPUTS.json').read_text(encoding='utf-8-sig'))
jobs=[j for j in s['jobs'] if j['id']!='V2_BASE_SOURCE_F3_INNER2'];jobs.sort(key=lambda j:(j['scope']!='source',j['fold'],j['inner']or 0))
files=[]
for j in jobs:
 p=Path(j.get('container',{}).get('path',C/'v2_base6_small_design/BASE6_SMALL8_F3_UNEVALUATED.bin'))
 files.append(dict(id=j['id'],local_path=str(p),path=R+'/containers/'+j['id']+'.bin',sha256=sha(p),bytes=p.stat().st_size))
mp=E/'PAIR_TRANSFER_MANIFEST.json';assert not mp.exists()
write(mp,dict(status='FROZEN_EXACT_SEVEN_ORIGINAL_PAIRS',files=files,total_bytes=sum(x['bytes']for x in files),no_F3INNER2=True,no_requantization=True,registered_at_unix=time.time(),wall_cap_seconds=480))
mod,client=a.connect();start=time.time();done=[]
try:
 with client.open_sftp()as ftp:
  ftp.get_channel().settimeout(30)
  for v in files:
   for p in [R,R+'/containers']:
    try:ftp.stat(p)
    except FileNotFoundError:ftp.mkdir(p)
   try:ftp.stat(v['path']);raise RuntimeError('DESTINATION_ALREADY_EXISTS')
   except FileNotFoundError:pass
   began=time.time();last=[0.]
   def progress(n,total):
    now=time.time()
    if now-last[0]>=5:
     last[0]=now;rate=n/max(now-began,.001);eta=(sum(z['bytes']for z in files)-sum(z['bytes']for z in done)-n)/max(rate,1)
     write(E/'PAIR_TRANSFER_PROGRESS.json',dict(job=v['id'],transferred=n,total=total,bytes_per_second=rate,remaining_all_seconds=eta,completed=[z['id']for z in done],at_unix=now))
     if now-began>=15 and eta>480:raise TimeoutError('TRANSFER_ETA_EXCEEDS_8MIN')
     if now-start>480:raise TimeoutError('TRANSFER_8MIN_CAP')
   ftp.put(v['local_path'],v['path']+'.partial',callback=progress,confirm=True)
   code="""from pathlib import Path
import hashlib,json,os
p=Path(P);assert p.stat().st_size==N;assert hashlib.sha256(p.read_bytes()).hexdigest()==H;os.replace(p,Path(T));print(json.dumps({'sha256':H,'verified':True}))
""".replace('P',repr(v['path']+'.partial')).replace('N',str(v['bytes'])).replace('H',repr(v['sha256'])).replace('T',repr(v['path']))
   # Avoid textual substitution into identifiers by emitting explicit code.
   code='from pathlib import Path\nimport hashlib,json,os\np=Path('+repr(v['path']+'.partial')+');assert p.stat().st_size=='+str(v['bytes'])+';assert hashlib.sha256(p.read_bytes()).hexdigest()=='+repr(v['sha256'])+';os.replace(p,'+repr(v['path'])+');print(json.dumps({"verified":True}))'
   assert a.old.execute(mod,client,code,30)['verified'];done.append(v)
   write(E/'PAIR_TRANSFER_COMPLETED.json',dict(files=done,seconds=time.time()-start,at_unix=time.time()))
 write(E/'PAIR_TRANSFER_DONE.json',dict(status='COMPLETE_SEVEN_ORIGINAL_PAIR_TRANSFER',files=done,seconds=time.time()-start))
except BaseException as e:
 write(E/'PAIR_TRANSFER_ERROR.json',dict(error=str(e),completed=done,seconds=time.time()-start));raise
finally:client.close()
