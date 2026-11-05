from pathlib import Path
import sys,json,hashlib,time
E=Path(__file__).resolve().parent;C=E.parent
sys.path.insert(0,str(C));import post_aux_cloud_access as a
R='/home/featurize/cuhkx_motion_transfer_20260911_v1/cloud_rethink_20260914'
s=json.loads((C/'v2_base6_small8_source_completion/INPUTS.json').read_text(encoding='utf-8-sig'))
paths={j['id']:R+'/'+Path(j.get('container',{}).get('path',C/'v2_base6_small_design/BASE6_SMALL8_F3_UNEVALUATED.bin')).relative_to(C).as_posix() for j in s['jobs']}
mod,client=a.connect()
try:
 code="""from pathlib import Path
import json,hashlib
paths=json.loads(PATHS);out={}
for k,p in paths.items():
 q=Path(p);out[k]={'path':p,'exists':q.is_file()}
 if q.is_file():out[k].update(bytes=q.stat().st_size,sha256=hashlib.sha256(q.read_bytes()).hexdigest())
print(json.dumps(out))
""".replace('PATHS',repr(json.dumps(paths)))
 r=a.old.execute(mod,client,code,60)
finally:client.close()
(E/'CLOUD_PAIR_INVENTORY.json').write_text(json.dumps(r,indent=2),encoding='utf-8');print(json.dumps(r))
