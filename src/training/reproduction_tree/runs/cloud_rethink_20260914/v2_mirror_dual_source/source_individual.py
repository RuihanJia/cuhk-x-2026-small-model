"""Descriptive complete local source endpoint only. No fit, no outer/Test."""
from pathlib import Path
import json,hashlib,time,numpy as np
E=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pair(y,u,a,b):
 x=a.argmax(1)==y;z=b.argmax(1)==y
 def nll(v):v=v.astype(np.float64);v-=v.max(1,keepdims=True);return float((np.log(np.exp(v).sum(1))-v[np.arange(len(y)),y]).mean())
 return dict(n=len(y),before=int(x.sum()),after=int(z.sum()),fix=int((~x&z).sum()),harm=int((x&~z).sum()),changes=int((a.argmax(1)!=b.argmax(1)).sum()),NLL_before=nll(a),NLL_after=nll(b),per_class={str(k):dict(n=int((y==k).sum()),before=int(x[y==k].sum()),after=int(z[y==k].sum()))for k in sorted(set(y.tolist()))})
s=read(E/'protocol.json');ph=sha(E/'protocol.json');out=Path(s['hosts']['local']['output_directory']);ct=read(out/'CONTROLLER_DONE.json');d=read(out/'DONE.json');assert ct['status']=='COMPLETE'and ct['exit_code']==0 and ct['protocol_sha256']==d['protocol_sha256']==ph and ct['terminal_sha256']==sha(out/'DONE.json')
j=next(j for j in s['jobs']if j['host']=='local');assert j['scope']=='source';p=out/j['id'];jd=read(p/'DONE.json');npz=p/'predictions.npz';assert sha(npz)==jd['artifacts']['predictions.npz']
with np.load(npz,allow_pickle=False)as v:a={k:v[k]for k in v.files}
r=dict(status='COMPLETE_LOCAL_SOURCE_MIRROR_DESCRIPTIVE_ONLY',protocol_sha256=ph,predictions_sha256=sha(npz),job=j['id'],Base=pair(a['targets'],a['users'],a['base6_original'],a['base6']),Small=pair(a['targets'],a['users'],a['small8_original'],a['small8']),no_fit=True,no_outer=True,no_Test=True,at_unix=time.time())
(E/'LOCAL_SOURCE_EARLY_REPORT.json').write_text(json.dumps(r,indent=2),encoding='utf-8');print(json.dumps({k:{n:v for n,v in r[k].items()if n!='per_class'}for k in ['Base','Small']}))
