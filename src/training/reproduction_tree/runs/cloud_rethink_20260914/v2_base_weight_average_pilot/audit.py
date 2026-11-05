"""Independent NumPy source-only comparisons after both actual readouts complete."""
from pathlib import Path
import hashlib,json,time
import numpy as np
E=Path(__file__).resolve().parent;C=E.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf8'))
def arr(p):
 with np.load(p,allow_pickle=False)as v:return {k:v[k]for k in v.files}
def softmax(v):
 v=v.astype(np.float64);v-=v.max(1,keepdims=True);p=np.exp(v);return p/p.sum(1,keepdims=True)
def paired(y,users,a,b):
 x=a.argmax(1)==y;z=b.argmax(1)==y;pa=softmax(a);pb=softmax(b)
 return dict(n=len(y),before=int(x.sum()),after=int(z.sum()),fix=int((~x&z).sum()),harm=int((x&~z).sum()),changes=int((a.argmax(1)!=b.argmax(1)).sum()),NLL_before=float(-np.log(pa[np.arange(len(y)),y].clip(1e-300)).mean()),NLL_after=float(-np.log(pb[np.arange(len(y)),y].clip(1e-300)).mean()),per_user={str(u):dict(n=int((users==u).sum()),before=int(x[users==u].sum()),after=int(z[users==u].sum()),fix=int((~x[users==u]&z[users==u]).sum()),harm=int((x[users==u]&~z[users==u]).sum()))for u in sorted(set(users.tolist()))})
def main():
 s=read(E/'protocol.json');ph=sha(E/'protocol.json');plan=read(E/'audit_protocol.json');assert plan['protocol_sha256']==ph and sha(E/'audit.py')==plan['audit_sha256']
 gate={};parts=[];frozen=[]
 for host in ['cloud','local']:
  r=E/'cloud'if host=='cloud'else E/'output/local';d=read(r/'DONE.json');ct=read(r/'CONTROLLER_DONE.json')
  assert d['protocol_sha256']==ct['protocol_sha256']==ph and ct['status']=='COMPLETE' and ct['terminal_sha256']==sha(r/'DONE.json') and d['status']=='COMPLETE_WEIGHT_AVERAGE_PILOT'
  for n,h in d['artifacts'].items():
   if not n.endswith('.pt'):assert sha(r/n)==h,n
  gate[host]=dict(controller_done_sha256=sha(r/'CONTROLLER_DONE.json'),done_sha256=sha(r/'DONE.json'))
 for n,v in plan['source_bindings'].items():assert sha(v['path'])==v['sha256'],n
 raw=arr(plan['source_bindings']['raw']['path']);control=arr(plan['source_bindings']['control']['path'])
 lookup={kind:{(int(f),cid):i for i,(f,cid)in enumerate(zip(v['source_fold'],v['clip_ids']))}for kind,v in [('raw',raw),('control',control)]}
 for j in s['jobs']:
  r=(E/'cloud'if j['host']=='cloud'else E/'output/local')/j['id'];d=read(r/'DONE.json');fr=read(r/'AVERAGE_FROZEN.json');v=arr(r/'predictions.npz')
  assert d['protocol_sha256']==fr['protocol_sha256']==ph and fr['alpha']==.5 and fr['quantized_matrices']==50 and fr['optimizer_updates']==0
  assert fr['raw']==j['raw'] and fr['control']==j['control'] and fr['checkpoint_sha256']==d['artifacts']['model_int8.pt']
  assert set(v['users'])==set(j['held_users']) and not set(v['users'])&set(j['outer_excluded_users']) and len(v['users'])==j['records']
  z=v['seven_view_logits'];replay=(.5*z[:,:5].mean(1)+.25*z[:,5]+.25*z[:,6]).astype(np.float32);assert np.array_equal(replay,v['videomae_logits']) and np.isfinite(z).all()
  ix={kind:np.array([lookup[kind][(j['fold'],cid)]for cid in v['clip_ids']])for kind in lookup}
  for kind,base in [('raw',raw),('control',control)]:
   for k in ['targets','users','clip_ids']:assert np.array_equal(base[k][ix[kind]],v[k])
  a=raw['quant_base'][ix['raw']];b=control['video'][ix['control']];y=v['targets'];u=v['users'];parts.append(dict(job=j['id'],fold=j['fold'],inner=j['inner'],vs_raw=paired(y,u,a,replay),vs_control=paired(y,u,b,replay)))
  frozen.append(dict(job=j['id'],average_frozen_sha256=sha(r/'AVERAGE_FROZEN.json'),predictions_sha256=sha(r/'predictions.npz'),checkpoint_sha256=fr['checkpoint_sha256']))
 out=E/'audit';out.mkdir(exist_ok=True);assert not(out/'RESULT.json').exists()
 result=dict(status='PASS_INDEPENDENT_FIXED_AVERAGE_SOURCE_PILOT',protocol_sha256=ph,gate=gate,source_bindings=plan['source_bindings'],jobs=parts,artifacts=frozen,optimizer_updates=0,calibration_fits=0,outer_rows_read=0,no_Test=True,independent_seven_view_formula_maxabs=0,cloud_checkpoint_bytes_not_downloaded=True,limitations=['Two adaptively exposed source-inner groups only; no claimed unbiased full validation.','Comparisons use frozen originalINT8 and CONTROL4INT8 source readouts, no fusion fit.','Cloud checkpoint midpoint execution is worker+bound-parent evidence; this audit independently replays predictions, not GPU forwards.'],at_unix=time.time())
 (out/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n','utf8');(out/'DONE.json').write_text(json.dumps(dict(status='COMPLETE',protocol_sha256=ph,result_sha256=sha(out/'RESULT.json')))+'\n','utf8')
 print(json.dumps({p['job']:{k:p[k]for k in ['vs_raw','vs_control']}for p in parts}))
if __name__=='__main__':main()
