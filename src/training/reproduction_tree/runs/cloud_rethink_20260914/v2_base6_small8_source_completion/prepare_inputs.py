"""CPU-only eight-parent identity audit and seven source-completion containers."""
from pathlib import Path
import os,json,hashlib,sys,time
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import torch
E=Path(__file__).resolve().parent;C=E.parent;D=C/'v2_base6_small_design';sys.path[:0]=[str(D),str(E)]
from container_writer import write_bound_pair
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
remote='/home/featurize/cuhkx_motion_transfer_20260911_v1/cloud_rethink_20260914/'
def mirror(p):return C/p[len(remote):].replace('/output/cloud','/cloud') if p.startswith(remote) else Path(p)
def bind(p,expected=None):
 h=sha(p);assert expected is None or h==expected,str(p);return dict(path=str(Path(p).resolve()),sha256=h,bytes=Path(p).stat().st_size)
def key(j):return (j['scope'],j['fold'],j.get('inner') if j['scope']=='source' else None)
def main():
 torch.set_num_threads(1);start=time.time();jobs=[];bp=read(C/'videomaev2_base_quant/protocol.json');sp=read(C/'videomaev2_quant/protocol.json');sm={key(j):j for j in sp['jobs']}
 registration=dict(status='FROZEN_CPU_PARENT_AND_CONTAINER_PREPARATION',registered_at_unix=start,wall_cap_seconds=300,no_GPU=True,no_Test=True,no_fit=True,base_quant_protocol=bind(C/'videomaev2_base_quant/protocol.json'),small_quant_protocol=bind(C/'videomaev2_quant/protocol.json'),scope='Eight registered K710 Base+K710 Small endpoints. Build seven new paired containers; F3 pilot remains single writer dino and will be reused.',correction='Initial filename-only source inventory mistakenly missed videomaev2_source. Actual source protocols and all16parent files exist; no model family changed.',code_sha256={str(p.relative_to(C)):sha(p)for p in [Path(__file__),E/'container_writer.py',D/'pair_codec.py']})
 registration['engineering_replay']=dict(prior_registration_sha256=sha(E/'CPU_PREPARATION_REGISTRATION.json'),reason='Outer inner-key None versus absent KeyError before any container generation. Normalize outer identity only.');reg=E/'CPU_PREPARATION_REGISTRATION_V2.json';assert not reg.exists();reg.write_text(json.dumps(registration,indent=2),encoding='utf-8');(E/'containers').mkdir(exist_ok=True)
 for bj in bp['jobs']:
  assert time.time()<start+300;sj=sm[key(bj)];assert all(bj[k]==sj[k]for k in ['train_users','held_users','outer_excluded_users','records'])
  parents={};cps={}
  for label,j,stage in [('Base',bj,'videomaev2_base_quant'),('Small',sj,'videomaev2_quant')]:
   root=mirror(j['parent_directory']);fr=read(root/'FROZEN.json');split=read(root/'SPLIT.json');qroot=C/stage/('cloud' if j['host']=='cloud' else 'output/local')/j['id'];qd=read(qroot/'DONE.json')
   assert fr['protocol_sha256']==j['parent_protocol_sha256']==split['protocol_sha256']==qd['parent_protocol_sha256']
   assert fr['epochs']==16 and fr['checkpoint_sha256']==qd['parent_checkpoint_sha256']
   assert fr['train_users']==split['train_users']==j['train_users'] and fr['held_users']==split['held_users']==j['held_users']
   assert set(j['train_users']).isdisjoint(set(j['held_users'])|set(j['outer_excluded_users'])) and set(j['held_users']).isdisjoint(j['outer_excluded_users'])
   cp=bind(root/'model_fp16.pt',fr['checkpoint_sha256']);q=bind(qroot/'model_int8.pt',qd['checkpoint_sha256'])
   parents[label]=dict(parent=cp,quant=q,frozen=bind(root/'FROZEN.json'),split=bind(root/'SPLIT.json'),quant_done=bind(qroot/'DONE.json'),old_five_predictions=bind(qroot/'predictions.npz',qd['predictions_sha256']))
   cps[label]=torch.load(cp['path'],map_location='cpu',weights_only=True)
   assert cps[label]['epochs']==16 and cps[label]['train_users']==j['train_users'] and cps[label]['held_users']==j['held_users']
   if label=='Small':cps['Small49']=torch.load(q['path'],map_location='cpu',weights_only=True);assert len(cps['Small49']['quantized_names'])==49
  old7root=C/'v2_temporal_halves'/('cloud'if bj['host']=='cloud'else'output/local')/bj['id'];old7done=read(old7root/'DONE.json')
  assert old7done['checkpoint_sha256']==parents['Base']['quant']['sha256']
  parents['Base']['old_seven_predictions']=bind(old7root/'predictions.npz',old7done['predictions_sha256'])
  j=dict(id=bj['id'],scope=bj['scope'],fold=bj['fold'],inner=bj.get('inner'),records=bj['records'],train_users=bj['train_users'],held_users=bj['held_users'],outer_excluded_users=bj['outer_excluded_users'],parents=parents)
  if bj['scope']=='outer'and bj['fold']==3:j['reuse_pilot']=True
  else:
   dest=E/'containers'/f"{bj['id']}.bin";pr=dict(scope='Eight-endpoint frozen K710 pair readout; no training, no Test',base_fp16_sha256=parents['Base']['parent']['sha256'],small_fp16_sha256=parents['Small']['parent']['sha256'],small49_sha256=parents['Small']['quant']['sha256'],base_protocol_sha256=bj['parent_protocol_sha256'],small_protocol_sha256=sj['parent_protocol_sha256'],train_users=bj['train_users'],held_users=bj['held_users'],outer_excluded_users=bj['outer_excluded_users'],epochs=16)
   stats=write_bound_pair(dest,cps['Base'],cps['Small'],cps['Small49'],pr);assert stats['file_bytes']+12656523+12000<=100000000
   j.update(container=bind(dest),container_verification=stats,reuse_pilot=False)
  jobs.append(j);del cps;print(json.dumps(dict(id=j['id'],records=j['records'],reuse_pilot=j['reuse_pilot'])),flush=True)
 result=dict(status='PASS_EIGHT_MATCHED_K710_PARENT_INVENTORY',registered_preparation_sha256=sha(reg),jobs=jobs,source_rows=sum(j['records']for j in jobs if j['scope']=='source'),outer_rows=sum(j['records']for j in jobs if j['scope']=='outer'),new_jobs=7,reused_pilot='V2_BASE_F3',new_model_file_bytes=sum(j.get('container',{}).get('bytes',0)for j in jobs),seconds=time.time()-start,no_GPU=True,no_Test=True,at_unix=time.time())
 assert result['source_rows']==4768 and result['outer_rows']==1094;(E/'INPUTS.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items()if k!='jobs'}))
if __name__=='__main__':main()
