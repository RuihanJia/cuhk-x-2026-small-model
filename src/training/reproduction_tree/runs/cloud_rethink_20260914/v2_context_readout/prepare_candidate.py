"""CPU-only identity receipts and non-executable candidate. Does not deploy."""
from pathlib import Path
import os,sys,json,hashlib,copy
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
E=Path(__file__).resolve().parent;C=E.parent;O=C.parent/'error_resolution_20260913';Q=C/'videomaev2_base_quant';H=C/'v2_temporal_halves_fusion'
sys.path[:0]=[str(E),str(Q)]
import torch
from codec import FORMAT
from quant_identity import tensor_manifest
REMOTE='/home/featurize/cuhkx_motion_transfer_20260911_v1/cloud_rethink_20260914'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def local(p):
 if p.startswith(REMOTE):return C/p[len(REMOTE)+1:].replace('/output/cloud','/cloud')
 return Path(p)

def main():
 torch.set_num_threads(1);qs=read(Q/'protocol.json');hs=read(C/'v2_temporal_halves/protocol.json')
 assert len(qs['jobs'])==8;receipts=E/'identity_receipts';receipts.mkdir(exist_ok=True)
 jobs=[];upload={};inventory=[]
 for old in qs['jobs']:
  j=copy.deepcopy(old);host='local' if old['id'] in ('V2_BASE_F3','V2_BASE_SOURCE_F3_INNER2') else 'cloud';j['host']=host
  qr=Q/('cloud' if old['host']=='cloud' else 'output/local')/old['id'];qd=read(qr/'DONE.json')
  cp=qr/'model_int8.pt';assert sha(cp)==qd['checkpoint_sha256'] and qd['protocol_sha256']==sha(Q/'protocol.json')
  parent=local(old['parent_directory']);td=read(parent/'TRAIN_DONE.json');fr=read(parent/'FROZEN.json');split=read(parent/'SPLIT.json')
  assert qd['parent_checkpoint_sha256']==td['artifacts']['model_fp16.pt']==fr['checkpoint_sha256']
  packed=torch.load(cp,map_location='cpu',weights_only=True)
  meta={k:packed[k] for k in ('protocol_sha256','pretrained_sha256','train_users','held_users','epochs','seed')}
  assert meta['train_users']==old['train_users'] and meta['held_users']==old['held_users'] and meta['epochs']==16
  receipt=dict(status='EXACT_PRIOR_INT8_TENSOR_RECEIPT',job_id=old['id'],original_quant_checkpoint_sha256=qd['checkpoint_sha256'],original_quant_checkpoint_bytes=cp.stat().st_size,parent_fp16_sha256=fr['checkpoint_sha256'],quant_protocol_sha256=sha(Q/'protocol.json'),codec_sha256=sha(Q/'codec.py'),storage_format=FORMAT,quantized_names=packed['quantized_names'],checkpoint_metadata=meta,tensor_manifest=tensor_manifest(packed),no_forward=True,no_CUDA=True)
  rp=receipts/(old['id']+'.json');write(rp,receipt);del packed
  j.update(parent_fp16_sha256=fr['checkpoint_sha256'],parent_train_done_sha256=sha(parent/'TRAIN_DONE.json'),parent_frozen_sha256=sha(parent/'FROZEN.json'),parent_split_sha256=sha(parent/'SPLIT.json'),original_quant_checkpoint_sha256=qd['checkpoint_sha256'],identity_receipt_sha256=sha(rp))
  if host=='local':j.update(parent_directory=str(parent),quant_mode='EXISTING_INT8',quant_checkpoint=str(cp),identity_receipt=str(rp))
  else:
   remote_parent=old['parent_directory'] if old['parent_directory'].startswith(REMOTE) else REMOTE+'/'+parent.relative_to(C).as_posix().replace('/cloud/','/output/cloud/')
   j.update(parent_directory=remote_parent,quant_mode='REPACK_FROZEN_FP16',identity_receipt=REMOTE+'/v2_context_readout/identity_receipts/'+rp.name)
   upload[j['identity_receipt']]=dict(local_path=str(rp),sha256=sha(rp),bytes=rp.stat().st_size)
   inventory.append(dict(job_id=j['id'],path=remote_parent+'/model_fp16.pt',sha256=fr['checkpoint_sha256'],bytes=fr['checkpoint_bytes'],status='ROOT_RESTORED_CLOUD_PARENT_CHECK_MANIFEST_BEFORE_LAUNCH'))
   for name in ('TRAIN_DONE.json','FROZEN.json','SPLIT.json'):
    p=parent/name;upload[remote_parent+'/'+name]=dict(local_path=str(p),sha256=sha(p),bytes=p.stat().st_size,upload_only_if_missing=True)
  jobs.append(j)
 code={k:v for k,v in hs['code_sha256'].items() if k.startswith('videomaev2_base/') or k.startswith('videomaev2_base_quant/')}
 for name in ('data.py','worker.py','quant_identity.py','cpu_qa.py','prepare_candidate.py'):
  code['v2_context_readout/'+name]=sha(E/name)
 hosts={}
 for host in ('local','cloud'):
  h=copy.deepcopy(hs['hosts'][host]);h['output_directory']=str(E/'output/local') if host=='local' else REMOTE+'/v2_context_readout/output/cloud'
  h['wall_cap_seconds']=3300;h['job_wall_cap_seconds']=900;h['eval_micro']=3;h['predecessor']=None
  # Inputs are original source/code authorization only; no prior expired execution window.
  hosts[host]=h
 cal_inputs=dict(source_fitted=str(H/'output/SOURCE_FITTED.npz'),source_fitted_sha256=sha(H/'output/SOURCE_FITTED.npz'),old_paired=str(H/'output/PAIRED.npz'),old_paired_sha256=sha(H/'output/PAIRED.npz'),old_protocol=str(H/'protocol.json'),old_protocol_sha256=sha(H/'protocol.json'),old_calibrators={str(f):{mode:dict(path=str(H/'output'/f'F{f}_{mode}.json'),sha256=sha(H/'output'/f'F{f}_{mode}.json')) for mode in ('global','classwise')} for f in (0,3)},original_whole3_paired=str(C/'videomaev2_base_fusion/output/PAIRED.npz'),original_whole3_paired_sha256=sha(C/'videomaev2_base_fusion/output/PAIRED.npz'))
 s=dict(status='PREPARED_NOT_EXECUTABLE',id='V2_BASE_CONTEXT_1P5_THREE_PHASE_SHARED_INT8',registered_at_unix=None,deadline_unix=None,proposal_global_cap_seconds=3600,proposal_cpu_calibration_cap_seconds=180,source_authorization_sha256=hs['source_authorization_sha256'],factor=1.5,phases=[.5,.25,.75],crop_method='CENTERED_SQUARE_EXPAND_1P5_CLAMP_TO_NATIVE_V1',optimizer_updates=0,storage_format=FORMAT,seed=hs['seed'],pose_inputs_sha256=hs['pose_inputs_sha256'],jobs=jobs,hosts=hosts,code_sha256=code,worker_sha256=sha(E/'worker.py'),controller_sha256=None,original_quant_protocol_sha256=sha(Q/'protocol.json'),original_temporal_protocol_sha256=sha(C/'v2_temporal_halves/protocol.json'),CPU_QA=dict(path=str(E/'CPU_QA.json'),sha256=sha(E/'CPU_QA.json')),calibration_inputs=cal_inputs,calibration_recipe=dict(branches=['T','I','BaseSevenFixed','ContextThreeMean'],steps=500,classwise_penalty=.1,calibrators=4,old_three_branch_calibrators='exact_existing_global_and_classwise_states_no_refit',all_four_new_calibrators_frozen_before_outer_context_arrays=True,no_NewWindow=True),model_size='One shared original INT8 Base; T/I unchanged. No new model weight. Calibration constant increment only; final package exact byte audit required.',readout='Reuse prior .5*whole5mean+.25*firsthalf+.25*secondhalf from frozen H SOURCE_FITTED/PAIRED.quant_base; only compute new1.5ROI phases .5/.25/.75; context feature is unweighted3logitmean.',no_Test=True,no_Public_fit=True,no_CSV=True,activation_requirements=['root explicit freeze','host predecessor terminal owned absent','all runtime bindings live verified','source F3INNER2 stays local; no checkpoint upload','whole stage including calibration<=3600seconds'],limitations='F0/F3 repeatedly inspected; selected after visual error audit. Not unseen confirmation. Fixed1.5 cannot solve all walk-out-of-ROI or fine-detail failures.')
 write(E/'protocol_candidate.json',s);write(E/'UPLOAD_CANDIDATE.json',dict(status='NOT_TRANSFERRED',files=upload,cloud_inventory=inventory,no_raw_data=True,no_checkpoint_upload=True,forbidden_source_parent_upload='V2_BASE_SOURCE_F3_INNER2'))
 print(json.dumps(dict(status='PREPARED_NOT_EXECUTABLE',candidate_sha256=sha(E/'protocol_candidate.json'),worker_sha256=s['worker_sha256'],receipts=8,cloud_existing_fp16=6,source_F3INNER2_stays_local=True,metadata_upload_bytes=sum(v['bytes'] for v in upload.values()))))
if __name__=='__main__':main()
