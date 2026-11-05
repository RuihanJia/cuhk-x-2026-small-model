import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,time,hashlib,shutil
E=Path(__file__).resolve().parent;C=E.parent;R=C.parents[1];D=C/'v2_base6_small_design';sys.path.insert(0,str(D))
import torch
from pair_codec import write_pair,verify_pair
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
start=time.time();torch.set_num_threads(1);roots={'Base':C/'videomaev2_base_production/cloud','Small':C/'videomaev2_production/cloud'};states={};orig={}
for kind,p in roots.items():
 f=read(p/'FROZEN.json');fp=p/'model_fp16.pt';assert sha(fp)==f['checkpoint_sha256'];cp=torch.load(fp,map_location='cpu',weights_only=True);assert cp['held_users']==[] and cp['epochs']==16 and cp['train_users']==list(range(1,10))+list(range(16,25));states[kind]=cp;orig[kind]=dict(path=str(fp),sha256=sha(fp),protocol_sha256=cp['protocol_sha256'])
assert states['Base']['train_users']==states['Small']['train_users'];out=E/'prepared_artifacts';out.mkdir(exist_ok=True);destination=out/'PAIR_FULL18_UNSELECTED.bin'
provenance=dict(scope='Full18 CPU storage qualification only; candidate not selected and no Test inference',base_fp16_sha256=orig['Base']['sha256'],small_fp16_sha256=orig['Small']['sha256'],base_protocol_sha256=orig['Base']['protocol_sha256'],small_protocol_sha256=orig['Small']['protocol_sha256'],train_users=states['Base']['train_users'],held_users=[],epochs=16)
registration=dict(status='REGISTERED_FULL18_STORAGE_ONLY_NO_TEST',at_unix=time.time(),sources=orig,codec_sha256=sha(D/'pair_codec.py'),scientific_precision='Baseall50symmetric6bit,Smallall50symmetric8bit,originalFP32rowscales,otherFP16',no_training=True,no_calibration=True,no_GPU=True,no_Test=True)
assert not(E/'STORAGE_QA_REGISTRATION.json').exists();(E/'STORAGE_QA_REGISTRATION.json').write_text(json.dumps(registration,indent=2)+'\n',encoding='utf-8')
stats=write_pair(destination,states['Base'],states['Small'],provenance);verification=verify_pair(destination,stats['tensor_checks']);P=R/'deliverables/videomaev2_base_temporal_halves_candidate_20260915/checkpoints';retained={n:dict(bytes=(P/n).stat().st_size,sha256=sha(P/n))for n in ['T.pt','I.pt','POSE.pt']};total=destination.stat().st_size+sum(v['bytes']for v in retained.values());assert total+12000<100000000
r=dict(status='PASS_FULL18_STORAGE_PREPARED_UNSELECTED_NO_TEST',container=str(destination),container_sha256=sha(destination),container_bytes=destination.stat().st_size,sources=orig,provenance=provenance,verification=verification,retained=retained,weight_total_before_final_calibration=total,remaining_calibration_capacity=100000000-total,calibration_reserve_bytes=12000,seconds=time.time()-start,no_Test=True,no_final_calibration=True,no_selection_claim=True)
(E/'FULL18_STORAGE_QA.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in r.items()if k not in ['sources','provenance','retained']}))
