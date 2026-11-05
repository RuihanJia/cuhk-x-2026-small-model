"""One fixed F3 pair, CPU quantization and serialization QA. No model forward."""
from pathlib import Path
import os,json,time,hashlib
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import torch
from pair_codec import write_pair,verify_pair
E=Path(__file__).resolve().parent;C=E.parent;R=C.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def main():
 start=time.time();torch.set_num_threads(1)
 bp=C/'videomaev2_base/cloud/V2_BASE_F3';sp=C/'videomaev2_small/output/local'
 bf=read(bp/'FROZEN.json');sf=read(sp/'FROZEN.json')
 bpath=bp/'model_fp16.pt';spath=sp/'model_fp16.pt';assert sha(bpath)==bf['checkpoint_sha256'];assert sha(spath)==sf['checkpoint_sha256']
 b=torch.load(bpath,map_location='cpu',weights_only=True);s=torch.load(spath,map_location='cpu',weights_only=True)
 assert b['train_users']==s['train_users'] and b['held_users']==s['held_users'] and b['epochs']==s['epochs']==16
 assert set(b['train_users']).isdisjoint(b['held_users'])
 dest=E/'BASE6_SMALL8_F3_UNEVALUATED.bin'
 provenance=dict(scope='F3 engineering prototype, unevaluated, no Test',base_fp16_sha256=bf['checkpoint_sha256'],small_fp16_sha256=sf['checkpoint_sha256'],base_protocol_sha256=b['protocol_sha256'],small_protocol_sha256=s['protocol_sha256'],train_users=b['train_users'],held_users=b['held_users'],epochs=16)
 stats=write_pair(dest,b,s,provenance);verification=verify_pair(dest,stats['tensor_checks'])
 D=R/'deliverables/videomaev2_base_temporal_halves_candidate_20260915/checkpoints'
 retained={n:dict(bytes=(D/n).stat().st_size,sha256=sha(D/n)) for n in ('T.pt','I.pt','POSE.pt')}
 total=stats['file_bytes']+sum(v['bytes'] for v in retained.values())
 result=dict(status='PASS_FIXED_CONTAINER_SIZE_AND_BYTE_QA_NOT_PREDICTION_QA',container=str(dest),container_sha256=sha(dest),codec_sha256=sha(E/'pair_codec.py'),**{k:v for k,v in stats.items() if k!='tensor_checks'},verification=verification,retained=retained,model_files_bytes_before_calibrator=total,headroom_before_calibrator=100000000-total,calibrator_reserve=12000,projected_model_files_bytes_with_reserve=total+12000,within100MB_with_reserve=total+12000<=100000000,provenance=provenance,no_GPU=True,no_fit=True,no_Test=True,seconds=time.time()-start,precision_limit='6bit Base and newly INT8 patch Small have not been evaluated. Container is lossless relative to these quantized tensors, not relative to original FP16.',size_limit='F3 prototype measured, full18 exact serialization still required; identical architecture and tensor counts.')
 (E/'CONTAINER_RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k not in ('retained','provenance')},ensure_ascii=False))
if __name__=='__main__':main()
