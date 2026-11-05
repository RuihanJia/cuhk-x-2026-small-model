import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,time,hashlib,importlib.util
E=Path(__file__).resolve().parent;C=E.parent;R=C.parents[1];D=C/'v2_base6_small_design';P=R/'deliverables/videomaev2_base_temporal_halves_candidate_20260915';O=R/'runs/error_resolution_20260913'
sys.path[:0]=[str(E),str(P),str(D),str(O),str(O/'bundle')]
import torch
from dual_model import PairVideo
from loader import load_state
from pose_view_data import PoseFiveViews
start=time.time();torch.set_num_threads(1);s=json.loads((O/'POSE_ADAPTER_INPUTS.json').read_text());row=next(r for r in s['rows']if r['user']==1);x=PoseFiveViews([row],'D:/Datasets/CUHK-X_Small_Model_Track/Small-Model-Track',s['regions'])[0]['x'][0:1];assert x.shape==(1,3,16,224,224)
results={};container=D/'BASE6_SMALL8_F3_UNEVALUATED.bin'
for kind,stage,cls in [('Base','videomaev2_base','VideoMAEBase'),('Small','videomaev2_small','VideoMAESmall')]:
 path=C/stage/'model.py';sp=importlib.util.spec_from_file_location('reference_'+kind,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);ref=getattr(m,cls)(classes=40,fast=True).eval();got=PairVideo(kind).eval();state,origin=load_state(container,kind);assert row['user']in origin['train_users'] and row['user']not in origin['held_users'];ref.load_state_dict(state,strict=True);got.load_state_dict(state,strict=True)
 assert set(ref.state_dict())==set(got.state_dict());assert all(torch.equal(ref.state_dict()[k],v)for k,v in got.state_dict().items())
 with torch.inference_mode():a=ref(x);b=got(x)
 error=float((a-b).abs().max());assert error==0.;results[kind]=dict(parameters=sum(p.numel()for p in got.parameters()),real_source_forward_max_abs=error,training_clip=row['clip_id'],model_code_sha256=hashlib.sha256(path.read_bytes()).hexdigest());del ref,got,state
r=dict(status='PASS_PRODUCTION_PAIR_CONSTRUCTORS_REAL_SOURCE_CPU',results=results,no_GPU=True,no_Test=True,no_final_calibration=True,seconds=time.time()-start,container_sha256=hashlib.sha256(container.read_bytes()).hexdigest());(E/'SOURCE_CONSTRUCTOR_QA.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(json.dumps(r))
