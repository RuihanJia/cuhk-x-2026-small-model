"""Actual training-clip parity for the selected full18 paired package; CPU only."""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='2'
import json,time,hashlib,sys,types,importlib.util
E=Path(__file__).resolve().parent;C=E.parent;R=C.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 s=read(E/'source_qa_protocol.json');start=time.time();D=Path(read(E/'protocol.json')['package']);assert s['status']=='FROZEN_FOR_EXECUTION'
 for p,h in s['bindings'].items():assert sha(p)==h,p
 O=C.parent/'error_resolution_20260913';P=C/'aux_public_regression_20260914/dynamic_interaction/photometric_candidate'
 sys.path[:0]=[str(D),str(P),str(O),str(O/'bundle'),str(C/'v2_temporal_halves')]
 import numpy as np,torch,infer
 from photometric_data_numpy import PhotometricFiveViews
 from data import TemporalHalves
 torch.set_num_threads(2);assert not torch.cuda.is_initialized()
 inputs=read(O/'POSE_ADAPTER_INPUTS.json');row=next(r for r in inputs['rows'] if r['clip_id']==s['source_clip']);cid=row['clip_id'];cfg=read(D/'release.json');data=Path(s['data_root'])
 class Modalities:
  def __truediv__(self,name):return data/row['streams'][0]['directory'].replace('\\','/').replace('/IR/','/'+name+'/')
 sample=types.SimpleNamespace(sample_dir=Modalities());checks=[]
 for kind,relative,cls in [('Base','videomaev2_base/model.py','VideoMAEBase')]:
  sp=importlib.util.spec_from_file_location('paired_source_reference_'+kind,C/relative);module=importlib.util.module_from_spec(sp);sp.loader.exec_module(module)
  packed=torch.load(D/cfg['checkpoints']['B']['path'],map_location='cpu',weights_only=True);model=infer.constructor(packed,cfg)
  reference=getattr(module,cls)(classes=40,fast=True);reference.load_state_dict(model.state_dict(),strict=True);reference.eval().requires_grad_(False)
  assert all(torch.equal(x,reference.state_dict()[key]) for key,x in model.state_dict().items())
  for missing in [False,True]:
   regions=json.loads(json.dumps(inputs['regions']));region=regions[cid][0]
   if missing:region['box']=None
   q=torch.cat([PhotometricFiveViews([row],str(data),regions)[0]['x'],TemporalHalves([row],str(data),regions)[0]['x']],0)
   item=infer.BaseViews([sample],[region])[0];assert torch.equal(q,item['mae'])
   n=7 if kind=='Base' else 5
   with torch.inference_mode():
    z,valid=infer.predict(model,item,'cpu',2,kind)
    raw=np.concatenate([reference(q[i:min(i+2,n)]).numpy()for i in range(0,n,2)])
   truth=.5*raw[:5].mean(0)+.25*raw[5]+.25*raw[6] if kind=='Base' else raw.mean(0)
   error=float(np.max(abs(z-truth)));assert valid and error<1e-4 and int(z.argmax())==int(truth.argmax())
   assert time.time()<s['deadline_unix'];checks.append(dict(kind=kind,source_clip=cid,missing_pose_box=missing,input_exact=True,logit_max_abs=error,argmax_equal=True));print(json.dumps(checks[-1]),flush=True)
  del model,reference
 result=dict(status='PASS_WEIGHT_AVERAGED_FULL18_SOURCE_CPU_PARITY',protocol_sha256=sha(E/'source_qa_protocol.json'),checks=checks,no_GPU=True,no_Test=True,optimizer_updates=0,seconds=time.time()-start)
 from dual_host_runtime.io import atomic
 atomic(E/'SOURCE_QA_RESULT.json',result);atomic(E/'SOURCE_QA_DONE.json',dict(status='PASS',protocol_sha256=sha(E/'source_qa_protocol.json'),artifacts={'SOURCE_QA_RESULT.json':sha(E/'SOURCE_QA_RESULT.json')}));print(json.dumps(result))
if __name__=='__main__':main()
