from pathlib import Path
import os,sys,json,hashlib,time,copy,types,importlib.util
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
E=Path(__file__).resolve().parent;C=E.parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 start=time.time();s=read(E/'protocol.json');q=read(E/'source_qa_protocol.json');D=Path(s['package']);assert q['status']=='FROZEN_FOR_EXECUTION'
 for p,h in q['bindings'].items():assert sha(p)==h,p
 sys.path[:0]=[str(D),str(C.parent/'error_resolution_20260913'),str(C.parent/'error_resolution_20260913/bundle'),str(C/'v2_base6_small_design')]
 import numpy as np,torch,infer
 from pose_view_data import PoseFiveViews
 from temporal_inputs import local_halves
 sp=importlib.util.spec_from_file_location('source_temporal_inputs_exact',C/'v2_base6_small_design/temporal_inputs.py');tm=importlib.util.module_from_spec(sp);sp.loader.exec_module(tm)
 from dual_host_runtime.io import atomic
 torch.set_num_threads(2);inp=read(C.parent/'error_resolution_20260913/POSE_ADAPTER_INPUTS.json');row=next(r for r in inp['rows']if r['clip_id']=='2_Comb_hair/1/1-1-1');cfg=read(D/'release.json');root=Path(s['data_root'])
 class Modalities:
  def __truediv__(self,name):return root/row['streams'][0]['directory'].replace('/IR/','/'+name+'/')
 sample=types.SimpleNamespace(sample_dir=Modalities());cases=[]
 for kind,code,cls in [('Base','videomaev2_base/model.py','VideoMAEBase'),('Small','videomaev2_small/model.py','VideoMAESmall')]:
  sp=importlib.util.spec_from_file_location('mirror_final_source_'+kind,C/code);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);model=infer.constructor(D/cfg['checkpoints']['PAIR']['path'],kind,cfg);ref=getattr(m,cls)(classes=40,fast=True).eval();ref.load_state_dict(model.state_dict(),strict=True);assert all(torch.equal(v,ref.state_dict()[k])for k,v in model.state_dict().items())
  for missing in [False,True]:
   regions=copy.deepcopy(inp['regions']);reg=regions[row['clip_id']][0]
   if missing:reg['box']=None
   x=torch.cat([PoseFiveViews([row],str(root),regions)[0]['x'],tm.TemporalHalves([row],str(root),regions)[0]['x']]);item=infer.BaseViews([sample],[reg])[0];assert torch.equal(x,item['mae'])
   v=torch.cat([x[:1],x[:1].flip(-1)])
   with torch.inference_mode():a=model(v);b=ref(v)
   assert torch.equal(a,b)and torch.isfinite(a).all();assert torch.equal(x.flip(-1).flip(-1),x)
   cases.append(dict(kind=kind,missing_pose_box=missing,source_input_exact=True,source_to_release_orig_and_mirror_maxabs=float((a-b).abs().max()),state_tensors=len(model.state_dict())))
  del model,ref,m
 # The source-stage CPU QA and preparation probe already cover all7/all5 aggregation.
 # This independent receipt identifies exactly the final calibration inputs without refitting.
 with np.load(s['source_npz'],allow_pickle=False)as ar:v={k:ar[k]for k in ar.files}
 z=np.concatenate([v['TI'],v['base_mixed'][:,None],v['small_mixed'][:,None]],1).astype(np.float32);mask=v['available'].astype(bool)
 atomic(E/'SOURCE_INPUT_RECEIPT.json',dict(source_fields=['TI','base_mixed','small_mixed'],source_rows=4768,source_npz_sha256=sha(s['source_npz']),source_logits_sha256=hashlib.sha256(z.tobytes()).hexdigest(),source_available_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),calibration_result_sha256=sha(E/'CALIBRATION_RESULT.json'),no_refit=True))
 r=dict(status='PASS_PAIRED_FULL18_SOURCE_CPU_PARITY',protocol_sha256=sha(E/'source_qa_protocol.json'),checks=cases,no_GPU=True,no_Test=True,optimizer_updates=0,seconds=time.time()-start);atomic(E/'SOURCE_QA_RESULT.json',r);atomic(E/'SOURCE_QA_DONE.json',dict(status='PASS',protocol_sha256=sha(E/'source_qa_protocol.json'),artifacts={n:sha(E/n)for n in ['SOURCE_QA_RESULT.json','SOURCE_INPUT_RECEIPT.json']}));print(json.dumps(r))
if __name__=='__main__':main()
