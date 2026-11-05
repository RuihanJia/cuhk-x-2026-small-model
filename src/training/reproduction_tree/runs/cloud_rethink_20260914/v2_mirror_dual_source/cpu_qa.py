from pathlib import Path
import sys,time,json,copy,importlib.util
E=Path(__file__).resolve().parent;C=E.parent;sys.path.insert(0,str(E))
import worker as w
def module(path,name):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def main():
 start=time.time();w.torch.set_num_threads(4);inp=w.read(C.parent/'error_resolution_20260913/POSE_ADAPTER_INPUTS.json');rows=[r for r in inp['rows']if r['user']==4][:1];assert len(rows)==1
 j=next(x for x in w.read(C/'v2_base6_small8_source_completion/INPUTS.json')['jobs']if x['id']=='V2_BASE_SOURCE_F3_INNER2');cp=j['container'];w.check(cp['path'],cp['sha256']);result=[]
 for no_roi in [False,True]:
  region=copy.deepcopy(inp['regions'])
  if no_roi:region[rows[0]['clip_id']][0]['box']=None
  x=w.Views(rows,'D:/Datasets/CUHK-X_Small_Model_Track/Small-Model-Track',region)[0]['x'];y=w.horizontal_mirror(x)
  assert w.torch.equal(w.horizontal_mirror(y),x);assert w.np.array_equal(y.numpy(),x.numpy()[...,::-1]);assert x.shape==(7,3,16,224,224)
  result.append(dict(no_roi=no_roi,pixels_mirror_numpy_exact=True,double_flip_exact=True,shape=list(x.shape)))
  if no_roi:continue
  for kind in ['Base','Small']:
   state,origin=w.load_state(cp['path'],kind);model=w.PairVideo(kind).eval();model.load_state_dict(state,strict=True)
   code=C/('videomaev2_base/model.py'if kind=='Base'else'videomaev2_small/model.py');m=module(code,'mirror_qa_source_'+kind);old=getattr(m,'VideoMAE'+kind)(classes=40,fast=True,seed=20260915).eval();old.load_state_dict(state,strict=True)
   assert all(w.torch.equal(v,state[k])for k,v in model.state_dict().items())
   with w.torch.inference_mode():a=model(x[:1]);b=old(x[:1]);c=model(y[:1]);d=old(y[:1])
   assert w.torch.equal(a,b)and w.torch.equal(c,d)and w.torch.isfinite(c).all()
   result[-1][kind]=dict(source_release_original_maxabs=float((a-b).abs().max()),source_release_mirror_maxabs=float((c-d).abs().max()),mirror_output_changes=float((a-c).abs().max()),state_tensors=len(state))
   del old,model,state,m
 w.atomic(E/'CPU_QA.json',dict(status='PASS_MIRROR_DUAL_SOURCE_PIXELS_AND_LOGITS',rows=[r['clip_id']for r in rows],container=cp,cases=result,optimizer_updates=0,no_Test=True,seconds=time.time()-start,at_unix=time.time()))
 print(json.dumps(result))
if __name__=='__main__':main()
