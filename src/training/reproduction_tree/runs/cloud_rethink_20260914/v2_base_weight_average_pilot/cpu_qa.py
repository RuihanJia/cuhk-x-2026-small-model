"""Bounded CPU-only verification of actual local parents, codec and one source clip."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util,json,time
from pathlib import Path
import worker as w
def main():
 start=time.time();w.torch.set_num_threads(2);C=w.C
 raw=C/'videomaev2_base/output/local/V2_BASE_SOURCE_F3_INNER2/model_fp16.pt';control=C/'v2_window_resume/output/local_v2/V2_BASE_SOURCE_F3_INNER2_CONTROL/model_fp16.pt'
 assert w.sha(raw)=='335807918a1bc7762bc4ce281c0573291e938ee91dbc2ef0c328ed7f964d10c4'
 assert w.sha(control)=='d6f4e67acdfee13a93a2a52e247d3b837bbece19f4f89cdffe488622e38ae785'
 a=w.torch.load(raw,map_location='cpu',weights_only=True);b=w.torch.load(control,map_location='cpu',weights_only=True)
 assert a['train_users']==b['train_users'] and a['held_users']==b['held_users'] and a['epochs']==16 and b['epochs']==4
 x,changed=w.average_state(a['model'],b['model']);assert all(w.torch.equal(x[k],((a['model'][k].float()+b['model'][k].float())*.5).half())for k in x)
 quant=w.pack(dict(model=x));path=w.E/'CPU_QA_AVERAGE_INT8.pt';assert not path.exists();w.save_torch(path,quant)
 deq=w.unpack(quant);loaded=w.unpack(w.torch.load(path,map_location='cpu',weights_only=True));assert all(w.torch.equal(deq[k],loaded[k])for k in deq)
 inp=w.read(w.O/'POSE_ADAPTER_INPUTS.json');rows=[r for r in inp['rows']if r['user']in a['train_users']];row=rows[0]
 spec=importlib.util.spec_from_file_location('old_window_data_cpu_qa',C/'v2_window_resume/data.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 root='D:/Datasets/CUHK-X_Small_Model_Track/Small-Model-Track';new=w.SevenViews([row],root,inp['regions'])[0];prior=old.SevenViews([row],root,inp['regions'])[0]
 assert new.keys()==prior.keys()
 for k,v in new.items():
  if w.torch.is_tensor(v):assert w.torch.equal(v,prior[k]),k
  else:assert v==prior[k],k
 result=dict(status='PASS_FIXED_AVERAGE_CPU_QA',raw_sha256=w.sha(raw),control_sha256=w.sha(control),tensors=len(x),parameters=sum(v.numel()for v in x.values()),changed_parent_tensors=changed,average_FP16_exact=True,quantized_matrices=len(quant['quantized_names']),INT8_disk_reload_exact=True,checkpoint_bytes=path.stat().st_size,qa_checkpoint_sha256=w.sha(path),new_seven_view_pixels_match_previous=True,source_clip_id=row['clip_id'],train_users=a['train_users'],held_users=a['held_users'],GPU_forwards=0,optimizer_updates=0,no_Test=True,seconds=time.time()-start)
 w.atomic(w.E/'CPU_QA.json',result);print(json.dumps(result))
if __name__=='__main__':main()
