"""Independent CPU-only storage audit; no labels, model forward, or Test."""
from pathlib import Path
import os,json,struct,hashlib,sys,importlib.util,argparse,time
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import numpy as np
import torch
E=Path(__file__).resolve().parent;C=E.parent;R=C.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m

def run(container,codec_dir,output):
 torch.set_num_threads(1);start=time.time();codec_dir=Path(codec_dir);sys.path.insert(0,str(codec_dir))
 codec=module(codec_dir/'pair_codec.py','pair_codec');loader=module(codec_dir/'loader.py','audited_pair_loader')
 bp=C/'videomaev2_base/cloud/V2_BASE_F3/model_fp16.pt'
 sp=C/'videomaev2_small/output/local/model_fp16.pt'
 sqp=C/'videomaev2_quant/output/local/V2_OUTER_F3/model_int8.pt'
 b=torch.load(bp,map_location='cpu',weights_only=True);s=torch.load(sp,map_location='cpu',weights_only=True);old=torch.load(sqp,map_location='cpu',weights_only=True)
 assert b['train_users']==s['train_users']==old['train_users'] and b['held_users']==s['held_users']==old['held_users']
 assert b['epochs']==s['epochs']==old['epochs']==16 and set(b['train_users']).isdisjoint(b['held_users'])
 raw=Path(container).read_bytes();offset=len(codec.MAGIC);assert raw[:offset]==codec.MAGIC
 n=struct.unpack_from('<I',raw,offset)[0];offset+=4;header=json.loads(raw[offset:offset+n]);offset+=n;header_bytes=offset
 assert header['format']==codec.FORMAT
 source={'Base':b,'Small':s};records=[];qnames=[];states={};boundary_samples=0
 for model in header['models']:
  label=model['name'];assert label in source;cp=source[label];loaded,_=loader.load_state(container,label);states[label]=loaded
  assert set(loaded)==set(cp['model'])
  for name,shape,bits in model['tensors']:
   value=cp['model'][name];assert tuple(shape)==tuple(value.shape);count=value.numel();record=dict(model=label,name=name,bits=bits,offset=offset)
   if bits==16:
    size=count*2;data=np.frombuffer(raw[offset:offset+size],'<f2').copy().reshape(shape);offset+=size
    assert np.array_equal(data,value.numpy()) and torch.equal(loaded[name],value.float())
   else:
    assert bits==(6 if label=='Base' else 8);limit=31 if bits==6 else 127
    scales=np.frombuffer(raw[offset:offset+shape[0]*4],'<f4').copy();offset+=shape[0]*4
    size=3*((count+3)//4) if bits==6 else count;payload=raw[offset:offset+size];offset+=size
    # Independent direct FP16 -> numpy ties-to-even quantization.
    w=value.numpy().astype(np.float32).reshape(shape[0],-1)
    expected_scale=np.maximum(np.abs(w).max(1),np.float32(1e-12))/np.float32(limit)
    expected_q=np.clip(np.rint(w/expected_scale[:,None]),-limit,limit).astype(np.int8).reshape(-1)
    assert np.array_equal(scales,expected_scale)
    if bits==6:
     q=codec.unpack6(payload,count);assert np.array_equal(q,expected_q)
     # Scalar generic little-endian six-bit reader at every row boundary plus interior probes.
     row=w.shape[1];ix={0,count-1}
     for j in range(shape[0]):
      ix.update([j*row,min(count-1,j*row+1),(j+1)*row-1])
     for j in sorted(ix):
      bit=j*6;byte=bit//8;word=int.from_bytes(payload[byte:byte+2],'little');v=((word>>(bit%8))&63)-31
      assert v==int(expected_q[j]);boundary_samples+=1
    else:
     q=np.frombuffer(payload,np.int8).copy();assert np.array_equal(q,expected_q)
     if name in old['quantized_names']:
      assert np.array_equal(q,old['model'][name]['q'].numpy().reshape(-1))
      assert np.array_equal(scales,old['model'][name]['scale'].numpy().reshape(-1));qnames.append(name)
     else:assert name=='video.patch_embed.proj.weight'
    actual=torch.from_numpy(q.reshape(shape)).float()*torch.from_numpy(scales).reshape(shape[0],*([1]*(len(shape)-1)))
    assert torch.equal(actual,loaded[name]);record.update(scale_sha256=hashlib.sha256(scales.tobytes()).hexdigest(),integer_sha256=hashlib.sha256(q.tobytes()).hexdigest())
   record['end_offset']=offset;records.append(record)
 assert offset==len(raw) and len(records)==324 and len(qnames)==49
 # Padding and non-aligned row examples exercise every represented signed code.
 cases=0
 for rows in [1,2,3,5]:
  for cols in [1,2,3,4,5,7,63,65]:
   q=(torch.arange(rows*cols).remainder(63)-31).to(torch.int8).reshape(rows,cols)
   payload=codec.pack6(q);decoded=codec.unpack6(payload,q.numel());assert np.array_equal(decoded,q.numpy().reshape(-1))
   assert len(payload)==3*((q.numel()+3)//4)
   for j,v in enumerate(q.reshape(-1).tolist()):
    bit=j*6;word=int.from_bytes(payload[bit//8:bit//8+2],'little');assert ((word>>(bit%8))&63)-31==v
   cases+=1
 retained=R/'deliverables/videomaev2_base_temporal_halves_candidate_20260915/checkpoints'
 aux={n:dict(bytes=(retained/n).stat().st_size,sha256=sha(retained/n))for n in ['T.pt','I.pt','POSE.pt']}
 total=len(raw)+sum(v['bytes']for v in aux.values());assert total+12000<=100000000
 result=dict(status='PASS_INDEPENDENT_CPU_CONTAINER_AUDIT',container=str(Path(container).resolve()),container_sha256=sha(container),codec_sha256=sha(codec_dir/'pair_codec.py'),loader_sha256=sha(codec_dir/'loader.py'),parent_sha256={'Base':sha(bp),'Small':sha(sp),'Small49':sha(sqp)},header_bytes=header_bytes,tensor_payload_bytes=len(raw)-header_bytes,container_bytes=len(raw),retained=aux,total_model_bytes_without_calibrator=total,calibrator_reserve=12000,total_with_reserve=total+12000,remaining_after_reserve=100000000-total-12000,FP32_scales_bitexact=True,all324_state_tensors_strictly_replayed=True,small_original49_q_scales_bitexact=True,small_only_new_quantized_tensor='video.patch_embed.proj.weight',scalar_sixbit_boundary_checks=boundary_samples,independent_padding_cases=cases,tensor_records=records,no_GPU=True,no_Test=True,no_forward=True,seconds=time.time()-start,at_unix=time.time(),limitation='F3 real container; actual full18 size and calibrator must still be measured. Equal storage weights establish identical mathematics, not GPU kernel/output parity.')
 Path(output).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items()if k!='tensor_records'}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--container',required=True);p.add_argument('--codec-dir',required=True);p.add_argument('--output',required=True);run(**vars(p.parse_args()))
