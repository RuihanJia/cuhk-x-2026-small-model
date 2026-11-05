"""Same frozen bit format; copy Small's original49 integer/scales directly."""
from pathlib import Path
import json,struct,hashlib
import torch
from pair_codec import MAGIC,FORMAT,quantized,pack6,verify_pair

def write_bound_pair(path,base_cp,small_cp,small_quant_cp,provenance):
 path=Path(path);assert not path.exists()
 assert len(small_quant_cp['quantized_names'])==49
 assert small_quant_cp['train_users']==small_cp['train_users']==base_cp['train_users']
 assert small_quant_cp['held_users']==small_cp['held_users']==base_cp['held_users']
 header=dict(format=FORMAT,models=[],provenance=provenance);checks=[];payload=0
 for label,bits,cp in [('Base',6,base_cp),('Small',8,small_cp)]:
  specs=[]
  for name,v in cp['model'].items():
   assert v.dtype==torch.float16 and v.device.type=='cpu' and torch.isfinite(v).all()
   specs.append([name,list(v.shape),bits if name.endswith('.weight') and v.ndim>=2 else 16])
  assert sum(a[2]!=16 for a in specs)==50;header['models'].append(dict(name=label,tensors=specs))
 raw=json.dumps(header,separators=(',',':'),ensure_ascii=True).encode()
 with path.open('xb')as f:
  f.write(MAGIC);f.write(struct.pack('<I',len(raw)));f.write(raw)
  for label,bits,cp in [('Base',6,base_cp),('Small',8,small_cp)]:
   for name,v in cp['model'].items():
    if name.endswith('.weight') and v.ndim>=2:
     if label=='Small' and name in small_quant_cp['quantized_names']:
      item=small_quant_cp['model'][name];q=item['q'];scale=item['scale'].reshape(-1)
      assert q.dtype==torch.int8 and scale.dtype==torch.float32 and q.shape==v.shape
     else:q,scale=quantized(v,bits)
     sc=scale.numpy().astype('<f4',copy=False).tobytes();encoded=pack6(q) if bits==6 else q.numpy().tobytes()
     f.write(sc);f.write(encoded);payload+=len(sc)+len(encoded)
     checks.append(dict(model=label,tensor=name,scale_sha256=hashlib.sha256(sc).hexdigest(),q_sha256=hashlib.sha256(q.numpy().tobytes()).hexdigest()))
    else:
     if label=='Small':assert torch.equal(small_quant_cp['model'][name]['value'],v)
     encoded=v.numpy().astype('<f2',copy=False).tobytes();f.write(encoded);payload+=len(encoded)
     checks.append(dict(model=label,tensor=name,value_sha256=hashlib.sha256(encoded).hexdigest()))
 return dict(file_bytes=path.stat().st_size,header_bytes=len(MAGIC)+4+len(raw),tensor_payload_bytes=payload,verification=verify_pair(path,checks),small49_direct_integer_copy=True)
