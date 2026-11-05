"""Single fixed Base6+Small8 tensor container; FP32 row scales, other FP16.

No compression or reduced metadata precision. Tensor names/shapes are explicit;
byte offsets are implicit from the declared fixed encoding, with no zip padding.
"""
from pathlib import Path
import json,struct,hashlib
import numpy as np
import torch

MAGIC=b'V2PAIR6S8V1\0'
FORMAT='VIDEOMAEV2_BASE6_SMALL8_ALL50_FP32ROWS_OTHERSFP16_V1'

def quantized(value,bits):
 assert bits in (6,8) and value.dtype==torch.float16 and value.ndim>=2
 limit=31 if bits==6 else 127;x=value.float();shape=[x.shape[0]]+[1]*(x.ndim-1)
 scale=x.flatten(1).abs().amax(1).clamp_min(1e-12).div(limit).reshape(shape)
 q=(x/scale).round().clamp(-limit,limit).to(torch.int8)
 return q,scale.reshape(-1).contiguous()

def pack6(q):
 values=q.numpy().reshape(-1).astype(np.int16)+31
 assert values.min()>=0 and values.max()<=62
 n=len(values);pad=(-n)%4
 if pad:values=np.pad(values,(0,pad),constant_values=31)
 a=values.astype(np.uint16).reshape(-1,4)
 b=np.empty((len(a),3),np.uint8)
 b[:,0]=(a[:,0]|(a[:,1]<<6))&255
 b[:,1]=((a[:,1]>>2)|(a[:,2]<<4))&255
 b[:,2]=((a[:,2]>>4)|(a[:,3]<<2))&255
 return b.tobytes()

def unpack6(raw,n):
 p=np.frombuffer(raw,np.uint8).reshape(-1,3).astype(np.uint16)
 a=np.empty((len(p),4),np.int16)
 a[:,0]=p[:,0]&63;a[:,1]=(p[:,0]>>6)|((p[:,1]&15)<<2)
 a[:,2]=(p[:,1]>>4)|((p[:,2]&3)<<4);a[:,3]=p[:,2]>>2
 a=a.reshape(-1)[:n];assert (a<=62).all()
 return (a-31).astype(np.int8)

def write_pair(path,base_cp,small_cp,provenance):
 path=Path(path);assert not path.exists();sources=[('Base',6,base_cp),('Small',8,small_cp)]
 header=dict(format=FORMAT,models=[],provenance=provenance)
 for label,bits,cp in sources:
  specs=[];count=0
  for name,v in cp['model'].items():
   assert v.device.type=='cpu' and v.dtype==torch.float16 and torch.isfinite(v).all()
   b=bits if name.endswith('.weight') and v.ndim>=2 else 16
   specs.append([name,list(v.shape),b]);count+=b!=16
  assert count==50;header['models'].append(dict(name=label,tensors=specs))
 raw=json.dumps(header,separators=(',',':'),ensure_ascii=True).encode();payload=0;checks=[]
 with path.open('xb')as out:
  out.write(MAGIC);out.write(struct.pack('<I',len(raw)));out.write(raw)
  for label,bits,cp in sources:
   for name,v in cp['model'].items():
    if name.endswith('.weight') and v.ndim>=2:
     q,scale=quantized(v,bits);sc=scale.numpy().astype('<f4',copy=False).tobytes()
     encoded=pack6(q) if bits==6 else q.numpy().tobytes()
     if bits==6:assert np.array_equal(unpack6(encoded,q.numel()),q.numpy().reshape(-1))
     out.write(sc);out.write(encoded);payload+=len(sc)+len(encoded)
     checks.append(dict(model=label,tensor=name,scale_sha256=hashlib.sha256(sc).hexdigest(),q_sha256=hashlib.sha256(q.numpy().tobytes()).hexdigest()))
    else:
     encoded=v.numpy().astype('<f2',copy=False).tobytes();out.write(encoded);payload+=len(encoded)
     checks.append(dict(model=label,tensor=name,value_sha256=hashlib.sha256(encoded).hexdigest()))
 return dict(header_bytes=len(MAGIC)+4+len(raw),tensor_payload_bytes=payload,file_bytes=path.stat().st_size,tensor_checks=checks)

def verify_pair(path,checks):
 expected={(x['model'],x['tensor']):x for x in checks};seen=0
 with Path(path).open('rb')as f:
  assert f.read(len(MAGIC))==MAGIC;n=struct.unpack('<I',f.read(4))[0];h=json.loads(f.read(n));assert h['format']==FORMAT
  for model in h['models']:
   for name,shape,bits in model['tensors']:
    count=int(np.prod(shape));old=expected[model['name'],name]
    if bits!=16:
     scales=f.read(shape[0]*4);assert hashlib.sha256(scales).hexdigest()==old['scale_sha256']
     raw=f.read(((count+3)//4)*3 if bits==6 else count)
     q=unpack6(raw,count) if bits==6 else np.frombuffer(raw,np.int8)
     assert hashlib.sha256(q.tobytes()).hexdigest()==old['q_sha256']
    else:
     raw=f.read(count*2);assert hashlib.sha256(raw).hexdigest()==old['value_sha256']
    seen+=1
  assert not f.read(1)
 assert seen==len(expected);return dict(status='PASS_LOSSLESS_CONTAINER_ROUNDTRIP',tensors=seen,scope='Exact q/scales/FP16 values; not evidence of quantization accuracy')
