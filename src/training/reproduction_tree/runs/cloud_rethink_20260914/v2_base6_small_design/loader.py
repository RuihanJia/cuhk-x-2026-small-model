"""Decode one declared model from the fixed pair, preserving quantized values."""
from pathlib import Path
import json,struct
import numpy as np
import torch
from pair_codec import MAGIC,FORMAT,unpack6

def load_state(path,wanted):
 assert wanted in ('Base','Small');state={}
 with Path(path).open('rb')as f:
  assert f.read(len(MAGIC))==MAGIC
  h=json.loads(f.read(struct.unpack('<I',f.read(4))[0]));assert h['format']==FORMAT
  for model in h['models']:
   for name,shape,bits in model['tensors']:
    n=int(np.prod(shape));size=n*2 if bits==16 else shape[0]*4+(((n+3)//4)*3 if bits==6 else n)
    if model['name']!=wanted:f.seek(size,1);continue
    if bits==16:
     v=np.frombuffer(f.read(size),dtype='<f2').copy().reshape(shape);state[name]=torch.from_numpy(v).float()
    else:
     scales=np.frombuffer(f.read(shape[0]*4),dtype='<f4').copy();raw=f.read(size-shape[0]*4)
     q=unpack6(raw,n) if bits==6 else np.frombuffer(raw,np.int8).copy()
     state[name]=torch.from_numpy(q.reshape(shape)).float()*torch.from_numpy(scales).reshape(shape[0],*([1]*(len(shape)-1)))
  assert not f.read(1)
 assert len(state)==162 and all(torch.isfinite(v).all() for v in state.values())
 return state,h['provenance']
