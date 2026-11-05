"""Private Thermal adaptation on the single frozen R6 visual backbone."""
import copy,math
import torch
from torch import nn
from torch.nn import functional as F
class LowRankLinear(nn.Module):
 def __init__(self,base,rank=8):
  super().__init__();self.base=base;self.base.requires_grad_(False);self.rank=rank;self.active=True
  self.A=nn.Parameter(torch.empty(rank,base.in_features));self.B=nn.Parameter(torch.zeros(base.out_features,rank));nn.init.kaiming_uniform_(self.A,a=math.sqrt(5))
 def forward(self,x):
  y=self.base(x)
  return y+F.linear(F.linear(x,self.A),self.B) if self.active else y
class ThermalAdapter(nn.Module):
 def __init__(self,base,rank=8):
  super().__init__();self.base=base;self.base.eval().requires_grad_(False)
  self.thermal_head=copy.deepcopy(base.video.head).requires_grad_(True)
  self.thermal_readout=copy.deepcopy(base.readout).requires_grad_(True)
  self.adapters=[]
  for block in list(self.base.video.blocks)[-4:]:
   block.attn.qkv=LowRankLinear(block.attn.qkv,rank)
   block.attn.project[0]=LowRankLinear(block.attn.project[0],rank)
   self.adapters.extend([block.attn.qkv,block.attn.project[0]])
 def train(self,mode=True):
  super().train(mode);self.base.eval();self.thermal_head.train(mode);self.thermal_readout.train(mode)
  return self
 def forward(self,x,thermal=True):
  for a in self.adapters:a.active=thermal
  if not thermal:return self.base(x)
  captured=[]
  h=self.base.video.norm.register_forward_hook(lambda m,a,o:captured.append(o))
  try:self.base.video(x)
  finally:h.remove()
  assert len(captured)==1 and captured[0].shape[1:]==(393,768)
  tokens=captured[0]
  return self.thermal_head(tokens[:,0])+self.thermal_readout(tokens[:,1:])
 def private_state(self):
  return {k:v.detach().cpu().half() for k,v in self.named_parameters() if v.requires_grad}
 def load_private(self,state):
  expected={k for k,p in self.named_parameters() if p.requires_grad};assert set(state)==expected
  params=dict(self.named_parameters())
  with torch.no_grad():
   for k,v in state.items():
    assert v.shape==params[k].shape and torch.isfinite(v).all();params[k].copy_(v)
 def frozen_digest(self):
  import hashlib
  h=hashlib.sha256()
  for k,p in self.named_parameters():
   if not p.requires_grad:h.update(k.encode());h.update(p.detach().cpu().numpy().tobytes())
  return h.hexdigest()
