"""Author VideoMAE-Small K400; only public pretraining source changes.

Upstream source is retained with its CC-BY-NC-4.0 license and exact hash.
Use the author's 6 heads and eps1e-6, not the inconsistent HF JSON16/1e-12.
No pretrained decoder or language component participates in this classifier.
"""
from pathlib import Path
import hashlib,importlib.util,sys,types
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent
WEIGHT_SHA='d69585904fec7e507bf2edfba4a7abe2b92def9afc76460d1dc14dbbf864bcfc'
AUTHOR_SHA='ce9a11ff05c7a1b5085addf6c73b71ffb68933564164f7b250dedad91f6e29b8'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def drop_path(x,drop_prob=0.,training=False):
 if not drop_prob or not training:return x
 keep=1-drop_prob;shape=(x.shape[0],)+(1,)*(x.ndim-1)
 mask=(keep+torch.rand(shape,dtype=x.dtype,device=x.device)).floor_()
 return x/keep*mask

def author_module():
 path=ROOT/'upstream_modeling_finetune.py';assert sha(path)==AUTHOR_SHA
 # Only three simple timm helpers are needed by this retained author file.
 # Temporary modules never change the installed environment or other workers.
 names=['timm','timm.models','timm.models.layers','timm.models.registry']
 prior={n:sys.modules.get(n) for n in names}
 try:
  for n in names:sys.modules[n]=types.ModuleType(n)
  layers=sys.modules['timm.models.layers'];layers.drop_path=drop_path;layers.to_2tuple=lambda x:x if isinstance(x,tuple) else (x,x);layers.trunc_normal_=nn.init.trunc_normal_
  sys.modules['timm.models.registry'].register_model=lambda f:f
  spec=importlib.util.spec_from_file_location('retained_author_videomae',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 finally:
  for n,v in prior.items():
   if v is None:sys.modules.pop(n,None)
   else:sys.modules[n]=v
 return module

def efficient_attention(self,x):
 b,n,c=x.shape
 bias=torch.cat([self.q_bias,torch.zeros_like(self.v_bias,requires_grad=False),self.v_bias])
 qkv=F.linear(x,self.qkv.weight,bias).reshape(b,n,3,self.num_heads,-1).permute(2,0,3,1,4)
 q,k,v=qkv.unbind(0)
 # Same noncausal attention as the author; no materialized N x N training map.
 output=F.scaled_dot_product_attention(q,k,v,dropout_p=self.attn_drop.p if self.training else 0.,is_causal=False,scale=self.scale)
 output=output.transpose(1,2).reshape(b,n,-1)
 return self.proj_drop(self.proj(output))

class VideoMAEBase(nn.Module):
 def __init__(self,classes=40,fast=True,seed=20260915):
  super().__init__();torch.manual_seed(seed);np.random.seed(seed)
  author=author_module()
  assert classes==40
  self.video=author.vit_base_patch16_224(num_classes=40,all_frames=16,drop_rate=0.,attn_drop_rate=0.,drop_path_rate=0.,use_mean_pooling=True)
  assert all(block.attn.num_heads==12 and block.norm1.eps==1e-6 for block in self.video.blocks)
  if fast:
   for block in self.video.blocks:block.attn.forward=types.MethodType(efficient_attention,block.attn)
  self.register_buffer('mean',torch.tensor([.485,.456,.406]).reshape(1,3,1,1,1),persistent=False)
  self.register_buffer('std',torch.tensor([.229,.224,.225]).reshape(1,3,1,1,1),persistent=False)
  self.fast=fast;self.classes=classes
 def forward(self,x):
  assert x.shape[1:]==(3,16,224,224)
  # Reuse the already-audited source clip/flip geometry; change only normalization.
  pixels=(x.float()*.225+.45).clamp(0,1)
  return self.video((pixels-self.mean)/self.std)
