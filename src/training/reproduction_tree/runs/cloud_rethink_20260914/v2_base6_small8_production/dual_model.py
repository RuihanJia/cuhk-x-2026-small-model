"""Production constructors for K710-distilled VideoMAEv2 Base and Small."""
import types
import numpy as np
import torch
from torch import nn
from mae_model import author_module,efficient_attention

class PairVideo(nn.Module):
 def __init__(self,kind):
  super().__init__();assert kind in ('Base','Small');author=author_module();ctor=author.vit_base_patch16_224 if kind=='Base' else author.vit_small_patch16_224
  self.video=ctor(num_classes=40,all_frames=16,drop_rate=0.,attn_drop_rate=0.,drop_path_rate=0.,use_mean_pooling=True)
  heads=12 if kind=='Base' else 6
  assert all(x.attn.num_heads==heads and x.norm1.eps==x.norm2.eps==1e-6 for x in self.video.blocks)
  for x in self.video.blocks:x.attn.forward=types.MethodType(efficient_attention,x.attn)
  self.register_buffer('mean',torch.tensor([.485,.456,.406]).reshape(1,3,1,1,1),persistent=False)
  self.register_buffer('std',torch.tensor([.229,.224,.225]).reshape(1,3,1,1,1),persistent=False)
  assert sum(p.numel()for p in self.parameters())==({'Base':86257960,'Small':21895336}[kind])
 def forward(self,x):
  assert x.shape[1:]==(3,16,224,224);pixels=(x.float()*.225+.45).clamp(0,1);return self.video((pixels-self.mean)/self.std)
