"""Compact spatial attention and ordered temporal readout of frozen visual tokens."""
import torch
from torch import nn

class DetailHead(nn.Module):
    def __init__(self, ordered=True):
        super().__init__();self.ordered=ordered
        self.project=nn.Sequential(nn.LayerNorm(384),nn.Linear(384,96),nn.GELU())
        self.spatial_position=nn.Parameter(torch.zeros(1,1,1,5,96))
        self.view_position=nn.Parameter(torch.zeros(1,4,1,1,96))
        self.spatial_gate=nn.Linear(96,1)
        if ordered:
            self.change=nn.Sequential(nn.Linear(288,96),nn.GELU())
            self.temporal=nn.Conv1d(96,96,3,padding=1,groups=96)
        self.temporal_gate=nn.Linear(96,1)
        self.view_gate=nn.Linear(96,1)
        self.dropout=nn.Dropout(.3)
        self.classifier=nn.Linear(96,40)
        nn.init.normal_(self.spatial_position,std=.01);nn.init.normal_(self.view_position,std=.01)
    def forward(self,x,available,drop_modalities=False):
        h=self.project(x)+self.spatial_position+self.view_position
        h=(h*self.spatial_gate(h).softmax(-2)).sum(-2)
        if self.ordered:
            delta=torch.cat([torch.zeros_like(h[:,:,:1]),h[:,:,1:]-h[:,:,:-1]],2)
            h=h+self.change(torch.cat([h,delta,delta.abs()],-1))
            t=self.temporal(h.flatten(0,1).transpose(1,2)).transpose(1,2).reshape_as(h)
            h=h+t
        h=(h*self.temporal_gate(h).softmax(-2)).sum(-2)
        v=self.classifier(self.dropout(h));g=self.view_gate(h).squeeze(-1)
        mask=available.clone()
        if self.training and drop_modalities:
            keep=torch.rand(mask.shape,device=mask.device)>.15
            # Keep at least one genuinely available view for every record.
            keep.scatter_(1,available.float().argmax(1,keepdim=True),True)
            mask=mask & keep
        g=g.masked_fill(~mask,-10000).softmax(-1)
        return (v*g.unsqueeze(-1)).sum(1),v,g
