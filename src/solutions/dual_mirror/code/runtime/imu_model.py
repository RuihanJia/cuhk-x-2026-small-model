import torch
from torch import nn

class DeviceIMU(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.Sequential(nn.Conv1d(3,64,7,2,3),nn.GroupNorm(8,64),nn.GELU(),nn.Conv1d(64,128,5,2,2),nn.GroupNorm(8,128),nn.GELU(),nn.Conv1d(128,128,3,2,1),nn.GroupNorm(8,128),nn.GELU(),nn.AdaptiveAvgPool1d(1));self.head=nn.Sequential(nn.Linear(645,128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,40))
    def forward(self,x,valid):
        x=torch.where(valid[:,:,None,None],x,0.);z=self.encoder(x.flatten(0,1)).reshape(len(x),5,128);z=z*valid[:,:,None]
        return self.head(torch.cat([z.flatten(1),valid.float()],1))
