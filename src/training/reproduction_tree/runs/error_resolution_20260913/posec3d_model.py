"""Native PyTorch rendering of the official 2M-parameter PoseC3D SlowOnly.

Adapted from OpenMMLab MMAction2 (Apache-2.0); source and license are retained
in assets/posec3d. State names intentionally match the official checkpoint.
"""
import torch
from torch import nn

class ConvModule(nn.Module):
 def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,bias=False,conv_cfg=None,norm_cfg=None,act_cfg=None,activate=None):
  super().__init__();self.conv=nn.Conv3d(in_channels,out_channels,kernel_size,stride,padding,dilation,bias=bias);self.bn=nn.BatchNorm3d(out_channels);self.activate=bool(act_cfg is not None) if activate is None else activate
 def forward(self,x):
  x=self.bn(self.conv(x));return x.relu() if self.activate else x

class Bottleneck(nn.Module):
 def __init__(self,inplanes,planes,spatial_stride,temporal_stride,inflate):
  super().__init__();self.conv1=ConvModule(inplanes,planes,(3 if inflate else 1,1,1),padding=(1 if inflate else 0,0,0),activate=True);self.conv2=ConvModule(planes,planes,(1,3,3),stride=(temporal_stride,spatial_stride,spatial_stride),padding=(0,1,1),activate=True);self.conv3=ConvModule(planes,planes*4,1,activate=False)
  self.downsample=ConvModule(inplanes,planes*4,1,stride=(temporal_stride,spatial_stride,spatial_stride),activate=False) if inplanes!=planes*4 or spatial_stride!=1 or temporal_stride!=1 else None
 def forward(self,x):return (self.conv3(self.conv2(self.conv1(x)))+(self.downsample(x) if self.downsample is not None else x)).relu()

class Backbone(nn.Module):
 def __init__(self):
  super().__init__();self.conv1=ConvModule(17,32,(1,7,7),padding=(0,3,3),activate=True);self.maxpool=nn.MaxPool3d((1,3,3),stride=(1,1,1),padding=(0,1,1));ci=32
  for stage,(planes,blocks,inflate) in enumerate([(32,4,False),(64,6,True),(128,3,True)],1):
   layers=[]
   for b in range(blocks):layers.append(Bottleneck(ci,planes,2 if b==0 else 1,2 if stage==3 and b==0 else 1,inflate));ci=planes*4
   setattr(self,f'layer{stage}',nn.Sequential(*layers))
 def forward(self,x):return self.layer3(self.layer2(self.layer1(self.maxpool(self.conv1(x)))))

class Head(nn.Module):
 def __init__(self,classes):super().__init__();self.dropout=nn.Dropout(.5);self.fc_cls=nn.Linear(512,classes)
 def forward(self,x):return self.fc_cls(self.dropout(x))

class PoseC3D(nn.Module):
 def __init__(self,classes=60):super().__init__();self.backbone=Backbone();self.cls_head=Head(classes)
 def features(self,x):return self.backbone(x).mean((2,3,4))
 def forward(self,x):return self.cls_head(self.features(x))
