"""Fixed storage-only symmetric int8 for every 2D VideoMAE linear weight."""
import torch
FORMAT='VIDEOMAE_ALL_LINEAR_OUTPUT_INT8_FP32_SCALE_OTHERS_FP16_V1'
def pack(cp):
 result={};names=[]
 for name,value in cp['model'].items():
  assert value.dtype==torch.float16
  if name.endswith('.weight') and value.ndim==2:
   x=value.float();scale=x.abs().amax(1,keepdim=True).clamp_min(1e-12)/127.
   result[name]=dict(q=(x/scale).round().clamp(-127,127).to(torch.int8),scale=scale);names.append(name)
  else:result[name]=dict(value=value)
 assert len(names)==49 and 'video.head.weight' in names
 return dict(**{k:v for k,v in cp.items() if k!='model'},model=result,storage_format=FORMAT,quantized_names=names)
def unpack(cp):
 assert cp['storage_format']==FORMAT
 return {k:(v['q'].float()*v['scale'] if 'q' in v else v['value'].float()) for k,v in cp['model'].items()}
