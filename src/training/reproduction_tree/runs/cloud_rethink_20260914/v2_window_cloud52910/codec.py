"""Fixed per-output-row int8 storage for all 49 linear weights and patch Conv3d."""
import torch
FORMAT='VIDEOMAE_BASE_ALL_50_WEIGHTS_OUTPUT_INT8_FP32_SCALE_OTHERS_FP16_V1'

def pack(cp):
 result={};names=[]
 for name,value in cp['model'].items():
  assert value.dtype==torch.float16 and torch.isfinite(value).all()
  if name.endswith('.weight') and value.ndim>=2:
   x=value.float();flat=x.reshape(x.shape[0],-1)
   scale=(flat.abs().amax(1).clamp_min(1e-12)/127.).reshape(x.shape[0],*([1]*(x.ndim-1)))
   result[name]=dict(q=(x/scale).round().clamp(-127,127).to(torch.int8),scale=scale);names.append(name)
  else:result[name]=dict(value=value)
 assert len(names)==50 and 'video.head.weight' in names and 'video.patch_embed.proj.weight' in names
 return dict(**{k:v for k,v in cp.items() if k!='model'},model=result,storage_format=FORMAT,quantized_names=names)

def unpack(cp):
 assert cp['storage_format']==FORMAT and len(cp['quantized_names'])==50
 return {k:(v['q'].float()*v['scale'] if 'q' in v else v['value'].float()) for k,v in cp['model'].items()}
