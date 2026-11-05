"""One fixed storage quantizer; computations remain FP32 after loading."""
import torch
NAMES=[f'video.blocks.{b}.mlp.{n}.weight' for b in [14,15] for n in [0,3]]
def pack(cp):
    state={}
    for k,v in cp['state_dict'].items():
        if k in NAMES:
            x=v.float();scale=x.abs().amax(1,keepdim=True).clamp_min(1e-12)/127.;state[k]=dict(q=(x/scale).round().clamp(-127,127).to(torch.int8),scale=scale)
        else:state[k]=dict(value=v)
    assert all(k in state and 'q' in state[k] for k in NAMES)
    return dict(**{k:v for k,v in cp.items() if k!='state_dict'},state_dict=state,storage_format='FP16_EXCEPT_LAST_TWO_MLP_OUTPUT_CHANNEL_INT8_FP32_SCALE')
def unpack(cp):return {k:(v['q'].float()*v['scale'] if 'q' in v else v['value'].float()) for k,v in cp['state_dict'].items()}
