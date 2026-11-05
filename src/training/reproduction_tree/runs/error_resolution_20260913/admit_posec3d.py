from pathlib import Path
import json,time,torch,sys,ast,__future__,hashlib
from torch import nn
from torch.nn.modules.utils import _ntuple,_triple
from torch.nn.modules.batchnorm import _BatchNorm
R=Path(__file__).resolve().parent;A=R/'assets/posec3d'
from posec3d_model import PoseC3D,ConvModule
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class BaseModule(nn.Module):
 def __init__(self,init_cfg=None):super().__init__()
def main():
 torch.set_num_threads(2);cp=torch.load(A/'source_checkpoint.pth',map_location='cpu',weights_only=True);sd=cp['state_dict'];m=PoseC3D();m.load_state_dict(sd,strict=True);m.eval()
 tree=ast.parse((A/'resnet3d.py').read_text(encoding='utf-8'));nodes=[]
 for node in tree.body:
  if isinstance(node,ast.ClassDef) and node.name in ['BasicBlock3d','Bottleneck3d','ResNet3d']:
   node.decorator_list=[];nodes.append(node)
 env=dict(torch=torch,nn=nn,Sequential=nn.Sequential,BaseModule=BaseModule,ConvModule=ConvModule,_ntuple=_ntuple,_triple=_triple,_BatchNorm=_BatchNorm,build_activation_layer=lambda d:nn.ReLU(inplace=d.get('inplace',False)))
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(A/'resnet3d.py'),'exec',flags=__future__.annotations.compiler_flag),env)
 reference=env['ResNet3d'](depth=50,in_channels=17,base_channels=32,num_stages=3,out_indices=(2,),stage_blocks=(4,6,3),conv1_kernel=(1,7,7),conv1_stride_s=1,pool1_stride_s=1,with_pool2=False,inflate=(0,1,1),spatial_strides=(2,2,2),temporal_strides=(1,1,2),dilations=(1,1,1));reference.load_state_dict({k.removeprefix('backbone.'):v for k,v in sd.items() if k.startswith('backbone.')},strict=True);reference.eval();torch.manual_seed(123);x=torch.rand(1,17,8,32,32)
 with torch.inference_mode():out=m.backbone(x);ref=reference(x)
 assert torch.equal(out,ref),float((out-ref).abs().max())
 torch.save(dict(state_dict={k:(v.half() if v.is_floating_point() else v) for k,v in sd.items()},architecture='POSEC3D_NTU60_2D_COCO17_SLOWONLY_2M'),A/'state_fp16.pt');q=PoseC3D();q.load_state_dict(torch.load(A/'state_fp16.pt',map_location='cpu',weights_only=True)['state_dict']);q.eval();assert all(torch.isfinite(v).all() for v in q.state_dict().values())
 qa=dict(status='PASS_STRICT_ALL_260_KEYS_AND_REFERENCE_GRAPH_BITWISE',official_reference_sha256=sha(A/'resnet3d.py'),native_code_sha256=sha(R/'posec3d_model.py'),QA_code_sha256=sha(__file__),reference_control_flow_verbatim=True,shared_primitive='Conv3d+BatchNorm3d+ReLU shims; noMMCVruntime',input_shape=list(x.shape),output_shape=list(out.shape),max_abs_error=float((out-ref).abs().max()),weights_loaded_safely=True,at_unix=time.time());(A/'QA.json').write_text(json.dumps(qa,indent=2))
 admit=dict(status='ADMITTED_PUBLIC_NONLANGUAGE_NTU60_SKELETON_ACTION',source=json.loads((A/'FETCH_PLAN.json').read_text())['checkpoint_url'],upstream_project='OpenMMLab MMAction2 official PoseC3D',license='Apache-2.0 project distribution; retain source and upstream license',pretraining='NTU RGB+D60 xsub official human action labels and COCO17 2D human pose; no text encoder or language teacher',use='Backbone features only, discard60classhead forTraining supervised fit; no manual semanticlabelmap',source_checkpoint_sha256=sha(A/'source_checkpoint.pth'),inference_state_sha256=sha(A/'state_fp16.pt'),inference_weight_bytes=(A/'state_fp16.pt').stat().st_size,all_tensor_parameters=sum(p.numel() for p in m.parameters()),QA_sha256=sha(A/'QA.json'),no_LLM_chain=True,disclosure_required=True,at_unix=time.time());(A/'ADMISSION.json').write_text(json.dumps(admit,indent=2));print(json.dumps(dict(QA=qa,admission=admit)))
if __name__=='__main__':main()
