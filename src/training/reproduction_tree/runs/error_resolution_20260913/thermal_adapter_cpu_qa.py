"""Real source checkpoint checks before registering any adapter training."""
from pathlib import Path
import sys,json,time,hashlib
import torch
from torch import nn
from torchvision.models.video import mvit_v2_s
R=Path(__file__).resolve().parent;ROOT=R.parents[1];sys.path.insert(0,str(R/'bundle'))
from activity_token_readout import ActivityTokenModel
from thermal_adapter_model import ThermalAdapter
torch.set_num_threads(2);torch.manual_seed(20260913);out=R/'thermal_engineering';out.mkdir(exist_ok=True)
p=ROOT/'runs/activity_token_stopgrad_20260913/cloud_mirror/output/C_TOKEN_STOPGRAD_F3_AUX025/model_fp16.pt'
cp=torch.load(p,map_location='cpu',weights_only=True);v=mvit_v2_s(weights=None);v.head=nn.Linear(768,40);base=ActivityTokenModel(v,'TOKEN_DIFF');base.load_state_dict(cp['state_dict']);del cp;base.eval()
x=torch.randn(1,3,16,224,224)
with torch.no_grad():before=base(x).clone()
m=ThermalAdapter(base);before_hash=m.frozen_digest();m.eval()
with torch.no_grad():original=m(x,False);zero=m(x,True)
assert torch.equal(before,original) and torch.equal(before,zero)
m.train();z=m(x);loss=z.square().mean();loss.backward();trainable={k:p for k,p in m.named_parameters() if p.requires_grad}
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable.values())
assert all(p.grad is None for p in m.parameters() if not p.requires_grad)
opt=torch.optim.SGD(trainable.values(),lr=1e-4);opt.step();assert m.frozen_digest()==before_hash
m.eval()
with torch.no_grad():preserved=m(x,False)
assert torch.equal(before,preserved),'PRIVATE_TRAINING_CHANGED_ORIGINAL_ROUTE'
saved=m.private_state();torch.save(saved,out/'private_qa_fp16.pt');m.load_private(torch.load(out/'private_qa_fp16.pt',map_location='cpu',weights_only=True))
with torch.no_grad():a=m(x,True)
m.load_private(saved)
with torch.no_grad():b=m(x,True)
assert torch.equal(a,b)
result=dict(status='PASS',at_unix=time.time(),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),trainable_parameters=sum(p.numel() for p in trainable.values()),private_serialized_bytes=(out/'private_qa_fp16.pt').stat().st_size,initial_routes_bitwise_equal=True,frozen_base_unchanged_after_gradient_step=True,original_IR_Depth_route_unchanged_after_private_update=True,serialized_private_reload_bitwise_equal=True,finite_gradients=True,training_fits=0,unit_engineering_gradient_steps=1,GPU_used=False,code_hashes={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in ['thermal_adapter_model.py','thermal_adapter_cpu_qa.py']})
(out/'CPU_QA.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
