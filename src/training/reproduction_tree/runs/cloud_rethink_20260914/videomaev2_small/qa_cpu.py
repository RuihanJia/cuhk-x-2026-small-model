from pathlib import Path
import hashlib,io,json,os,sys,time
os.environ['CUDA_VISIBLE_DEVICES']=''
E=Path(__file__).resolve().parent;C=E.parent;O=C.parent/'error_resolution_20260913'
sys.path[:0]=[str(E),str(O),str(O/'bundle')]
import torch
from model import VideoMAESmall
from pose_view_data import PoseClips
from dual_host_runtime.io import atomic,read
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
torch.set_num_threads(2);start=time.time();s=read(E/'QA_PROTOCOL.json');assert sha(E/'qa_cpu.py')==s['worker_sha256'] and sha(E/'model.py')==s['model_sha256']
a=read(O/'POSE_ADAPTER_INPUTS.json');assert sha(O/'POSE_ADAPTER_INPUTS.json')==s['pose_inputs_sha256'];row=next(r for r in a['rows'] if r['clip_id']==s['source_clip_id']);assert row['user'] in s['source_users'] and row['user'] not in s['held_users']
ds=PoseClips([row],s['data_root'],20260915,a['regions']);x=ds[0]['x'][None];y=torch.tensor([row['label']])
ref=VideoMAESmall(classes=710,fast=False).eval()
with torch.inference_mode():expected=ref(x)
del ref
import model as vm
av2=vm.author_module(E/'source_v2.py',s['v2_author_sha256'])
reference=av2.vit_small_patch16_224(num_classes=710,all_frames=16,drop_rate=0.,attn_drop_rate=0.,drop_path_rate=0.,use_mean_pooling=True).eval()
reference.load_state_dict(vm.converted_state(E/'vit_s_k710_dl_from_giant.pth'),strict=True)
with torch.inference_mode():
 pixels=(x.float()*.225+.45).clamp(0,1);z2=reference((pixels-torch.tensor([.485,.456,.406]).reshape(1,3,1,1,1))/torch.tensor([.229,.224,.225]).reshape(1,3,1,1,1))
assert torch.allclose(expected,z2,rtol=1e-5,atol=1e-5) and torch.equal(expected.argmax(-1),z2.argmax(-1))
v2_error=float((expected-z2).abs().max());del reference
fast=VideoMAESmall(classes=710,fast=True).eval()
with torch.inference_mode():actual=fast(x)
assert torch.allclose(expected,actual,rtol=1e-4,atol=1e-4) and torch.equal(expected.argmax(-1),actual.argmax(-1));difference=float((expected-actual).abs().max());del fast
model=VideoMAESmall(classes=40,fast=True).train();initial={k:v.detach().clone() for k,v in model.state_dict().items()};logits=model(x);loss=torch.nn.functional.cross_entropy(logits,y,label_smoothing=.1);loss.backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters());assert all(torch.equal(v,initial[k]) for k,v in model.state_dict().items());assert model.video.patch_embed.proj.weight.grad.norm()>0
# Compare only freshly initialized40classhead with the frozen original constructor.
import importlib.util
sp=importlib.util.spec_from_file_location('w_model_reference',C/'videomae_small/model.py');wm=importlib.util.module_from_spec(sp);sp.loader.exec_module(wm);old=wm.VideoMAESmall(classes=40,fast=True)
assert torch.equal(old.video.head.weight,model.video.head.weight) and torch.equal(old.video.head.bias,model.video.head.bias)
buf=io.BytesIO();torch.save({k:v.half() for k,v in initial.items()},buf)
r=dict(status='PASS',at_unix=time.time(),seconds=time.time()-start,protocol_sha256=sha(E/'QA_PROTOCOL.json'),model_sha256=sha(E/'model.py'),public_weight_sha256=sha(E/'vit_s_k710_dl_from_giant.pth'),source_clip_id=row['clip_id'],source_only=True,no_held_forward=True,no_Test=True,no_GPU=True,optimizer_updates=0,all_weights_strict_mapped=True,all_parameter_gradients_finite=True,parameters_unchanged=True,fresh40head_exact_original_W=True,author_heads=6,author_layernorm_eps=1e-6,author_vs_SDPA_FP32_max_abs=difference,author_vs_SDPA_argmax_equal=True,v2_author_vs_original_runtime_max_abs=v2_error,parameters=sum(p.numel() for p in model.parameters()),FP16_serialized_bytes=len(buf.getvalue()))
assert r['seconds']<s['wall_cap_seconds'];atomic(E/'QA_RESULT.json',r);print(json.dumps(r))
