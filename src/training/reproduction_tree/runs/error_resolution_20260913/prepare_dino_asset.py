"""Fetch the public non-language DINOv2-S/14 visual backbone; no task fitting."""
import json,hashlib,time,urllib.request,zipfile,sys,os
from pathlib import Path
R=Path(__file__).resolve().parent;A=R/'assets/dino';A.mkdir(parents=True,exist_ok=True)
COMMIT='7764ea0f912e53c92e82eb78a2a1631e92725fc8'
urls={'source.zip':f'https://codeload.github.com/facebookresearch/dinov2/zip/{COMMIT}', 'dinov2_vits14_pretrain.pth':'https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth'}
records=[]
for name,url in urls.items():
 p=A/name
 if not p.exists():
  start=time.time()
  with urllib.request.urlopen(url,timeout=45) as f,p.with_suffix(p.suffix+'.part').open('wb') as out:
   while True:
    b=f.read(1024*1024)
    if not b:break
    if time.time()-start>600:raise TimeoutError('ASSET_DOWNLOAD_CAP')
    out.write(b)
  p.with_suffix(p.suffix+'.part').replace(p)
 records.append(dict(file=name,url=url,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 print(json.dumps(records[-1]),flush=True)
src=A/'source';src.mkdir(exist_ok=True)
with zipfile.ZipFile(A/'source.zip') as z:
 for name in z.namelist():
  rel=Path(*Path(name).parts[1:])
  if not rel.parts or name.endswith('/'):continue
  if rel.suffix!='.py' and rel.name not in ['LICENSE','MODEL_CARD.md','README.md']:continue
  dest=(src/rel).resolve();assert dest.is_relative_to(src.resolve());dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(name))
os.environ['XFORMERS_DISABLED']='1';sys.path.insert(0,str(src))
import torch
from dinov2.hub.backbones import dinov2_vits14
torch.set_num_threads(2)
m=dinov2_vits14(pretrained=False)
sd=torch.load(A/'dinov2_vits14_pretrain.pth',map_location='cpu',weights_only=True);m.load_state_dict(sd,strict=True)
packed={};dequant={}
for k,v in sd.items():
 v=v.float()
 if k.endswith('weight') and v.ndim>=2:
  shape=[v.shape[0]]+[1]*(v.ndim-1);scale=v.abs().flatten(1).amax(1).clamp_min(1e-8).div(127).half().reshape(shape)
  q=(v/scale.float()).round().clamp(-127,127).to(torch.int8);packed[k]={'q':q,'scale':scale};dequant[k]=q.float()*scale.float()
 else:packed[k]={'value':v.half()};dequant[k]=v.half().float()
cp=A/'dino_int8.pt';torch.save(dict(format='PER_OUTPUT_INT8_WEIGHT_FP16_OTHER_V1',state_dict=packed,source=records,code_commit=COMMIT),cp)
m.load_state_dict(dequant,strict=True)
record=dict(status='ADMITTED_PUBLIC_VISUAL_NON_LLM',at_unix=time.time(),model='DINOv2 ViT-S/14',code_commit=COMMIT,source=records,license='Apache-2.0',model_card_sha256=hashlib.sha256((src/'MODEL_CARD.md').read_bytes()).hexdigest(),license_sha256=hashlib.sha256((src/'LICENSE').read_bytes()).hexdigest(),rule_evidence='evidence/rules/source_20260904/external_data_host_clarification.png',rule_evidence_sha256='252F48D087F7DCAEE539476A7062E3AC8224575F921887B8047921D14DB6CE24',public_free_model_access=True,disclosure_required=True,LLM_in_chain=False,task_training_fits=0,parameters=sum(p.numel() for p in m.parameters()),int8_bytes=cp.stat().st_size,int8_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),existing_R6_bytes=76193153,remaining_head_budget=100000000-76193153-cp.stat().st_size)
assert record['remaining_head_budget']>200000
(A/'ADMISSION.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8');print(json.dumps(record),flush=True)
