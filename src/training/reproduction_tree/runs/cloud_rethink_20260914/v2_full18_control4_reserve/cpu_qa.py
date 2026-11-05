from pathlib import Path
import os,sys,json,hashlib,ast
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
E=Path(__file__).resolve().parent;C=E.parent;O=C.parent/'error_resolution_20260913';PHOTO=C/'aux_public_regression_20260914/dynamic_interaction/photometric_candidate'
sys.path[:0]=[str(C/'v2_window_cloud52910'),str(PHOTO),str(O),str(O/'bundle')]
import torch
from photometric_data_numpy import PhotometricClips,train_metadata
from data import WindowClips
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
class Epoch:
 value=1

def main():
 torch.set_num_threads(1);parent=C/'videomaev2_base_production/cloud';cp=torch.load(parent/'model_fp16.pt',map_location='cpu',weights_only=True);fr=read(parent/'FROZEN.json');td=read(parent/'TRAIN_DONE.json');sp=read(parent/'SPLIT.json');inp=read(O/'POSE_ADAPTER_INPUTS.json')
 assert cp['protocol_sha256']==fr['protocol_sha256']==td['protocol_sha256']==sp['protocol_sha256'] and cp['epochs']==fr['epochs']==16
 assert sha(parent/'model_fp16.pt')==fr['checkpoint_sha256']==td['artifacts']['model_fp16.pt']=='7a5beb6db683507c2143644ca95f21b6f7cd7ef5ef74d28bf9937e533ce01a65'
 assert cp['held_users']==sp['held_users']==[] and len(cp['train_users'])==18 and cp['train_users']==sorted({r['user'] for r in inp['rows']})
 assert len(sp['train_ids'])==len(inp['rows'])==2931 and not sp['held_ids'] and sp['train_ids']==[r['clip_id']for r in inp['rows']]
 assert all(v.dtype==torch.float16 and torch.isfinite(v).all()for v in cp['model'].values());del cp
 row=inp['rows'][0];shared=Epoch();comparison=[]
 for epoch in (1,4):
  shared.value=epoch;a=WindowClips([row],'D:/Datasets/CUHK-X_Small_Model_Track/Small-Model-Track',20260915,inp['regions'],training=True,epoch_value=shared,arm='CONTROL')[0]
  b=PhotometricClips([row],'D:/Datasets/CUHK-X_Small_Model_Track/Small-Model-Track',20260915,inp['regions'],training=True,epoch=epoch+16,arm='RAW_CONTROL')[0]
  assert torch.equal(a['x'],b['x']) and torch.equal(a['frame_indices'],b['frame_indices'])and a['flip']==b['flip']
  comparison.append(dict(epoch=epoch,augmentation_epoch=epoch+16,pixels_exact=True,indices=a['frame_indices'].tolist(),flip=a['flip']))
 expected=[];gen=torch.Generator()
 for epoch in range(1,5):
  gen.manual_seed(20260915+epoch)
  if epoch==1:torch.empty((),dtype=torch.int64).random_(generator=gen)
  order=torch.randperm(2931,generator=gen).tolist();dig=hashlib.sha256()
  for ix in order:
   r=inp['rows'][ix];idx,flip=train_metadata(20260915,16+epoch,r['clip_id'],len(r['streams'][0]['files']),True);dig.update(json.dumps([r['clip_id'],idx.tolist(),bool(flip)],separators=(',',':')).encode())
  expected.append(dict(epoch=epoch,records=2931,updates=184*epoch,sampling_sha256=dig.hexdigest()))
 for p in E.glob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'))
 report=dict(status='PASS_FULL18_ORIGINAL_PARENT_AND_CONTROL_INTERFACE',parent_sha256=fr['checkpoint_sha256'],parent_protocol_sha256=fr['protocol_sha256'],source_users=fr['train_users'],source_rows=2931,held_rows=0,real_pixel_qa=comparison,expected_history=expected,training_stack_sha256=sha(C/'videomaev2_base/worker.py'),data_sha256=sha(C/'v2_window_cloud52910/data.py'),training_science='freshAdamW4epochs; augmentation epochs17to20; shuffle seed+stageEpoch with persistent workers base-seed draw only first epoch; exact original fit function',no_Test=True,no_GPU=True)
 (E/'CPU_QA.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(status=report['status'],parent_sha256=report['parent_sha256'],source_rows=2931,real_pixel_matches=2,expected_updates=736,sha256=sha(E/'CPU_QA.json'))))
if __name__=='__main__':main()
