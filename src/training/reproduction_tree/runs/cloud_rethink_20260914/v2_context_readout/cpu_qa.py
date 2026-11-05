"""Only new spatial input difference: four locked Training source clips, CPU."""
from pathlib import Path
import os,sys,time,json,hashlib,copy
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
E=Path(__file__).resolve().parent;C=E.parent;O=C.parent/'error_resolution_20260913'
sys.path[:0]=[str(E),str(O),str(O/'bundle')]
import torch
from torch.utils.data import default_collate
from pose_view_data import PoseFiveViews
from data import ContextThreeViews,expand_box,PHASES
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))

def main():
 started=time.time();torch.set_num_threads(2)
 inputs=O/'POSE_ADAPTER_INPUTS.json';inp=read(inputs)
 spec=read(C/'videomaev2_base_source/protocol.json')
 job=next(j for j in spec['jobs'] if j['fold']==0)
 allowed=set(job['train_users']);rows=sorted([r for r in inp['rows'] if r['user'] in allowed],key=lambda r:r['clip_id'])[:4]
 root=Path(r'D:\Datasets\CUHK-X_Small_Model_Track\Small-Model-Track')
 assert len(rows)==4 and not set(r['user'] for r in rows)&set(job['held_users'])
 old=PoseFiveViews(rows,root,inp['regions']);identity=ContextThreeViews(rows,root,inp['regions'],1.)
 candidate=ContextThreeViews(rows,root,inp['regions']);facts=[]
 for i,r in enumerate(rows):
  assert r['streams'][0]['directory'].startswith('Training/')
  a=old[i]['x'][[0,3,4]];b=identity[i];v=candidate[i]
  assert torch.equal(a,b['x']) and v['x'].shape==(3,3,16,224,224) and torch.isfinite(v['x']).all()
  mapped=copy.deepcopy(inp['regions']);mapped[r['clip_id']][0]['box']=v['context_box'].tolist()
  reference=PoseFiveViews([r],root,mapped)[0]['x'][[0,3,4]]
  assert torch.equal(reference,v['x']) and not torch.equal(a,v['x'])
  facts.append(dict(clip_id=r['clip_id'],user=r['user'],original_box=inp['regions'][r['clip_id']][0]['box'],context_box=v['context_box'].tolist(),identity_maxabs=float((a-b['x']).abs().max()),reference_maxabs=float((reference-v['x']).abs().max()),frame_indices=v['frame_indices'].tolist()))
  assert time.time()-started<180
 # Existing global-center fallback is unchanged for each of the three phases.
 fallback=copy.deepcopy(inp['regions']);fallback[rows[0]['clip_id']][0]['box']=None
 a=ContextThreeViews(rows[:1],root,fallback)[0]
 b=PoseFiveViews(rows[:1],root,fallback)[0]['x'][[0,3,4]]
 assert a['fallback'] and torch.equal(a['x'],b)
 collated=default_collate([candidate[0],candidate[1]]);assert collated['x'].shape==(2,3,3,16,224,224)
 assert expand_box([0,0,120,120],[640,480])==[0,0,180,180]
 assert expand_box([520,360,120,120],[640,480])==[460,300,180,180]
 assert expand_box([0,0,480,480],[640,480])==[0,0,480,480]
 result=dict(status='PASS_CONTEXT_SPATIAL_CPU_QA',data_sha256=sha(E/'data.py'),inputs_sha256=sha(inputs),source_protocol_sha256=sha(C/'videomaev2_base_source/protocol.json'),source_job_id=job['id'],train_users=job['train_users'],held_users=job['held_users'],rows=facts,phases=PHASES,no_model=True,no_CUDA=True,no_Test=True,identity_exact=True,expanded_reference_exact=True,edge_clamp_pass=True,None_fallback_exact=True,default_collate_pass=True,seconds=time.time()-started)
 assert not (E/'CPU_QA.json').exists();(E/'CPU_QA.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
