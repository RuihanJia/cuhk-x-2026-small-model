"""Registered production only: all-source final cal and self-contained paired package."""
from pathlib import Path
import os,sys,json,time,hashlib,shutil,importlib.util,struct
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
E=Path(__file__).resolve().parent;C=E.parent;R=C.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def main():
 start=time.time();s=read(E/'protocol.json');assert s['status']=='FROZEN_FOR_EXECUTION' and s['production_selected_after_complete_source_validation'];assert start<s['deadline_unix'];D=Path(s['package']);P=R/'deliverables/videomaev2_base_temporal_halves_candidate_20260915'
 for p,h in s['bindings'].items():assert sha(p)==h,p
 assert not D.exists()
 sys.path.insert(0,str(C));import frozen_fusion_math as fm
 sp=importlib.util.spec_from_file_location('paired_final_classwise4',C/'v2_context_readout/classwise4.py');cw=importlib.util.module_from_spec(sp);sp.loader.exec_module(cw)
 import numpy as np,torch
 torch.set_num_threads(1);v=dict(np.load(s['source_npz'],allow_pickle=False));src=dict(np.load(s['base_source_npz'],allow_pickle=False))
 assert len(v['targets'])==4768 and len(set(v['clip_ids']))==2931 and set(v['users'])==set(s['training_users']);assert v['available'].shape==(4768,4)
 index={(int(f),str(cid)):i for i,(f,cid) in enumerate(zip(src['source_fold'],src['original_clip_ids']))};assert len(index)==4768;keys=[(int(f),str(cid))for f,cid in zip(v['source_fold'],v['clip_ids'])];assert len(set(keys))==4768
 ix=np.asarray([index[k]for k in keys]);assert np.array_equal(src['users'][ix],v['users']) and np.array_equal(src['targets'][ix],v['targets'])
 d=dict(logits=np.concatenate([v['TI'],v['base_mixed'][:,None],v['small_mixed'][:,None]],1),available=v['available'],targets=v['targets'],users=v['users'],depth=src['depth'][ix],ir=src['ir'][ix]);assert d['logits'].shape==(4768,4,40)
 class Clock:
  def check(self,**kw):assert time.time()<s['deadline_unix']
 clock=Clock();fm.STRATEGIES['convex_BASE6_SMALL8_TI4']=dict(branches=[0,1,2,3],mode='convex',availability=False);g=fm.fit(d,'convex_BASE6_SMALL8_TI4',clock,500)
 assert s['selected_fusion'] in ('global4','classwise4')
 cal=cw.fit(d['logits'],d['available'],d['targets'],d['users'],g,clock) if s['selected_fusion']=='classwise4' else {k:v for k,v in g.items() if k!='trace'}
 (E/'FULL_SOURCE_GLOBAL_FIT.json').write_text(json.dumps(g,indent=2)+'\n',encoding='utf-8')
 from audit_math import replay
 ind=replay(d['logits'],d['available'],cal)
 if s['selected_fusion']=='classwise4':ref=cw.predict(d['logits'],d['available'],cal)
 else:
  model=fm.Calibrator(cal['description']);model.load_state_dict({k:torch.tensor(v,dtype=torch.float64)for k,v in cal['state'].items()},strict=True)
  with torch.no_grad():ref=model(dict(logits=torch.tensor(d['logits'],dtype=torch.float64),mask=torch.tensor(d['available']))).numpy().astype(np.float32)
 assert np.max(abs(ind-ref))<1e-6 and np.array_equal(ind.argmax(1),ref.argmax(1))
 shutil.copytree(P,D,ignore=shutil.ignore_patterns('__pycache__','*.pyc','runtime','submission.csv*','FINAL_VALIDATION.json','inference_manifest.json','calibrator.json','B.pt'))
 for source,target in [(E/'paired_infer.py','infer.py'),(E/'dual_model.py','dual_model.py'),(C/'v2_base6_small_design/loader.py','loader.py'),(C/'v2_base6_small_design/pair_codec.py','pair_codec.py'),(C/'v2_context_readout/classwise4.py','classwise_fusion.py'),(C/'videomaev2_small/ADMISSION.json','VIDEOMAEV2_SMALL_ADMISSION.json')]:shutil.copy2(source,D/target)
 shutil.copy2(E/'mirror_predict.py',D/'mirror_predict.py');shutil.copy2(s['container'],D/'checkpoints/PAIR.bin');(D/'calibrator.json').write_text(json.dumps(cal,separators=(',',':'))+'\n',encoding='utf-8');storage=read(s['storage_qa']);assert storage['container_sha256']==sha(D/'checkpoints/PAIR.bin');sys.path.insert(0,str(C/'v2_base6_small_design'));from pair_codec import MAGIC
 with (D/'checkpoints/PAIR.bin').open('rb') as f:
  assert f.read(len(MAGIC))==MAGIC;header=json.loads(f.read(struct.unpack('<I',f.read(4))[0]))
 provenance=header['provenance'];assert provenance['train_users']==s['training_users'] and provenance['held_users']==[] and provenance['epochs']==16
 cfg=read(D/'release.json');cfg['checkpoints'].pop('B');cfg['checkpoints']['PAIR']=dict(path='checkpoints/PAIR.bin',sha256=sha(D/'checkpoints/PAIR.bin'),bytes=(D/'checkpoints/PAIR.bin').stat().st_size,format='VIDEOMAEV2_BASE6_SMALL8_ALL50_FP32ROWS_OTHERSFP16_V1');cfg.update(id='VIDEOMAEV2_BASE6_SMALL8_TI_'+s['selected_fusion'].upper(),selected_fusion=s['selected_fusion'],quantization_method=s['quantization_method'],pair_provenance=provenance,source_storage_qa_sha256=sha(s['storage_qa']),calibrator_sha256=sha(D/'calibrator.json'),serialized_weight_bytes=sum((D/x['path']).stat().st_size for x in cfg['checkpoints'].values())+(D/'calibrator.json').stat().st_size,branch_order=['T','I','Base6','Small8'],temporal_readout=dict(Base='halfweighted7',Small='uniform5'),provisional=False,calibration_pending=False)
 cfg.update(id='VIDEOMAEV2_DUAL_EQUAL_MIRROR_CLASSWISE4',optional_original_video_cache_sha256=s['original_video_cache_sha256'],original_video_cache_pair_sha256=s['container_sha256'],mirror_readout=dict(Base='0.5 original7 +0.5 mirrored7',Small='0.5 original5 +0.5 mirrored5',final_source_fields=['TI','base_mixed','small_mixed']));assert cfg['serialized_weight_bytes']<100000000;(D/'release.json').write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf-8')
 r=dict(status='PASS_PAIRED_FINAL_SOURCE_CALIBRATION_AND_PACKAGE_PREPARATION',protocol_sha256=sha(E/'protocol.json'),source_rows=4768,unique_clips=2931,source_users=18,calibrator_sha256=sha(D/'calibrator.json'),independent_max_abs=float(np.max(abs(ind-ref))),actual_total_model_bytes=cfg['serialized_weight_bytes'],source_fit_not_validation=True,selected_fusion=s['selected_fusion'],no_Test_yet=True,seconds=time.time()-start);(E/'CALIBRATION_RESULT.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(json.dumps(r))
if __name__=='__main__':main()
