from pathlib import Path
import os,sys,json,time,hashlib,shutil,importlib.util
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
E=Path(__file__).resolve().parent;C=E.parent;R=C.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))

def main():
 start=time.time();s=read(E/'protocol.json');assert s['status']=='FROZEN_FOR_EXECUTION'and s['production_selected_after_complete_source_validation'];assert start<s['deadline_unix'];D=Path(s['package']);P=Path(s['template'])
 for p,h in s['bindings'].items():assert sha(p)==h,p
 assert not D.exists()
 sys.path.insert(0,str(C));import frozen_fusion_math as fm
 sp=importlib.util.spec_from_file_location('weight_average_final_classwise3',C/'videomaev2_base_fusion/classwise3.py');cw=importlib.util.module_from_spec(sp);sp.loader.exec_module(cw)
 import numpy as np,torch
 torch.set_num_threads(1);v=dict(np.load(s['source_npz'],allow_pickle=False));src=dict(np.load(s['base_source_npz'],allow_pickle=False))
 assert len(v['targets'])==4768 and len(set(v['clip_ids']))==2931 and set(v['users'])==set(s['training_users']);assert v['available'].shape==(4768,3)
 index={(int(f),str(cid)):i for i,(f,cid)in enumerate(zip(src['source_fold'],src['original_clip_ids']))};keys=[(int(f),str(cid))for f,cid in zip(v['source_fold'],v['clip_ids'])];assert len(index)==len(set(keys))==4768
 ix=np.asarray([index[k]for k in keys]);assert np.array_equal(src['users'][ix],v['users'])and np.array_equal(src['targets'][ix],v['targets'])
 logits=np.concatenate([v['TI'],v['video'][:,None]],1);assert logits.shape==(4768,3,40)
 d=dict(logits=logits,available=v['available'],targets=v['targets'],users=v['users'],depth=src['depth'][ix],ir=src['ir'][ix])
 class Clock:
  def check(self,**kw):assert time.time()<s['deadline_unix']
 clock=Clock();fm.STRATEGIES['convex_T_I_AVERAGE3']=dict(branches=[0,1,2],mode='convex',availability=False);g=fm.fit(d,'convex_T_I_AVERAGE3',clock,500)
 assert s['selected_fusion']=='classwise3';cal=cw.fit(logits,v['available'],v['targets'],v['users'],g,clock)
 (E/'FULL_SOURCE_GLOBAL_FIT.json').write_text(json.dumps(g,indent=2)+'\n',encoding='utf-8')
 from audit_math import replay
 independent=replay(logits,v['available'],cal);ref=cw.predict(logits,v['available'],cal)
 assert np.max(abs(independent-ref))<1e-6 and np.array_equal(independent.argmax(1),ref.argmax(1))
 shutil.copytree(P,D,ignore=shutil.ignore_patterns('__pycache__','*.pyc','runtime','submission.csv*','FINAL_VALIDATION.json','inference_manifest.json','calibrator.json','B.pt'))
 shutil.copy2(E/'selected_infer.py',D/'infer.py');shutil.copy2(C/'videomaev2_base_fusion/classwise3.py',D/'classwise_fusion.py');shutil.copy2(s['checkpoint'],D/'checkpoints/B.pt')
 (D/'calibrator.json').write_text(json.dumps(cal,separators=(',',':'))+'\n',encoding='utf-8')
 cp=torch.load(D/'checkpoints/B.pt',map_location='cpu',weights_only=True);assert cp['protocol_sha256']==s['checkpoint_protocol_sha256'] and cp['train_users']==s['training_users']and cp['held_users']==[]
 provenance={k:cp[k]for k in ('epochs','epoch_semantics','raw_epochs','control_epochs','control_optimizer','raw_protocol_sha256','control_protocol_sha256','seed','alpha','raw_sha256','control_sha256','arithmetic')}
 assert provenance['raw_epochs']==16 and provenance['control_epochs']==4 and provenance['control_optimizer']=='freshAdamW'and provenance['alpha']==.5
 cfg=read(D/'release.json');cfg['checkpoints']['B']=dict(path='checkpoints/B.pt',sha256=sha(D/'checkpoints/B.pt'),bytes=(D/'checkpoints/B.pt').stat().st_size,protocol_sha256=cp['protocol_sha256'])
 cfg.update(id='VIDEOMAEV2_WEIGHT_AVERAGE_INT8_TI_CLASSWISE3',selected_fusion='classwise3',checkpoint_provenance=provenance,source_storage_audit_sha256=sha(s['storage_audit']),calibrator_sha256=sha(D/'calibrator.json'),serialized_weight_bytes=sum((D/x['path']).stat().st_size for x in cfg['checkpoints'].values())+(D/'calibrator.json').stat().st_size,branch_order=['T','I','BaseAverage'],provisional=False,calibration_pending=False)
 assert cfg['serialized_weight_bytes']<=100000000;(D/'release.json').write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf-8')
 report=dict(status='PASS_WEIGHT_AVERAGE_FINAL_SOURCE_CALIBRATION_AND_PACKAGE_PREPARATION',protocol_sha256=sha(E/'protocol.json'),source_rows=4768,unique_clips=2931,source_users=18,calibrator_sha256=sha(D/'calibrator.json'),independent_max_abs=float(np.max(abs(independent-ref))),actual_total_model_bytes=cfg['serialized_weight_bytes'],source_fit_not_validation=True,selected_fusion='classwise3',no_Test_yet=True,seconds=time.time()-start)
 (E/'CALIBRATION_RESULT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
 inputs=[E/'protocol.json',E/'source_qa.py',E/'CALIBRATION_RESULT.json',D/'infer.py',D/'release.json',D/'checkpoints/B.pt',D/'calibrator.json',C/'videomaev2_base/model.py',C.parent/'error_resolution_20260913/POSE_ADAPTER_INPUTS.json']
 sq=dict(status='FROZEN_FOR_EXECUTION',registered_at_unix=time.time(),deadline_unix=s['deadline_unix'],source_clip='2_Comb_hair/1/1-1-1',data_root=s['data_root'],no_GPU=True,no_Test=True,bindings={str(p):sha(p)for p in inputs})
 (E/'source_qa_protocol.json').write_text(json.dumps(sq,indent=2),encoding='utf-8');print(json.dumps(report))
if __name__=='__main__':main()
