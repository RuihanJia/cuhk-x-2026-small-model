from pathlib import Path
import json,time,hashlib
E=Path(__file__).resolve().parent;C=E.parent;R=C.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def main():
 s=read(E/'protocol.json');assert s['status']=='FROZEN_FOR_EXECUTION'and s['production_selected_after_complete_source_validation'];D=Path(s['package'])
 q=read(E/'SOURCE_QA_DONE.json');assert q['status']=='PASS'and q['protocol_sha256']==sha(E/'source_qa_protocol.json')
 for n,h in q['artifacts'].items():assert sha(E/n)==h
 cal=read(E/'CALIBRATION_RESULT.json');assert cal['status']=='PASS_WEIGHT_AVERAGE_FINAL_SOURCE_CALIBRATION_AND_PACKAGE_PREPARATION'and cal['protocol_sha256']==sha(E/'protocol.json')
 cfg=read(D/'release.json');assert cfg['serialized_weight_bytes']<=100000000
 (D/'README.md').write_text('''# VideoMAEv2 fixed weight average with Thermal and IMU

The single Base video encoder is the fixed 50/50 FP32 midpoint of saved FP16 raw16 and fresh-optimizer CONTROL4 endpoints, rounded to FP16 and stored by the original ALL50 INT8 codec. CONTROL4 uses a fresh AdamW optimizer; this is not a continuous original-optimizer epoch20 checkpoint. Both parent hashes, protocols, epoch semantics and training users are retained in release.json and the checkpoint. All49 Linear matrices and patchConv3d use per-output FP32 scales; other parameters are FP16. Decoding and inference use FP32.

Readout is0.5 originalfive-view mean plus0.25 firsthalf and0.25 lasthalf. Thermal, IMU and the shared pose locator are included in the serialized size. Final classwise3 fusion uses only4,768 sourcefold/clip rows from2,931 distinct clips and18 trainingusers: original500-iteration globalconvex calibration then original centered3x40 regularization0.1. Source fitting is not validation. The inspected seven-user development gives930/1094 vsH928,3fix1harm, with mixed per-user/NLL evidence. No Public score is guaranteed.

Run:

`python infer.py --data-dir /path/to/Small-Model-Track --output submission.csv --workers 4 --micro 4`

By default, infer discovers the supplied samples, computes Thermal/IMU and pose geometry from raw input, and accepts a different sample count. Original405 geometry and optional external TI cache are used only with explicit --strict-original-template and exact hash/order checks. No labels or LLM participate in inference; this program does not upload to Kaggle. Install requirements.txt and requirements_pose_dual.txt in a compatible PyTorch environment. Included admission/license records retain pretrained model provenance. FINAL_VALIDATION.json is generated only after actual405 inference and independent numerical, template and byte validation.
''',encoding='utf-8')
 (D/'PRODUCTION_PROVENANCE.json').write_text(json.dumps(dict(production_protocol_sha256=sha(E/'protocol.json'),source_qa_protocol_sha256=sha(E/'source_qa_protocol.json'),source_qa_result_sha256=sha(E/'SOURCE_QA_RESULT.json'),source_validation=s['validation'],selected_fusion=s['selected_fusion'],checkpoint_provenance=cfg['checkpoint_provenance'],Public_score_unknown=True,no_Test_labels=True),indent=2),encoding='utf-8')
 files={}
 for p in sorted(D.rglob('*')):
  rel=p.relative_to(D)
  if p.is_file()and '__pycache__'not in rel.parts and 'runtime'not in rel.parts and p.suffix!='.pyc'and p.name!='inference_manifest.json':files[rel.as_posix()]=dict(bytes=p.stat().st_size,sha256=sha(p))
 (D/'inference_manifest.json').write_text(json.dumps(dict(status='FROZEN_BEFORE_FINAL405',files=files,at_unix=time.time()),indent=2),encoding='utf-8')
 # Source-only work need not idle while predecessor protocol is being registered.
 pre=s['gpu_predecessor'];previous=Path(pre['directory']);predprotocol=Path(pre['protocol_file'])
 while not predprotocol.exists():
  assert time.time()<s['deadline_unix'],'MEAN6_PREDECESSOR_PROTOCOL_MISSING';time.sleep(2)
 pred=read(predprotocol);assert pred['status']=='FROZEN_FOR_EXECUTION'and Path(pred['hosts']['local']['output_directory']).resolve()==previous.resolve()
 bindings={}
 for p in [E/'protocol.json',E/'SOURCE_QA_RESULT.json',E/'SOURCE_QA_DONE.json',E/'CALIBRATION_RESULT.json',E/'worker.py',E/'controller.py',E/'resilient_io.py',E/'final_audit.py',E/'audit_math.py',D/'inference_manifest.json',D/'release.json',D/'calibrator.json',predprotocol]:bindings[str(p.relative_to(R))]=sha(p)
 now=time.time();ip=dict(status='FROZEN_FOR_EXECUTION',registered_at_unix=now,deadline_unix=s['deadline_unix'],wall_cap_seconds=5400,source_authorization_sha256=s['source_authorization_sha256'],hosts=dict(local=dict(output_directory=str(E/'output/local'),wall_cap_seconds=5400,authorization_file=str(C/'AUTHORIZATION_DUAL_RESOURCE_20260914.json'),predecessor=dict(directory=str(previous),protocol_sha256=sha(predprotocol),protocol_file=str(predprotocol),success_status=pre['success_status'],success_only=True))),worker_sha256=sha(E/'worker.py'),controller_sha256=sha(E/'controller.py'),bindings=bindings,data_root=s['data_root'],cache_path=s['sensor_cache'],optimizer_updates=0,no_Test_labels=True,no_Public_fit=True,scope='Selected fixed average singleBase; actual405, no upload')
 assert ip['deadline_unix']>now and not(E/'inference_protocol.json').exists();(E/'inference_protocol.json').write_text(json.dumps(ip,indent=2),encoding='utf-8');print(json.dumps(dict(status='FROZEN_WEIGHT_AVERAGE_405',protocol_sha256=sha(E/'inference_protocol.json'),files=len(files),model_bytes=cfg['serialized_weight_bytes'],predecessor_sha256=sha(predprotocol))))
if __name__=='__main__':main()
