"""After selected source QA: describe, bind and freeze the actual final package."""
from pathlib import Path
import json,time,hashlib
E=Path(__file__).resolve().parent;C=E.parent;R=C.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def main():
 s=read(E/'protocol.json');assert s['status']=='FROZEN_FOR_EXECUTION' and s['production_selected_after_complete_source_validation'];D=Path(s['package'])
 q=read(E/'SOURCE_QA_DONE.json');assert q['status']=='PASS' and q['protocol_sha256']==sha(E/'source_qa_protocol.json')
 for n,h in q['artifacts'].items():assert sha(E/n)==h
 cal=read(E/'CALIBRATION_RESULT.json');assert cal['status']=='PASS_PAIRED_FINAL_SOURCE_CALIBRATION_AND_PACKAGE_PREPARATION' and cal['protocol_sha256']==sha(E/'protocol.json')
 cfg=read(D/'release.json');assert cfg['serialized_weight_bytes']<100000000
 (D/'README.md').write_text("""# VideoMAEv2 Base6 + Small8 with Thermal and IMU

This package contains two separately trained video encoders: VideoMAEv2 K710 Base with signed 6-bit row quantization, and VideoMAEv2 K710 Small with signed 8-bit row quantization. All 49 Linear matrices and the patch Conv3d in each encoder are quantized. Every scale is FP32; remaining parameters are FP16. The PAIR container stores both models and is decoded into FP32 for inference. Thermal, IMU and a shared pose locator are included and counted in the model size.

Base uses 0.5 times the original five-view mean plus 0.25 each first/last temporal half. Small uses its original five-view mean. The four branches are ordered Thermal, IMU, Base6, Small8. All fusion parameters were fitted exclusively on source predictions: 4,768 fold/clip rows, 2,931 distinct clips from 18 training users. Source fitting is not heldout validation. Development users have been adaptively inspected; no Public score is guaranteed. The selected fusion, model provenance and actual model-plus-calibration bytes are in release.json.

Run on raw competition data:

`python infer.py --data-dir /path/to/Small-Model-Track --output submission.csv --workers 4 --micro 4`

The default path discovers the supplied sample manifest order, computes Thermal and IMU, and derives pose geometry from input images. It supports a different number of samples. An optional geometry cache belongs only to the original 405 samples and is ignored by default. Use `--strict-original-template` only for exact original-template replay; an optional external Thermal/IMU prediction cache also requires that flag and its exact recorded hash. No labels or LLM participate in inference, and the program performs no Kaggle upload.

Install requirements.txt and requirements_pose_dual.txt in a compatible PyTorch environment. The supplied VideoMAEv2, Thermal and pose admission/license records describe the pretraining provenance. FINAL_VALIDATION.json is created only after actual final inference and independent CSV/model-size validation succeed.
""",encoding='utf-8')
 (D/'PRODUCTION_PROVENANCE.json').write_text(json.dumps(dict(production_protocol_sha256=sha(E/'protocol.json'),source_qa_protocol_sha256=sha(E/'source_qa_protocol.json'),source_qa_result_sha256=sha(E/'SOURCE_QA_RESULT.json'),source_validation=s['validation'],selected_fusion=s['selected_fusion'],public_score_unknown=True,no_Test_labels=True),indent=2)+'\n',encoding='utf-8')
 files={}
 for p in sorted(D.rglob('*')):
  rel=p.relative_to(D)
  if p.is_file() and '__pycache__' not in rel.parts and 'runtime' not in rel.parts and p.suffix!='.pyc' and p.name!='inference_manifest.json':files[rel.as_posix()]=dict(bytes=p.stat().st_size,sha256=sha(p))
 (D/'inference_manifest.json').write_text(json.dumps(dict(status='FROZEN_BEFORE_FINAL405',files=files,at_unix=time.time()),indent=2)+'\n',encoding='utf-8')
 now=time.time();out=E/'output/local';previous=Path(s['gpu_predecessor']['directory']);prior=read(previous/'CONTROLLER_DONE.json');assert prior['status']=='COMPLETE' and prior['protocol_sha256']==s['gpu_predecessor']['protocol_sha256']
 bindings={}
 for p in [E/'protocol.json',E/'SOURCE_QA_RESULT.json',E/'SOURCE_QA_DONE.json',E/'CALIBRATION_RESULT.json',E/'worker.py',E/'controller.py',E/'resilient_io.py',E/'final_audit.py',E/'audit_math.py',D/'inference_manifest.json',D/'release.json',D/'calibrator.json']:bindings[str(p.relative_to(R))]=sha(p)
 ip=dict(status='FROZEN_FOR_EXECUTION',registered_at_unix=now,deadline_unix=min(now+3600,s['deadline_unix']),wall_cap_seconds=3600,source_authorization_sha256=s['source_authorization_sha256'],hosts=dict(local=dict(output_directory=str(out),wall_cap_seconds=3600,authorization_file=str(C/'AUTHORIZATION_DUAL_RESOURCE_20260914.json'),predecessor=dict(directory=str(previous),protocol_sha256=prior['protocol_sha256'],success_only=True))),worker_sha256=sha(E/'worker.py'),controller_sha256=sha(E/'controller.py'),bindings=bindings,data_root=s['data_root'],cache_path=s['sensor_cache'],optimizer_updates=0,no_Test_labels=True,no_Public_fit=True,scope='Selected complete-source paired model; exact original405 inputs, no upload.')
 assert ip['deadline_unix']>now
 (E/'inference_protocol.json').write_text(json.dumps(ip,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status='FROZEN_ACTUAL_PAIRED405_INFERENCE',protocol_sha256=sha(E/'inference_protocol.json'),files=len(files),model_bytes=cfg['serialized_weight_bytes'])))
if __name__=='__main__':main()
