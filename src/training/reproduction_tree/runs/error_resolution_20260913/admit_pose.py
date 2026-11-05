from pathlib import Path
import os,sys,json,hashlib,time
ROOT=Path(__file__).resolve().parents[2];r=ROOT/'assets/pose';(r/'settings').mkdir(exist_ok=True)
os.environ['YOLO_CONFIG_DIR']=str(r/'settings');os.environ['YOLO_AUTOINSTALL']='false';os.environ['PYTEST_CURRENT_TEST']='pose_geometry_audit';sys.path.insert(0,str(r/'site'))
import torch
from ultralytics import YOLO
from ultralytics.utils import SETTINGS
SETTINGS.update({'sync':False});torch.set_num_threads(2);source=json.loads((r/'FETCH.json').read_text());assert hashlib.sha256((r/'yolo11n-pose.pt').read_bytes()).hexdigest()==source['files']['yolo11n-pose.pt']['sha256']
m=YOLO(str(r/'yolo11n-pose.pt'));assert m.task=='pose' and list(m.model.model[-1].kpt_shape)==[17,3];assert m.names=={0:'person'}
cp=dict(state_dict={k:v.cpu().half() if v.is_floating_point() else v.cpu() for k,v in m.model.state_dict().items()},yaml=m.model.yaml,task='pose',source_sha256=source['files']['yolo11n-pose.pt']['sha256']);torch.save(cp,r/'pose_state_fp16.pt');n=sum(p.numel() for p in m.model.parameters())
admit=dict(status='ADMITTED_PUBLIC_NONLANGUAGE_COCO_POSE',at_unix=time.time(),official_files=source['files'],model='YOLO11n-pose',task='COCO person and17keypoint localization',pretraining='OfficialCOCO-Pose supervised dataset; notlanguage and noLLM',license='AGPL-3.0; preserve source andlicense in anydistributed implementation',parameters=n,inference_weight_file='pose_state_fp16.pt',inference_weight_bytes=(r/'pose_state_fp16.pt').stat().st_size,inference_weight_sha256=hashlib.sha256((r/'pose_state_fp16.pt').read_bytes()).hexdigest(),use='Onlylabel-independent geometry, nevermanualactorlabels orTestlabels; admissiondoesnotproveIRdomainaccuracy',model_selection='No YOLOsize or confidencegrid; onefixednmodel',total_with_existing_R6_bytes=76193153+(r/'pose_state_fp16.pt').stat().st_size,rules_evidence='evidence/rules/RULES_SNAPSHOT_20260904.md',GPU_use=False)
(r/'ADMISSION.json').write_text(json.dumps(admit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(admit,ensure_ascii=False))
