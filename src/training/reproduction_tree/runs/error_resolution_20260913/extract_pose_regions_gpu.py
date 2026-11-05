"""Fixed CPU geometry on Training images, labels never passed to the detector."""
from pathlib import Path
import os,sys,json,time,hashlib
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parent;ROOT=R.parents[1];A=R/'assets/pose_gpu';sys.path.insert(0,str(R/'pose_site'));sys.path.insert(0,str(R/'bundle'))
os.environ['YOLO_CONFIG_DIR']=str(A/'settings');os.environ['YOLO_AUTOINSTALL']='false';os.environ['PYTEST_CURRENT_TEST']='pose_geometry_audit'
import torch,cv2
from ultralytics import YOLO
from ultralytics.nn.tasks import PoseModel
from ultralytics.utils import SETTINGS
from ultralytics.utils import torch_utils
from dual_host_runtime.io import atomic
from dual_host_runtime.health import Guard
from pose_roi import box_from_detections
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 SETTINGS.update({'sync':False});torch_utils.NUM_THREADS=2;torch.set_num_threads(2);cv2.setNumThreads(1);s=json.loads((R/'POSE_REGIONS_GPU_PROTOCOL.json').read_text());ph=sha(R/'POSE_REGIONS_GPU_PROTOCOL.json');assert s['worker_sha256']==sha(__file__) and s['geometry_sha256']==sha(R/'pose_roi.py');assert s['model_sha256']==sha(A/'pose_state_fp16.pt');assert s['inputs_sha256']==sha(R/'IR_ADAPTER_INPUTS.json');out=R/'pose_regions_gpu';out.mkdir(exist_ok=True)
 if not(out/'BUDGET.json').exists():atomic(out/'BUDGET.json',dict(first_started_unix=time.time(),deadline_unix=min(s['deadline_unix'],time.time()+s['CPU_wall_cap_seconds']),protocol_sha256=ph))
 budget=json.loads((out/'BUDGET.json').read_text());assert budget['protocol_sha256']==ph;guard=Guard(out,budget['deadline_unix'],out/'TELEMETRY.json');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 cp=torch.load(A/'pose_state_fp16.pt',map_location='cpu',weights_only=True);net=PoseModel(cp['yaml'],ch=3,nc=1,data_kpt_shape=(17,3),verbose=False);net.load_state_dict(cp['state_dict']);net.kpt_shape=[17,3];net.names={0:'person'};net.eval().requires_grad_(False);model=YOLO(str(A/'yolo11n-pose.pt'));model.model=net
 rows=json.loads((R/'IR_ADAPTER_INPUTS.json').read_text())['rows'];root=Path(s['data_root']);results={};cache=out/'PARTIAL.json'
 if cache.exists():
  q=json.loads(cache.read_text());assert q['protocol_sha256']==ph;results=q['regions']
 start=time.time();new=0
 for i,row in enumerate(rows):
  key=row['clip_id']
  if key in results:continue
  assert time.time()<budget['deadline_unix'],'CPU_GEOMETRY_BUDGET';assert not(out/'STOP_REQUESTED').exists()
  st=row['streams'][0];assert Path(st['directory']).parts[:5]==('Training','data','HAR','data','IR');indices=sorted(set(min(len(st['files'])-1,int((k+.5)*len(st['files'])/8)) for k in range(8)));frames=[];hashes=[]
  for index in indices:
   p=root/st['directory']/st['files'][index];a=np.asarray(Image.open(p).convert('RGB'));frames.append(np.round(np.sqrt(a.astype(np.float32)/255)*255).astype(np.uint8)[:,:,::-1].copy());hashes.append(dict(index=index,sha256=sha(p)))
  h,w=frames[0].shape[:2];preds=model.predict(frames,imgsz=640,conf=.25,iou=.7,max_det=1,device='cuda:0',half=False,verbose=False,save=False);detections=[dict(boxes=p.boxes.xyxy.tolist(),confidence=p.boxes.conf.tolist(),keypoints=p.keypoints.data.tolist()) for p in preds];box,reason=box_from_detections(detections,w,h);guard.check(phase='LABEL_FREE_POSE_EXTRACTION',records=len(results)+1,total=len(rows));results[key]=dict(box=box,reason=reason,native_size=[w,h],sampled_frame_hashes=hashes,detections=detections);new+=1
  if new%25==0:
   atomic(cache,dict(protocol_sha256=ph,regions=results));atomic(out/'PROGRESS.json',dict(completed=len(results),total=len(rows),new_completed=new,new_seconds=time.time()-start,at_unix=time.time(),host_threads=torch.get_num_threads(),inference_device='cuda:0',precision='FP32'));print(json.dumps(dict(completed=len(results),total=len(rows),seconds=time.time()-start)),flush=True)
 atomic(out/'POSE_REGIONS.json',dict(protocol_sha256=ph,regions=results));atomic(out/'DONE.json',dict(status='COMPLETE_TRAINING_GEOMETRY_ONLY',protocol_sha256=ph,regions_sha256=sha(out/'POSE_REGIONS.json'),clips=len(results),no_person=sum(v['box'] is None for v in results.values()),seconds=time.time()-start,host_threads=torch.get_num_threads(),inference_device='cuda:0',precision='FP32',no_Test=True,action_labels_used=False));print(json.dumps(dict(completed=len(results),done=True)),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  import traceback
  atomic(R/'pose_regions_gpu/ERROR.json',dict(error=str(e),at_unix=time.time(),traceback=traceback.format_exc()));raise
