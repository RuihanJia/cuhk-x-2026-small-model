"""Label-free dense IR pose, lossless per-clip numeric cache for temporal models."""
from pathlib import Path
import os,sys,time,json,hashlib,io,traceback
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parent
from extract_pose_regions_gpu import torch,cv2,YOLO,PoseModel,SETTINGS,torch_utils,A,atomic,Guard

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def decode(p):
 b=p.read_bytes()
 with Image.open(io.BytesIO(b)) as im:a=np.asarray(im.convert('RGB'))
 a=np.round(np.sqrt(a.astype(np.float32)/255)*255).astype(np.uint8)[:,:,::-1].copy()
 return a,hashlib.sha256(b).hexdigest()
def main():
 spec=json.loads((R/'POSE_DENSE_PROTOCOL.json').read_text());ph=sha(R/'POSE_DENSE_PROTOCOL.json');assert sha(__file__)==spec['worker_sha256'];assert sha(A/'pose_state_fp16.pt')==spec['model_sha256'];assert sha(R/'IR_ADAPTER_INPUTS.json')==spec['inputs_sha256'];out=R/'pose_dense_output';out.mkdir(exist_ok=True);clips=out/'clips';clips.mkdir(exist_ok=True)
 start=time.time();guard=Guard(out,min(spec['deadline_unix'],start+spec['wall_cap_seconds']),out/'TELEMETRY.json');SETTINGS.update({'sync':False});torch_utils.NUM_THREADS=2;torch.set_num_threads(2);cv2.setNumThreads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 cp=torch.load(A/'pose_state_fp16.pt',map_location='cpu',weights_only=True);net=PoseModel(cp['yaml'],ch=3,nc=1,data_kpt_shape=(17,3),verbose=False);net.load_state_dict(cp['state_dict']);net.kpt_shape=[17,3];net.names={0:'person'};net.eval().requires_grad_(False);model=YOLO(str(A/'yolo11n-pose.pt'));model.model=net
 rows=json.loads((R/'IR_ADAPTER_INPUTS.json').read_text())['rows'];root=Path(spec['data_root']);records=[];frames_total=0
 with ThreadPoolExecutor(max_workers=8) as pool:
  for row in rows:
   guard.bounds();key=row['clip_id'];fn=hashlib.sha256(key.encode()).hexdigest()+'.npz';dest=clips/fn;meta=dest.with_suffix('.json');st=row['streams'][0];assert Path(st['directory']).parts[:5]==('Training','data','HAR','data','IR')
   if meta.exists():
    rec=json.loads(meta.read_text());assert rec['protocol_sha256']==ph and sha(dest)==rec['sha256'];records.append(rec);frames_total+=rec['frames'];continue
   vals=list(pool.map(decode,[root/st['directory']/f for f in st['files']]));kpts=[];boxes=[];pc=[]
   for begin in range(0,len(vals),32):
    batch=[x[0] for x in vals[begin:begin+32]];preds=model.predict(batch,imgsz=640,conf=.25,iou=.7,max_det=1,device='cuda:0',half=False,verbose=False,save=False)
    for p in preds:
     if len(p.boxes):
      k=p.keypoints.data[0].cpu().numpy().copy();k[(k[:,:2]<=0).any(1)]=0;kpts.append(k);boxes.append(p.boxes.xyxy[0].cpu().numpy());pc.append(float(p.boxes.conf[0]))
     else:kpts.append(np.zeros((17,3),np.float32));boxes.append(np.zeros(4,np.float32));pc.append(0.)
   np.savez_compressed(dest,keypoints=np.asarray(kpts,np.float32),boxes=np.asarray(boxes,np.float32),person_confidence=np.asarray(pc,np.float32));rec=dict(clip_id=key,file='clips/'+fn,sha256=sha(dest),frames=len(vals),native_size=[vals[0][0].shape[1],vals[0][0].shape[0]],frame_hashes=[x[1] for x in vals],protocol_sha256=ph);atomic(meta,rec);records.append(rec);frames_total+=len(vals)
   guard.check(phase='DENSE_LABEL_FREE_POSE',clips=len(records),total=len(rows),frames=frames_total)
   if len(records)%100==0:atomic(out/'PROGRESS.json',dict(clips=len(records),total=len(rows),frames=frames_total,seconds=time.time()-start,at_unix=time.time()));print(json.dumps(dict(clips=len(records),frames=frames_total,seconds=time.time()-start)),flush=True)
 atomic(out/'INDEX.json',dict(protocol_sha256=ph,records=records));atomic(out/'DONE.json',dict(status='COMPLETE',protocol_sha256=ph,index_sha256=sha(out/'INDEX.json'),clips=len(records),frames=frames_total,seconds=time.time()-start,no_Test=True,action_labels_used=False))
if __name__=='__main__':
 try:main()
 except BaseException as e:
  (R/'pose_dense_output').mkdir(exist_ok=True);atomic(R/'pose_dense_output/ERROR.json',dict(error=str(e),traceback=traceback.format_exc(),at_unix=time.time()));raise
