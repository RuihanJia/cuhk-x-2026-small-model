"""Label-free person/head/arm geometry from fixed COCO pose predictions."""
import numpy as np
def box_from_detections(detections,width,height):
 points=[];valid=0;fallback=[]
 for d in detections:
  if not d['boxes']:continue
  b=np.asarray(d['boxes'][0]);fallback.extend([[b[0],b[1]],[b[2],b[1]+.75*(b[3]-b[1])]])
  k=np.asarray(d['keypoints'][0]);ok=(k[:,2]>=.5)&(k[:,0]>0)&(k[:,1]>0)
  # Ultralytics Keypoints explicitly zeroes x/y below0.5 confidence. Zero
  # coordinates denote missing observations and must never enlarge the hull.
  if ok[[5,6]].all() and ok[[9,10]].any():valid+=1;points.extend(k[:11,:2][ok[:11]].tolist())
 if valid<2:points=fallback
 if len(points)<4:return None,'NO_RELIABLE_PERSON'
 a=np.asarray(points);lo=a.min(0);hi=a.max(0);side=min(min(width,height),max(.25*min(width,height),1.35*float((hi-lo).max())));cx,cy=(lo+hi)/2;x=max(0,min(width-side,cx-side/2));y=max(0,min(height-side,cy-side/2))
 return [round(x),round(y),round(side),round(side)],'POSE_HEAD_ARMS' if valid>=2 else 'PERSON_UPPER_BODY_FALLBACK'
