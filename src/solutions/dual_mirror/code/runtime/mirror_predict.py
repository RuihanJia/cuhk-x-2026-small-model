"""Equal original/mirror readouts; Base keeps seven views and Small keeps five."""
import numpy as np
import torch

@torch.inference_mode()
def predict(model,item,device,micro=4,kind='Base'):
 x=item['mae']
 if not x.numel():return np.zeros(40,np.float32),False
 assert x.shape==(7,3,16,224,224)and kind in ('Base','Small')
 x=x if kind=='Base'else x[:5]
 duplicate=torch.equal(x[0],x[1])and torch.equal(x[0],x[2]);selection=[0,3,4,5,6]if kind=='Base'else[0,3,4];chosen=x[selection]if duplicate else x
 def branch(values):
  logits=[]
  for i in range(0,len(values),micro):
   z=model(values[i:i+micro].to(device,non_blocking=True));assert z.dtype==torch.float32 and torch.isfinite(z).all();logits.append(z.cpu().numpy())
  z=np.concatenate(logits)
  if duplicate:z=z[[0,0,0,1,2,3,4]if kind=='Base'else[0,0,0,1,2]]
  return ((.5*z[:5].mean(0)+.25*z[5]+.25*z[6])if kind=='Base'else z.mean(0)).astype(np.float32)
 original=branch(chosen);mirrored=branch(chosen.flip(-1))
 return (.5*original+.5*mirrored).astype(np.float32),True

@torch.inference_mode()
def predict_mirror_only(model,item,device,micro=4,kind='Base'):
 x=item['mae']
 if not x.numel():return np.zeros(40,np.float32),False
 assert x.shape==(7,3,16,224,224)and kind in ('Base','Small')
 x=x if kind=='Base'else x[:5];duplicate=torch.equal(x[0],x[1])and torch.equal(x[0],x[2]);selection=[0,3,4,5,6]if kind=='Base'else[0,3,4];chosen=(x[selection]if duplicate else x).flip(-1);values=[]
 for i in range(0,len(chosen),micro):
  z=model(chosen[i:i+micro].to(device,non_blocking=True));assert z.dtype==torch.float32 and torch.isfinite(z).all();values.append(z.cpu().numpy())
 z=np.concatenate(values)
 if duplicate:z=z[[0,0,0,1,2,3,4]if kind=='Base'else[0,0,0,1,2]]
 return ((.5*z[:5].mean(0)+.25*z[5]+.25*z[6])if kind=='Base'else z.mean(0)).astype(np.float32),True
