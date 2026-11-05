"""Regularized class-specific deviations from a source-fitted global mixture.

No class is named or manually routed. Only source out-of-fold labels fit the
160 deviations; temperatures and global prior are frozen source estimates.
"""
import numpy as np
import torch
from torch import nn

PENALTY=0.1

def logsumexp(x,axis,keepdims=False):
 m=np.max(x,axis=axis,keepdims=True)
 v=m+np.log(np.exp(x-m).sum(axis=axis,keepdims=True))
 return v if keepdims else np.squeeze(v,axis=axis)

class Classwise(nn.Module):
 def __init__(self,global_cal):
  super().__init__()
  assert global_cal['description']['mode']=='convex' and len(global_cal['temperature'])==4
  self.register_buffer('temperature',torch.tensor(global_cal['temperature'],dtype=torch.float64))
  self.register_buffer('prior',torch.tensor(global_cal['state']['bias'],dtype=torch.float64))
  self.delta=nn.Parameter(torch.zeros((4,40),dtype=torch.float64))
 def centered(self):return self.delta-self.delta.mean(0,keepdim=True)
 def forward(self,logits,mask):
  assert mask.any(1).all()
  logp=(logits/self.temperature[None,:,None]).log_softmax(-1)
  scores=self.prior[None,:,None]+self.centered()[None]
  lw=scores.expand(len(logits),-1,-1).masked_fill(~mask[:,:,None],-torch.inf).log_softmax(1)
  result=(lw+logp).logsumexp(1)
  return result-result.logsumexp(-1,keepdim=True)

def predict(logits,mask,cal):
 z=np.asarray(logits,dtype=np.float64);mask=np.asarray(mask,bool)
 assert z.ndim==3 and z.shape[1:]==(4,40) and mask.shape==z.shape[:2] and mask.any(1).all()
 temp=np.asarray(cal['temperature']);prior=np.asarray(cal['global_bias']);delta=np.asarray(cal['delta']);delta=delta-delta.mean(0,keepdims=True)
 p=z/temp[None,:,None];lp=p-logsumexp(p,-1,True)
 scores=np.broadcast_to(prior[None,:,None]+delta[None],z.shape).copy();scores[~mask]=-np.inf
 lw=scores-logsumexp(scores,1,True);out=logsumexp(lw+lp,1);out=out-logsumexp(out,-1,True)
 assert np.isfinite(out).all()
 return out.astype(np.float32)

def fit(logits,mask,targets,users,global_cal,clock):
 torch.set_num_threads(1)
 z=torch.tensor(logits,dtype=torch.float64);m=torch.tensor(mask,dtype=torch.bool);y=torch.tensor(targets,dtype=torch.int64)
 unique,counts=np.unique(users,return_counts=True);freq=dict(zip(unique,counts));weight=np.asarray([1./freq[u] for u in users],dtype=np.float64);weight/=weight.sum();weight=torch.from_numpy(weight)
 model=Classwise(global_cal);opt=torch.optim.LBFGS([model.delta],lr=1.,max_iter=500,tolerance_grad=1e-9,tolerance_change=1e-12,line_search_fn='strong_wolfe');calls=0
 def closure():
  nonlocal calls
  clock.check();opt.zero_grad();lp=model(z,m);ce=-(lp[torch.arange(len(y)),y]*weight).sum();loss=ce+PENALTY*model.centered().square().mean()
  assert torch.isfinite(loss);loss.backward();assert torch.isfinite(model.delta.grad).all();calls+=1;return loss
 opt.step(closure);clock.check()
 with torch.no_grad():lp=model(z,m);nll=float(-(lp[torch.arange(len(y)),y]*weight).sum())
 cal=dict(kind='SOURCE_REGULARIZED_CLASSWISE_CONVEX4_V1',temperature=model.temperature.tolist(),global_bias=model.prior.tolist(),delta=model.delta.detach().tolist(),penalty=PENALTY,parameters=160,optimizer='FP64_LBFGS500',closure_calls=calls,calibration_users=sorted(int(u) for u in unique),calibration_rows=len(users),outer_labels_used=False,source_user_balanced_nll=nll,source_fit_is_not_validation=True)
 q=predict(logits,mask,cal);assert np.max(np.abs(q-lp.numpy().astype(np.float32)))<1e-6 and np.array_equal(q.argmax(1),lp.numpy().argmax(1))
 return cal
