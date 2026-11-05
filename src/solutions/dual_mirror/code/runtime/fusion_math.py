"""Small CPU-FP64 calibration models, fitted only to caller-supplied inner OOF arrays."""
import math
import numpy as np
import torch
from torch import nn

STRATEGIES={
    'convex_vs':dict(branches=[0,1],mode='convex',availability=False),
    'convex_vst':dict(branches=[0,1,2],mode='convex',availability=False),
    'gate_vst':dict(branches=[0,1,2],mode='gate',availability=True),
    'gate_vst_no_availability':dict(branches=[0,1,2],mode='gate',availability=False),
    'convex_vsti':dict(branches=[0,1,2,3],mode='convex',availability=False),
    'gate_vsti':dict(branches=[0,1,2,3],mode='gate',availability=True),
    'gate_vsti_no_availability':dict(branches=[0,1,2,3],mode='gate',availability=False),
}

def quality(logits,depth,ir,available):
    """Class-agnostic raw-probability entropy/margin and video Depth/IR JS distance."""
    z=torch.as_tensor(logits,dtype=torch.float64);p=z.softmax(-1)
    entropy=-(p*z.log_softmax(-1)).sum(-1)/math.log(z.shape[-1]);top=p.topk(2,dim=-1).values
    q=torch.stack([entropy,top[:,:,0]-top[:,:,1],torch.zeros_like(entropy)],-1)
    a=torch.as_tensor(depth,dtype=torch.float64).softmax(-1);b=torch.as_tensor(ir,dtype=torch.float64).softmax(-1);mid=(a+b)/2
    tiny=torch.finfo(torch.float64).tiny
    q[:,0,2]=(.5*(a*(a.clamp_min(tiny).log()-mid.clamp_min(tiny).log())).sum(-1)+.5*(b*(b.clamp_min(tiny).log()-mid.clamp_min(tiny).log())).sum(-1))/math.log(2)
    return torch.where(torch.as_tensor(available,dtype=torch.bool)[:,:,None],q,0.).numpy()

def prepare(data,branches,normalizer=None):
    logits=np.asarray(data['logits'])[:,branches];mask=np.asarray(data['available'],bool)[:,branches]
    assert np.isfinite(logits).all() and mask.any(1).all()
    q=quality(np.asarray(data['logits']),data['depth'],data['ir'],data['available'])[:,branches]
    if normalizer is None:
        assert mask.any(0).all(),'An entirely absent calibration branch needs a prospectively amended model'
        mu=np.stack([q[mask[:,j],j].mean(0) for j in range(len(branches))]);sd=np.stack([q[mask[:,j],j].std(0) for j in range(len(branches))]).clip(1e-6)
        normalizer=dict(quality_mean=mu.tolist(),quality_std=sd.tolist(),availability_mean=mask.mean(0).tolist())
    q=np.where(mask[:,:,None],(q-np.asarray(normalizer['quality_mean']))/np.asarray(normalizer['quality_std']),0.)
    return dict(logits=torch.tensor(logits,dtype=torch.float64),mask=torch.tensor(mask),quality=torch.tensor(q,dtype=torch.float64),availability=torch.tensor(mask.astype(float)-np.asarray(normalizer['availability_mean']),dtype=torch.float64)),normalizer

class Calibrator(nn.Module):
    def __init__(self,description):
        super().__init__();self.description=description;n=len(description['branches'])
        # Exactly T=1, equal branch weights, and zero quality/availability slopes at initialization.
        t=(1.-.25)/(8.-.25);self.temperature_raw=nn.Parameter(torch.full((n,),math.log(t/(1-t)),dtype=torch.float64));self.bias=nn.Parameter(torch.zeros(n,dtype=torch.float64))
        self.quality_weight=nn.Parameter(torch.zeros(n,3,dtype=torch.float64)) if description['mode']=='gate' else None
        self.availability_weight=nn.Parameter(torch.zeros(n,n,dtype=torch.float64)) if description['availability'] else None
    def forward(self,batch,return_weights=False):
        temperature=.25+7.75*self.temperature_raw.sigmoid();score=self.bias[None,:].expand(batch['logits'].shape[0],-1)
        if self.quality_weight is not None:score=score+(batch['quality']*self.quality_weight).sum(-1)
        if self.availability_weight is not None:score=score+batch['availability']@self.availability_weight.T
        score=score.masked_fill(~batch['mask'],-torch.inf);lw=score.log_softmax(1)
        logp=(lw[:,:,None]+(batch['logits']/temperature[None,:,None]).log_softmax(-1)).logsumexp(1)
        return (logp,lw.exp()) if return_weights else logp
    def penalty(self):
        values=[v for v in [self.quality_weight,self.availability_weight] if v is not None]
        return .01*sum(v.square().sum() for v in values)/sum(v.numel() for v in values) if values else self.bias.sum()*0

def fit(data,strategy,guard=None,max_iter=500):
    description=STRATEGIES[strategy];batch,normalizer=prepare(data,description['branches']);model=Calibrator(description)
    y=torch.as_tensor(data['targets'],dtype=torch.long);users=np.asarray(data['users']);order,counts=np.unique(users,return_counts=True);n=dict(zip(order,counts))
    weight=torch.tensor([1./n[u]/len(order) for u in users],dtype=torch.float64);assert abs(float(weight.sum())-1)<1e-12
    optimizer=torch.optim.LBFGS(model.parameters(),lr=1,max_iter=max_iter,max_eval=math.ceil(1.5*max_iter),history_size=50,tolerance_grad=1e-7,tolerance_change=1e-9,line_search_fn='strong_wolfe')
    trace=[]
    def closure():
        optimizer.zero_grad();logp=model(batch);nll=-(weight*logp[torch.arange(len(y)),y]).sum();loss=nll+model.penalty()
        assert torch.isfinite(loss),'NONFINITE_CALIBRATION_LOSS';loss.backward()
        assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters()),'NONFINITE_CALIBRATION_GRADIENT'
        trace.append(dict(evaluation=len(trace)+1,nll=float(nll.detach()),objective=float(loss.detach())))
        if guard:guard.check(phase='INNER_ONLY_FUSION_FIT',strategy=strategy,evaluations=len(trace),calibration_records=len(y))
        return loss
    optimizer.step(closure)
    with torch.no_grad():
        logp=model(batch);final_nll=float(-(weight*logp[torch.arange(len(y)),y]).sum());temperature=(.25+7.75*model.temperature_raw.sigmoid()).tolist()
    state={k:v.detach().tolist() for k,v in model.state_dict().items()};iterations=int(optimizer.state[model.temperature_raw].get('n_iter',0))
    return dict(strategy=strategy,description=description,normalizer=normalizer,state=state,temperature=temperature,calibration_users=order.tolist(),calibration_records=len(y),final_user_balanced_nll=final_nll,iterations=iterations,evaluations=len(trace),trace=trace,optimization_limit_reached=iterations>=max_iter,outer_labels_used=False)

def predict(data,value,return_weights=False):
    batch,_=prepare(data,value['description']['branches'],value['normalizer']);model=Calibrator(value['description']);model.load_state_dict({k:torch.tensor(v,dtype=torch.float64) for k,v in value['state'].items()},strict=True);model.eval()
    with torch.no_grad():logp,weights=model(batch,True)
    assert torch.isfinite(logp).all() and torch.allclose(logp.exp().sum(1),torch.ones(len(logp),dtype=torch.float64),atol=1e-10,rtol=0)
    return (logp.numpy().astype(np.float32),weights.numpy()) if return_weights else logp.numpy().astype(np.float32)
