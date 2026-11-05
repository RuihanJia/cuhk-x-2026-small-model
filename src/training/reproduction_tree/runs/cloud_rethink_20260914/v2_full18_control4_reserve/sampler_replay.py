from pathlib import Path
import os,json,hashlib,time
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import torch
from torch.utils.data import DataLoader
E=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(1);g=torch.Generator();loader=DataLoader(list(range(2931)),batch_size=8,shuffle=True,generator=g,num_workers=2,prefetch_factor=1,persistent_workers=True);reports=[]
 for ep in range(1,5):
  g.manual_seed(20260915+ep);actual=torch.cat(list(loader))
  expected=torch.Generator();expected.manual_seed(20260915+ep)
  if ep==1:torch.empty((),dtype=torch.int64).random_(generator=expected)
  predicted=torch.randperm(2931,generator=expected);assert torch.equal(actual,predicted)
  reports.append(dict(epoch=ep,records=len(actual),order_sha256=hashlib.sha256(actual.numpy().tobytes()).hexdigest(),expected_formula_exact=True))
 del loader
 p=E/'SAMPLER_INDEPENDENT_QA.json';p.write_text(json.dumps(dict(status='PASS_ACTUAL_PERSISTENT_DATALOADER_ORDER',epochs=reports,no_GPU=True,no_images=True,at_unix=time.time()),indent=2));print(p.read_text())
if __name__=='__main__':main()
