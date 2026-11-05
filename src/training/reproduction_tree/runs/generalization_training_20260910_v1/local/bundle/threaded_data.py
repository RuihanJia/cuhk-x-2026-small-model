"""Bounded CPU decoding threads with the exact existing DataLoader sample order."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import torch
from torch.utils.data._utils.collate import default_collate

class ThreadedLoader:
    def __init__(self,ds,batch,workers,shuffle):
        self.ds=ds;self.batch=batch;self.workers=min(4,max(1,workers));generator=torch.Generator().manual_seed(ds.seed+ds.epoch*1000003)
        self.order_source=torch.utils.data.DataLoader(ds,batch_size=batch,shuffle=shuffle,num_workers=0,generator=generator,drop_last=False)
    def __len__(self):return len(self.order_source)
    def __iter__(self):
        dl=self.order_source
        # _BaseDataLoaderIter consumes one64-bit base seed before RandomSampler draws.
        torch.empty((),dtype=torch.int64).random_(generator=dl.generator)
        batches=iter(dl.batch_sampler);pending=deque()
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            def enqueue():
                try:indices=next(batches)
                except StopIteration:return False
                pending.append([pool.submit(self.ds.__getitem__,i) for i in indices]);return True
            for _ in range(3):
                if not enqueue():break
            while pending:
                value=default_collate([future.result() for future in pending.popleft()]);enqueue()
                if torch.cuda.is_available():value={k:v.pin_memory() if isinstance(v,torch.Tensor) else v for k,v in value.items()}
                yield value

def loader(ds,batch=2,workers=4,shuffle=False):return ThreadedLoader(ds,batch,workers,shuffle)
