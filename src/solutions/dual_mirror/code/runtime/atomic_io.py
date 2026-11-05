"""Single-writer atomic files with bounded Windows sharing retries."""
import json, os, time
from pathlib import Path
def retry(fn):
    for i in range(25):
        try:return fn()
        except PermissionError:
            if i==24:raise
            time.sleep(.02)
def read(path):
    return retry(lambda:json.loads(Path(path).read_text(encoding='utf-8-sig')))
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.partial')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    retry(lambda:os.replace(tmp,path))
def save_torch(path,value):
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.partial')
    with tmp.open('wb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
    retry(lambda:os.replace(tmp,path))
