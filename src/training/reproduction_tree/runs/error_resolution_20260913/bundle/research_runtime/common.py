from __future__ import annotations
import hashlib, json, os, random, time
from pathlib import Path
import numpy as np
import torch

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest().upper()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest().upper()

def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)

def save_torch(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    torch.save(value, tmp); os.replace(tmp, path)

def seed(value):
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(value)

def configure():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

def restore_rng(value):
    random.setstate(value['python']); np.random.set_state(value['numpy']); torch.set_rng_state(value['torch'])
    if value['cuda']: torch.cuda.set_rng_state_all(value['cuda'])

def runtime():
    import platform, torchvision
    return dict(python=platform.python_version(), torch=torch.__version__, torchvision=torchvision.__version__,
                numpy=np.__version__, platform=platform.platform(), cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

def verify_bundle(bundle):
    bundle = Path(bundle).resolve(); manifest = read(bundle / 'BUNDLE_MANIFEST.json')
    for name, expected in manifest['sha256'].items():
        path = (bundle / name).resolve()
        if not path.is_relative_to(bundle) or sha(path) != expected:
            raise ValueError('BUNDLE_HASH_MISMATCH:' + name)
    p = read(bundle / 'protocol.json'); p['_sha'] = sha(bundle / 'protocol.json')
    for name, expected in p['code_sha256'].items():
        if sha(bundle/name)!=expected:raise ValueError('PROTOCOL_CODE_HASH_MISMATCH:'+name)
    p['_bundle'] = str(bundle)
    return p

def assert_disjoint(train_users, held_users):
    overlap = set(map(int, train_users)) & set(map(int, held_users))
    if overlap: raise ValueError('USER_LEAKAGE:' + str(sorted(overlap)))

def prior_floor(labels, classes=40, smoothing=.1):
    q = (1-smoothing)*np.bincount(labels, minlength=classes)/len(labels) + smoothing/classes
    return float(-(q*np.log(q)).sum())

from dual_host_runtime.io import atomic,save_torch
