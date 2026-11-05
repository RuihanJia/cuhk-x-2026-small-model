from __future__ import annotations
import hashlib, json, random
from pathlib import Path, PurePosixPath
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from .common import assert_disjoint, digest, read, sha

def safe_path(root, relative):
    parts = PurePosixPath(relative).parts
    if not parts or parts[0] != 'Training' or '..' in parts or '\\' in relative or ':' in relative:
        raise ValueError('TRAINING_PATH_REQUIRED:' + relative)
    root = Path(root).resolve(); result = root.joinpath(*parts).resolve()
    if not result.is_relative_to(root): raise ValueError('PATH_ESCAPES_DATA_ROOT')
    return result

def rows_from_bundle(bundle):
    return read(Path(bundle) / 'data/frame_index.json')

def verify_frames(rows, data_root, sample_only=False):
    # Enumerate all training image names/sizes; sampled payload hashes also bind actual image bytes.
    checked = 0
    for row in rows:
        for stream in row['streams']:
            directory = safe_path(data_root, stream['directory'])
            if not sample_only:
                actual = sorted((p.name, p.stat().st_size) for p in directory.glob('*.png'))
                if digest(actual) != stream['inventory_sha256']: raise ValueError('FRAME_INVENTORY:' + row['clip_id'])
            sample = directory / stream['sample_name']
            if sha(sample) != stream['sample_sha256']: raise ValueError('FRAME_PAYLOAD:' + row['clip_id'])
            checked += 1
    return dict(clips=len(rows), streams_checked=checked, all_names_and_sizes=not sample_only,
                limitation='Only one image payload per stream is hashed; all image names and sizes are checked.')

class Clips(torch.utils.data.Dataset):
    def __init__(self, rows, data_root, seed_value, training=False, epoch=0):
        self.rows, self.root, self.seed, self.training, self.epoch = rows, str(data_root), seed_value, training, epoch
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        key = f'{self.seed}:{self.epoch}:{row["clip_id"]}'.encode()
        rng = random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8], 'big'))
        relative = [rng.random() if self.training else .5 for _ in range(16)]
        flip = self.training and rng.random() < .5; views = []
        for stream in row['streams']:
            directory = safe_path(self.root, stream['directory']); files = stream['files']
            indices = [min(len(files)-1, int((k+relative[k])*len(files)/16)) for k in range(16)]
            frames = []
            for k in indices:
                with Image.open(directory / files[k]) as im: frames.append(TF.pil_to_tensor(im.convert('RGB')))
            value = torch.stack(frames).float().div_(255)
            value = TF.center_crop(TF.resize(value, [256], antialias=True), [224, 224])
            if flip: value = value.flip(-1)
            views.append(TF.normalize(value, [.45]*3, [.225]*3).permute(1,0,2,3).contiguous())
        return dict(x=torch.stack(views), y=row['label'], user=row['user'], clip_id=row['clip_id'])

def loader(ds, batch=2, workers=0, shuffle=False):
    generator = torch.Generator().manual_seed(ds.seed + ds.epoch*1000003)
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=workers,
                                     pin_memory=torch.cuda.is_available(), generator=generator, drop_last=False)

def split_rows(rows, train_users, predict_users):
    assert_disjoint(train_users, predict_users)
    train = [r for r in rows if r['user'] in train_users]
    pred = [r for r in rows if r['user'] in predict_users]
    if not train or not pred: raise ValueError('EMPTY_SPLIT')
    return train, pred

def sensor_arrays(bundle, train_users):
    raw = dict(np.load(Path(bundle)/'data/measured_sensors.npz', allow_pickle=False))
    allowed = {'clip_ids','targets','users','skeleton','imu','imu_valid'}
    if set(raw) != allowed: raise ValueError('MEASURED_ONLY_SENSOR_SCHEMA')
    train = np.flatnonzero(np.isin(raw['users'], train_users)); valid = raw['imu_valid']
    fit = train[valid[train]]
    if not len(fit): raise ValueError('NO_VALID_TRAIN_IMU')
    mean = raw['imu'][fit].mean(axis=(0,2), keepdims=True)
    std = raw['imu'][fit].std(axis=(0,2), keepdims=True).clip(1e-4)
    imu = ((raw['imu']-mean)/std).astype(np.float32); imu[~valid] = 0
    imu = np.concatenate([imu, np.broadcast_to(valid[:,None,None], (len(valid),1,256))], axis=1).astype(np.float32)
    return raw, train, fit, mean, std, torch.from_numpy(raw['skeleton'].transpose(0,2,1).copy()), torch.from_numpy(imu)
