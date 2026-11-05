import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.video import mvit_v2_s
from .common import assert_disjoint, read, sha

def freeze_bn_statistics(model):
    # Called AFTER every model.train(); keeps pretrained running statistics, retains trainable affine parameters.
    names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            if not module.track_running_stats: raise ValueError('BN_REQUIRES_RUNNING_STATISTICS')
            module.eval(); names.append(name)
    return names

def set_train_mode(model, architecture):
    model.train()
    if architecture == 'uniformer':
        freeze_bn_statistics(model)
        assert all(not m.training for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm))

def build_video(p, source, held_users=(), architecture='mvit'):
    from pathlib import Path
    bundle = Path(p['_bundle']); spec = p['assets'][source]; path = bundle/spec['path']
    if sha(path) != spec['sha256']: raise ValueError('INITIALIZATION_HASH_MISMATCH')
    assert_disjoint(spec.get('train_users', []), held_users)
    cp = torch.load(path, map_location='cpu', weights_only=True)
    if architecture == 'uniformer':
        from .vendor.uniformer import Uniformer, config
        model = Uniformer(config()); state = cp.get('model', cp.get('model_state', cp))
        model.load_state_dict(state, strict=True); model.head = nn.Linear(512, 40)
        freeze_bn_statistics(model)
    else:
        model = mvit_v2_s(weights=None)
        if source == 'mvit_public':
            model.load_state_dict(cp, strict=True); model.head = nn.Linear(768, 40)
        else:
            model.head = nn.Linear(768, 40)
            if cp['protocol_sha256'] != spec['source_protocol_sha256']: raise ValueError('SOURCE_PROTOCOL_MISMATCH')
            model.load_state_dict(cp['state_dict'], strict=True)
    return model

def view_logits(model, batch, device, architecture='mvit', non_blocking=False):
    x = batch['x'].to(device, non_blocking=non_blocking); n, views = x.shape[:2]
    flat = x.flatten(0,1)
    return model([flat] if architecture == 'uniformer' else flat).reshape(n, views, 40)

def objective(logits, y, arm, smoothing=.1):
    if arm == 'paired': return F.cross_entropy(logits.mean(1), y, label_smoothing=smoothing)
    if arm != 'individual': raise ValueError('UNKNOWN_OBJECTIVE')
    return F.cross_entropy(logits.flatten(0,1), y.repeat_interleave(logits.shape[1]), label_smoothing=smoothing)

def accumulation_weight(batch_n, batch_index, records, effective=8, accumulation=4):
    return batch_n / min(effective, records-(batch_index//accumulation)*effective)
