"""Independent NumPy replay of the saved, source-fitted four-branch model."""
import numpy as np


def replay(logits, available, cal):
    logits = np.asarray(logits, dtype=np.float64)
    available = np.asarray(available, dtype=bool)
    assert logits.ndim == 3 and logits.shape[1:] == (4, 40)
    assert available.shape == logits.shape[:2] and available.any(1).all()
    if 'delta' in cal:
        temperature = np.asarray(cal['temperature'], dtype=np.float64)
        delta = np.asarray(cal['delta'], dtype=np.float64)
        delta = delta - delta.mean(0, keepdims=True)
        score = np.asarray(cal['global_bias'])[None, :, None] + delta[None]
    else:
        state = cal['state']
        raw = np.asarray(state['temperature_raw'], dtype=np.float64)
        temperature = .25 + 7.75 / (1. + np.exp(-raw))
        score = np.asarray(state['bias'], dtype=np.float64)[None, :, None]
    z = logits / temperature[None, :, None]
    z -= z.max(-1, keepdims=True)
    prob = np.exp(z)
    prob /= prob.sum(-1, keepdims=True)
    score = np.broadcast_to(score, logits.shape).copy()
    score[~available] = -np.inf
    score -= score.max(1, keepdims=True)
    weights = np.exp(score)
    weights /= weights.sum(1, keepdims=True)
    mixed = (weights * prob).sum(1)
    mixed /= mixed.sum(-1, keepdims=True)
    result = np.log(mixed).astype(np.float32)
    assert np.isfinite(result).all() and not weights[~available].any()
    return result
