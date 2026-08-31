"""Within-user linear blend of FM logits and train-only ranking priors.

Weights are chosen on valid primary only (no test peek). evaluate.py is unchanged.
"""
import numpy as np
from evaluate import evaluate


def zscore_by_user(users, scores):
    scores = np.asarray(scores, dtype=np.float64)
    out = np.zeros(len(scores), dtype=np.float64)
    byu = {}
    for i, u in enumerate(users):
        byu.setdefault(u, []).append(i)
    for idxs in byu.values():
        s = scores[idxs]
        sd = float(s.std())
        if sd < 1e-8:
            out[idxs] = 0.0
        else:
            out[idxs] = (s - s.mean()) / sd
    return out


def combine(users, parts, weights):
    acc = None
    for name, w in weights.items():
        if abs(w) < 1e-12:
            continue
        z = zscore_by_user(users, parts[name])
        acc = w * z if acc is None else acc + w * z
    if acc is None:
        return np.zeros(len(users), dtype=np.float64)
    return acc


def tune_weights(users, labels, parts, rounds=3, seed=0):
    """Coordinate ascent on valid primary. `parts` must include 'fm'."""
    names = list(parts.keys())
    if 'fm' in names:
        names = ['fm'] + [n for n in names if n != 'fm']
    rng = np.random.default_rng(seed)
    w = {n: (1.0 if n == 'fm' else 0.0) for n in names}
    best = evaluate(users, labels, combine(users, parts, w))['primary']
    coarse = (-0.4, -0.2, 0.0, 0.15, 0.3, 0.5, 0.8, 1.2)
    for _ in range(rounds):
        order = names[:]
        rng.shuffle(order)
        for name in order:
            local_best_w, local_best = w[name], best
            for cand in coarse:
                w[name] = cand
                p = evaluate(users, labels, combine(users, parts, w))['primary']
                if p > local_best + 1e-6:
                    local_best, local_best_w = p, cand
            center = local_best_w
            for cand in np.linspace(center - 0.25, center + 0.25, 7):
                w[name] = float(cand)
                p = evaluate(users, labels, combine(users, parts, w))['primary']
                if p > local_best + 1e-6:
                    local_best, local_best_w = p, float(cand)
            w[name] = local_best_w
            best = local_best
    return w, best
