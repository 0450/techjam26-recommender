"""
KuaiRand-Pure baselines.

  --model pop   : item popularity (official baseline, pure statistics, no training)
  --model fm    : Factorization Machine (starter model, students improve from here)
  --model random: random scoring (lower bound, for sanity check to ensure evaluation code is not broken)

Only depends on numpy. See README.md for usage.
"""
import argparse, collections, time
import numpy as np
from data import load, encode, FIELDS
from evaluate import evaluate

def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

# ---------------- item popularity (official baseline) ----------------
def run_pop(splits, prior=20.0):
    pos, imp = collections.Counter(), collections.Counter()
    for x in splits['train']:
        imp[x[2]] += 1; pos[x[2]] += x[6]
    gmean = sum(pos.values()) / sum(imp.values())
    score = lambda v: (pos[v] + prior * gmean) / (imp[v] + prior) if imp[v] else gmean
    out = {}
    for name in ('valid', 'test'):
        rws = splits[name]
        out[name] = evaluate([x[1] for x in rws], [x[6] for x in rws],
                             [score(x[2]) for x in rws])
    return out

def run_random(splits, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for name in ('valid', 'test'):
        rws = splits[name]
        out[name] = evaluate([x[1] for x in rws], [x[6] for x in rws],
                             rng.random(len(rws)))
    return out

# ---------------- Factorization Machine ----------------
class FM:
    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0, warmup_steps=100):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        # FIXED: Single per-feature bias (not double bu/bi which caused double counting)
        self.b_feat = np.zeros(dim, dtype=np.float32)

        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.warmup_steps = warmup_steps
        # Adam moments for parameters
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.mb_feat = np.zeros_like(self.b_feat); self.vb_feat = np.zeros_like(self.b_feat)
        self.t = 0

    def _get_lr(self):
        """Linear warmup for first warmup_steps, then constant."""
        return self.lr * min(1.0, self.t / max(1, self.warmup_steps))

    def logits(self, X):
        E = self.V[X]                                   # (B,F,k)
        S = E.sum(1)                                    # (B,k)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        # sum linear term W over row features, add single per-feature bias
        return self.b + self.W[X].sum(1) + self.b_feat[X].sum(1) + inter, E, S

    def step(self, X, y):
        B = len(y)
        z, E, S = self.logits(X)
        g = ((sigmoid(z) - y) / B).astype(np.float32)    # (B,)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        gb_feat = np.zeros_like(self.b_feat)
        np.add.at(gW, X, g[:, None])
        np.add.at(gb_feat, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        gV += self.l2 * self.V; gW += self.l2 * self.W
        bias_l2 = self.l2 * 0.1
        gb_feat += bias_l2 * self.b_feat

        self.t += 1
        current_lr = self._get_lr()
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= current_lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        # update bias with Adam-style steps
        M, Vv = self.mb_feat, self.vb_feat
        M *= b1; M += (1 - b1) * gb_feat
        Vv *= b2; Vv += (1 - b2) * (gb_feat * gb_feat)
        self.b_feat -= current_lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)

        self.b -= current_lr * g.sum()
        return float(-np.mean(y * np.log(sigmoid(z) + 1e-9) + (1 - y) * np.log(1 - sigmoid(z) + 1e-9)))

    def _adam(self, gV, gW, gb_feat):
        self.t += 1
        current_lr = self._get_lr()
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW),
                            (self.b_feat, gb_feat, self.mb_feat, self.vb_feat)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= current_lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)

    def step_listwise(self, X, y, uids):
        """ListNet-style softmax over each user's impressions in the batch (matches GAUC/nDCG).
        FIXED: Gradient scaled by batch size per user to avoid scale explosion."""
        B = len(y)
        z, E, S = self.logits(X)
        g = np.zeros(B, dtype=np.float32)
        loss = 0.0
        n_u = 0
        byu = {}
        for i, u in enumerate(uids):
            byu.setdefault(u, []).append(i)
        for idxs in byu.values():
            idxs = np.asarray(idxs)
            yy = y[idxs]
            npos = float(yy.sum())
            nneg = float(len(idxs) - npos)
            if npos <= 0 or nneg <= 0:
                continue
            zz = z[idxs]
            zz = zz - zz.max()
            p = np.exp(zz); p = p / (p.sum() + 1e-9)
            q = yy / npos
            # FIXED: Scale gradient by user batch size to avoid magnitude explosion
            g[idxs] = (p - q) / len(idxs)
            loss += float(-np.sum(q * np.log(p + 1e-9)))
            n_u += 1
        if n_u == 0:
            return self.step(X, y)
        g = (g / n_u).astype(np.float32)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        gb_feat = np.zeros_like(self.b_feat)
        np.add.at(gW, X, g[:, None])
        np.add.at(gb_feat, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        gV += self.l2 * self.V; gW += self.l2 * self.W
        bias_l2 = self.l2 * 0.1
        gb_feat += bias_l2 * self.b_feat
        self._adam(gV, gW, gb_feat)
        self.b -= self._get_lr() * g.sum()
        return loss / n_u

    def step_pair(self, Xp, Xn):
        """True BPR pairwise update: minimizes -log(sigmoid(z_pos - z_neg))."""
        B = len(Xp)
        zp, Ep, Sp = self.logits(Xp)
        zn, En, Sn = self.logits(Xn)
        d = zp - zn
        s = sigmoid(d)
        g = ((s - 1.0) / B).astype(np.float32)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        gb_feat = np.zeros_like(self.b_feat)
        np.add.at(gW, Xp, g[:, None]);  np.add.at(gW, Xn, -g[:, None])
        np.add.at(gb_feat, Xp, g[:, None]); np.add.at(gb_feat, Xn, -g[:, None])
        np.add.at(gV, Xp, g[:, None, None] * (Sp[:, None, :] - Ep))
        np.add.at(gV, Xn, -g[:, None, None] * (Sn[:, None, :] - En))
        gV += self.l2 * self.V; gW += self.l2 * self.W
        bias_l2 = self.l2 * 0.1
        gb_feat += bias_l2 * self.b_feat

        self.t += 1
        current_lr = self._get_lr()
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= current_lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        M, Vv = self.mb_feat, self.vb_feat
        M *= b1; M += (1 - b1) * gb_feat
        Vv *= b2; Vv += (1 - b2) * (gb_feat * gb_feat)
        self.b_feat -= current_lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        return float(-np.mean(np.log(s + 1e-9)))

    def predict(self, X, bs=200_000):
        return np.concatenate([self.logits(X[i:i + bs])[0] for i in range(0, len(X), bs)])

def listwise_batches(X, y, users, rng, bs, target_users=32):
    """FIXED: Yield batches with approximately target_users unique users per batch.
    This ensures more stable per-user gradient variance."""
    byu = {}
    for i, u in enumerate(users):
        byu.setdefault(u, []).append(i)
    ulist = list(byu.keys())
    rng.shuffle(ulist)
    batch_idx = []
    batch_users = set()
    for u in ulist:
        batch_idx.extend(byu[u])
        batch_users.add(u)
        if len(batch_users) >= target_users or len(batch_idx) >= bs:
            ii = np.asarray(batch_idx)
            yield X[ii], y[ii], [users[j] for j in ii]
            batch_idx = []
            batch_users = set()
    if batch_idx:
        ii = np.asarray(batch_idx)
        yield X[ii], y[ii], [users[j] for j in ii]


def run_fm(splits, k=16, lr=0.001, epochs=40, bs=8192, patience=4, seed=0, verbose=True,
           loss='pointwise', neg_per_pos=1, extra=False):
    """
    loss: 'pointwise' | 'bpr' (true pairwise) | 'listwise' (per-user softmax)
    extra: sequence / hour / pop-bucket fields (not the failed static CWM user buckets)
    """
    enc, dim = encode(splits, extra=extra)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    m = FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0

    user_neg = {}
    if loss in ('bpr', 'bpr_pw'):
        for idx, u in enumerate(utr):
            if ytr[idx] == 0.0:
                user_neg.setdefault(u, []).append(idx)

    for ep in range(1, epochs + 1):
        t0 = time.time()
        if loss == 'pointwise':
            idx = rng.permutation(len(ytr))
            losses = [m.step(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]]) for i in range(0, len(idx), bs)]
        elif loss == 'listwise':
            losses = [m.step_listwise(Xb, yb, ub) for Xb, yb, ub in listwise_batches(Xtr, ytr, utr, rng, bs)]
        elif loss == 'bpr_pw':
            pos_indices = [i for i, val in enumerate(ytr) if val == 1.0]
            rng.shuffle(pos_indices)
            losses = []
            for start in range(0, len(pos_indices), max(1, bs)):
                chunk = pos_indices[start:start + bs]
                rows, labels = [], []
                for p_idx in chunk:
                    u = utr[p_idx]
                    pool = user_neg.get(u)
                    for _ in range(neg_per_pos):
                        if pool:
                            n_idx = rng.choice(pool)
                        else:
                            n_idx = rng.integers(len(ytr))
                            while n_idx == p_idx:
                                n_idx = rng.integers(len(ytr))
                        rows.append(Xtr[n_idx]); labels.append(0.0)
                    rows.append(Xtr[p_idx]); labels.append(1.0)
                if rows:
                    losses.append(m.step(np.stack(rows), np.array(labels, dtype=np.float32)))
            if not losses:
                idx = rng.permutation(len(ytr))
                losses = [m.step(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]]) for i in range(0, len(idx), bs)]
        else:
            pos_indices = [i for i, val in enumerate(ytr) if val == 1.0]
            rng.shuffle(pos_indices)
            losses = []
            for start in range(0, len(pos_indices), max(1, bs)):
                batch_pos = pos_indices[start:start + bs]
                pos_rows, neg_rows = [], []
                for p_idx in batch_pos:
                    u = utr[p_idx]
                    neg_pool = user_neg.get(u)
                    for _ in range(neg_per_pos):
                        if neg_pool and len(neg_pool) > 0:
                            n_idx = rng.choice(neg_pool)
                        else:
                            n_idx = rng.integers(len(ytr))
                            while n_idx == p_idx:
                                n_idx = rng.integers(len(ytr))
                        pos_rows.append(Xtr[p_idx]); neg_rows.append(Xtr[n_idx])
                if pos_rows:
                    losses.append(m.step_pair(np.stack(pos_rows), np.stack(neg_rows)))
            if not losses:
                idx = rng.permutation(len(ytr))
                losses = [m.step(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]]) for i in range(0, len(idx), bs)]

        va = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b), m.b_feat.copy())
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  early stop at epoch {ep}")
                break
    if best_state is not None:
        m.V, m.W, m.b, m.b_feat = best_state
    return {'valid': evaluate(uva, yva, m.predict(Xva)),
            'test':  evaluate(ute, yte, m.predict(Xte))}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data',
                    help='KuaiRand-Pure data directory after extraction')
    ap.add_argument('--model', default='fm', choices=['pop', 'fm', 'random'])
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--bs', type=int, default=8192)
    ap.add_argument('--patience', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--loss', choices=['pointwise','bpr','listwise','bpr_pw'], default='pointwise', help='Training loss style')
    ap.add_argument('--neg-per-pos', type=int, default=1, help='When using bpr, negatives sampled per positive')
    ap.add_argument('--extra', action='store_true', help='Add sequence/hour/pop-bucket fields (usually hurts)')
    a = ap.parse_args()
    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    used = (['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']
            + (['hour_bucket', 'pop_bucket', 'last_video', 'last_pos_video'] if a.extra else []))
    print({k_: len(v) for k_, v in splits.items()}, f"fields={used}")
    res = {'pop': run_pop, 'random': lambda s: run_random(s, a.seed),
           'fm': lambda s: run_fm(s, k=a.k, lr=a.lr, epochs=a.epochs, bs=a.bs,
                                  patience=a.patience, seed=a.seed, verbose=True,
                                  loss=a.loss, neg_per_pos=a.neg_per_pos,
                                  extra=a.extra)}[a.model](splits)
    print(f"\n=== {a.model} (seed={a.seed}, loss={a.loss}) ===")
    for sp in ('valid', 'test'):
        r = res[sp]
        print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")