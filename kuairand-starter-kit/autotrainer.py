#!/usr/bin/env python3
"""
autotrainer.py - improved autotrainer for the KuaiRand starter kit.

This version prints a minimal console summary by default and supports
verbosity levels: minimal (default), normal, verbose.

Features:
- trains FM models (pointwise or bpr) across seeds
- small built-in grid proposer (optional)
- safety checks before promotion
- saves per-seed .npz models and run metrics (JSON)
- writes a submission CSV for the test split when promoting
- optionally runs submit.py to validate & score promoted submissions
"""
import argparse
import os
import json
import time
import datetime
import statistics
import numpy as np
import subprocess
import sys

from data import load, encode, pop_scores_for_split
from baseline import FM, listwise_batches
from evaluate import evaluate

# ---------------- utils ----------------
def mkdir(path):
    os.makedirs(path, exist_ok=True)

def save_npz_model(m, path):
    # Save V, W, global bias b, and per-feature bias b_feat
    np.savez(path, V=m.V, W=m.W, b=m.b, b_feat=m.b_feat)

def load_npz_model(path):
    d = np.load(path, allow_pickle=True)
    V = d['V']
    W = d['W']
    b = float(d['b']) if 'b' in d else 0.0
    b_feat = d['b_feat'] if 'b_feat' in d else np.zeros_like(W)
    return V, W, b, b_feat

def _to_serializable(x):
    # Recursively convert numpy types -> Python natives so json.dump works.
    if x is None:
        return None
    if isinstance(x, (str, bool, int, float)):
        return x
    # numpy scalar
    if hasattr(x, "item") and not isinstance(x, (list, tuple, dict)):
        try:
            return x.item()
        except Exception:
            pass
    # numpy arrays
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): _to_serializable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_serializable(v) for v in x]
    # fallback: try to coerce to float or string
    try:
        return float(x)
    except Exception:
        return str(x)

def save_json(obj, path):
    serial = _to_serializable(obj)
    with open(path, 'w', encoding='utf8') as fh:
        json.dump(serial, fh, indent=2, ensure_ascii=False)

def append_jsonl(obj, path):
    serial = _to_serializable(obj)
    with open(path, 'a', encoding='utf8') as fh:
        fh.write(json.dumps(serial, ensure_ascii=False) + '\n')

def make_submission_csv(path, rows, scores):
    # rows: original rows list from splits[split]; scores aligned to rows
    import csv
    HEADER = ['row_id', 'user_id', 'video_id', 'score']
    with open(path, 'w', newline='', encoding='utf8') as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        for i, (row, sc) in enumerate(zip(rows, scores)):
            w.writerow([i, row[1], row[2], f"{float(sc):.6g}"])

# ---------------- safety checks ----------------
def safety_checks(scores, topk=5):
    # scores: numpy array
    if not np.all(np.isfinite(scores)):
        return False, "NaN/Inf in scores"
    if np.std(scores) < 1e-8:
        return False, "scores collapsed to near-constant"
    return True, "ok"

# ---------------- proposer (small grid) ----------------
def grid_proposals():
    # small cheap grid — used for quick proxy trials
    grid = []
    for k in (8, 16):
        for lr in (0.0005, 0.001):
            for loss in ('pointwise', 'bpr', 'listwise'):
                grid.append({'k': k, 'lr': lr, 'loss': loss})
    return grid

# ---------------- verbosity helper ----------------
_VERB_LEVEL = {'minimal': 0, 'normal': 1, 'verbose': 2}
def _make_logger(level_name):
    level = _VERB_LEVEL.get(level_name, 0)
    def _log(msg, lvl=1):
        if lvl <= level:
            print(msg)
    return _log

# ---------------- training wrapper ----------------
def _zscore(x):
    mu, sd = float(np.mean(x)), float(np.std(x))
    if sd < 1e-8:
        return np.zeros_like(x)
    return (x - mu) / sd


def blend_with_pop(fm_scores, pop_scores, alpha):
    return _zscore(fm_scores) + alpha * _zscore(pop_scores)


def tune_pop_alpha(user_ids, labels, fm_scores, pop_scores, grid=None):
    """Item-pop is a strong within-user prior; blend weight chosen on valid only."""
    if grid is None:
        grid = (0.0, 0.15, 0.3, 0.5, 0.8, 1.2)
    best_a, best_p = 0.0, -1.0
    for a in grid:
        p = evaluate(user_ids, labels, blend_with_pop(fm_scores, pop_scores, a))['primary']
        if p > best_p:
            best_p, best_a = p, a
    return best_a, best_p


def train_fm_on_enc(enc, dim, args, seed, pop_va=None, pop_te=None):
    """Train a single FM instance using arrays returned from encode()."""
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']
    Xte, yte, ute = enc['test']

    rng = np.random.default_rng(seed)
    m = FM(dim, k=args.k, lr=args.lr, seed=seed)
    best = -1.0; best_state = None; bad = 0
    bs = args.bs
    start_seed = time.time()

    user_neg = {}
    if args.loss == 'bpr':
        for idx, u in enumerate(utr):
            if ytr[idx] == 0.0:
                user_neg.setdefault(u, []).append(idx)

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        if args.loss == 'pointwise':
            idx = rng.permutation(len(ytr))
            losses = [m.step(Xtr[idx[i:i+bs]], ytr[idx[i:i+bs]]) for i in range(0, len(idx), bs)]
        elif args.loss == 'listwise':
            losses = [m.step_listwise(Xb, yb, ub) for Xb, yb, ub in listwise_batches(Xtr, ytr, utr, rng, bs)]
        else:
            pos_indices = [i for i, v in enumerate(ytr) if v == 1.0]
            rng.shuffle(pos_indices)
            losses = []
            for start in range(0, len(pos_indices), max(1, bs)):
                chunk = pos_indices[start:start+bs]
                pos_rows, neg_rows = [], []
                for p_idx in chunk:
                    u = utr[p_idx]
                    pool = user_neg.get(u)
                    for _ in range(args.neg_per_pos):
                        if pool and len(pool) > 0:
                            n_idx = rng.choice(pool)
                        else:
                            n_idx = rng.integers(len(ytr))
                            while n_idx == p_idx:
                                n_idx = rng.integers(len(ytr))
                        pos_rows.append(Xtr[p_idx]); neg_rows.append(Xtr[n_idx])
                if pos_rows:
                    losses.append(m.step_pair(np.stack(pos_rows), np.stack(neg_rows)))
            if not losses:
                idx = rng.permutation(len(ytr))
                losses = [m.step(Xtr[idx[i:i+bs]], ytr[idx[i:i+bs]]) for i in range(0, len(idx), bs)]

        va_scores = m.predict(Xva)
        alpha = 0.0
        if pop_va is not None and getattr(args, 'pop_blend', True):
            alpha, _ = tune_pop_alpha(uva, yva, va_scores, pop_va)
            va = evaluate(uva, yva, blend_with_pop(va_scores, pop_va, alpha))
        else:
            va = evaluate(uva, yva, va_scores)

        if args.verbosity == 'verbose':
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | pop_alpha {alpha:.2f} | {time.time()-t0:.1f}s")
        elif args.verbosity == 'normal' and va['primary'] > best + 1e-6:
            print(f"  epoch {ep:2d} primary improved -> {va['primary']:.6f} (time {time.time()-t0:.1f}s)")

        if va['primary'] > best + 1e-5:
            best = va['primary']; bad = 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b), m.b_feat.copy(), float(alpha))
        else:
            bad += 1
            if bad >= args.patience:
                if args.verbosity != 'minimal':
                    print(f"  early stop at epoch {ep}")
                break

    pop_alpha = 0.0
    if best_state is not None:
        m.V, m.W, m.b, m.b_feat, pop_alpha = best_state
    va_scores = m.predict(Xva)
    te_scores = m.predict(Xte)
    if pop_va is not None and getattr(args, 'pop_blend', True):
        valid_metrics = evaluate(uva, yva, blend_with_pop(va_scores, pop_va, pop_alpha))
        test_metrics = evaluate(ute, yte, blend_with_pop(te_scores, pop_te, pop_alpha))
    else:
        valid_metrics = evaluate(uva, yva, va_scores)
        test_metrics = evaluate(ute, yte, te_scores)
    elapsed = time.time() - start_seed
    return m, valid_metrics, test_metrics, elapsed, pop_alpha

# ---------------- helper: run submit.py on a submission ----------------
def run_submit_check_and_score(sub_path, data_dir):
    submit_py = os.path.join(os.path.dirname(__file__), "submit.py")
    if not os.path.exists(submit_py):
        print("submit.py not found at", submit_py, " — skipping auto-check/score.")
        return
    print("Running submit.py --check on submission:", sub_path)
    rc = subprocess.run([sys.executable, submit_py, sub_path, "--check", "--data_dir", data_dir, "--split", "test"]).returncode
    if rc == 0:
        print("Check passed — running submit.py --score ...")
        subprocess.run([sys.executable, submit_py, sub_path, "--score", "--data_dir", data_dir, "--split", "test"])
    else:
        print("Submission check failed (submit.py returned code", rc, "). Not running score.")

# ---------------- main orchestrator ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--bs', type=int, default=8192)
    ap.add_argument('--patience', type=int, default=4)
    ap.add_argument('--loss', choices=['pointwise','bpr','listwise'], default='pointwise')
    ap.add_argument('--neg-per-pos', type=int, default=1)
    ap.add_argument('--out-dir', default='artifacts')
    ap.add_argument('--epsilon', type=float, default=0.002)
    ap.add_argument('--keep-models', action='store_true')
    ap.add_argument('--no-extra', action='store_true', help='Official 5 fields only (no sequence/hour/pop buckets)')
    ap.add_argument('--no-pop-blend', dest='pop_blend', action='store_false', help='Disable FM+pop score blend')
    ap.set_defaults(pop_blend=True)
    ap.add_argument('--grid', action='store_true', help='Run small built-in grid of candidates (quick proxy)')
    ap.add_argument('--quick-epochs', type=int, default=6, help='short epoch count for quick proxy runs (grid)')
    ap.add_argument('--verbosity', choices=['minimal','normal','verbose'], default='minimal',
                    help='Output verbosity (minimal=compact, normal=some details, verbose=per-epoch)')
    # Auto-submit behavior: on by default; use --no-auto-submit to turn off
    ap.add_argument('--no-auto-submit', dest='auto_submit', action='store_false',
                    help='Do not automatically run submit.py on promoted submission CSV (default: run it).')
    ap.set_defaults(auto_submit=True)
    args = ap.parse_args()

    mkdir(args.out_dir)
    log_path = os.path.join(args.out_dir, 'experiments_log.jsonl')
    best_file = os.path.join(args.out_dir, 'best_metrics.json')

    print("Loading data and encoding (this may take a moment)...")
    splits = load(args.data_dir)
    enc, dim = encode(splits, extra=not args.no_extra)
    pop_va = pop_scores_for_split(splits, 'valid')
    pop_te = pop_scores_for_split(splits, 'test')
    rows_test = splits['test']
    print(f"fields={enc.get('_fields')} dim={dim} pop_blend={args.pop_blend}")

    base_seeds = [int(s) for s in args.seeds.split(',') if s.strip()!='']

    # decide candidate list
    candidates = []
    if args.grid:
        for c in grid_proposals():
            # use quick epochs and single seed for the proxy
            c.update({'epochs': args.quick_epochs, 'seeds': [base_seeds[0]], 'bs': args.bs})
            candidates.append(c)
    else:
        candidates.append({'k': args.k, 'lr': args.lr, 'loss': args.loss,
                           'epochs': args.epochs, 'seeds': base_seeds, 'bs': args.bs,
                           'neg_per_pos': args.neg_per_pos})

    # load current champion metrics (primary + GAUC + nDCG) if present
    best_known = -1.0
    best_known_gauc = None
    best_known_ndcg = None
    if os.path.exists(best_file):
        try:
            with open(best_file,'r',encoding='utf8') as fh:
                best_record = json.load(fh)
                best_known = float(best_record.get('mean_valid_primary', -1.0))
                best_known_gauc = best_record.get('mean_valid_gauc', None)
                best_known_ndcg = best_record.get('mean_valid_ndcg', None)
        except Exception:
            best_known = -1.0
            best_known_gauc = None
            best_known_ndcg = None
    else:
        best_known = -1.0

    print("Current best known primary:", best_known,
          "GAUC:", best_known_gauc if best_known_gauc is not None else "N/A",
          "nDCG@5:", best_known_ndcg if best_known_ndcg is not None else "N/A")

    # iterate candidates
    for cand in candidates:
        cand_epochs = int(cand.get('epochs', args.epochs))
        cand_seeds = cand.get('seeds', base_seeds)
        cand_k = int(cand.get('k', args.k))
        cand_lr = float(cand.get('lr', args.lr))
        cand_loss = cand.get('loss', args.loss)
        cand_neg = int(cand.get('neg_per_pos', args.neg_per_pos))
        cand_bs = int(cand.get('bs', args.bs))

        print("Running candidate:", {'k': cand_k, 'lr': cand_lr, 'loss': cand_loss,
                                     'epochs': cand_epochs, 'seeds': cand_seeds, 'bs': cand_bs})

        per_seed_results = []
        model_paths = []
        start_all = time.time()
        for s in cand_seeds:
            run_args = argparse.Namespace(k=cand_k, lr=cand_lr, epochs=cand_epochs,
                                          bs=cand_bs, patience=args.patience, loss=cand_loss,
                                          neg_per_pos=cand_neg, verbosity=args.verbosity,
                                          pop_blend=args.pop_blend)
            if args.verbosity != 'minimal':
                print(f" Training seed {s} (epochs={cand_epochs}) ...")
            m, valid_metrics, test_metrics, elapsed_seed, pop_alpha = train_fm_on_enc(
                enc, dim, run_args, s, pop_va=pop_va, pop_te=pop_te)
            stamp = int(time.time())
            model_path = os.path.join(args.out_dir, f"model_k{cand_k}_lr{cand_lr}_{cand_loss}_s{s}_{stamp}.npz")
            save_npz_model(m, model_path)
            model_paths.append(model_path)
            per_seed_results.append({'seed': s, 'valid': valid_metrics, 'test': test_metrics,
                                     'model': model_path, 'elapsed_s': elapsed_seed, 'pop_alpha': pop_alpha})
            elapsed_str = str(datetime.timedelta(seconds=int(elapsed_seed)))
            print(f" seed {s} — valid={valid_metrics['primary']:.6f} test={test_metrics['primary']:.6f} elapsed={elapsed_str}")

        elapsed = time.time() - start_all
        mean_valid = statistics.mean(float(x['valid']['primary']) for x in per_seed_results)
        mean_test  = statistics.mean(float(x['test']['primary']) for x in per_seed_results)

        # compute mean GAUC and mean nDCG@5 across seeds for the validation and test splits
        mean_valid_gauc = statistics.mean(float(x['valid']['GAUC']) for x in per_seed_results)
        mean_valid_ndcg = statistics.mean(float(x['valid']['nDCG@5']) for x in per_seed_results)
        mean_test_gauc = statistics.mean(float(x['test']['GAUC']) for x in per_seed_results)
        mean_test_ndcg = statistics.mean(float(x['test']['nDCG@5']) for x in per_seed_results)

        # print a compact banner similar to the screenshot
        sep = "=" * 80
        print(sep)
        print(f"Val GAUC  : {mean_valid_gauc:0.4f} | Val nDCG@5 : {mean_valid_ndcg:0.4f} | Val Primary : {mean_valid:0.4f}")
        print(f"Test GAUC : {mean_test_gauc:0.4f} | Test nDCG@5 : {mean_test_ndcg:0.4f} | Test Primary: {mean_test:0.4f}")
        print(sep)

        # save run metrics
        run_summary = {
            'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'candidate': {'k': cand_k, 'lr': cand_lr, 'loss': cand_loss, 'epochs': cand_epochs, 'bs': cand_bs},
            'per_seed': per_seed_results,
            'mean_valid_primary': mean_valid,
            'mean_valid_gauc': mean_valid_gauc,
            'mean_valid_ndcg': mean_valid_ndcg,
            'mean_test_primary': mean_test,
            'mean_test_gauc': mean_test_gauc,
            'mean_test_ndcg': mean_test_ndcg,
            'elapsed_seconds': elapsed,
        }
        run_filename = os.path.join(args.out_dir, f"run_{int(time.time())}.json")
        save_json(run_summary, run_filename)
        append_jsonl({'ts': run_summary['timestamp'], 'run_file': run_filename, 'mean_valid_primary': mean_valid}, log_path)
        if args.verbosity != 'minimal':
            print("Saved run metrics:", run_filename)

        # decide promotion
        promoted = False
        print(f"Candidate mean_valid_primary={mean_valid:.6f}, best_known={best_known:.6f}, epsilon={args.epsilon}")
        # quick safety check: test inference on test split with the first model (as representative)
        first_model = model_paths[0]
        V,W,b,b_feat = load_npz_model(first_model)
        def fm_predict_with_params(X):
            E = V[X]  # (N,F,k)
            S = E.sum(1)
            inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1,2)))
            return b + W[X].sum(1) + b_feat[X].sum(1) + inter
        Xte_arr = enc['test'][0]
        test_scores = fm_predict_with_params(Xte_arr)
        ok, reason = safety_checks(test_scores)
        if not ok:
            print("Safety check failed for candidate:", reason)
            if not args.keep_models:
                for p in model_paths:
                    try: os.remove(p)
                    except: pass
            continue

        mean_alpha = statistics.mean(float(x.get('pop_alpha', 0.0)) for x in per_seed_results)
        logits = None
        for p in model_paths:
            Vp, Wp, bp, bp_feat = load_npz_model(p)
            E = Vp[Xte_arr]; S = E.sum(1)
            inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
            cur = bp + Wp[Xte_arr].sum(1) + bp_feat[Xte_arr].sum(1) + inter
            logits = cur if logits is None else logits + cur
        logits /= len(model_paths)
        if args.pop_blend:
            logits = blend_with_pop(logits, pop_te, mean_alpha)
        sub_path = os.path.join(args.out_dir, f"submission_{int(time.time())}.csv")
        make_submission_csv(sub_path, rows_test, logits)
        print("Wrote submission for test split:", sub_path)

        if mean_valid > best_known + args.epsilon:
            promoted = True
            print("PROMOTE candidate -> new best!")
            best_known = mean_valid
            best_models_dir = os.path.join(args.out_dir, 'best_models')
            mkdir(best_models_dir)
            promoted_paths = []
            for p in model_paths:
                dest = os.path.join(best_models_dir, os.path.basename(p))
                with open(p,'rb') as rf, open(dest,'wb') as wf:
                    wf.write(rf.read())
                promoted_paths.append(dest)
            best_record = {
                'timestamp': run_summary['timestamp'],
                'candidate': run_summary['candidate'],
                'mean_valid_primary': mean_valid,
                'mean_valid_gauc': mean_valid_gauc,
                'mean_valid_ndcg': mean_valid_ndcg,
                'mean_test_primary': mean_test,
                'promoted_models': promoted_paths,
                'run_metrics': run_filename
            }
            save_json(best_record, best_file)
            print("Updated best metrics:", best_file)

        else:
            print("Candidate did not beat best_known; not promoting.")
            if not args.keep_models:
                for p in model_paths:
                    try: os.remove(p)
                    except: pass

        if getattr(args, 'auto_submit', True):
            run_submit_check_and_score(sub_path, args.data_dir)

        # detailed run summary with improvement vs previous best
        val_primaries = [float(x['valid']['primary']) for x in per_seed_results]
        test_primaries = [float(x['test']['primary']) for x in per_seed_results]
        n_seeds = len(val_primaries)
        sd_valid = statistics.pstdev(val_primaries) if n_seeds > 1 else 0.0
        sd_test  = statistics.pstdev(test_primaries) if n_seeds > 1 else 0.0

        sd_valid_gauc = statistics.pstdev([float(x['valid']['GAUC']) for x in per_seed_results]) if n_seeds > 1 else 0.0
        sd_valid_ndcg = statistics.pstdev([float(x['valid']['nDCG@5']) for x in per_seed_results]) if n_seeds > 1 else 0.0

        delta = mean_valid - best_known if best_known is not None else mean_valid
        pct = (delta / best_known * 100.0) if (best_known and best_known > 0) else None

        elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))
        promoted_flag = "PROMOTED" if promoted else "NOT_PROMOTED"

        print("RUN SUMMARY:")
        print(f"  validation primary: {mean_valid:.6f} ± {sd_valid:.6f} (n={n_seeds})")
        print(f"  test primary      : {mean_test:.6f} ± {sd_test:.6f}")
        print(f"  validation GAUC   : {mean_valid_gauc:.6f} ± {sd_valid_gauc:.6f}")
        print(f"  validation nDCG@5 : {mean_valid_ndcg:.6f} ± {sd_valid_ndcg:.6f}")
        if best_known is not None and best_known >= 0:
            if pct is not None:
                print(f"  improvement vs best: d={delta:.6f} (~{pct:.3f}%)  [epsilon={args.epsilon}]")
            else:
                print(f"  improvement vs best: d={delta:.6f}  [epsilon={args.epsilon}]")
        else:
            print(f"  improvement vs best: d={delta:.6f}  [no previous champion]")
        print(f"  elapsed: {elapsed_str}  => {promoted_flag}")

    print("All candidates processed. Current best:", best_known)

if __name__ == '__main__':
    main()