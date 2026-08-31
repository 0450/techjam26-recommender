"""run_grid.py

Two-stage pipeline:

run a cheap proxy grid (quick epochs, proxy seeds) for a set of candidate configs
pick top-K proxy candidates by mean_valid_primary
run a full multi-seed evaluation for each top candidate
Example (PowerShell single-line): .venv\Scripts\python .\runGrid.py --data-dir .\KuaiRand-Pure\data --out-dir artifacts_pipeline --quick-epochs 6 --proxy-seeds 0 --full-seeds 0,1,2 --full-epochs 40 --top-k 2 --verbosity minimal

If you want to split the command across lines in PowerShell, use the backtick ` at the end of each line (not backslash). """

import argparse
import json
import os
import subprocess
import sys
import time
from glob import glob

def run_cmd(cmd, cwd=None):
    print(">>>", " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd)

def find_latest_run_json(outdir):
    files = glob(os.path.join(outdir, "run_*.json"))
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def load_run_mean(run_json):
    try:
        with open(run_json, encoding='utf8') as fh:
            r = json.load(fh)
            return float(r.get('mean_valid_primary', r.get('mean_valid', None,)))
    except Exception:
        return None

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def build_candidate_name(c):
    parts = [f"k{c['k']}", f"lr{c['lr']}", c['loss']]
    if c.get('loss') == 'bpr':
        parts.append(f"neg{c.get('neg_per_pos',1)}")
    return "_".join(parts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--out-dir', default='artifacts_pipeline')
    ap.add_argument('--quick-epochs', type=int, default=6, help='epochs for proxy runs')
    ap.add_argument('--proxy-seeds', default='0', help='comma-separated seeds for proxy (usually single seed)')
    ap.add_argument('--full-seeds', default='0,1,2', help='comma-separated seeds for full evaluation')
    ap.add_argument('--full-epochs', type=int, default=40, help='epochs for full evaluation')
    ap.add_argument('--top-k', type=int, default=1, help='how many top proxy candidates to full-evaluate')
    ap.add_argument('--verbosity', choices=['minimal','normal','verbose'], default='minimal')
    ap.add_argument('--python', default=sys.executable, help='Python interpreter to run autotrainer (use venv python)')
    args = ap.parse_args()

    python = args.python
    data_dir = args.data_dir
    base_out = args.out_dir
    quick_epochs = args.quick_epochs
    proxy_seeds = args.proxy_seeds
    full_seeds = args.full_seeds
    full_epochs = args.full_epochs
    top_k = args.top_k
    verbosity = args.verbosity

    ensure_dir(base_out)
    timestamp = int(time.time())
    proxy_base = os.path.join(base_out, f"proxy_{timestamp}")
    ensure_dir(proxy_base)

    # Define a small candidate grid to try as proxies (edit as desired)
    grid = []
    for k in (16, 32):
        for lr in (0.001, 0.0005):
            grid.append({'k': k, 'lr': lr, 'loss': 'pointwise'})
            grid.append({'k': k, 'lr': lr, 'loss': 'bpr', 'neg_per_pos': 1})
            grid.append({'k': k, 'lr': lr, 'loss': 'listwise'})

    proxy_results = []

    # 1) Run proxy candidates
    print("Running proxy grid: %d candidates -> outdir: %s" % (len(grid), proxy_base))
    for i, cand in enumerate(grid):
        name = f"{i:02d}_{build_candidate_name(cand)}"
        outdir = os.path.join(proxy_base, name)
        ensure_dir(outdir)
        cmd = [python, "autotrainer.py",
               "--data-dir", data_dir,
               "--k", str(cand['k']),
               "--lr", str(cand['lr']),
               "--loss", cand['loss'],
               "--epochs", str(quick_epochs),
               "--seeds", proxy_seeds,
               "--out-dir", outdir,
               "--verbosity", verbosity]
        if cand.get('loss') == 'bpr':
            cmd += ["--neg-per-pos", str(cand.get('neg_per_pos', 1))]
        # run proxy
        try:
            run_cmd(cmd)
        except subprocess.CalledProcessError as e:
            print("Proxy run failed for", name, "-> skipping. Error:", e)
            continue

        # find the run json and read mean_valid_primary
        run_json = find_latest_run_json(outdir)
        mean = None
        if run_json:
            try:
                with open(run_json, encoding='utf8') as fh:
                    r = json.load(fh)
                    mean = float(r.get('mean_valid_primary', float('nan')))
            except Exception:
                mean = None
        proxy_results.append({'cand': cand, 'outdir': outdir, 'run_json': run_json, 'mean_valid_primary': mean})
        print(f" Proxy candidate {name} -> mean_valid_primary={mean}")

    # sort proxies by mean_valid_primary desc
    proxy_results = [p for p in proxy_results if p.get('mean_valid_primary') is not None]
    proxy_results.sort(key=lambda x: x['mean_valid_primary'], reverse=True)

    if not proxy_results:
        print("No successful proxy runs found. Exiting.")
        return

    # write proxy summary
    proxy_summary = os.path.join(base_out, "proxy_summary.json")
    with open(proxy_summary, 'w', encoding='utf8') as fh:
        json.dump(proxy_results, fh, indent=2, ensure_ascii=False)
    print("Wrote proxy summary:", proxy_summary)

    # 2) pick top-K and run full evaluation
    topk = proxy_results[:top_k]
    full_results = []
    for rank, p in enumerate(topk, start=1):
        cand = p['cand']
        name = f"full_{rank:02d}_{build_candidate_name(cand)}"
        full_outdir = os.path.join(base_out, name)
        ensure_dir(full_outdir)
        cmd = [python, "autotrainer.py",
               "--data-dir", data_dir,
               "--k", str(cand['k']),
               "--lr", str(cand['lr']),
               "--loss", cand['loss'],
               "--epochs", str(full_epochs),
               "--seeds", full_seeds,
               "--out-dir", full_outdir,
               "--verbosity", verbosity]
        if cand.get('loss') == 'bpr':
            cmd += ["--neg-per-pos", str(cand.get('neg_per_pos', 1))]
        print("Launching full evaluation for rank %d candidate -> %s" % (rank, name))
        try:
            run_cmd(cmd)
        except subprocess.CalledProcessError as e:
            print("Full run failed for", name, "-> skipping. Error:", e)
            continue

        # collect run info
        run_json = find_latest_run_json(full_outdir)
        best_metrics = None
        if os.path.exists(os.path.join(full_outdir, "best_metrics.json")):
            try:
                with open(os.path.join(full_outdir, "best_metrics.json"), encoding='utf8') as fh:
                    best_metrics = json.load(fh)
            except Exception:
                best_metrics = None
        mean_full = None
        if run_json:
            try:
                with open(run_json, encoding='utf8') as fh:
                    r = json.load(fh)
                    mean_full = float(r.get('mean_valid_primary', None))
            except Exception:
                mean_full = None
        full_results.append({'cand': cand, 'full_outdir': full_outdir, 'run_json': run_json,
                             'best_metrics': best_metrics, 'mean_valid_primary': mean_full})
        print(f" Full candidate {name} -> mean_valid_primary={mean_full}")

    summary = {
        'timestamp': int(time.time()),
        'proxy_base': proxy_base,
        'proxy_results': proxy_results,
        'full_results': full_results,
    }
    summary_path = os.path.join(base_out, "grid_pipeline_summary.json")
    with open(summary_path, 'w', encoding='utf8') as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print("Pipeline finished. Summary written to:", summary_path)
    print("Top proxy candidates:")
    for i, p in enumerate(proxy_results[:top_k], start=1):
        print(f" {i}. {build_candidate_name(p['cand'])} mean_valid_primary={p['mean_valid_primary']} outdir={p['outdir']}")

if __name__ == '__main__':
    main()