"""
runGrid.py
----------
Two-Stage Hyperparameter & Architecture Orchestrator.
- Stage 1: Fast proxy evaluation over individual models & learning rates.
- Stage 2: Full multi-seed execution and heterogeneous blending.
"""

import os
import argparse
import subprocess
import torch

from data import load, encode
from utils import grid_proposals, mkdir, save_json, load_auxiliary_targets, train_architecture


def main():
    parser = argparse.ArgumentParser(description="Two-Stage Heterogeneous Grid Search Orchestrator")
    parser.add_argument("--data-dir", type=str, default="./KuaiRand-Pure/data", help="Path to data directory")
    parser.add_argument("--out-dir", type=str, default="artifacts_pipeline", help="Output directory")
    parser.add_argument("--quick-epochs", type=int, default=3, help="Stage 1 quick proxy epochs")
    parser.add_argument("--proxy-seeds", type=str, default="42", help="Stage 1 seed")
    parser.add_argument("--full-seeds", type=str, default="42,1024,2026,7,999", help="Stage 2 seeds")
    parser.add_argument("--full-epochs", type=int, default=12, help="Stage 2 full training epochs")
    args = parser.parse_args()

    mkdir(args.out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("STAGE 1: Running Quick Proxy Grid Search")
    print("=" * 60)

    splits = load(args.data_dir)
    enc, num_features = encode(splits)
    aux_targets = load_auxiliary_targets(args.data_dir, splits)

    proposals = grid_proposals()
    proxy_seeds = [int(s) for s in args.proxy_seeds.split(",")]

    results = []
    for idx, cfg in enumerate(proposals, 1):
        print(f"\nEvaluating Proxy Configuration {idx}/{len(proposals)}: {cfg}")
        _, _, val_res = train_architecture(
            cfg["model_type"], proxy_seeds, enc, aux_targets, num_features, device,
            epochs=args.quick_epochs, batch_size=8192, patience=2, lr=cfg["lr"]
        )
        score = float(val_res["primary"])
        print(f"Candidate {idx} Proxy Primary Score: {score:.4f}")
        results.append({"config": cfg, "proxy_score": score})

    results.sort(key=lambda x: x["proxy_score"], reverse=True)
    save_json(results, os.path.join(args.out_dir, "proxy_summary.json"))

    print("\n" + "=" * 60)
    print("STAGE 2: Running Full Multi-Seed Heterogeneous Training & Blending")
    print("=" * 60)

    cmd = [
        "python3", "autotrainer.py",
        "--data-dir", args.data_dir,
        "--out-dir", args.out_dir,
        "--epochs", str(args.full_epochs),
        "--seeds", args.full_seeds
    ]
    subprocess.run(cmd, check=True)

    print("\nTwo-Stage Heterogeneous Grid Pipeline Execution Complete!")


if __name__ == "__main__":
    main()