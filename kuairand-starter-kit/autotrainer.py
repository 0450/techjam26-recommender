"""
autotrainer.py
--------------
Main orchestrator for multi-seed heterogeneous model training, blending,
prediction validation, submission writing, and champion promotion.
"""

import os
import argparse
import time
import subprocess
import json
import torch

from data import load, encode
from submit import write_submission, read_submission
from utils import (
    mkdir, save_json, load_auxiliary_targets,
    train_architecture, optimize_blend, safety_checks
)


def main():
    parser = argparse.ArgumentParser(description="Heterogeneous Automated Recommender Trainer & Blender")
    parser.add_argument("--data-dir", type=str, default="./KuaiRand-Pure/data", help="Path to data directory")
    parser.add_argument("--out-dir", type=str, default="artifacts", help="Output directory for predictions and logs")
    parser.add_argument("--epochs", type=int, default=12, help="Max training epochs per seed")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--batch-size", type=int, default=8192, help="Training batch size")
    parser.add_argument("--seeds", type=str, default="42,1024,2026,7,999", help="Comma-separated seed integers")
    parser.add_argument("--epsilon", type=float, default=0.001, help="Score improvement threshold for champion promotion")
    args = parser.parse_args()

    mkdir(args.out_dir)
    seeds = [int(s) for s in args.seeds.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    print("Loading data splits and auxiliary MTL targets...")
    splits = load(args.data_dir)
    enc, num_features = encode(splits)
    aux_targets = load_auxiliary_targets(args.data_dir, splits)

    uva = enc['valid'][2]
    yva = enc['valid'][1]

    # Train 1st Architecture: Compact SENet DeepFM
    val_senet, test_senet, _ = train_architecture(
        'senet', seeds, enc, aux_targets, num_features, device,
        epochs=args.epochs, batch_size=args.batch_size, patience=args.patience, lr=4e-4
    )

    # Train 2nd Architecture: Low-Rank DCNv2 DeepFM
    val_dcn, test_dcn, _ = train_architecture(
        'lowrank_dcn', seeds, enc, aux_targets, num_features, device,
        epochs=args.epochs, batch_size=args.batch_size, patience=args.patience, lr=2e-4
    )

    # Optimize Power-Rank Blending
    print("\n" + "=" * 60)
    print("=== OPTIMIZING HETEROGENEOUS BLEND ON VALIDATION SET ===")
    print("=" * 60)
    blended_test_preds, best_val_res, w1, w2, power = optimize_blend(
        val_senet, val_dcn, test_senet, test_dcn, uva, yva
    )

    print("\n" + "=" * 60)
    print("=== FINAL OPTIMAL BLEND METRICS ===")
    print("=" * 60)
    print(f"  * SENet DeepFM Weight  : {w1:.2f}")
    print(f"  * Low-Rank DCNv2 Weight: {w2:.2f}")
    print(f"  * Optimal Power Exponent: {power:.1f}")
    print("-" * 60)
    print(f"  * Blended Val GAUC     : {best_val_res['GAUC']:.4f}")
    print(f"  * Blended Val nDCG@5   : {best_val_res['nDCG@5']:.4f}")
    print(f"  * Blended Val Primary  : {best_val_res['primary']:.4f}")
    print("=" * 60)

    # Validate output predictions
    safety_checks(blended_test_preds)

    timestamp = int(time.time())
    sub_path = os.path.join(args.out_dir, f"submission_heterogeneous_{timestamp}.csv")
    
    # Save submission using official hackathon write_submission
    write_submission(sub_path, splits['test'], blended_test_preds)
    print(f"\nWrote submission: {sub_path}")

    # Validate submission alignment using official submit.py read_submission
    read_submission(sub_path, splits['test'])
    print("✓ Format and row-alignment validation passed successfully!")

    # Champion Promotion Tracking
    champion_file = os.path.join(args.out_dir, "best_metrics.json")
    best_primary = -1.0

    if os.path.exists(champion_file):
        with open(champion_file, "r") as f:
            champion_data = json.load(f)
            best_primary = champion_data.get("mean_valid_primary", -1.0)

    candidate_primary = best_val_res["primary"]
    print(f"\nCandidate Primary Score: {candidate_primary:.4f} | Champion Primary Score: {best_primary:.4f}")

    if candidate_primary > best_primary + args.epsilon:
        print(" Candidate promoted to NEW CHAMPION!")
        champ_record = {
            "seeds": seeds,
            "epochs": args.epochs,
            "mean_valid_primary": candidate_primary,
            "valid_metrics": best_val_res,
            "senet_weight": w1,
            "dcn_weight": w2,
            "power_exponent": power,
            "submission_path": sub_path,
            "timestamp": timestamp
        }
        save_json(champ_record, champion_file)
    else:
        print("Candidate score did not exceed champion promotion threshold.")


if __name__ == "__main__":
    main()