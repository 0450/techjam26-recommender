# KuaiRand Baseline — Getting Started

A minimal reproducible project for a KuaiRand-style ranking baseline:
- Synthetic data generator
- LightGBM ranking baseline (feature engineering, validation, metrics)
- Diagnostic helpers for GAUC / nDCG@5

This README is written for a first-time user on Windows (PowerShell). Linux/macOS commands are shown where different.

---

## Quick summary

1. Activate your Python environment (see below).  
2. Generate sample data: `python generate_synthetic_data.py --out data/train.csv --n 2000`  
3. Inspect dataset: `python diag_stats.py data/train.csv`  
4. Train LightGBM baseline: `python train_lightgbm.py --data data/train.csv --out lightgbm_model.joblib`  
5. Inspect results or iterate.

---

## Prerequisites

- VS Code (you already have)
- Python 3.9–3.11
- Git (optional)
- Recommended: create a virtual environment (venv) or use conda for reproducibility

We expect you to run the commands from the project root (where this README lives).

---

## Setup (Windows PowerShell)

1. Create & activate a venv (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1