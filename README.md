# techjam26-recommender

If you have trouble extracting the KuaiRand-Pure.tar.gz, I have uploaded the extracted folder to `https://files.catbox.moe/ej9pmm.7z`. Just extract internal folder and drop in into .\kuairand-starter-kit


This repository contains a KuaiRand recommender benchmark project built around the official starter kit and a validation-driven model training pipeline.

The goal is to reproduce the organizer baseline, measure performance with the official evaluator, and iterate on model design using only the public train/validation split before selecting a final test submission.

## Repository layout

- `kuairand-starter-kit/` — main benchmark implementation
  - `data.py` — data loading, official split logic, and feature encoding
  - `evaluate.py` — organizer scoring protocol; treat as the ground truth
  - `baseline.py` — baseline recommender experiments (`pop`, `random`, `fm`)
  - `submit.py` — submission generation and validation
  - `train_heterogeneous_blend.py` — advanced multi-seed heterogeneous ensemble trainer
  - `KuaiRand-Pure/` — extracted benchmark dataset
- `tests/` — project test area (keep additional checks here)

## Quick start

### 1) Clone and enter the repo

```powershell
cd d:\techjam26-recommender
```

### 2) Create a local Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 3) Install dependencies

The starter benchmark itself runs with NumPy, but the full model training pipeline uses additional packages.

```powershell
pip install numpy pandas scipy torch
```

If you only need to reproduce the organizer baseline, `numpy` is the main requirement. If you want to run the heterogeneous model trainer, install the full stack above.

### 4) Prepare the dataset

The dataset must be placed under `kuairand-starter-kit/KuaiRand-Pure/data`.

You can either:

- extract the official archive into `kuairand-starter-kit/`, or
- place the extracted `KuaiRand-Pure` folder there directly.

If you have trouble extracting the source archive, the extracted folder is also available here:

`https://files.catbox.moe/ej9pmm.7z`

After extraction, the structure should look like this:

```text
kuairand-starter-kit/
  KuaiRand-Pure/
    data/
      log_standard_4_08_to_4_21_pure.csv
      log_standard_4_22_to_5_08_pure.csv
      video_features_basic_pure.csv
      ...
```

## How to navigate the project

### Benchmark logic

- The benchmark definition lives in `kuairand-starter-kit/data.py` and `kuairand-starter-kit/evaluate.py`.
- Do not change the evaluator logic unless you are intentionally doing a local research experiment and understand what you are changing.
- The task is a within-user ranking problem, not a flat classification problem.

### Baselines and checks

- `baseline.py` is the cleanest place to reproduce the organizer baseline and sanity-check metrics.
- `submit.py` is the validation layer for CSV submission format and score computation.

### Advanced training

- `train_heterogeneous_blend.py` contains the more advanced ensemble pipeline with multi-seed training, rank normalization, and validation-based blending.
- This is the file to use when experimenting with structure changes, blend weights, or training objectives.

## Running the benchmark

Run commands from the repo root or from inside `kuairand-starter-kit` depending on the script.

### Sanity check: random baseline

```powershell
cd d:\techjam26-recommender
.\.venv\Scripts\python.exe .\kuairand-starter-kit\baseline.py --model random --data_dir .\kuairand-starter-kit\KuaiRand-Pure\data
```

Expected behavior: this should produce a valid score near the benchmark’s random lower bound without crashing.

### Official FM baseline

```powershell
cd d:\techjam26-recommender
.\.venv\Scripts\python.exe .\kuairand-starter-kit\baseline.py --model fm --data_dir .\kuairand-starter-kit\KuaiRand-Pure\data
```

This is the baseline to beat.

### Advanced ensemble training

```powershell
cd d:\techjam26-recommender\kuairand-starter-kit
.\.venv\Scripts\python.exe .\train_heterogeneous_blend.py
```

This script will:

- load the dataset,
- train the SENet and DCNv2-style model families,
- ensemble across seeds,
- search over power-rank normalization exponents and blend weights,
- and write a final submission file.

## Submission and local validation workflow

### Generate a submission from the FM baseline

```powershell
cd d:\techjam26-recommender\kuairand-starter-kit
.\.venv\Scripts\python.exe .\submit.py --make --split valid .\valid_submission.csv --data_dir .\KuaiRand-Pure\data
```

### Validate the CSV format and alignment

```powershell
cd d:\techjam26-recommender\kuairand-starter-kit
.\.venv\Scripts\python.exe .\submit.py --check --split valid .\valid_submission.csv --data_dir .\KuaiRand-Pure\data
```

### Score the submission locally on validation data

```powershell
cd d:\techjam26-recommender\kuairand-starter-kit
.\.venv\Scripts\python.exe .\submit.py --score --split valid .\valid_submission.csv --data_dir .\KuaiRand-Pure\data
```

This is the standard local loop for deciding whether a model change is promising before using test-set outputs.

## What the benchmark expects

The project follows the organizer rules:

- use the official KuaiRand splits,
- evaluate with the official `evaluate.py` scorer,
- optimize on the public validation split,
- keep the metric selection rule aligned with the benchmark’s definition of `primary` score,
- and only submit final test predictions once validation is promising.

The metrics are calculated at the user level in a within-user ranking setting:

- `GAUC`
- `nDCG@5`
- `primary = (GAUC + nDCG@5) / 2`

## Recommended workflow

1. Reproduce the random or FM sanity check.
2. Confirm the dataset and file structure are correct.
3. Run the FM baseline and record the validation metrics.
4. Try the advanced trainer in `train_heterogeneous_blend.py`.
5. Validate submission structure with `submit.py --check`.
6. Use validation metrics to decide whether to keep or reject an experiment.
7. Only then generate the final test submission.

## Troubleshooting

### Dataset not found

Check that the extracted folder is at:

```text
kuairand-starter-kit/KuaiRand-Pure/data
```

### Python package errors

Install the project dependencies explicitly:

```powershell
pip install numpy pandas scipy torch
```

### Submission validation fails

Common causes:

- wrong header
- wrong number of rows
- non-consecutive `row_id`
- mismatched `user_id` / `video_id`
- NaN or Infinity values in the score column

Use `submit.py --check` to catch these issues before submission.

## Notes

- The `evaluate.py` script is the benchmark ground truth and should be considered fixed.
- The starter baseline is a useful reference point, but the advanced ensemble pipeline is the main experimental system in this repo.
- A real no-frills workflow here is: benchmark -> baseline -> validation -> iterate -> final submission.

## Verified commands

The following commands were validated in this environment:

```powershell
.\.venv\Scripts\python.exe .\kuairand-starter-kit\baseline.py --model random --data_dir .\kuairand-starter-kit\KuaiRand-Pure\data
```

This produced a valid run with metrics for the public validation split, confirming the setup and data path are working in the checked-out workspace.