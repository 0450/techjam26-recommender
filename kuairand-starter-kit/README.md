# KuaiRand-Pure Starter Kit

## Dependencies

Python 3.9+ and numpy. **Nothing else.** No need for torch, pandas, or sklearn.

## Data

Download from https://kuairand.com (Zenodo direct link, no registration needed):

```bash
# Execute in the Starter Kit directory. After extraction, ./KuaiRand-Pure/ will be created.
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz
```

## Running

```bash
python3 baseline.py --model fm
```

`--data_dir` defaults to `./KuaiRand-Pure/data`; specify it explicitly if your data is elsewhere.

`--model` options: `fm` (official baseline) / `pop` (trivial baseline) / `random` (lower bound, used to sanity-check evaluation code).

The full FM run takes about 40 seconds (CPU, single core).

## Task Definition — conventions are hard-coded; do not modify

| | |
|---|---|
| Task | **Within-user ranking** — for each user, only rank that user's impressions in the evaluation set; no full-corpus retrieval |
| Relevance label | `long_view` (native column, 0/1) |
| Metrics | `GAUC`, `nDCG@5`; **primary score = average of the two** |
| Data split | train `20220408–20220421` / valid `20220422–20220428` / test `20220429–20220508` |
| Users with zero positives | nDCG is recorded as 0.0 and included in the average; GAUC only counts users with `0 < number of positives < number of impressions`, weighted by number of positives |
| nDCG gain | `2^rel − 1` (equivalent to identity for binary labels) |

See `evaluate.py` for the implementation; all conventions are stated in the file header comments.

## Baseline Ladder

Scores on the test set. **The target to beat is the FM row.**

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| random (lower bound, for self-checking) | 0.4996 | 0.4511 | 0.4753 |
| item popularity (trivial) | 0.6308 | 0.5121 | 0.5715 |
| **FM (official baseline)** | **0.6610** | **0.5282** | **0.5946** |

### ⚠️ True metric range: the nDCG@5 ceiling is 0.729, not 1.0

Among the 23,875 users in the test set:

| | Percentage | Impact on metrics |
|---|---|---|
| All-negative users (all impressions for that user are not `long_view`) | **27.1%** | nDCG is always **0**; no model can fix this; excluded from GAUC |
| All-positive users | **9.2%** | nDCG is always **1**; excluded from GAUC |
| Discriminative users | **63.7%** | The actual samples used by GAUC |

Therefore, even using the true labels as prediction scores (oracle / perfect ranking) can only achieve:

| | random | FM baseline | **oracle ceiling** | Range consumed by FM |
|---|---|---|---|---|
| GAUC | 0.4996 | 0.6610 | **1.0000** | 32.3% |
| nDCG@5 | 0.4511 | 0.5282 | **0.7289** | 27.8% |
| **primary** | 0.4753 | **0.5946** | **0.8645** | **30.7%** |

**Please evaluate progress using the oracle as the denominator.** Thinking 0.5946 is "far from a perfect score of 1.0" is misleading — 
the baseline has already consumed about one third of the usable range, and the remaining headroom is 0.27 rather than 0.41.

Across 5 random seeds, the standard deviation of FM is **0.0008** for all metrics. Based on this, the convergence rule is set to **ε = 0.002 (≈2.5σ), N = 3**:
Stop when the validation primary score improves by no more than 0.002 for 3 consecutive iterations.

> Self-check: If running your evaluation code with `--model random` does not give primary ≈ 0.475 (±0.001), your harness is broken; fix it first.

## Submission Format

CSV with a header; each row corresponds to one row in the evaluation set:

```
row_id,user_id,video_id,score
0,0,7531,-3.34176
1,0,4214,-1.4955
...
```

| Field | Description |
|---|---|
| `row_id` | Consecutive 0-based row index corresponding to the row order of `data.load()[split]`. (Deterministic: first reads `log_standard_4_08_to_4_21_pure.csv`, then `log_standard_4_22_to_5_08_pure.csv`; after filtering by date, preserves the original file order) |
| `user_id` / `video_id` | Redundant fields used only to verify alignment |
| `score` | Score assigned by your model to this row. Any real number is allowed; only relative ordering matters. NaN / Inf is not allowed |

> **Why `row_id` is required:** `(user_id, video_id)` is **not unique** in the evaluation set —
> the test set has 3.06% duplicate pairs, with up to 12 repeats. Therefore, it cannot be used as the primary key.

Generation and validation:

```bash
python3 submit.py --make --split test submission.csv # generate an example submission using the official FM baseline
python3 submit.py --check --split test submission.csv # validate format and alignment
python3 submit.py --score --split valid submission.csv # validate and score locally (valid only)
```

`--check` will reject: wrong header, wrong number of rows, non-consecutive `row_id`, `user_id`/`video_id` misalignment with the evaluation set,
non-numeric scores, or NaN/Inf scores. **Run `--check` yourself before submitting.**

## Where to start modifying

The ranking below is **based on actual experiments**, not guesses. Dead ends already tested by the organizers are clearly marked; do not repeat them.

### Already tested: these two have no benefit, do not waste iterations

| Tried | Result |
|---|---|
| **Adding static features** — bringing in all 13 CWM feature fields (+`music_id`/`video_type`/`upload_type` + 6 coarse user-side buckets) | primary **0.5940** vs **0.5950** for the 5-field setup; no difference within noise, even slightly worse |
| **Increasing model capacity** — embedding dimension k = 8 / 16 / 32 | 0.5895 / 0.5902 / 0.5887; almost no movement |

Reason: the `user_id × video_id` interaction already captures most of the learnable signal. Coarse buckets such as `follow_user_num_range`
are redundant given `user_id`; and 1.14 million rows are not enough to support larger capacity. **The bottleneck is not features or capacity.**

⚠️ Also note: **first-order terms of pure user-side features contribute exactly 0 to the score.** Because ranking is performed within each user,
any term that is constant within a user does not change the within-group ordering (empirically, `item_pop × user bias` produces scores identical to pure `item_pop`). User-side features can only matter through
**interaction terms with item-side features**.

### Not explored: the headroom should be here

In our judgment, ordered by likelihood (**these have not been tested by the organizers; they are left for you**):

1. **Change the loss function.** The current one is pointwise logloss, but the metrics (GAUC / nDCG) are **ranking metrics**.
   Switch to pairwise (BPR) or listwise (softmax over that user's impressions) — aligning the training objective with the evaluation protocol
   is what we think is most likely to help.
2. **User behavior sequences.** Existing features do **not use behavior sequences at all**. In KuaiRand, each user has hundreds to thousands of train interactions.
   Interest modeling such as DIN / SIM is a completely unexplored direction here.
3. **Multi-objective learning.** The logs also contain `is_click`, `is_like`, `is_follow`, `is_comment`, `is_forward`, and `play_time_ms`;
   these can be used as auxiliary tasks for the main `long_view` task.
4. **Watch-time modeling.** The contribution of [CWM](https://github.com/hyz20/CWM) is precisely this: it models watch time via **censored regression**
   (when a video is watched to completion, the true watch time is censored, so it uses a one-sided loss rather than squared error). This is a research-rich direction.
5. **Change the model.** DeepFM / DCN / xDeepFM. Since capacity has empirically not been the bottleneck, put this after 1–4 in priority.
6. **Time features and distribution shift.** `hourmin`, `date`, and drift between train and test.
7. **Unbiased validation (advanced).** `log_random_4_22_to_5_08_pure.csv` is a random-exposure log (1.18 million rows),
   which can be used as an additional unbiased validation set to check whether the model is merely overfitting to biased traffic.

## Using your own model, including CWM

`evaluate.py` is completely decoupled from the model. It only needs three arrays of equal length:

```python
from evaluate import evaluate
print(evaluate(user_ids, labels, scores)) # scores can come from any model
```

- `user_ids`: `user_id` for each row in the evaluation set
- `labels`: `long_view` for that row (0/1)
- `scores`: score assigned by your model to that row; any real number, only relative ordering matters

So you do not have to use `baseline.py` at all. You can switch to PyTorch, LightGBM, or [CWM](https://github.com/hyz20/CWM)'s xDeepFM,
as long as you eventually pass `scores` to `evaluate()`. **The scoring protocol is determined solely by `evaluate.py`.**

> Notes on using CWM: it depends on `torch==1.6.0` (a 2020 version that likely will not install on newer GPUs),
> and its loss optimizes counterfactual watch time, while its evaluation label is a self-reconstructed `long_view2`.
> It is research code for a watch-time debiasing paper and can serve as an **advanced reference**, but is not recommended as a starting point.

## Files

| | |
|---|---|
| `evaluate.py` | Metric implementation + all convention definitions. **Do not modify.** |
| `data.py` | Data loading, official split, feature encoding. Modify this to add features. |
| `baseline.py` | Three baselines. FM is the one to beat. |
| `baseline_scores.json` | Official released scores + seed variance + convergence parameters. |
| `submit.py` | Generate / validate submission files. |
| `ablation_features.py` | Feature ablation experiment; reproduces the result that adding features does not help. |
| `perplexity_agent.py` | Autonomous FM hyperparameter search driven by the Perplexity Agent API. See below. |

## Perplexity Agent integration (`perplexity_agent.py`)

An autonomous research loop that asks the [Perplexity Agent API](https://docs.perplexity.ai)
for the next FM hyperparameter configuration to try, trains it, and feeds the validation
result back for the next round.

**Setup:**
```bash
pip install -r requirements-perplexity.txt   # official `perplexityai` SDK; nothing else
export PERPLEXITY_API_KEY=your_key_here      # get one at https://console.perplexity.ai
```
Never commit this key or paste it anywhere public. If it's ever exposed, rotate it in the
API Console immediately.

**Run:**
```bash
python3 perplexity_agent.py --max-iterations 6 --split test --output submission.csv
python3 perplexity_agent.py --no-agent        # local heuristic planner only, no API/network calls
```

**Test-set discipline is enforced in code, not just by convention:** the search loop
(`train_fm_once`, `run_experiment`, `baseline_experiment`) only ever reads
`enc['train']` / `enc['valid']`; the hidden/held-out split is never passed into a metric
call anywhere in the file, and nothing about it is ever sent to the Perplexity API. The
final submission CSV is produced by scoring the *features* of the requested `--split`
(same as `submit.py --make`) — its labels are never read.

Each round sends the last few validation results back to the model and asks for the next
`k` / `lr` / `epochs` / `patience` configuration via a strict JSON schema
(`response_format`), so responses don't need free-text parsing. The `web_search` tool is
enabled so the model can ground suggestions in public FM/CTR-tuning literature. Conversation
state carries over between rounds via `previous_response_id`, so only new information is
sent each call. If the API key is missing or a call fails (e.g. no network), the script logs
it and falls back to a local heuristic planner rather than crashing the run.
```
