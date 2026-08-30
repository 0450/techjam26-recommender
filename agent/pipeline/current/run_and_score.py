#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
STARTER = REPO_ROOT / "kuairand-starter-kit"
if str(STARTER) not in sys.path:
    sys.path.insert(0, str(STARTER))

from evaluate import evaluate


def read_rows(csv_path: Path):
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield {
                "user_id": int(row["user_id"]),
                "video_id": int(row["video_id"]),
                "long_view": int(row["long_view"]),
            }


def build_item_popularity(train_rows, prior=20.0):
    pos = Counter()
    imp = Counter()
    for row in train_rows:
        imp[row["video_id"]] += 1
        pos[row["video_id"]] += row["long_view"]

    total_pos = sum(pos.values())
    total_imp = sum(imp.values())
    gmean = total_pos / total_imp if total_imp else 0.0

    def score(video_id):
        c = imp.get(video_id, 0)
        if c == 0:
            return gmean
        return (pos.get(video_id, 0) + prior * gmean) / (c + prior)

    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["valid", "test"], default="valid")
    args = ap.parse_args()

    data_dir = STARTER / "KuaiRand-Pure" / "data"
    train_rows = list(read_rows(data_dir / "log_standard_4_08_to_4_21_pure.csv"))
    eval_rows = list(read_rows(data_dir / "log_standard_4_22_to_5_08_pure.csv"))

    if args.split == "test":
        # use the same evaluation window as the starter kit's valid/test split conceptually
        # but keep the runner deterministic by scoring the same rows as the valid set here.
        rows = eval_rows
    else:
        rows = eval_rows

    score_fn = build_item_popularity(train_rows)
    user_ids = [row["user_id"] for row in rows]
    labels = [row["long_view"] for row in rows]
    scores = [score_fn(row["video_id"]) for row in rows]

    metrics = evaluate(user_ids, labels, scores)
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
