#!/usr/bin/env python3
"""
Synthetic KuaiRand-style dataset generator.

Generates a CSV with columns:
- user_id (string)
- item_id (string)
- creator_id (string)
- category_id (string)
- timestamp (ISO 8601)
- label (0/1)       <- long_view target
- click (0/1)
- like (0/1)
- follow (0/1)
- play_time (float seconds)

Example:
  python generate_synthetic_data.py --out data/train.csv --n 2000

"""
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def make_generator(args):
    rng = np.random.RandomState(args.seed)

    # cardinalities
    num_users = args.users
    num_items = args.items
    num_creators = args.creators
    num_cats = args.categories

    # latent embeddings
    dim = 8
    user_emb = rng.normal(scale=1.0, size=(num_users, dim))
    item_emb = rng.normal(scale=1.0, size=(num_items, dim))
    creator_emb = rng.normal(scale=1.0, size=(num_creators, dim))
    cat_emb = rng.normal(scale=1.0, size=(num_cats, dim))

    # biases
    user_bias = rng.normal(scale=0.2, size=num_users)
    item_bias = rng.normal(scale=0.2, size=num_items)
    creator_bias = rng.normal(scale=0.1, size=num_creators)
    cat_bias = rng.normal(scale=0.1, size=num_cats)

    # activity skew for sampling users/items (some heavy users/items)
    user_weights = rng.exponential(scale=1.0, size=num_users)
    user_probs = user_weights / user_weights.sum()
    item_weights = rng.exponential(scale=1.0, size=num_items)
    item_probs = item_weights / item_weights.sum()

    # time window
    start_dt = datetime.fromisoformat(args.start_date)
    days = args.days

    def sample_one():
        # sample user/item/creator/category
        u = rng.choice(num_users, p=user_probs)
        i = rng.choice(num_items, p=item_probs)
        c = rng.randint(num_creators)
        cat = rng.randint(num_cats)

        # timestamp uniform in window
        sec_offset = rng.randint(0, days * 24 * 3600)
        ts = start_dt + timedelta(seconds=int(sec_offset))

        # latent affinity signals
        affinity = np.dot(user_emb[u], item_emb[i]) \
                   + 0.5 * np.dot(user_emb[u], creator_emb[c]) \
                   + 0.3 * np.dot(user_emb[u], cat_emb[cat]) \
                   + user_bias[u] + item_bias[i] + creator_bias[c] + cat_bias[cat]

        # make long_view probability (primary label) moderately sparse
        p_long_view = sigmoid(0.4 * affinity + rng.normal(scale=0.3))
        # click probability correlated but not identical
        p_click = sigmoid(0.6 * affinity + rng.normal(scale=0.6) - 0.5)
        # play_time depends strongly on long_view and affinity
        base_play = np.maximum(0.0, 5.0 + 3.0 * affinity + rng.normal(scale=2.0))
        # sample actual events
        long_view = rng.rand() < p_long_view
        click = rng.rand() < p_click
        # likes and follows are rarer and more likely if long_view or click
        p_like = 0.02 + 0.25 * p_long_view + 0.15 * p_click
        p_follow = 0.01 + 0.15 * p_long_view + 0.10 * p_click
        like = rng.rand() < p_like
        follow = rng.rand() < p_follow
        # play_time: if long_view true -> longer on average
        if long_view:
            play_time = max(0.0, rng.normal(loc=base_play + 6.0, scale=4.0))
        else:
            play_time = max(0.0, rng.normal(loc=base_play * 0.4, scale=2.5))

        row = {
            'user_id': f'u{u}',
            'item_id': f'i{i}',
            'creator_id': f'c{c}',
            'category_id': f'cat{cat}',
            'timestamp': ts.isoformat(sep=' '),
            'label': int(long_view),
            'click': int(click),
            'like': int(like),
            'follow': int(follow),
            'play_time': float(round(play_time, 3)),
        }
        return row

    return sample_one

def generate(args):
    rng = np.random.RandomState(args.seed)
    sampler = make_generator(args)

    rows = [sampler() for _ in range(args.n)]
    df = pd.DataFrame(rows)

    # minor post-processing: ensure some users have few/all positives/negatives
    # purposely create a small number of all-negative users to test GAUC exclusion behavior
    if args.ensure_all_neg_users > 0:
        # pick some users to force all-zero labels
        chosen = rng.choice(df['user_id'].unique(), size=min(args.ensure_all_neg_users, df['user_id'].nunique()), replace=False)
        for u in chosen:
            df.loc[df['user_id'] == u, 'label'] = 0

    # shuffle rows
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # save
    args.out.parent.mkdir(parents=True, exist_ok=True) if hasattr(args, 'out') else None
    df.to_csv(str(args.out), index=False)
    # print summary stats
    print(f"Wrote {len(df)} rows to {args.out}")
    print("Unique users:", df['user_id'].nunique())
    print("Unique items:", df['item_id'].nunique())
    print("Label distribution (long_view):")
    print(df['label'].value_counts().to_dict())
    print("Click distribution:")
    print(df['click'].value_counts().to_dict())
    print("Average play_time: %.3f" % df['play_time'].mean())
    print("\nSample rows:")
    print(df.head(8).to_string(index=False))

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=str, default='data/train.csv', help='Output CSV path')
    p.add_argument('--n', type=int, default=2000, help='Number of rows to generate')
    p.add_argument('--users', type=int, default=200, help='Number of distinct users')
    p.add_argument('--items', type=int, default=500, help='Number of distinct items')
    p.add_argument('--creators', type=int, default=150, help='Number of distinct creators')
    p.add_argument('--categories', type=int, default=12, help='Number of categories')
    p.add_argument('--start-date', type=str, default='2026-01-01', help='Start date (ISO) for timestamps')
    p.add_argument('--days', type=int, default=30, help='Span (days) to spread timestamps over')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    p.add_argument('--ensure-all-neg-users', type=int, default=3,
                   help='Force this many users to have all-negative labels (for metric edge cases)')
    args = p.parse_args()
    # make out a Path-like object string -> Path
    from pathlib import Path
    args.out = Path(args.out)
    return args

if __name__ == '__main__':
    args = _parse_args()
    generate(args)