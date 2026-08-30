#!/usr/bin/env python3
"""
train_lightgbm.py

LightGBM baseline for KuaiRand-style data.

Features and behavior:
- Encodes categorical IDs consistently on the full dataset (avoids train/val mismatch).
- Robust timestamp parsing (supports epoch seconds or ISO strings).
- Supports time-based split (default), random split (--random-split), or per-user holdout (--per-user-holdout).
- Uses LightGBM callbacks for early stopping and logging.
- Prints diagnostic stats about train/validation user distributions before training.
- Computes Validation ROC AUC, GAUC, and nDCG@5 after training.
"""
import argparse
import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import callback
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib
from metrics import gauc, ndcg_at_k_grouped

def prepare_features_full_df(df):
    df = df.copy()
    # Ensure IDs are strings then create consistent integer mappings across full df
    cat_cols = [c for c in ['user_id', 'item_id', 'creator_id', 'category_id'] if c in df.columns]
    for c in cat_cols:
        df[c] = df[c].astype(str)
        uniques = pd.unique(df[c])
        mapping = {v: i for i, v in enumerate(uniques)}
        df[c] = df[c].map(mapping).astype(int)

    # Robust timestamp parsing: support epoch seconds (int) or ISO strings
    if 'timestamp' in df.columns:
        if pd.api.types.is_integer_dtype(df['timestamp'].dtype) or pd.api.types.is_float_dtype(df['timestamp'].dtype):
            ts = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
        else:
            ts = pd.to_datetime(df['timestamp'], errors='coerce')
        df['hour'] = ts.dt.hour.fillna(-1).astype(int)
        df['dayofweek'] = ts.dt.dayofweek.fillna(-1).astype(int)

    df = df.fillna(-1)
    return df

def print_split_diagnostics(train_df, val_df, label_col='label'):
    print("\n--- Split diagnostics ---")
    print("Train rows:", len(train_df), "Val rows:", len(val_df))
    # Per-user stats in validation
    if 'user_id' in val_df.columns:
        vc = val_df.groupby('user_id')[label_col].agg(['count','sum']).reset_index()
        total_users = vc.shape[0]
        users_more_than_one = (vc['count'] > 1).sum()
        users_with_both = ((vc['sum'] > 0) & (vc['sum'] < vc['count'])).sum()
        print(f"Validation users: {total_users}")
        print(f"Users with >1 impression in val: {users_more_than_one}")
        print(f"Users with both pos & neg in val (counts used for GAUC): {users_with_both}")
        print("Sample per-user stats (first 10):")
        print(vc.head(10).to_string(index=False))
    else:
        print("No user_id column for per-user diagnostics.")
    print("-------------------------\n")

def main(args):
    df = pd.read_csv(args.data)
    label_col = args.label_col
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {args.data}. Available columns: {list(df.columns)}")

    # Encode IDs and create time features on the full dataset
    df = prepare_features_full_df(df)

    # Choose split strategy
    if args.per_user_holdout:
        if 'timestamp' not in df.columns:
            raise ValueError("per-user holdout requires a 'timestamp' column in the data.")
        df = df.sort_values(['user_id', 'timestamp'])
        # take last `holdout_last` rows per user as validation
        val_idx = df.groupby('user_id').tail(args.holdout_last).index
        val_df = df.loc[val_idx].reset_index(drop=True)
        train_df = df.drop(val_idx).reset_index(drop=True)
    elif 'timestamp' in df.columns and not args.random_split:
        df = df.sort_values('timestamp')
        split_idx = int(len(df) * (1 - args.val_frac))
        train_df = df.iloc[:split_idx].reset_index(drop=True)
        val_df = df.iloc[split_idx:].reset_index(drop=True)
    else:
        train_df, val_df = train_test_split(df, test_size=args.val_frac, random_state=42, shuffle=True)

    # diagnostic prints
    print_split_diagnostics(train_df, val_df, label_col=label_col)

    # features: all columns except label and timestamp (keep encoded ID ints)
    exclude = {label_col, 'timestamp'}
    feature_cols = [c for c in train_df.columns if c not in exclude]

    # sanity checks
    if len(feature_cols) == 0:
        raise RuntimeError("No feature columns detected after excluding label/timestamp. Columns available: " + ", ".join(train_df.columns))

    y_train = train_df[label_col].astype(int)
    y_val = val_df[label_col].astype(int)

    dtrain = lgb.Dataset(train_df[feature_cols], label=y_train)
    dval = lgb.Dataset(val_df[feature_cols], label=y_val, reference=dtrain)

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'seed': 42,
        'verbosity': -1
    }

    callbacks = [
        callback.early_stopping(stopping_rounds=50),
        callback.log_evaluation(period=50),
    ]

    print("Starting LightGBM training...")
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=args.num_rounds,
        valid_sets=[dtrain, dval],
        callbacks=callbacks
    )

    best_iter = getattr(model, "best_iteration", None)
    if best_iter is None or best_iter <= 0:
        val_df['score'] = model.predict(val_df[feature_cols])
    else:
        val_df['score'] = model.predict(val_df[feature_cols], num_iteration=best_iter)

    # compute metrics
    try:
        global_auc = roc_auc_score(y_val, val_df['score'])
    except Exception:
        global_auc = float('nan')

    print("\n=== Validation metrics ===")
    print("Validation ROC AUC (global):", global_auc)
    print("Validation GAUC:", gauc(val_df.rename(columns={label_col: 'label'}), user_col='user_id', label_col='label', score_col='score'))
    print("Validation nDCG@5:", ndcg_at_k_grouped(val_df.rename(columns={label_col: 'label'}), user_col='user_id', label_col='label', score_col='score', k=5))
    print("==========================\n")

    # feature importance (top 20)
    try:
        names = model.feature_name()
        imps = model.feature_importance(importance_type='gain')
        print("Top feature importances (gain):")
        for n, i in sorted(zip(names, imps), key=lambda x: -x[1])[:20]:
            print(f"  {n}: {i:.2f}")
    except Exception:
        pass

    joblib.dump(model, args.out)
    print("Saved model to", args.out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to CSV data')
    parser.add_argument('--out', default='model.joblib', help='Output model path')
    parser.add_argument('--val-frac', type=float, default=0.2, help='Validation fraction for time/random split')
    parser.add_argument('--label-col', default='label', help='Name of label column (default: label)')
    parser.add_argument('--per-user-holdout', dest='per_user_holdout', action='store_true', help='Use last-k per-user holdout for validation')
    parser.add_argument('--holdout-last', type=int, default=1, help='Number of last interactions per user to hold out as validation (when --per-user-holdout is used)')
    parser.add_argument('--random-split', dest='random_split', action='store_true', help='Use random split instead of time split')
    parser.add_argument('--num-rounds', type=int, default=2000, help='Number of boosting rounds')
    args = parser.parse_args()
    main(args)