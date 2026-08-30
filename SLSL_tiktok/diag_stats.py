#!/usr/bin/env python3
"""
Simple diagnostics for data/train.csv
Usage:
  python diag_stats.py data/train.csv
Prints:
- row & column counts
- first few rows
- per-user counts and positive counts (based on time-based 80/20 split)
"""
import pandas as pd
import sys

def main(path):
    df = pd.read_csv(path)
    print("Path:", path)
    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print("\nFirst rows:")
    print(df.head().to_string(index=False))

    # convert timestamp if present (try epoch seconds)
    if 'timestamp' in df.columns:
        try:
            if pd.api.types.is_integer_dtype(df['timestamp'].dtype) or pd.api.types.is_float_dtype(df['timestamp'].dtype):
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
            else:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            print("\nTimestamp range:", df['timestamp'].min(), "to", df['timestamp'].max())
        except Exception as e:
            print("Warning: timestamp parsing failed:", e)

        # 80/20 time split diagnostics
        df_sorted = df.sort_values('timestamp')
        split_idx = int(len(df_sorted) * 0.8)
        val = df_sorted.iloc[split_idx:]
        print("\nValidation rows (time-based 80/20):", len(val))
        if 'user_id' in val.columns and 'label' in val.columns:
            vc = val.groupby('user_id')['label'].agg(['count', 'sum']).reset_index()
            total_users = vc.shape[0]
            users_more_than_one = (vc['count'] > 1).sum()
            users_with_both = ((vc['sum'] > 0) & (vc['sum'] < vc['count'])).sum()
            print("Validation users:", total_users)
            print("Users with >1 impression in val:", users_more_than_one)
            print("Users with both pos & neg in val:", users_with_both)
            print("\nSample per-user stats (first 10):")
            print(vc.head(10).to_string(index=False))
        else:
            print("Validation does not have columns 'user_id' and 'label' for per-user diagnostics.")
    else:
        print("No 'timestamp' column found — cannot run time-split diagnostics.")

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/train.csv'
    main(path)