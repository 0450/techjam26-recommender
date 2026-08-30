import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def gauc(df, user_col='user_id', label_col='label', score_col='score'):
    """
    Compute GAUC as described: per-user AUC averaged with each user's positive count as weight.
    Users with all-positive or all-negative labels are excluded from AUC computation.
    df: DataFrame containing user, label (0/1), and score.
    Returns: GAUC (float)
    """
    grouped = df.groupby(user_col)
    aucs = []
    weights = []
    for user, g in grouped:
        y = g[label_col].values
        s = g[score_col].values
        if np.all(y == 0) or np.all(y == 1):
            continue
        try:
            auc = roc_auc_score(y, s)
        except Exception:
            continue
        aucs.append(auc)
        weights.append(y.sum())  # weight by positive count
    if len(aucs) == 0:
        return 0.0
    aucs = np.array(aucs)
    weights = np.array(weights, dtype=float)
    return float((aucs * weights).sum() / weights.sum())

def ndcg_at_k_grouped(df, user_col='user_id', label_col='label', score_col='score', k=5):
    """
    nDCG@k averaged over users. Users with no positives score 0 (and are included in the average).
    """
    def dcg(rels):
        # rels: list or array of relevance (0/1)
        gains = np.array(rels)
        discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
        return (gains * discounts).sum()

    grouped = df.groupby(user_col)
    ndcgs = []
    for user, g in grouped:
        # sort by score desc
        g_sorted = g.sort_values(by=score_col, ascending=False).head(k)
        rels = g_sorted[label_col].values
        dcg_val = dcg(rels)
        # ideal (sorted by label)
        ideal = np.sort(g[label_col].values)[::-1][:k]
        idcg = dcg(ideal)
        if idcg == 0:
            ndcgs.append(0.0)
        else:
            ndcgs.append(dcg_val / idcg)
    return float(np.mean(ndcgs))