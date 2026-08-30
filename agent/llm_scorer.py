"""
LLM Scorer module for evaluating recommendations using KuaiRand-Pure metrics.
Imports evaluation functions from kuairand-starter-kit and provides a scoring interface.
"""
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import math
import collections


# Import scoring utilities from kuairand-starter-kit
def load_kuairand_evaluate():
    """Dynamically load evaluate module from kuairand-starter-kit."""
    kit_path = Path(__file__).parent.parent / "kuairand-starter-kit"
    if kit_path not in sys.path:
        sys.path.insert(0, str(kit_path))
    
    try:
        from evaluate import evaluate as evaluate_func
        return evaluate_func
    except ImportError:
        # Fallback: define evaluate locally
        return define_evaluate()


def define_evaluate():
    """
    Fallback implementation of evaluate function.
    Returns {'GAUC': float, 'nDCG@5': float, 'primary': float, 'users': int, 'rows': int}
    """
    def auc(labels, scores):
        """Mann-Whitney U test with tie correction."""
        pairs = sorted(zip(scores, labels))
        ranks = [0.0] * len(pairs)
        i = 0
        while i < len(pairs):
            j = i
            while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[k] = avg
            i = j + 1
        npos = sum(l for _, l in pairs)
        nneg = len(pairs) - npos
        if npos == 0 or nneg == 0:
            return 0.5
        srank = sum(r for r, (_, l) in zip(ranks, pairs) if l == 1)
        return (srank - npos * (npos + 1) / 2.0) / (npos * nneg)

    def ndcg_at_k(labels, k):
        """Normalized Discounted Cumulative Gain@k."""
        disc = [math.log2(i + 2) for i in range(k)]
        dcg = sum(((2 ** t) - 1) / disc[i] for i, t in enumerate(labels[:k]))
        ideal = sorted(labels, reverse=True)[:k]
        idcg = sum(((2 ** t) - 1) / disc[i] for i, t in enumerate(ideal))
        return 0.0 if idcg == 0 else dcg / idcg

    def evaluate(user_ids, labels, scores, k=5):
        """
        Evaluate ranking performance using GAUC and nDCG@k.
        
        Args:
            user_ids: List of user IDs
            labels: List of binary relevance labels
            scores: List of prediction scores
            k: NDCG cutoff (default 5)
            
        Returns:
            Dict with GAUC, nDCG@k, primary (average), users count, and rows count
        """
        byu = collections.defaultdict(list)
        for u, y, s in zip(user_ids, labels, scores):
            byu[u].append((s, y))
        gnum = gden = 0.0
        nd = []
        for u, lst in byu.items():
            lst.sort(key=lambda x: -x[0])
            labs = [y for _, y in lst]
            npos = sum(labs)
            if 0 < npos < len(labs):
                gnum += npos * auc(labs, [s for s, _ in lst])
                gden += npos
            nd.append(ndcg_at_k(labs, k))
        gauc = gnum / gden if gden else 0.5
        ndcg = sum(nd) / len(nd) if nd else 0.0
        return {
            'GAUC': gauc,
            f'nDCG@{k}': ndcg,
            'primary': (gauc + ndcg) / 2.0,
            'users': len(byu),
            'rows': len(labels)
        }
    
    return evaluate


# Get the evaluate function
_evaluate_func = load_kuairand_evaluate()


def load_baseline_scores() -> Dict[str, Any]:
    """Load baseline scores from kuairand-starter-kit."""
    kit_path = Path(__file__).parent.parent / "kuairand-starter-kit" / "baseline_scores.json"
    if kit_path.exists():
        with open(kit_path, encoding='utf-8') as f:
            return json.load(f)
    return {
        "scores": {
            "item_popularity": {"valid": {"primary": 0.5807}},
            "fm_official": {"valid": {"primary": 0.6016}},
            "oracle_ceiling": {"valid": {"primary": 0.8484}},
        }
    }


class LLMScorer:
    """Scorer for LLM-generated recommendations using KuaiRand metrics."""

    def __init__(self):
        """Initialize scorer with baseline metrics."""
        self.baseline_scores = load_baseline_scores()
        self.evaluate = _evaluate_func
        
        # Extract key baselines
        scores = self.baseline_scores.get("scores", {})
        self.baseline_primary = scores.get("fm_official", {}).get("valid", {}).get("primary", 0.6016)
        self.oracle_ceiling = scores.get("oracle_ceiling", {}).get("valid", {}).get("primary", 0.8484)

    def score(
        self,
        user_ids: List[int],
        labels: List[int],
        scores: List[float],
        k: int = 5,
    ) -> Dict[str, float]:
        """
        Score predictions using KuaiRand-Pure metrics.
        
        Args:
            user_ids: List of user IDs
            labels: List of binary relevance labels (0/1)
            scores: List of prediction scores
            k: NDCG cutoff (default 5)
            
        Returns:
            Dict with GAUC, nDCG@k, primary, users, rows
        """
        return self.evaluate(user_ids, labels, scores, k=k)

    def compare_to_baseline(
        self,
        current_score: float,
        metric_name: str = "primary",
    ) -> Dict[str, Any]:
        """
        Compare current score to baseline and oracle.
        
        Args:
            current_score: Current metric value
            metric_name: Metric to compare (default: primary)
            
        Returns:
            Dict with comparison info: gain, pct_gain, vs_oracle, status
        """
        gain = current_score - self.baseline_primary
        pct_gain = (gain / self.baseline_primary * 100) if self.baseline_primary else 0
        vs_oracle = self.oracle_ceiling - current_score
        pct_vs_oracle = (vs_oracle / self.oracle_ceiling * 100) if self.oracle_ceiling else 0
        
        return {
            "current": current_score,
            "baseline": self.baseline_primary,
            "oracle": self.oracle_ceiling,
            "gain": gain,
            "pct_gain": pct_gain,
            "vs_oracle": vs_oracle,
            "pct_vs_oracle": pct_vs_oracle,
            "status": "exceeds_baseline" if gain > 0 else "below_baseline",
        }

    def get_improvement_potential(self) -> Dict[str, float]:
        """Get the potential improvement range."""
        room_for_improvement = self.oracle_ceiling - self.baseline_primary
        return {
            "baseline": self.baseline_primary,
            "oracle": self.oracle_ceiling,
            "room": room_for_improvement,
            "pct_room": (room_for_improvement / self.oracle_ceiling * 100) if self.oracle_ceiling else 0,
        }

    def is_convergent(
        self,
        recent_gains: List[float],
        epsilon: float = 0.002,
        n: int = 3,
    ) -> bool:
        """
        Check if optimization has converged based on recent gains.
        
        Args:
            recent_gains: List of recent gains (last N iterations)
            epsilon: Minimum gain threshold for convergence
            n: Number of recent iterations to check
            
        Returns:
            True if last N gains are all below epsilon
        """
        if len(recent_gains) < n:
            return False
        return all(g < epsilon for g in recent_gains[-n:])
