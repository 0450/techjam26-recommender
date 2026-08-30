"""
Judge module for evaluating and accepting/rejecting iterations.
Uses LLM scoring and convergence detection from kuairand-starter-kit.
"""
from llm_scorer import LLMScorer

# Convergence parameters from baseline_scores.json
EPSILON, N = 0.002, 3

# Initialize scorer
_scorer = LLMScorer()


def judge(result: dict, current_best: float, recent_gains: list[float]) -> dict:
    """
    Judge whether an iteration's result should be accepted and if optimization converged.
    
    Args:
        result: {'status': 'ok'|'error'|..., 'metrics': {'primary': float, ...}, 'dir': str}
        current_best: Best primary metric achieved so far
        recent_gains: List of recent gains
    
    Returns:
        {
            'accept': bool,
            'gain': float,
            'new_best': float,
            'recent_gains': list[float],
            'converged': bool,
            'reason': str,
            'score_details': dict (if ok)
        }
    """
    # Handle non-ok status
    if result["status"] != "ok":
        return {
            "accept": False,
            "reason": f"Execution failed: {result['status']}",
            "new_best": current_best,
            "recent_gains": recent_gains,
            "converged": False,
        }

    try:
        score = result["metrics"]["primary"]
    except (KeyError, TypeError):
        return {
            "accept": False,
            "reason": "Missing or invalid primary metric",
            "new_best": current_best,
            "recent_gains": recent_gains,
            "converged": False,
        }

    # Calculate gain and acceptance decision
    gain = score - current_best
    accept = gain > 0
    new_best = max(score, current_best)
    updated_gains = (recent_gains + [gain])[-N:]

    # Check for convergence
    converged = len(updated_gains) == N and all(g < EPSILON for g in updated_gains)

    # Get score comparison details
    score_comparison = _scorer.compare_to_baseline(score)
    improvement_potential = _scorer.get_improvement_potential()

    return {
        "accept": accept,
        "gain": gain,
        "new_best": new_best,
        "recent_gains": updated_gains,
        "converged": converged,
        "reason": f"Gain: {gain:.6f}, Total improvements: {new_best - score_comparison['baseline']:.6f}",
        "score_details": score_comparison,
        "improvement_potential": improvement_potential,
    }


def get_convergence_status() -> dict:
    """Get current convergence parameters and baseline info."""
    return {
        "epsilon": EPSILON,
        "N": N,
        "baseline": _scorer.baseline_primary,
        "oracle": _scorer.oracle_ceiling,
        "room_for_improvement": _scorer.oracle_ceiling - _scorer.baseline_primary,
    }
