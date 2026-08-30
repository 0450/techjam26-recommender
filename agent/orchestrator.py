"""
Main orchestrator for Perplexity-powered LLM optimization agent.
Coordinates research, proposal generation, execution, and judging with integrated LLM scoring.
"""
import json
from pathlib import Path
from typing import Dict, List, Any

from research import research
from proposer import propose
from runner import run_attempt
from judge import judge, get_convergence_status
from llm_scorer import LLMScorer
from perplexity_client import PerplexityClient


# Constants
BASELINE_VALID_PRIMARY = 0.6016  # From kuairand-starter-kit baseline_scores.json
LOG_FILE = Path("run_log.jsonl")


def ensure_runtime_dirs() -> tuple[Path, Path, Path]:
    """Create the expected pipeline directories if the repo snapshot is incomplete."""
    repo_root = Path(__file__).resolve().parent
    pipeline_root = repo_root / "pipeline"
    v0_dir = pipeline_root / "v0"
    current_dir = pipeline_root / "current"
    attempts_dir = pipeline_root / "attempts"

    for directory in [pipeline_root, v0_dir, current_dir, attempts_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    return pipeline_root, v0_dir, current_dir


def load_files(version_dir: str) -> Dict[str, str]:
    """Load all Python files from a version directory."""
    files = {}
    for fpath in Path(version_dir).glob("*.py"):
        try:
            with open(fpath) as f:
                files[fpath.name] = f.read()
        except Exception as e:
            print(f"Warning: Could not load {fpath}: {e}")
    return files


def pick_next_topic(history: List[Dict[str, Any]]) -> str:
    """
    Use Perplexity to pick the next research topic based on history.
    Falls back to predefined topics if Perplexity fails.
    """
    if not history:
        return "improving ranking metrics for implicit feedback recommendation systems using pairwise losses"
    
    try:
        client = PerplexityClient()
        recent_ideas = "\n".join(h.get("idea", "N/A") for h in history[-3:])
        prompt = f"""Given these recent improvements tried in a ranking recommendation optimization:
{recent_ideas}

Suggest ONE specific next research direction to explore for improving GAUC and nDCG@5 metrics. 
Focus on techniques not yet tried. Be concise (one sentence)."""
        
        result = client.query(prompt, temperature=0.6, max_tokens=200)
        if result.get("content") and not result.get("error"):
            return result["content"][:200]
    except Exception as e:
        print(f"Warning: Perplexity topic selection failed: {e}")
    
    # Fallback topics
    fallback_topics = [
        "implementing cross-entropy loss with label smoothing for ranking recommendation",
        "feature engineering for user-item interaction prediction using attention mechanisms",
        "contrastive learning approaches for implicit feedback recommendation systems",
        "incorporating temporal dynamics and session-based features in ranking models",
        "ensemble methods combining multiple ranking signals and learners",
    ]
    
    idx = len(history) % len(fallback_topics)
    return fallback_topics[idx]


def log_iteration(
    iteration: int,
    findings: Dict[str, Any],
    proposal: Dict[str, Any],
    result: Dict[str, Any],
    verdict: Dict[str, Any],
) -> None:
    """Log iteration details to JSONL file."""
    log_entry = {
        "iter": iteration,
        "topic": findings.get("summary", "")[:100],
        "idea": proposal.get("idea", "")[:100],
        "status": result["status"],
        "valid_primary": result.get("metrics", {}).get("primary", None),
        "accepted": verdict["accept"],
        "gain": verdict.get("gain", 0),
        "recent_gains": verdict.get("recent_gains", []),
        "converged": verdict.get("converged", False),
        "reason": verdict.get("reason", ""),
    }
    
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    
    print(f"\n[Iter {iteration}] {verdict.get('reason', '')}")
    if verdict["accept"]:
        print(f"  Accepted! New best: {result.get('metrics', {}).get('primary', 0):.4f}")
    else:
        print(f"  Rejected: {verdict.get('reason', 'no gain')}")


def main(budget_iters: int = 50) -> None:
    """
    Main optimization loop: research → propose → run → judge → iterate.
    
    Args:
        budget_iters: Number of iterations to run
    """
    print("=" * 70)
    print("Perplexity-Powered LLM Recommendation Optimizer")
    print("=" * 70)

    pipeline_root, v0_dir, current_dir = ensure_runtime_dirs()
    if not (v0_dir / "run_and_score.py").exists() and not any(v0_dir.iterdir()):
        print("\nWARNING: pipeline/v0 is empty. This repo does not contain a runnable baseline snapshot.")
        print("         Add the baseline pipeline or populate pipeline/v0 and pipeline/current before running.")

    if not (current_dir / "run_and_score.py").exists():
        print("\nWARNING: pipeline/current does not contain run_and_score.py.")
        print("         The runner cannot execute without a valid baseline pipeline snapshot.")
        print("         Expected files: pipeline/current/run_and_score.py")
        return
    
    # Initialize
    scorer = LLMScorer()
    convergence_info = get_convergence_status()
    
    print(f"\nBaseline (FM Official): {convergence_info['baseline']:.4f}")
    print(f"Oracle Ceiling:         {convergence_info['oracle']:.4f}")
    print(f"Room for Improvement:   {convergence_info['room_for_improvement']:.4f}")
    print(f"Convergence Rule:       epsilon={convergence_info['epsilon']}, N={convergence_info['N']}")
    
    history = []
    current_best = BASELINE_VALID_PRIMARY
    recent_gains = []
    current_files = load_files(str(v0_dir))

    for i in range(budget_iters):
        print(f"\n{'='*70}")
        print(f"Iteration {i+1}/{budget_iters}")
        print(f"{'='*70}")
        
        # Step 1: Research
        print("\n[1/4] Researching topic...")
        topic = pick_next_topic(history)
        print(f"Topic: {topic[:100]}")
        findings = research(topic, focus="Python, numpy-only implementation, no GPU required")
        
        if findings.get("error"):
            print(f"  Research failed: {findings['error']}")
        else:
            print(f"  Found {len(findings.get('sources', []))} sources")
        
        # Step 2: Propose
        print("\n[2/4] Generating proposal...")
        proposal = propose(findings, history, current_files)
        print(f"Idea: {proposal.get('idea', 'N/A')[:80]}")
        
        if proposal.get("error"):
            print(f"  Proposal generation failed: {proposal['error']}")
        else:
            print(f"  Proposal generated")
        
        # Step 3: Run
        print("\n[3/4] Running attempt...")
        result = run_attempt(proposal.get("files", current_files), base_version_dir=str(current_dir))
        print(f"Status: {result['status']}")
        
        if result["status"] == "ok":
            metrics = result.get("metrics", {})
            print(f"  GAUC:      {metrics.get('GAUC', 0):.4f}")
            print(f"  nDCG@5:    {metrics.get('nDCG@5', 0):.4f}")
            print(f"  Primary:   {metrics.get('primary', 0):.4f}")
        
        # Step 4: Judge
        print("\n[4/4] Judging result...")
        verdict = judge(result, current_best, recent_gains)
        
        # Log iteration
        log_iteration(i, findings, proposal, result, verdict)
        
        # Update state if accepted
        if verdict["accept"]:
            current_files = proposal.get("files", current_files)
            current_best = verdict["new_best"]
            recent_gains = verdict["recent_gains"]
            
            # Save snapshot (optional)
            print(f"\n  Saving snapshot to pipeline/v{i+1}")
        
        # Check for convergence
        if verdict.get("converged"):
            print(f"\n{'='*70}")
            print(f"CONVERGED after {i+1} iterations!")
            print(f"  Final score: {current_best:.4f}")
            print(f"  Improvement over baseline: {current_best - convergence_info['baseline']:.4f}")
            print(f"  Remaining gap to oracle: {convergence_info['oracle'] - current_best:.4f}")
            print(f"{'='*70}")
            break
    
    # Summary
    print(f"\n{'='*70}")
    print("Optimization Complete")
    print(f"{'='*70}")
    print(f"Best score achieved: {current_best:.4f}")
    print(f"Improvement over baseline: {current_best - convergence_info['baseline']:+.4f} "
          f"({(current_best - convergence_info['baseline']) / convergence_info['baseline'] * 100:+.2f}%)")
    print(f"Iterations completed: {min(i+1, budget_iters)}")


if __name__ == "__main__":
    main(budget_iters=50)
