"""
Quick Start Example: Running the Perplexity-powered Optimizer
"""

# Example 1: Run full optimization with default settings
if __name__ == "__main__":
    from orchestrator import main
    
    # Run for 50 iterations (or until convergence)
    main(budget_iters=50)


# Example 2: Monitor progress during optimization
"""
import json
from pathlib import Path

# Read optimization log
log_file = Path("run_log.jsonl")
iterations = [json.loads(line) for line in log_file.read_text().strip().split('\n') if line]

# Print summary
print(f"Iterations completed: {len(iterations)}")
for it in iterations[-5:]:  # Last 5
    print(f"  Iter {it['iter']}: primary={it['valid_primary']:.4f}, "
          f"accepted={it['accepted']}, converged={it['converged']}")
"""


# Example 3: Access scoring functionality directly
"""
from llm_scorer import LLMScorer

scorer = LLMScorer()

# Score some predictions
user_ids = [1, 1, 1, 2, 2, 2]
labels = [1, 0, 0, 1, 1, 0]
scores = [0.9, 0.7, 0.5, 0.8, 0.6, 0.4]

result = scorer.score(user_ids, labels, scores, k=5)
print(f"GAUC: {result['GAUC']:.4f}")
print(f"nDCG@5: {result['nDCG@5']:.4f}")
print(f"Primary: {result['primary']:.4f}")

# Compare to baseline
comparison = scorer.compare_to_baseline(result['primary'])
print(f"Improvement over baseline: {comparison['gain']:+.4f}")
"""


# Example 4: Use Perplexity client directly
"""
from perplexity_client import PerplexityClient

client = PerplexityClient()

# Conduct research
findings = client.research(
    topic="contrastive learning for recommendation systems",
    focus="applicable to implicit feedback with binary labels"
)

print("Summary:", findings['summary'][:500])
print("Sources:", findings['sources'][:3])

# Query for specific advice
advice = client.query(
    "How can I improve nDCG@5 in a recommendation ranking task?",
    temperature=0.3,
    max_tokens=500
)

print("Advice:", advice['content'])
"""


# Example 5: Check convergence status
"""
from judge import get_convergence_status

status = get_convergence_status()
print(f"Baseline: {status['baseline']:.4f}")
print(f"Oracle: {status['oracle']:.4f}")
print(f"Room: {status['room_for_improvement']:.4f}")
print(f"Epsilon: {status['epsilon']}")
print(f"N: {status['N']}")
"""


# Example 6: View improvement potential
"""
from llm_scorer import LLMScorer

scorer = LLMScorer()
potential = scorer.get_improvement_potential()

print(f"Baseline primary: {potential['baseline']:.4f}")
print(f"Oracle ceiling: {potential['oracle']:.4f}")
print(f"Total room for improvement: {potential['room']:.4f}")
print(f"Percentage of oracle to reach: {potential['pct_room']:.1f}%")
"""


# Example 7: Verify setup before running
"""
import subprocess
import sys

result = subprocess.run([sys.executable, "setup.py"], cwd="agent")
sys.exit(result.returncode)
"""


if __name__ == "__main__":
    print(__doc__)
    print("\nUncomment examples above to run them!")
