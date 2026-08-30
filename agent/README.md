# Perplexity-Powered LLM Recommendation Optimizer

This agent folder contains an automated optimization system that uses Perplexity AI to improve recommendation ranking models on the KuaiRand-Pure dataset.

## Architecture

### Core Components

1. **`orchestrator.py`** - Main entry point
   - Coordinates the optimization loop: research → propose → run → judge
   - Manages iteration history and convergence tracking
   - Integrates all modules with LLM scoring from kuairand-starter-kit

2. **`perplexity_client.py`** - Perplexity API Integration
   - Handles all communication with Perplexity AI API
   - Methods:
     - `query()` - Send raw queries to Perplexity
     - `research()` - Conduct targeted research on topics
     - `propose_improvements()` - Generate code improvement proposals
     - `evaluate_approach()` - Evaluate approach feasibility

3. **`llm_scorer.py`** - LLM Scoring Module
   - Integrates scoring functions from `kuairand-starter-kit/evaluate.py`
   - Implements KuaiRand-Pure metrics:
     - **GAUC** (Group AUC) - Weighted AUC per user
     - **nDCG@5** - Normalized Discounted Cumulative Gain
     - **Primary** - Average of GAUC and nDCG@5
   - Methods:
     - `score()` - Evaluate predictions using KuaiRand metrics
     - `compare_to_baseline()` - Compare to FM Official baseline (0.6016)
     - `get_improvement_potential()` - Show room to oracle (0.8484)
     - `is_convergent()` - Check convergence (ε=0.002, N=3)

4. **`research.py`** - Research Module
   - Queries Perplexity for topic-driven insights
   - Returns: summary, sources/citations, API usage, raw response

5. **`proposer.py`** - Proposal Generation
   - Uses Perplexity to generate code improvement proposals
   - Analyzes current code, research findings, and history
   - Returns: idea, rationale, proposed file changes

6. **`judge.py`** - Evaluation & Convergence
   - Judges iteration results against baseline
   - Detects convergence based on recent gains
   - Integrates with LLMScorer for detailed metrics

7. **`runner.py`** - Execution (existing)
   - Copies baseline, applies proposed changes, runs pipeline
   - Executes `run_and_score.py` on validation split

8. **`coder_client.py`** - LLM Code Generation (existing)
   - Can be extended for code-to-code transformations

## Key Metrics & Baselines

From `kuairand-starter-kit/baseline_scores.json`:

| Baseline | GAUC | nDCG@5 | Primary |
|----------|------|--------|---------|
| Random | 0.4993 | 0.4675 | 0.4834 |
| Item Popularity | 0.6387 | 0.5227 | 0.5807 |
| **FM Official** | 0.6674 | 0.5357 | **0.6016** |
| Oracle Ceiling | 1.0 | 0.6968 | **0.8484** |

**Convergence Rule**: Last 3 gains all < 0.002

## Setup

1. **Install dependencies**:
   ```bash
   cd agent
   pip install -r requirements.txt
   ```

2. **Set environment variables**:
   - `PERPLEXITY_API_KEY` - Your Perplexity API key (in `.env` or exported)
   - Already available from: `kuairand-starter-kit/.env`

3. **Prepare pipeline directories**:
   - `pipeline/v0/` - Baseline snapshot
   - `pipeline/current/` - Current working version
   - `pipeline/attempts/` - Attempt results (auto-created)

## Running the Optimizer

```bash
cd agent
python orchestrator.py
```

Or specify iteration budget:
```python
from orchestrator import main
main(budget_iters=100)
```

## Iteration Loop

Each iteration follows this pattern:

```
[1/4] Research
  ↓ Perplexity selects topic based on history
  ↓ Conducts web research with citations
  
[2/4] Propose
  ↓ Perplexity analyzes findings + current code
  ↓ Generates improvement proposals
  
[3/4] Run
  ↓ Creates attempt directory
  ↓ Applies proposed changes
  ↓ Executes pipeline/current/run_and_score.py
  ↓ Returns: GAUC, nDCG@5, Primary metrics
  
[4/4] Judge
  ↓ Compares primary metric to current_best
  ↓ Calculates gain
  ↓ Checks convergence (3 gains < 0.002)
  ↓ Accept/Reject + iterate or stop
```

## Logging

Each iteration is logged to `run_log.jsonl`:

```json
{
  "iter": 0,
  "topic": "pairwise ranking loss...",
  "idea": "implement BPR loss...",
  "status": "ok",
  "valid_primary": 0.6021,
  "accepted": true,
  "gain": 0.0068,
  "recent_gains": [0.0068],
  "converged": false,
  "reason": "Gain: 0.006800, Total improvements: 0.000500"
}
```

## Integration with kuairand-starter-kit

The agent **reads from but does not modify** kuairand-starter-kit:

- ✅ Imports scoring functions: `evaluate.py`
- ✅ Reads baseline metrics: `baseline_scores.json`
- ✅ Reads API key: `.env`
- ❌ Does NOT modify any files in kuairand-starter-kit

## Error Handling

- **Research fails**: Falls back to predefined topics
- **Proposal fails**: Returns current code unchanged
- **Execution fails**: Logged as rejected iteration
- **API failures**: Graceful degradation with error messages

## Monitoring

Monitor optimization progress:

```python
import json

# Read latest log entry
with open("run_log.jsonl") as f:
    latest = json.loads(f.readlines()[-1])
    print(f"Best: {latest['valid_primary']:.4f}, Converged: {latest['converged']}")
```

## Extending the Agent

### Add custom research topic selector
Replace `pick_next_topic()` in `orchestrator.py` with domain-specific logic

### Integrate other LLM providers
Extend `PerplexityClient` or create `{Provider}Client` with same interface

### Custom scoring metrics
Modify `LLMScorer.score()` or add new methods

### Parse Perplexity proposals into code
Enhance `proposer.py` to extract actual code changes from LLM rationale
