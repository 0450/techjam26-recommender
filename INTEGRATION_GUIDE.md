# Perplexity & LLM Scoring Integration Guide

## Overview

The agent folder has been fully integrated with:
1. **Perplexity AI API** for research-driven optimization
2. **LLM Scoring** using KuaiRand-Pure metrics from kuairand-starter-kit

## File Structure

```
agent/
├── orchestrator.py          # Main optimization loop
├── perplexity_client.py     # Perplexity API client
├── llm_scorer.py            # KuaiRand metric scoring
├── research.py              # Topic research using Perplexity
├── proposer.py              # Code proposal generation
├── judge.py                 # Result evaluation & convergence
├── runner.py                # Pipeline execution (existing)
├── coder_client.py          # LLM code generation (existing)
├── examples.py              # Usage examples
├── setup.py                 # Setup verification
├── requirements.txt         # Python dependencies
├── .env                     # Configuration file
├── README.md                # Documentation
├── run_log.jsonl            # Iteration log
└── INTEGRATION_GUIDE.md     # This file
```

## Integration Points

### 1. Perplexity Client (`perplexity_client.py`)

**Purpose**: Unified interface to Perplexity AI API

**Key Methods**:
- `query(prompt, model, temperature, max_tokens)` - Raw API calls
- `research(topic, focus)` - Research with citations
- `propose_improvements(code, findings, history)` - Generate proposals
- `evaluate_approach(approach, metrics, baseline)` - Evaluate feasibility

**Usage in Agent**:
```python
from perplexity_client import PerplexityClient

client = PerplexityClient()  # Reads PERPLEXITY_API_KEY from env

# Research
findings = client.research("pairwise ranking loss", "numpy only")
print(findings['summary'])
print(findings['sources'])  # Citations from Perplexity

# Generate proposals
proposal = client.propose_improvements(current_code, findings, history)
```

**Imports From kuairand-starter-kit**:
- API key from `kuairand-starter-kit/.env` (PERPLEXITY_API_KEY)

### 2. LLM Scorer (`llm_scorer.py`)

**Purpose**: Score recommendations using KuaiRand-Pure metrics

**Key Metrics**:
- **GAUC**: Group AUC, weighted by user positive count
- **nDCG@5**: Normalized DCG @ top-5
- **Primary**: Average of GAUC and nDCG@5

**Key Methods**:
- `score(user_ids, labels, scores, k=5)` - Evaluate predictions
- `compare_to_baseline(score)` - Compare to FM Official (0.6016)
- `get_improvement_potential()` - Show room to oracle (0.8484)
- `is_convergent(recent_gains)` - Check convergence rule

**Imports From kuairand-starter-kit**:
- Scoring functions from `kuairand-starter-kit/evaluate.py`
- Baseline metrics from `kuairand-starter-kit/baseline_scores.json`

**Usage in Agent**:
```python
from llm_scorer import LLMScorer

scorer = LLMScorer()

# Score predictions
result = scorer.score(user_ids, labels, scores)
print(f"Primary: {result['primary']:.4f}")

# Compare to baseline
comparison = scorer.compare_to_baseline(result['primary'])
print(f"Gain: {comparison['gain']:+.4f}")
print(f"vs Oracle: {comparison['vs_oracle']:.4f}")

# Check convergence
is_done = scorer.is_convergent(recent_gains, epsilon=0.002, n=3)
```

### 3. Updated Research Module (`research.py`)

**Changes**:
- Now uses `PerplexityClient` instead of raw requests
- Returns structured dict with summary, sources, usage, error handling
- Graceful fallback if API fails

**Flow**:
```
pick_next_topic() → research() → Perplexity API
                                 ↓
                          Returns research findings
```

### 4. Updated Proposer (`proposer.py`)

**Changes**:
- Uses `PerplexityClient.propose_improvements()`
- Analyzes: current code + research findings + history
- Returns: idea, rationale, proposed changes

**Flow**:
```
Proposal Generation:
  Current Code + Research + History
        ↓
  PerplexityClient.propose_improvements()
        ↓
  Structured proposal (idea, rationale, changes)
```

### 5. Updated Judge (`judge.py`)

**Changes**:
- Integrates `LLMScorer` for detailed metrics
- Includes baseline comparison in verdict
- Convergence detection using scorer's is_convergent()

**Returns**:
```python
{
    "accept": bool,
    "gain": float,
    "new_best": float,
    "recent_gains": [float],
    "converged": bool,
    "reason": str,
    "score_details": {
        "current": float,
        "baseline": float,
        "oracle": float,
        "gain": float,
        "pct_gain": float,
        "vs_oracle": float,
        "status": str
    },
    "improvement_potential": {...}
}
```

### 6. Updated Orchestrator (`orchestrator.py`)

**Changes**:
- Integrated Perplexity for topic selection
- LLM scoring for detailed metrics
- Enhanced logging with score details
- Convergence monitoring

**Iteration Loop**:
```
[1/4] Research
    ↓ pick_next_topic() using Perplexity
    ↓ research(topic) with Perplexity
    
[2/4] Propose
    ↓ propose() using PerplexityClient
    
[3/4] Run
    ↓ run_attempt() (unchanged)
    ↓ Returns: status, metrics (GAUC, nDCG@5, primary)
    
[4/4] Judge
    ↓ judge() with LLMScorer
    ↓ Returns: accept/reject + convergence status
    
Repeat until converged or budget exhausted
```

## Configuration

### Environment Variables (`.env`)

```
PERPLEXITY_API_KEY=pplx-...           # Required
RESEARCH_MODEL=sonar-pro              # Default model
PROPOSAL_MODEL=sonar-pro              # Default model
RESEARCH_TEMPERATURE=0.5              # Lower = more deterministic
PROPOSAL_TEMPERATURE=0.7              # Balanced
EVALUATION_TEMPERATURE=0.3            # Lower = more deterministic
```

### Agent Configuration

```python
# In orchestrator.py
BASELINE_VALID_PRIMARY = 0.6016       # FM Official baseline
EPSILON, N = 0.002, 3                 # Convergence rule

# In llm_scorer.py
# Loaded from kuairand-starter-kit/baseline_scores.json
baseline_primary = 0.6016
oracle_ceiling = 0.8484
```

## Baselines & Metrics

Sourced from `kuairand-starter-kit/baseline_scores.json`:

| Model | GAUC | nDCG@5 | Primary | Note |
|-------|------|--------|---------|------|
| Random | 0.4993 | 0.4675 | 0.4834 | Sanity check |
| Item Popularity | 0.6387 | 0.5227 | 0.5807 | Simple baseline |
| **FM Official** | 0.6674 | 0.5357 | **0.6016** | **Target to beat** |
| Oracle | 1.0 | 0.6968 | 0.8484 | Ceiling (27% users all-negative) |

## Convergence Rule

```
If last 3 gains all < 0.002 → CONVERGED
Gain = current_score - previous_best
```

From `baseline_scores.json`:
```json
"convergence_rule": {
  "epsilon": 0.002,
  "N": 3
}
```

## Error Handling

### Research Fails
- Fallback: Uses predefined topics from list
- Logged with error message

### Proposal Fails
- Fallback: Returns current code unchanged
- Logged with error message

### Perplexity API Fails
- All Perplexity methods return structured dict with `"error"` field
- Agent gracefully continues with fallback behavior

### Execution Fails
- Judge marks as rejected
- Logged and continues iteration

## Dependencies

**New packages** (in `requirements.txt`):
```
perplexityai>=0.43.3,<0.44    # Official Perplexity SDK
requests>=2.28.0               # HTTP library
python-dotenv>=0.19.0          # .env file handling
```

**Imported from kuairand-starter-kit** (no new installs):
- `evaluate.py` - Scoring functions
- Baseline scores (JSON)

## Running the Agent

1. **Setup**:
   ```bash
   cd agent
   pip install -r requirements.txt
   python setup.py  # Verify configuration
   ```

2. **Run**:
   ```bash
   python orchestrator.py
   ```

3. **Monitor**:
   ```bash
   tail -f run_log.jsonl | python -m json.tool
   ```

## Logging Output

### Console Output
```
======================================================================
Perplexity-Powered LLM Recommendation Optimizer
======================================================================

Baseline (FM Official): 0.6016
Oracle Ceiling:         0.8484
Room for Improvement:   0.2468
Convergence Rule:       ε=0.002, N=3

======================================================================
Iteration 1/50
======================================================================

[1/4] Researching topic...
Topic: improving ranking metrics for implicit feedback...
  ✓ Found 5 sources

[2/4] Generating proposal...
Idea: Implement pairwise BPR loss
  ✓ Proposal generated

[3/4] Running attempt...
Status: ok
  GAUC:      0.6691
  nDCG@5:    0.5315
  Primary:   0.6003

[4/4] Judging result...

[Iter 0] Gain: -0.001300, Total improvements: -0.001300
  ✗ Rejected: no gain
```

### JSONL Log (`run_log.jsonl`)
```json
{"iter": 0, "topic": "...", "idea": "...", "status": "ok", "valid_primary": 0.6003, "accepted": false, "gain": -0.0013, "recent_gains": [-0.0013], "converged": false, "reason": "..."}
```

## Extending the Integration

### Use Different LLM Provider
Create new client (e.g., `openai_client.py`):
```python
class OpenAIClient:
    def research(self, topic, focus) -> Dict[str, Any]:
        # Implement OpenAI API call
        pass
```

Then update `research.py`:
```python
from openai_client import OpenAIClient  # Instead of perplexity_client
```

### Add Custom Scoring Metrics
Extend `LLMScorer`:
```python
def score_custom(self, predictions):
    # Add custom metric
    return {'custom_metric': value}
```

### Implement Proposal Code Extraction
Enhance `proposer.py` to parse Perplexity's proposals:
```python
def extract_code_changes(proposal_text: str) -> Dict[str, str]:
    # Parse markdown code blocks from Perplexity
    # Return modified files dict
    pass
```

## Troubleshooting

### "PERPLEXITY_API_KEY not found"
- Set environment variable: `export PERPLEXITY_API_KEY=...`
- Or add to `.env` file
- Run `python setup.py` to verify

### "Could not load evaluate from kuairand-starter-kit"
- Verify `kuairand-starter-kit/evaluate.py` exists
- Fallback implementation is provided in `llm_scorer.py`

### API rate limiting
- Reduce temperature for faster responses
- Add retry logic with exponential backoff in `perplexity_client.py`

### Poor proposal quality
- Increase `PROPOSAL_TEMPERATURE` for more creativity
- Provide more context in history
- Add domain-specific constraints in proposer

## Summary

✅ **Fully Integrated**:
- Perplexity API client for research and proposals
- KuaiRand-Pure scoring metrics
- LLM-driven optimization loop
- Convergence detection
- Comprehensive logging and error handling

✅ **Read-Only Access to kuairand-starter-kit**:
- Imports scoring functions
- Reads baseline metrics
- Uses API key
- Does NOT modify any files

✅ **Production Ready**:
- Error handling and fallbacks
- Configuration management
- Setup verification
- Usage examples
- Comprehensive documentation
