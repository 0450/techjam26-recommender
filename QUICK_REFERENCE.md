# Perplexity Agent - Quick Reference Card

## 🚀 Start Here

```bash
cd agent
pip install -r requirements.txt      # ✅ Already done
export PERPLEXITY_API_KEY=pplx-...   # Set your API key
python setup.py                      # Verify setup
python orchestrator.py               # Start optimization
```

## 📊 Key Metrics

```
Baseline (FM Official):  0.6016
Oracle (Theoretical Max): 0.8484
Room for Improvement:    0.2468
Convergence Rule:        Last 3 gains < 0.002
```

## 🔧 Core Components

| Component | File | Purpose |
|-----------|------|---------|
| **PerplexityClient** | `perplexity_client.py` | API access, research, proposals |
| **LLMScorer** | `llm_scorer.py` | GAUC, nDCG@5, Primary metrics |
| **Orchestrator** | `orchestrator.py` | Main optimization loop |
| **Research** | `research.py` | Web research with citations |
| **Proposer** | `proposer.py` | Generate improvements |
| **Judge** | `judge.py` | Evaluate and decide |

## 💻 Code Examples

### Score predictions
```python
from llm_scorer import LLMScorer

scorer = LLMScorer()
result = scorer.score(user_ids, labels, scores)
print(f"Primary: {result['primary']:.4f}")
print(f"GAUC: {result['GAUC']:.4f}, nDCG@5: {result['nDCG@5']:.4f}")
```

### Research a topic
```python
from perplexity_client import PerplexityClient

client = PerplexityClient()
findings = client.research("ranking loss for recommendation", "numpy-only")
print(findings['summary'])
print(f"Sources: {findings['sources']}")
```

### Check convergence
```python
from judge import get_convergence_status

status = get_convergence_status()
print(f"Baseline: {status['baseline']:.4f}")
print(f"Oracle: {status['oracle']:.4f}")
```

### Run full optimization
```python
from orchestrator import main

main(budget_iters=50)  # Run 50 iterations max
```

## 📁 File Structure

```
agent/
├── Core Modules
│   ├── orchestrator.py          # Main loop
│   ├── perplexity_client.py     # Perplexity API
│   ├── llm_scorer.py            # Scoring
│   ├── research.py              # Web research
│   ├── proposer.py              # Generate proposals
│   └── judge.py                 # Evaluate results
│
├── Setup & Testing
│   ├── setup.py                 # Verification script
│   ├── test_integration.py      # Integration test
│   ├── examples.py              # Usage examples
│   ├── requirements.txt         # Dependencies
│   └── .env                     # Configuration
│
└── Documentation
    ├── README.md                # Architecture
    └── run_log.jsonl            # Iteration log
```

## 📝 Log Output

Monitor progress:
```bash
tail -f run_log.jsonl | python -m json.tool
```

Each iteration logs:
```json
{
  "iter": 0,
  "status": "ok",
  "valid_primary": 0.6021,
  "accepted": true,
  "gain": 0.0068,
  "converged": false
}
```

## 🔍 Troubleshooting

| Issue | Solution |
|-------|----------|
| `PERPLEXITY_API_KEY not found` | Export: `export PERPLEXITY_API_KEY=pplx-...` |
| Module import fails | Run: `pip install -r requirements.txt` |
| Setup verification fails | Run: `python setup.py` for diagnostics |
| Poor proposal quality | Increase `PROPOSAL_TEMPERATURE` in `.env` |

## ⚡ Performance

| Stage | Time |
|-------|------|
| Research | 3-10 sec |
| Proposal | 3-5 sec |
| Execution | 10-30 sec |
| Per Iteration | ~20-50 sec |
| **50 Iterations** | **~20-40 min** |
| **With convergence** | **~5-10 min** |

## ✨ Features

✅ **Perplexity Integration**
- Web research with citations
- Intelligent proposal generation
- Approach evaluation

✅ **LLM Scoring**
- GAUC metric (user-weighted AUC)
- nDCG@5 (ranking quality)
- Baseline & oracle tracking
- Convergence detection

✅ **Production Ready**
- Error handling & fallbacks
- Comprehensive logging
- Setup verification
- Zero impact on kuairand-starter-kit

## 📚 More Info

- **Full Docs**: `README.md` or `INTEGRATION_GUIDE.md`
- **Integration Details**: `INTEGRATION_GUIDE.md`
- **Code Examples**: `examples.py`
- **Troubleshooting**: `setup.py`

---

**Status**: ✅ Ready to Use
**All Components**: ✅ Verified
**Integration**: ✅ Complete
