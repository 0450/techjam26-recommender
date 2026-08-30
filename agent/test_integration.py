"""
Integration test for Perplexity-powered agent.
Verifies all components are working correctly.
"""

from llm_scorer import LLMScorer
from judge import get_convergence_status
from perplexity_client import PerplexityClient
import os


def main():
    print('='*70)
    print('Perplexity Agent Integration Test')
    print('='*70)

    # Test 1: LLM Scorer
    print()
    print('[1/4] Testing LLM Scorer...')
    try:
        scorer = LLMScorer()
        potential = scorer.get_improvement_potential()
        
        print(f'  Baseline: {potential["baseline"]:.4f}')
        print(f'  Oracle:   {potential["oracle"]:.4f}')
        print(f'  Room:     {potential["room"]:.4f}')
        print('  [OK] LLM Scorer initialized')
    except Exception as e:
        print(f'  [FAIL] {e}')

    # Test 2: Convergence Status
    print()
    print('[2/4] Testing Convergence Status...')
    try:
        status = get_convergence_status()
        print(f'  Epsilon: {status["epsilon"]}')
        print(f'  N:       {status["N"]}')
        print(f'  Baseline: {status["baseline"]:.4f}')
        print('  [OK] Convergence status loaded')
    except Exception as e:
        print(f'  [FAIL] {e}')

    # Test 3: Perplexity Client
    print()
    print('[3/4] Testing Perplexity Client...')
    try:
        api_key = os.getenv('PERPLEXITY_API_KEY')
        if not api_key:
            print('  [SKIP] PERPLEXITY_API_KEY not set')
        else:
            client = PerplexityClient()
            print(f'  API Key: {api_key[:15]}...{api_key[-5:]}')
            print('  [OK] Perplexity client initialized')
    except Exception as e:
        print(f'  [FAIL] {e}')

    # Test 4: File Paths
    print()
    print('[4/4] Testing File Paths...')
    try:
        from pathlib import Path
        
        required_files = {
            'evaluate.py': Path('../kuairand-starter-kit/evaluate.py'),
            'baseline_scores.json': Path('../kuairand-starter-kit/baseline_scores.json'),
        }
        
        all_ok = True
        for name, path in required_files.items():
            if path.exists():
                print(f'  [OK] {name}')
            else:
                print(f'  [WARN] {name} not found at {path}')
                all_ok = False
        
        if all_ok:
            print('  [OK] All required files found')
    except Exception as e:
        print(f'  [FAIL] {e}')

    print()
    print('='*70)
    print('Integration test complete!')
    print('='*70)


if __name__ == '__main__':
    main()
