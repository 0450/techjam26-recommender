"""
Setup and verification script for Perplexity-powered agent.
Validates environment, dependencies, and connectivity.
"""
import sys
from pathlib import Path


def check_dependencies():
    """Check if all required packages are installed."""
    print("\n[1/5] Checking dependencies...")
    
    required = {
        'requests': 'requests',
        'perplexityai': 'perplexityai',
        'dotenv': 'dotenv',
    }
    
    missing = []
    for pkg_name, import_name in required.items():
        try:
            __import__(import_name)
            print(f"  [OK] {pkg_name}")
        except ImportError:
            print(f"  [MISSING] {pkg_name}")
            missing.append(pkg_name)
    
    if missing:
        print(f"\nInstall missing packages:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    return True


def check_api_key():
    """Check if Perplexity API key is available."""
    print("\n[2/5] Checking Perplexity API key...")
    
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if api_key:
        masked = api_key[:10] + "..." + api_key[-5:]
        print(f"  [OK] Found API key: {masked}")
        return True
    
    print("  [MISSING] PERPLEXITY_API_KEY not found")
    print("    Set it in .env or as environment variable:")
    print("    export PERPLEXITY_API_KEY=your_key_here")
    return False


def check_kuairand_integration():
    """Check if kuairand-starter-kit files are accessible."""
    print("\n[3/5] Checking kuairand-starter-kit integration...")
    
    kit_path = Path(__file__).parent.parent / "kuairand-starter-kit"
    required_files = [
        "evaluate.py",
        "baseline_scores.json",
        ".env",
    ]
    
    all_found = True
    for fname in required_files:
        fpath = kit_path / fname
        if fpath.exists():
            print(f"  [OK] {fname}")
        else:
            print(f"  [MISSING] {fname}")
            all_found = False
    
    if not all_found:
        print(f"\n  Please ensure kuairand-starter-kit is properly set up at: {kit_path}")
        return False
    
    return True


def test_perplexity_connection():
    """Test connectivity to Perplexity API."""
    print("\n[4/5] Testing Perplexity API connection...")
    
    try:
        from perplexity_client import PerplexityClient
        
        client = PerplexityClient()
        result = client.query("What is a recommendation system?", max_tokens=100)
        
        if result.get("error"):
            print(f"  [FAIL] API Error: {result['error']}")
            return False
        
        if result.get("content"):
            print(f"  [OK] API responded successfully")
            print(f"    Token usage: {result['usage']}")
            return True
        else:
            print("  [FAIL] Empty response from API")
            return False
            
    except Exception as e:
        print(f"  [FAIL] Connection failed: {e}")
        return False


def check_pipeline_dirs():
    """Check if pipeline directories exist."""
    print("\n[5/5] Checking pipeline directories...")
    
    from pathlib import Path
    
    dirs = [
        "pipeline/v0",
        "pipeline/current",
        "pipeline/attempts",
    ]
    
    all_found = True
    for dname in dirs:
        dpath = Path(dname)
        if dpath.exists():
            print(f"  [OK] {dname}/")
        else:
            print(f"  [WARN] {dname}/ (not found - will be created during run)")
    
    return True


def main():
    """Run all checks."""
    print("=" * 60)
    print("Perplexity Agent Setup Verification")
    print("=" * 60)
    
    checks = [
        ("Dependencies", check_dependencies),
        ("API Key", check_api_key),
        ("KuaiRand Integration", check_kuairand_integration),
        ("Perplexity Connection", test_perplexity_connection),
        ("Pipeline Directories", check_pipeline_dirs),
    ]
    
    results = []
    for name, check_fn in checks:
        try:
            passed = check_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\n  [FAIL] Check failed with error: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Setup Verification Summary")
    print("=" * 60)

    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status}: {name}")

    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)

    print(f"\n{passed_count}/{total_count} checks passed")

    if passed_count == total_count:
        print("\n[READY] Ready to run optimizer!")
        print("  python orchestrator.py")
        return 0
    else:
        print("\n[FIX] Fix the failed checks above before running optimizer")
        return 1


if __name__ == "__main__":
    sys.exit(main())
