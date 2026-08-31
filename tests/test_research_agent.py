import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parents[1] / "kuairand-starter-kit"))

from research_agent import run_experiment


def test_run_experiment_parses_validation_metrics(tmp_path):
    command = [
        sys.executable,
        "-c",
        "print('valid GAUC 0.7000 | nDCG@5 0.5500 | primary 0.6250')",
    ]

    metrics, output = run_experiment(command, tmp_path)

    assert metrics == {"gauc": 0.7, "ndcg": 0.55, "primary": 0.625}
    assert "valid GAUC" in output


def test_run_experiment_parses_heterogeneous_metrics(tmp_path):
    command = [
        sys.executable,
        "-c",
        "print('* Blended Val GAUC     : 0.7100\\n* Blended Val nDCG@5   : 0.5600\\n* Blended Val Primary  : 0.6350')",
    ]

    metrics, _ = run_experiment(command, tmp_path)

    assert metrics == {"gauc": 0.71, "ndcg": 0.56, "primary": 0.635}