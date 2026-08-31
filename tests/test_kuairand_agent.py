import json
from pathlib import Path

import numpy as np

from kuairand_agent import ResearchAgent


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'kuairand-starter-kit' / 'KuaiRand-Pure' / 'data'


def test_agent_reproduces_reference_validation_baseline():
    agent = ResearchAgent(data_dir=str(DATA_DIR), output_dir=str(ROOT / 'results'))
    run = agent.run_autonomous_loop()

    assert 'validation_best' in run
    assert run['validation_best']['primary'] > 0.59
    assert 0.0 <= run['validation_best']['GAUC'] <= 1.0
    assert 0.0 <= run['validation_best']['nDCG@5'] <= 1.0
    assert run['iterations_used'] >= 1


def test_agent_logs_and_saves_results():
    agent = ResearchAgent(data_dir=str(DATA_DIR), output_dir=str(ROOT / 'results'))
    run = agent.run_autonomous_loop()

    assert run['results_path'].exists()
    assert run['log_path'].exists()
    assert run['checkpoint_path'].exists()

    payload = json.loads(run['results_path'].read_text())
    assert payload['validation_best']['primary'] >= 0.0
    assert payload['resource_report']['iterations_used'] >= 1
