from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def load_env_file(env_path: str | Path | None = None) -> None:
    """Load a local .env file into os.environ without requiring shell export."""
    path = Path(env_path) if env_path else Path(__file__).resolve().parent / '.env'
    if not path.exists():
        return
    for raw_line in path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


load_env_file()
DEFAULT_GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-3.6-flash')

# RecBole 1.2.x still references NumPy aliases removed in NumPy 2.x.
# Keep the compatibility shim local to this project so the adapter works
# without rewriting the third-party library itself.
for name, value in {
    'float_': np.float64,
    'complex_': np.complex128,
    'int_': np.int64,
    'bool_': np.bool_,
    'unicode_': np.str_,
    'str_': np.str_,
    'object_': np.object_,
    'float': float,
    'complex': complex,
    'int': int,
    'bool': bool,
    'long': int,
    'unicode': str,
}.items():
    if not hasattr(np, name):
        setattr(np, name, value)

try:
    from google import genai as google_genai
except Exception:  # pragma: no cover - optional dependency
    google_genai = None

try:
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.model.general_recommender.bpr import BPR
    from recbole.trainer import Trainer
except Exception:  # pragma: no cover - optional dependency
    Config = None
    create_dataset = None
    data_preparation = None
    BPR = None
    Trainer = None


class GeminiClient:
    """Minimal Gemini client wrapper for planning/debugging prompts."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.client = None
        if self.api_key and google_genai is not None:
            try:
                self.client = google_genai.Client(api_key=self.api_key)
            except Exception:
                self.client = None

    def generate_plan(self, prompt: str) -> str:
        if not self.client:
            return (
                'Gemini is not configured; using local heuristic planning. '
                'Next step: test the RecBole adapter on a small split, then extend the model zoo.'
            )
        try:
            response = self.client.models.generate_content(
                model=os.getenv('GEMINI_MODEL', DEFAULT_GEMINI_MODEL),
                contents=prompt,
            )
            text = getattr(response, 'text', None) or str(response)
            return text
        except Exception:
            return 'Gemini call failed; proceeding with local planner fallback.'


class RecBoleAdapter:
    """Thin adapter that converts the starter-kit split into a RecBole dataset."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.temp_dir = Path(tempfile.mkdtemp(prefix='kuairand_recbole_'))

    @staticmethod
    def _row_to_recbole(row: Tuple[Any, ...]) -> Dict[str, Any]:
        date, user_id, item_id, author_id, tab, duration_ms, label = row
        return {
            'user_id': str(user_id),
            'item_id': str(item_id),
            'label': float(label),
            'timestamp': int(date),
            'author_id': str(author_id),
            'tab': str(tab),
            'duration_ms': float(duration_ms),
        }

    def build_dataset(self, splits: Dict[str, List[Tuple[Any, ...]]]) -> Dict[str, Path]:
        if Config is None or create_dataset is None or data_preparation is None or BPR is None or Trainer is None:
            raise RuntimeError('RecBole is not installed in this environment.')

        dataset_dir = self.temp_dir / 'recbole_dataset'
        dataset_dir.mkdir(parents=True, exist_ok=True)
        paths = {}
        for split_name in ('train', 'valid', 'test'):
            path = dataset_dir / f'{split_name}.csv'
            with path.open('w', newline='') as fh:
                header = ['user_id', 'item_id', 'label', 'timestamp', 'author_id', 'tab', 'duration_ms']
                import csv
                writer = csv.DictWriter(fh, fieldnames=header)
                writer.writeheader()
                for row in splits[split_name]:
                    writer.writerow(self._row_to_recbole(row))
            paths[split_name] = path
        return paths

    def train_bpr(self, splits: Dict[str, List[Tuple[Any, ...]]]) -> Dict[str, Any]:
        paths = self.build_dataset(splits)
        dataset_name = 'kuairand_recbole'
        config_dict = {
            'model': 'BPR',
            'dataset': dataset_name,
            'data_path': str(self.temp_dir / 'recbole_dataset'),
            'field_separator': ',',
            'USER_ID_FIELD': 'user_id',
            'ITEM_ID_FIELD': 'item_id',
            'RATING_FIELD': 'label',
            'TIME_FIELD': 'timestamp',
            'load_col': {'interactions': ['user_id', 'item_id', 'label', 'timestamp', 'author_id', 'tab', 'duration_ms']},
            'epochs': 1,
            'train_batch_size': 2048,
            'eval_batch_size': 4096,
            'learning_rate': 0.001,
            'embedding_size': 16,
            'show_progress': False,
            'eval_args': {'split': {'valid': 'valid', 'test': 'test'}, 'order': 'RO', 'group_by': 'user', 'mode': 'full'},
            'neg_sampling': None,
        }
        config = Config(model='BPR', dataset=dataset_name, config_dict=config_dict)
        dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)
        model = BPR(config, train_data.dataset)
        trainer = Trainer(config, model)
        trainer.fit(train_data, valid_data, show_progress=False)
        return {
            'dataset': dataset_name,
            'train_path': paths['train'],
            'valid_path': paths['valid'],
            'test_path': paths['test'],
            'model': 'BPR',
            'status': 'ok',
        }


class ResearchAgent:
    """A compact autonomous research loop for the KuaiRand-Pure benchmark."""

    def __init__(self, data_dir: str | None = None, output_dir: str | None = None, model_name: str = 'fm', llm_provider: str = 'gemini'):
        self.root = Path(__file__).resolve().parent
        self.starter_dir = self.root / 'kuairand-starter-kit'
        self.data_dir = Path(data_dir) if data_dir else self.starter_dir / 'KuaiRand-Pure' / 'data'
        self.output_dir = Path(output_dir) if output_dir else self.root / 'results'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.llm_provider = llm_provider.lower()
        self.gemini = GeminiClient() if self.llm_provider == 'gemini' else None
        self._log: List[str] = []
        self.iterations_used = 0
        self.start_time = time.time()

    @staticmethod
    def _to_jsonable(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): ResearchAgent._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [ResearchAgent._to_jsonable(v) for v in value]
        if isinstance(value, np.ndarray):
            return [ResearchAgent._to_jsonable(v) for v in value.tolist()]
        if isinstance(value, (np.floating, np.integer, np.bool_)):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        return value

    def _log_line(self, message: str) -> None:
        self._log.append(message)
        print(message)

    def _ensure_dataset(self) -> None:
        if self.data_dir.exists():
            return

        archive = self.starter_dir / 'KuaiRand-Pure.tar.gz'
        if not archive.exists():
            url = 'https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz'
            self._log_line(f'Downloading archive from {url} ...')
            with urllib.request.urlopen(url, timeout=60) as resp, open(archive, 'wb') as fh:
                fh.write(resp.read())

        self._log_line(f'Extracting {archive} to {self.starter_dir} ...')
        with tarfile.open(archive, 'r:gz') as tar:
            tar.extractall(self.starter_dir)

    def _import_starter_modules(self):
        sys.path.insert(0, str(self.starter_dir))
        import baseline as B
        from data import encode, load
        from evaluate import evaluate

        return B, load, encode, evaluate

    def _build_llm_plan(self, splits: Dict[str, List]) -> str:
        if self.llm_provider != 'gemini':
            return 'Local planner active: compare FM and RecBole baselines on validation, keep the best score, then export a checkpoint.'
        prompt = (
            'You are a recommender-system research agent. '
            f'The benchmark is KuaiRand-Pure with {len(splits["train"])} train rows and {len(splits["valid"])} validation rows. '
            'Use the starter-kit evaluation conventions exactly and prefer a thin RecBole adapter rather than changing the evaluator. '
            'Keep the first pass simple: validate the baseline, then try a small RecBole model zoo selection.'
        )
        return self.gemini.generate_plan(prompt) if self.gemini else 'Local planner active: no Gemini key found.'

    def _run_recbole_loop(self, splits: Dict[str, List]) -> Dict[str, Any]:
        adapter = RecBoleAdapter(self.data_dir)
        self._log_line('Routing to RecBole adapter with a thin data conversion layer.')
        try:
            status = adapter.train_bpr(splits)
            self._log_line(f"RecBole model status: {status['status']} | model={status['model']} | dataset={status['dataset']}")
        except Exception as exc:  # pragma: no cover - environment-dependent
            self._log_line(f'RecBole path failed: {exc}')
            status = {'status': 'failed', 'model': 'BPR', 'dataset': 'kuairand_recbole', 'error': str(exc)}

        validation_best = {'GAUC': 0.0, 'nDCG@5': 0.0, 'primary': 0.0}
        test_best = {'GAUC': 0.0, 'nDCG@5': 0.0, 'primary': 0.0}
        return {
            'validation_best': validation_best,
            'test_best': test_best,
            'results_path': self.output_dir / 'results.json',
            'log_path': self.output_dir / 'research_run_log.txt',
            'checkpoint_path': self.output_dir / 'best_fm_checkpoint.npz',
            'iterations_used': self.iterations_used,
            'resource_report': {'total_llm_tokens': 0, 'total_wall_clock_seconds': round(time.time() - self.start_time, 2), 'iterations_used': self.iterations_used, 'gpu_hours': 0.0},
            'selected_model': 'BPR',
            'recbole_status': status,
        }

    def _train_fm_variant(self, splits: Dict[str, List], cfg: Dict[str, Any], evaluate_fn):
        B, load_fn, encode_fn, evaluate_fn = self._import_starter_modules()
        enc, dim = encode_fn(splits)
        Xtr, ytr, _ = enc['train']
        Xva, yva, uva = enc['valid']
        Xte, yte, ute = enc['test']

        rng = np.random.default_rng(cfg['seed'])
        model = B.FM(dim, k=cfg['k'], lr=cfg['lr'], l2=cfg.get('l2', 1e-6), seed=cfg['seed'])
        best_primary = -1.0
        best_state = None
        bad = 0

        for ep in range(1, cfg['epochs'] + 1):
            idx = rng.permutation(len(ytr))
            for i in range(0, len(idx), cfg['batch_size']):
                batch = idx[i:i + cfg['batch_size']]
                model.step(Xtr[batch], ytr[batch])
            valid = evaluate_fn(uva, yva, model.predict(Xva))
            if valid['primary'] > best_primary + 1e-5:
                best_primary = valid['primary']
                best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
                bad = 0
            else:
                bad += 1
                if bad >= cfg.get('patience', 4):
                    break

        if best_state is not None:
            model.V, model.W, model.b = best_state

        valid = evaluate_fn(uva, yva, model.predict(Xva))
        test = evaluate_fn(ute, yte, model.predict(Xte))
        return {
            'config': cfg,
            'model': model,
            'valid': valid,
            'test': test,
            'best_primary': best_primary,
        }

    def _candidate_configs(self) -> List[Dict[str, Any]]:
        return [
            {'name': 'fm_reference', 'k': 16, 'lr': 0.001, 'epochs': 40, 'batch_size': 8192, 'patience': 4, 'seed': 0},
            {'name': 'fm_lower_lr', 'k': 16, 'lr': 0.0005, 'epochs': 40, 'batch_size': 8192, 'patience': 4, 'seed': 0},
            {'name': 'fm_higher_lr', 'k': 16, 'lr': 0.002, 'epochs': 40, 'batch_size': 8192, 'patience': 4, 'seed': 0},
            {'name': 'fm_k8', 'k': 8, 'lr': 0.001, 'epochs': 40, 'batch_size': 8192, 'patience': 4, 'seed': 0},
            {'name': 'fm_k32', 'k': 32, 'lr': 0.001, 'epochs': 40, 'batch_size': 8192, 'patience': 4, 'seed': 0},
        ]

    def run_autonomous_loop(self) -> Dict[str, Any]:
        self._ensure_dataset()
        B, load_fn, encode_fn, evaluate_fn = self._import_starter_modules()
        splits = load_fn(str(self.data_dir))
        baseline_valid = {
            'GAUC': 0.6674,
            'nDCG@5': 0.5357,
            'primary': 0.6016,
        }
        baseline_test = {
            'GAUC': 0.6610,
            'nDCG@5': 0.5282,
            'primary': 0.5946,
        }

        self._log_line(f'Loaded dataset with rows: { {k: len(v) for k, v in splits.items()} }')
        self._log_line(self._build_llm_plan(splits))
        if self.model_name.lower() == 'recbole':
            return self._run_recbole_loop(splits)

        self._log_line('Starting autonomous FM search over validation-only configurations.')

        best_run = None
        best_valid_primary = -1.0
        results = []

        candidates = self._candidate_configs()
        for item in candidates:
            self.iterations_used += 1
            run = self._train_fm_variant(splits, item, evaluate_fn)
            results.append({
                'name': item['name'],
                'config': item,
                'valid': run['valid'],
                'test': run['test'],
            })
            self._log_line(
                f"iteration {self.iterations_used} | {item['name']} | "
                f"valid GAUC {run['valid']['GAUC']:.4f} nDCG@5 {run['valid']['nDCG@5']:.4f} primary {run['valid']['primary']:.4f} | "
                f"test GAUC {run['test']['GAUC']:.4f} nDCG@5 {run['test']['nDCG@5']:.4f} primary {run['test']['primary']:.4f}"
            )
            if run['valid']['primary'] > best_valid_primary:
                best_valid_primary = run['valid']['primary']
                best_run = run

        if best_run is None:
            raise RuntimeError('No valid FM configuration produced a valid score.')

        best_results = {
            'GAUC': float(best_run['valid']['GAUC']),
            'nDCG@5': float(best_run['valid']['nDCG@5']),
            'primary': float(best_run['valid']['primary']),
        }
        best_test = {
            'GAUC': float(best_run['test']['GAUC']),
            'nDCG@5': float(best_run['test']['nDCG@5']),
            'primary': float(best_run['test']['primary']),
        }

        # Save the best model checkpoint.
        checkpoint_path = self.output_dir / 'best_fm_checkpoint.npz'
        model = best_run['model']
        np.savez(checkpoint_path, V=model.V, W=model.W, b=np.asarray(model.b, dtype=np.float32))

        # Save the run log.
        log_path = self.output_dir / 'research_run_log.txt'
        log_path.write_text('\n'.join(self._log) + '\n', encoding='utf-8')

        # Save structured results.
        chosen_name = best_run['config']['name']
        resource_report = {
            'total_llm_tokens': 0,
            'total_wall_clock_seconds': round(time.time() - self.start_time, 2),
            'iterations_used': self.iterations_used,
            'gpu_hours': 0.0,
        }
        results_payload = {
            'dataset': 'KuaiRand-Pure',
            'selected_model': chosen_name,
            'selected_config': best_run['config'],
            'validation_best': best_results,
            'test_best': best_test,
            'delta_over_baseline_valid': {
                'GAUC': float(best_results['GAUC'] - baseline_valid['GAUC']),
                'nDCG@5': float(best_results['nDCG@5'] - baseline_valid['nDCG@5']),
                'primary': float(best_results['primary'] - baseline_valid['primary']),
            },
            'delta_over_baseline_test': {
                'GAUC': float(best_test['GAUC'] - baseline_test['GAUC']),
                'nDCG@5': float(best_test['nDCG@5'] - baseline_test['nDCG@5']),
                'primary': float(best_test['primary'] - baseline_test['primary']),
            },
            'resource_report': resource_report,
            'all_trials': results,
            'checkpoint_path': str(checkpoint_path),
            'log_path': str(log_path),
        }
        jsonable_payload = self._to_jsonable(results_payload)
        results_path = self.output_dir / 'results.json'
        results_path.write_text(json.dumps(jsonable_payload, indent=2), encoding='utf-8')

        return {
            'validation_best': best_results,
            'test_best': best_test,
            'results_path': results_path,
            'log_path': log_path,
            'checkpoint_path': checkpoint_path,
            'iterations_used': self.iterations_used,
            'resource_report': resource_report,
            'selected_model': chosen_name,
        }


def _print_summary(result: Dict[str, Any]) -> None:
    val = result.get('validation_best', {})
    test = result.get('test_best', {})
    elapsed = result.get('resource_report', {}).get('total_wall_clock_seconds', 0.0)
    selected = result.get('selected_model', 'unknown')
    line = '=' * 80

    print(line)
    print(f"seed 0 — valid={val.get('primary', 0.0):.6f} test={test.get('primary', 0.0):.6f} elapsed={elapsed:.2f}s")
    print(line)
    print(f"Val GAUC : {val.get('GAUC', 0.0):.4f} | Val nDCG@5 : {val.get('nDCG@5', 0.0):.4f} | Val Primary : {val.get('primary', 0.0):.4f}")
    print(f"Test GAUC : {test.get('GAUC', 0.0):.4f} | Test nDCG@5 : {test.get('nDCG@5', 0.0):.4f} | Test Primary : {test.get('primary', 0.0):.4f}")
    print(line)
    print(f"Selected model: {selected}")
    print(line)


def main() -> None:
    ap = argparse.ArgumentParser(description='Autonomous KuaiRand research runner.')
    ap.add_argument('--model', choices=['fm', 'recbole'], default='fm', help='Model family to run.')
    ap.add_argument('--llm-provider', choices=['gemini', 'none'], default='gemini', help='LLM provider for planning; Gemini is supported via GEMINI_API_KEY.')
    ap.add_argument('--data-dir', default=None, help='Override the KuaiRand data directory.')
    ap.add_argument('--output-dir', default=None, help='Directory for results and checkpoints.')
    a = ap.parse_args()

    agent = ResearchAgent(data_dir=a.data_dir, output_dir=a.output_dir, model_name=a.model, llm_provider=a.llm_provider)
    result = agent.run_autonomous_loop()
    _print_summary(result)


if __name__ == '__main__':
    main()
