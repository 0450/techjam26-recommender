#!/usr/bin/env python3
"""Autonomous FM research agent for KuaiRand-Pure, driven by the Perplexity Agent API.

Scope discipline (enforced in code, not just comments):
  - The optimization loop only ever reads splits['train'] and splits['valid'].
  - splits['test'] (the hidden/held-out period) is never loaded into a metric call
    anywhere in this file. It is not scored, not printed, not sent to the model.
  - The only thing produced for a 'test' run is a submission CSV of predicted
    scores (features only, no labels needed) -- the same thing submit.py --make
    already does for the official baseline.

Perplexity integration:
  - Uses the official `perplexityai` SDK (`pip install perplexityai`), not a
    hand-rolled HTTP client.
  - Calls `client.responses.create(...)`, which hits POST /v1/agent under the
    OpenAI-compatible `/v1/responses` alias documented for the Agent API.
  - Requests structured JSON via `response_format` (json_schema) instead of
    regex-scraping a JSON blob out of free text.
  - Keeps conversation state across iterations with `previous_response_id`
    instead of resending the full experiment history every call.
  - Enables the `web_search` tool so the model can ground hyperparameter/
    architecture suggestions in public literature (allowed by the task scope).

The API key is never hardcoded. It is resolved from the PERPLEXITY_API_KEY
environment variable and is never printed, logged, or written to disk.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv

from baseline import FM
from data import load, encode
from evaluate import evaluate
load_dotenv()  # loads PERPLEXITY_API_KEY 
DEFAULT_MODEL_PRESET = "medium"  # bundles model + tools + limits; overridable via --preset/--model

# ---------------------------------------------------------------------------
# API key handling
# ---------------------------------------------------------------------------


def resolve_api_key() -> str:
    """Return the API key from the environment without printing or logging it."""
    api_key = (os.getenv("PERPLEXITY_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError(
            "Missing PERPLEXITY_API_KEY. Create one in the Perplexity API Console "
            "(https://console.perplexity.ai), then export it in your own terminal, e.g.: "
            "export PERPLEXITY_API_KEY=your_key_here. Do not paste it into this chat. "
            "If it has ever been exposed, rotate it immediately in the console."
        )
    return api_key


# ---------------------------------------------------------------------------
# Perplexity Agent API client (official SDK)
# ---------------------------------------------------------------------------

PLAN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "best_hypothesis": {"type": "string"},
        "next_experiments": {
            "type": "array",
            "minItems": 1,
            "maxItems": 4,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "why": {"type": "string"},
                    "k": {"type": "integer", "minimum": 2, "maximum": 64},
                    "lr": {"type": "number", "exclusiveMinimum": 0, "maximum": 0.05},
                    "epochs": {"type": "integer", "minimum": 5, "maximum": 80},
                    "patience": {"type": "integer", "minimum": 2, "maximum": 10},
                },
                "required": ["name", "why", "k", "lr", "epochs", "patience"],
            },
        },
        "stop_rule": {"type": "string"},
    },
    "required": ["best_hypothesis", "next_experiments", "stop_rule"],
}


class PerplexityAgentClient:
    """Thin wrapper around the official `perplexity` SDK's Agent (`responses`) API."""

    def __init__(self, api_key: str):
        from perplexity import Perplexity  # imported lazily so --no-agent needs no SDK/network

        self._client = Perplexity(api_key=api_key)
        self._last_response_id: Optional[str] = None

    def ask_for_plan(self, prompt: str, *, model: Optional[str] = None,
                      preset: Optional[str] = None, use_web_search: bool = True) -> Dict[str, Any]:
        """Send `prompt` to the Agent API and return the plan as a parsed dict.

        Uses `previous_response_id` so later calls only need to send new
        information rather than the full running history.
        """
        kwargs: Dict[str, Any] = {
            "input": prompt,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "research_plan", "schema": PLAN_SCHEMA, "strict": True},
            },
        }
        if use_web_search:
            kwargs["tools"] = [{"type": "web_search"}]
        if model:
            kwargs["model"] = model
        elif preset:
            kwargs["preset"] = preset
        if self._last_response_id:
            kwargs["previous_response_id"] = self._last_response_id

        response = self._client.responses.create(**kwargs)
        self._last_response_id = getattr(response, "id", None)

        text = getattr(response, "output_text", None)
        if not text:
            raise ValueError("Perplexity response had no output_text to parse.")
        return json.loads(text)


def default_agent_plan() -> Dict[str, Any]:
    """Local fallback plan used when the Agent API is disabled or unavailable."""
    return {
        "best_hypothesis": (
            "Increase learning-rate stability and rank-aware regularization around the FM "
            "encoder before considering deeper architectures."
        ),
        "next_experiments": [
            {"name": "fm_k08_lr3e4", "why": "smaller embedding, lower lr: check underfitting boundary",
             "k": 8, "lr": 3e-4, "epochs": 50, "patience": 5},
            {"name": "fm_k16_lr1e3", "why": "current default configuration as a control arm",
             "k": 16, "lr": 1e-3, "epochs": 40, "patience": 4},
            {"name": "fm_k24_lr2e3", "why": "slightly higher capacity with a faster schedule",
             "k": 24, "lr": 2e-3, "epochs": 35, "patience": 4},
        ],
        "stop_rule": "Stop when validation primary stays within 0.002 of the best score for 3 consecutive iterations.",
    }


# ---------------------------------------------------------------------------
# Training / evaluation -- VALIDATION SPLIT ONLY
# ---------------------------------------------------------------------------


def train_fm_once(enc: Dict[str, Any], dim: int, *, k: int, lr: float, epochs: int, bs: int = 8192,
                   seed: int = 0, patience: int = 4, verbose: bool = False) -> Dict[str, Any]:
    """Train one FM configuration and score it on the VALID split only.

    `enc`/`dim` must come from data.encode(); this function never reads enc['test'].
    """
    Xtr, ytr, _ = enc['train']
    Xva, yva, uva = enc['valid']
    model = FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best = -1.0
    best_state = None
    bad = 0
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(ytr))
        losses = []
        for start in range(0, len(idx), bs):
            batch = idx[start:start + bs]
            losses.append(model.step(Xtr[batch], ytr[batch]))
        valid_metrics = evaluate(uva, yva, model.predict(Xva))
        if verbose:
            print(
                f"  epoch {ep:2d} | loss {float(np.mean(losses)):.4f} | "
                f"valid GAUC {valid_metrics['GAUC']:.4f} "
                f"nDCG@5 {valid_metrics['nDCG@5']:.4f} "
                f"primary {valid_metrics['primary']:.4f}"
            )
        if valid_metrics['primary'] > best + 1e-5:
            best = valid_metrics['primary']
            bad = 0
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  early stop at epoch {ep}")
                break
    if best_state is None:
        best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
    model.V, model.W, model.b = best_state
    valid_metrics = evaluate(uva, yva, model.predict(Xva))
    return {'valid': valid_metrics, 'model_state': best_state, 'config': {'k': k, 'lr': lr, 'epochs': epochs,
                                                                           'bs': bs, 'seed': seed, 'patience': patience}}


def baseline_experiment(enc: Dict[str, Any], dim: int) -> Dict[str, Any]:
    return train_fm_once(enc, dim, k=16, lr=1e-3, epochs=40, bs=8192, seed=0, patience=4)


def run_experiment(enc: Dict[str, Any], dim: int, cfg: Dict[str, Any], seed: int) -> Dict[str, Any]:
    result = train_fm_once(
        enc,
        dim,
        k=int(cfg.get('k', 16)),
        lr=float(cfg.get('lr', 1e-3)),
        epochs=int(cfg.get('epochs', 40)),
        patience=int(cfg.get('patience', 4)),
        seed=seed,
    )
    result['name'] = cfg.get('name', f"cfg_k{cfg.get('k')}_lr{cfg.get('lr')}")
    result['primary'] = result['valid']['primary']
    return result


def summarize_results(history: List[Dict[str, Any]]) -> str:
    lines = []
    for item in history:
        m = item['valid']
        cfg = item.get('config', {})
        lines.append(
            f"{item['name']}: k={cfg.get('k')} lr={cfg.get('lr')} epochs={cfg.get('epochs')} "
            f"-> GAUC={m['GAUC']:.4f} nDCG@5={m['nDCG@5']:.4f} primary={m['primary']:.4f}"
        )
    return "\n".join(lines)


def build_agent_prompt(history: List[Dict[str, Any]], baseline_valid_primary: float, iteration: int,
                        max_iterations: int) -> str:
    if iteration == 1:
        return f"""You are an autonomous ML research agent tuning a Factorization Machine (FM) for the
KuaiRand-Pure recommendation benchmark (within-user ranking; label = long_view; primary
metric = mean(GAUC, nDCG@5)).

Hard rules:
- You only ever see validation-split feedback. The hidden test split is never shown to you
  and must never be referenced.
- Propose FM hyperparameter configurations only: k (embedding dim), lr, epochs, patience.
- Improvement need not be monotonic, but your proposals should show a clear intent to search
  past the current best rather than repeating it.

Current information:
- Official FM baseline, public validation primary: {baseline_valid_primary:.4f}
- Iteration budget: {max_iterations}
- This is iteration {iteration}; no experiments have run yet.

Propose {min(3, max_iterations)} FM configurations to try next, each with a brief rationale.
You may use web search to ground your rationale in published FM/CTR-ranking tuning practice.
Respond only with the structured JSON object described by the schema."""
    return f"""Here are the results since the last message (validation-split only; no test data):
{summarize_results(history[-3:])}

Best validation primary so far: {max(h['primary'] for h in history):.4f}
Iteration {iteration} of {max_iterations}.

Propose the next FM configuration(s) to try (k, lr, epochs, patience), building on what worked
or explicitly moving away from what didn't. Respond only with the structured JSON object
described by the schema."""


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_agent_loop(data_dir: str, output_path: str, max_iterations: int, split: str, use_agent: bool,
                    model: Optional[str], preset: Optional[str], dry_run: bool = False) -> Dict[str, Any]:
    splits = load(data_dir)
    print(f"loaded {len(splits['train'])} train rows, {len(splits['valid'])} valid rows "
          f"(test split loaded but withheld from every metric call below)")
    enc, dim = encode(splits)

    baseline = baseline_experiment(enc, dim)
    print(f"baseline valid => GAUC {baseline['valid']['GAUC']:.4f} | "
          f"nDCG@5 {baseline['valid']['nDCG@5']:.4f} | primary {baseline['valid']['primary']:.4f}")

    history: List[Dict[str, Any]] = [{**baseline, 'name': 'baseline_fm', 'primary': baseline['valid']['primary']}]
    best = baseline
    plateau = 0

    agent: Optional[PerplexityAgentClient] = None
    if use_agent:
        try:
            agent = PerplexityAgentClient(resolve_api_key())
        except Exception as exc:
            print(f"Perplexity Agent API unavailable, falling back to local heuristic planner: {exc}")
            agent = None

    for iteration in range(1, max_iterations + 1):
        if agent is not None:
            try:
                prompt = build_agent_prompt(history, baseline['valid']['primary'], iteration, max_iterations)
                plan = agent.ask_for_plan(prompt, model=model, preset=preset)
                print(f"[iter {iteration}] Perplexity hypothesis: {plan.get('best_hypothesis', '')}")
            except Exception as exc:
                print(f"[iter {iteration}] Perplexity call failed ({exc}); using local heuristic plan instead")
                plan = default_agent_plan()
        else:
            plan = default_agent_plan()

        experiments = plan.get('next_experiments') or default_agent_plan()['next_experiments']
        for cfg in experiments:
            result = run_experiment(enc, dim, cfg, seed=iteration)
            history.append(result)
            improved = result['primary'] > best['valid']['primary'] + 1e-5
            if improved:
                best = result
                plateau = 0
            else:
                plateau += 1
            print(
                f"iteration {iteration} / {result['name']}: "
                f"GAUC {result['valid']['GAUC']:.4f} | nDCG@5 {result['valid']['nDCG@5']:.4f} | "
                f"primary {result['valid']['primary']:.4f}"
                + ("  <-- new best" if improved else "")
            )
            if plateau >= 3:
                print("Convergence reached: validation improvement has stalled for 3 iterations; stopping.")
                break
        if plateau >= 3:
            break

    best_name = best.get('name', 'baseline_fm')
    print(f"final selection: {best_name} with valid primary {best['valid']['primary']:.4f} "
          f"(delta vs baseline: {best['valid']['primary'] - baseline['valid']['primary']:+.4f})")

    if dry_run:
        return {'best': best, 'history': history, 'best_name': best_name}

    # Refit the selected configuration and score the requested split.
    # NOTE: this only ever uses X (features) of the target split, never its labels --
    # identical in spirit to submit.py --make. If split == 'test', no test label is read.
    cfg = best.get('config', {'k': 16, 'lr': 1e-3, 'epochs': 40, 'patience': 4})
    Xtr, ytr, _ = enc['train']
    Xva, yva, uva = enc['valid']
    X_target, _y_target_unused, _u_target = enc[split]
    model_fm = FM(dim, k=cfg['k'], lr=cfg['lr'], seed=0)
    rng = np.random.default_rng(0)
    best_seen = -1.0
    best_state = None
    for _ in range(cfg['epochs']):
        idx = rng.permutation(len(ytr))
        for start in range(0, len(idx), cfg.get('bs', 8192)):
            batch = idx[start:start + cfg.get('bs', 8192)]
            model_fm.step(Xtr[batch], ytr[batch])
        score = evaluate(uva, yva, model_fm.predict(Xva))['primary']
        if score > best_seen + 1e-5:
            best_seen = score
            best_state = (model_fm.V.copy(), model_fm.W.copy(), np.float32(model_fm.b))
    if best_state is not None:
        model_fm.V, model_fm.W, model_fm.b = best_state

    final_scores = model_fm.predict(X_target)
    rows = splits[split]
    with open(output_path, 'w', newline='') as fh:
        fh.write('row_id,user_id,video_id,score\n')
        for idx, row in enumerate(rows):
            fh.write(f"{idx},{row[1]},{row[2]},{float(final_scores[idx]):.6g}\n")
    print(f"wrote {split}-split submission to {output_path} (labels of that split were never read)")

    return {'best': best, 'history': history, 'best_name': best_name, 'output_path': output_path}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perplexity-Agent-backed KuaiRand-Pure FM research agent.')
    parser.add_argument('--data_dir', default='./KuaiRand-Pure/data', help='KuaiRand-Pure data directory')
    parser.add_argument('--split', default='valid', choices=['valid', 'test'],
                         help='Split to write a submission CSV for. Never used to score during the search loop.')
    parser.add_argument('--output', default='submission.csv', help='Submission CSV path.')
    parser.add_argument('--max-iterations', type=int, default=6, help='Maximum optimization rounds.')
    parser.add_argument('--no-agent', action='store_true',
                         help='Skip the Perplexity Agent API and use the local heuristic planner only.')
    parser.add_argument('--model', default=None, help='Explicit Perplexity model id (overrides --preset).')
    parser.add_argument('--preset', default=DEFAULT_MODEL_PRESET,
                         help='Perplexity Agent API preset (e.g. low/medium/high). Ignored if --model is set.')
    parser.add_argument('--dry-run', action='store_true', help='Run the search loop without writing a submission CSV.')
    args = parser.parse_args()

    try:
        run_agent_loop(
            data_dir=args.data_dir,
            output_path=args.output,
            max_iterations=args.max_iterations,
            split=args.split,
            use_agent=not args.no_agent,
            model=args.model,
            preset=args.preset,
            dry_run=args.dry_run,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:  # pragma: no cover - guardrail for unexpected issues
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
