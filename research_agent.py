"""Run the Gemini-guided KuaiRand research loop.

Gemini proposes and evaluates ideas; the configured experiment command does the
actual work. Suggestions are logged for a human to apply because Gemini cannot
run or verify code changes.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# The starter kit is intentionally a script directory rather than a package.
sys.path.insert(0, str(Path(__file__).parent / "kuairand-starter-kit"))
from gemini_agent import GeminiResearchAgent, Iteration, ResearchContext


ROOT = Path(__file__).parent
STARTER_KIT = ROOT / "kuairand-starter-kit"
BASELINE_METRIC_RE = re.compile(
    r"valid\s+GAUC\s+(?P<gauc>[0-9.]+)\s+\|\s+nDCG@5\s+"
    r"(?P<ndcg>[0-9.]+)\s+\|\s+primary\s+(?P<primary>[0-9.]+)"
)
HETEROGENEOUS_METRIC_RE = re.compile(
    r"\* Blended Val GAUC\s*:\s*(?P<gauc>[0-9.]+).*?"
    r"\* Blended Val nDCG@5\s*:\s*(?P<ndcg>[0-9.]+).*?"
    r"\* Blended Val Primary\s*:\s*(?P<primary>[0-9.]+)",
    re.DOTALL,
)


def run_experiment(command: list[str], cwd: Path) -> tuple[dict[str, float], str]:
    """Run one existing experiment and extract its final validation metrics."""
    process = subprocess.Popen(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output_lines = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        output_lines.append(line)
    process.wait()
    output = "".join(output_lines)
    matches = list(BASELINE_METRIC_RE.finditer(output))
    if not matches:
        matches = list(HETEROGENEOUS_METRIC_RE.finditer(output))
    if process.returncode != 0:
        raise RuntimeError(f"Experiment failed with exit code {process.returncode}.\n{output}")
    if not matches:
        raise RuntimeError(f"Could not parse validation metrics from experiment output.\n{output}")
    match = matches[-1]
    return {key: float(value) for key, value in match.groupdict().items()}, output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--model", choices=["fm", "pop", "random"], default="fm",
                        help="Used only with --experiment baseline")
    parser.add_argument("--experiment", choices=["heterogeneous", "baseline"], default="heterogeneous")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--senet-embedding-dim", type=int, default=16)
    parser.add_argument("--dcn-embedding-dim", type=int, default=20)
    parser.add_argument("--eval-interval", type=int, default=3)
    parser.add_argument("--seed-count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "research_history.json")
    args = parser.parse_args()

    if args.iterations < 1:
        parser.error("--iterations must be at least 1")

    pipeline_path = (STARTER_KIT / "train_heterogeneous_blend.py"
                     if args.experiment == "heterogeneous" else STARTER_KIT / "baseline.py")
    pipeline_code = pipeline_path.read_text(encoding="utf-8")
    advisor = GeminiResearchAgent()
    baseline_primary = 0.5946
    history: list[Iteration] = []
    if args.experiment == "heterogeneous":
        experiment = [
            sys.executable,
            "-u",
            "train_heterogeneous_blend.py",
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--batch-size",
            str(args.batch_size),
            "--senet-embedding-dim",
            str(args.senet_embedding_dim),
            "--dcn-embedding-dim",
            str(args.dcn_embedding_dim),
            "--eval-interval",
            str(args.eval_interval),
            "--seed-count",
            str(args.seed_count),
        ]
    else:
        experiment = [sys.executable, "-u", "baseline.py", "--model", args.model, "--seed", str(args.seed)]

    for number in range(1, args.iterations + 1):
        context = ResearchContext(
            pipeline_code=pipeline_code,
            iterations=history,
            baseline_primary=baseline_primary,
            current_primary=history[-1].metrics["primary"] if history else None,
        )
        hypothesis = advisor.advise(context, "hypothesis")
        print(f"\n=== Iteration {number}: Gemini hypothesis ===\n{hypothesis}\n")

        metrics, output = run_experiment(experiment, STARTER_KIT)
        item = Iteration(number=number, metrics=metrics, hypothesis=hypothesis, outcome=output)
        history.append(item)
        print(
            f"Iteration {number} metrics: GAUC={metrics['gauc']:.4f}, "
            f"nDCG@5={metrics['ndcg']:.4f}, primary={metrics['primary']:.4f}"
        )

        decision = advisor.advise(
            ResearchContext(
                pipeline_code=pipeline_code,
                iterations=history,
                baseline_primary=baseline_primary,
                current_primary=metrics["primary"],
            ),
            "stop",
        )
        print(f"\n=== Gemini stop decision ===\n{decision}\n")

    args.output.write_text(
        json.dumps(
            {
                "baseline_primary": baseline_primary,
                "model": args.model,
                "iterations": [item.__dict__ for item in history],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved research history to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())