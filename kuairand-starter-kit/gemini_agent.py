"""Optional Gemini advisor for the KuaiRand research loop.

The advisor proposes experiments and explains failures, but never executes or
modifies code. Existing training and evaluation scripts remain independent of
the Gemini SDK and can still run without an API key.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Union

from dotenv import load_dotenv


Action = Literal["hypothesis", "code", "error", "stop"]


@dataclass
class Iteration:
    """One completed research iteration supplied to Gemini as history."""

    number: int
    metrics: dict[str, float]
    hypothesis: str = ""
    outcome: str = ""


@dataclass
class ResearchContext:
    """Run state used to construct a bounded, reproducible advisor prompt."""

    pipeline_code: str
    iterations: list[Iteration] = field(default_factory=list)
    baseline_primary: Optional[float] = None
    current_primary: Optional[float] = None
    eda_output: str = ""
    traceback: str = ""


class GeminiResearchAgent:
    """Call Gemini as the text-only ML engineer in a research loop."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        dotenv_path: Optional[Union[str, Path]] = None,
    ) -> None:
        env_file = Path(dotenv_path) if dotenv_path else Path(__file__).parents[1] / ".env"
        load_dotenv(env_file)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
        self._client = None

    def _client_for_request(self):
        if not self.api_key:
            raise RuntimeError(
                "Gemini is not configured. Set GEMINI_API_KEY in the project .env file."
            )
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise RuntimeError(
                    "Gemini support requires google-genai. Install requirements.txt."
                ) from exc
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def build_prompt(self, context: ResearchContext, action: Action) -> str:
        """Build the complete context Gemini needs for one decision."""
        gap = "unknown"
        if context.baseline_primary is not None and context.current_primary is not None:
            gap = f"{context.current_primary - context.baseline_primary:+.6f}"
        history = "\n".join(
            f"- Iteration {item.number}: metrics={item.metrics}; "
            f"hypothesis={item.hypothesis or 'n/a'}; outcome={item.outcome or 'n/a'}"
            for item in context.iterations[-5:]
        ) or "- No previous iterations."
        request = {
            "hypothesis": "Propose one testable ML hypothesis and the smallest experiment that can test it.",
            "code": "Propose a focused code change. Return a unified diff or a complete replacement for the relevant function, plus rationale.",
            "error": "Interpret the traceback, identify the likely root cause, and propose the smallest fix. Do not invent unseen files.",
            "stop": "Decide whether to continue or stop. Use the convergence rule and evidence; state the decision and reason.",
        }[action]
        return f"""You are the ML engineer driving a recommender-system research loop.
You cannot run commands, inspect files, or verify results. Base every claim only on
the context below. Preserve the existing evaluation protocol and avoid changing
unrelated behavior.

Requested action: {action}
{request}

Baseline primary score: {context.baseline_primary if context.baseline_primary is not None else 'unknown'}
Current primary score: {context.current_primary if context.current_primary is not None else 'unknown'}
Gap to baseline (current - baseline): {gap}

Last iterations:
{history}

EDA output:
{context.eda_output or 'None provided.'}

Traceback:
{context.traceback or 'None provided.'}

Current pipeline code:
```python
{context.pipeline_code}
```

Return concise, actionable text. Keep generated code limited to the requested
change and call out assumptions explicitly.
"""

    def advise(self, context: ResearchContext, action: Action = "hypothesis") -> str:
        """Send one context-rich request to Gemini and return its text response."""
        prompt = self.build_prompt(context, action)
        for attempt in range(3):
            try:
                response = self._client_for_request().models.generate_content(
                    model=self.model,
                    contents=prompt,
                )
                break
            except Exception as exc:
                status_code = getattr(exc, "status_code", None)
                if status_code not in (429, 500, 502, 503, 504) or attempt == 2:
                    raise
                delay = 5 * (2 ** attempt)
                print(
                    f"Gemini temporarily unavailable ({status_code}); "
                    f"retrying in {delay}s ({attempt + 1}/2)...",
                    flush=True,
                )
                time.sleep(delay)
        text = getattr(response, "text", None)
        if not text:
            raise RuntimeError("Gemini returned an empty response.")
        return text