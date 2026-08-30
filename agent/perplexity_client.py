"""
Perplexity API client for research-driven LLM interaction.
Provides methods for research queries, content generation, and citations.
"""
import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from typing import Optional, Dict, List, Any


def _load_env_file() -> None:
    """Load .env from the agent folder or repo root if present."""
    base_dir = Path(__file__).resolve().parent
    for candidate in [base_dir / ".env", base_dir.parent / ".env"]:
        if candidate.exists():
            load_dotenv(candidate, override=False)


_load_env_file()


class PerplexityClient:
    """Client for interacting with Perplexity AI API."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Perplexity client.
        
        Args:
            api_key: Perplexity API key. If None, reads from PERPLEXITY_API_KEY env var.
        """
        _load_env_file()
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise ValueError(
                "PERPLEXITY_API_KEY not found. Set it as environment variable, add it to agent/.env, or pass it explicitly."
            )
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.default_model = "sonar-pro"

    def query(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> Dict[str, Any]:
        """
        Send a query to Perplexity API.
        
        Args:
            prompt: Query text
            model: Model to use (default: sonar-pro)
            temperature: Temperature for generation
            max_tokens: Maximum tokens in response
            
        Returns:
            Response dict with 'content', 'citations', 'usage', 'model', and 'raw'
        """
        model = model or self.default_model
        
        try:
            resp = requests.post(
                self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=60,
            )
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            return {
                "content": "",
                "citations": [],
                "usage": {},
                "model": model,
                "error": str(e),
                "raw": {},
            }

        data = resp.json()
        
        return {
            "content": data.get("choices", [{}])[0].get("message", {}).get("content", ""),
            "citations": data.get("citations", []),
            "usage": data.get("usage", {}),
            "model": model,
            "error": None,
            "raw": data,
        }

    def research(self, topic: str, focus: str = "") -> Dict[str, Any]:
        """
        Conduct research on a topic using Perplexity.
        
        Args:
            topic: Main research topic
            focus: Optional additional constraints or focus areas
            
        Returns:
            Dict with 'summary', 'sources', 'usage', and 'raw' response
        """
        prompt = f"{topic}. {focus}".strip()
        result = self.query(prompt, temperature=0.5)
        
        return {
            "summary": result["content"],
            "sources": result["citations"],
            "usage": result["usage"],
            "raw": result["raw"],
            "error": result.get("error"),
        }

    def propose_improvements(
        self,
        current_code: Dict[str, str],
        research_findings: str,
        history: List[Dict[str, Any]],
        constraint: str = "",
    ) -> Dict[str, Any]:
        """
        Use Perplexity to propose code improvements based on research findings.
        
        Args:
            current_code: Dict of filename -> file contents
            research_findings: Summary of research findings
            history: List of previous attempts and their results
            constraint: Additional constraints for the proposal
            
        Returns:
            Dict with 'idea', 'rationale', 'proposed_changes', and 'raw' response
        """
        # Build a summary of current code and history
        code_summary = "\n".join(
            f"--- {fname} ({len(content)} chars) ---\n{content[:500]}..."
            for fname, content in list(current_code.items())[:3]
        )
        
        history_summary = "\n".join(
            f"Attempt {i}: {h.get('idea', 'N/A')} → "
            f"score={h.get('score', 'N/A')}, accepted={h.get('accepted', False)}"
            for i, h in enumerate(history[-3:])  # Last 3 attempts
        )
        
        prompt = f"""Based on the following research findings and code context, propose specific improvements:

Research Findings:
{research_findings}

Current Code Structure (sample):
{code_summary}

Previous Attempts (last 3):
{history_summary}

Constraint: {constraint}

Propose concrete, testable improvements with specific code changes needed."""
        
        result = self.query(prompt, temperature=0.7, max_tokens=3000)
        
        return {
            "idea": result["content"][:200],  # First 200 chars as idea summary
            "rationale": result["content"],
            "proposed_changes": result["content"],  # Full response
            "citations": result["citations"],
            "usage": result["usage"],
            "raw": result["raw"],
            "error": result.get("error"),
        }

    def evaluate_approach(
        self,
        approach: str,
        current_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Use Perplexity to evaluate whether an approach is promising.
        
        Args:
            approach: Description of the approach
            current_metrics: Current performance metrics
            baseline_metrics: Baseline/best known metrics
            
        Returns:
            Dict with 'assessment', 'confidence', 'recommendations', and 'raw' response
        """
        prompt = f"""Evaluate this ML approach:

Approach:
{approach}

Current Metrics:
{current_metrics}

Baseline/Target Metrics:
{baseline_metrics}

Provide:
1. Brief assessment of promise
2. Confidence level (0-1)
3. Specific recommendations for improvement"""
        
        result = self.query(prompt, temperature=0.3, max_tokens=1500)
        
        return {
            "assessment": result["content"],
            "confidence": 0.5,  # Can be parsed from response if needed
            "recommendations": result["content"],
            "citations": result["citations"],
            "usage": result["usage"],
            "raw": result["raw"],
            "error": result.get("error"),
        }
