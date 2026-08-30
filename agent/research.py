"""
Research module using Perplexity AI for topic-driven insights.
Provides research functionality to support LLM-based optimization.
"""
import os
from perplexity_client import PerplexityClient


def research(topic: str, focus: str = "") -> dict:
    """
    Conduct research on a topic using Perplexity AI.
    
    Args:
        topic: e.g. "pairwise ranking loss for implicit feedback recommendation"
        focus: optional extra constraint, e.g. "must work with plain numpy, no GPU"
    
    Returns:
        {
            'summary': str - main research findings
            'sources': list[str] - citations/sources
            'usage': dict - API usage stats
            'raw': dict - full API response
            'error': str or None
        }
    """
    try:
        client = PerplexityClient()
        result = client.research(topic, focus)
        return result
    except Exception as e:
        return {
            "summary": "",
            "sources": [],
            "usage": {},
            "raw": {},
            "error": str(e),
        }
