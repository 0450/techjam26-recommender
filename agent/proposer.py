"""
Proposer module using Perplexity AI to generate code improvement proposals.
Leverages research findings and historical results to recommend changes.
"""
from perplexity_client import PerplexityClient


def propose(research_findings: dict, history: list[dict], current_code: dict[str, str]) -> dict:
    """
    Use Perplexity to propose code improvements based on research findings.
    
    Args:
        research_findings: {'summary': str, 'sources': list, 'usage': dict, 'error': str|None}
        history: list of past iterations, each like
            {'idea': str, 'files_changed': [...], 'valid_primary': float, 'accepted': bool, 'error': str|None}
        current_code: {'data.py': "<contents>", 'baseline.py': "<contents>", ...} of the last-accepted version
    
    Returns:
        {'idea': str, 'rationale': str, 'files': {'data.py': '<new full contents>', ...}, 'error': str|None}
    """
    try:
        client = PerplexityClient()
        
        # Handle case where research_findings has an error
        if research_findings.get("error"):
            return {
                "idea": "Fallback proposal",
                "rationale": f"Research failed: {research_findings['error']}. Using baseline modifications.",
                "files": current_code,
                "error": research_findings["error"],
            }
        
        findings_text = research_findings.get("summary", "")
        
        # Get Perplexity's proposal
        proposal = client.propose_improvements(
            current_code=current_code,
            research_findings=findings_text,
            history=history,
            constraint="Must maintain backward compatibility with existing interface.",
        )
        
        if proposal.get("error"):
            return {
                "idea": "Fallback proposal",
                "rationale": f"Perplexity request failed: {proposal['error']}",
                "files": current_code,
                "error": proposal["error"],
            }
        
        # Extract proposed changes from Perplexity response
        # For now, return the proposal as-is (in real scenario, parse it and extract file changes)
        return {
            "idea": proposal.get("idea", "Unspecified improvement")[:100],
            "rationale": proposal.get("rationale", ""),
            "files": current_code,  # TODO: Parse rationale and apply code changes
            "citations": proposal.get("citations", []),
            "usage": proposal.get("usage", {}),
            "error": None,
        }
        
    except Exception as e:
        return {
            "idea": "Error in proposal",
            "rationale": f"Proposal generation failed: {str(e)}",
            "files": current_code,
            "error": str(e),
        }
