import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "kuairand-starter-kit"))

from gemini_agent import GeminiResearchAgent, Iteration, ResearchContext


def test_prompt_contains_iteration_context():
    context = ResearchContext(
        pipeline_code="def train(): pass",
        iterations=[Iteration(1, {"primary": 0.6}, "try BPR", "validation improved")],
        baseline_primary=0.5946,
        current_primary=0.6,
        eda_output="rows=100",
    )

    prompt = GeminiResearchAgent(api_key="test-key").build_prompt(context, "hypothesis")

    assert "def train(): pass" in prompt
    assert "try BPR" in prompt
    assert "Gap to baseline" in prompt
    assert "rows=100" in prompt


def test_advise_forwards_prompt_to_gemini():
    class FakeModels:
        def generate_content(self, **kwargs):
            self.kwargs = kwargs
            return type("Response", (), {"text": "use pairwise loss"})()

    class FakeClient:
        def __init__(self):
            self.models = FakeModels()

    agent = GeminiResearchAgent(api_key="test-key")
    agent._client = FakeClient()
    response = agent.advise(ResearchContext(pipeline_code="pass"), "code")

    assert response == "use pairwise loss"
    assert agent._client.models.kwargs["model"] == "gemini-3.6-flash"
    assert "Current pipeline code" in agent._client.models.kwargs["contents"]


def test_advise_retries_transient_gemini_errors():
    class TemporaryError(Exception):
        status_code = 503

    class FakeModels:
        def __init__(self):
            self.calls = 0

        def generate_content(self, **kwargs):
            self.calls += 1
            if self.calls < 2:
                raise TemporaryError("busy")
            return type("Response", (), {"text": "retry succeeded"})()

    class FakeClient:
        def __init__(self):
            self.models = FakeModels()

    agent = GeminiResearchAgent(api_key="test-key")
    agent._client = FakeClient()
    agent.advise(ResearchContext(pipeline_code="pass"))

    assert agent._client.models.calls == 2