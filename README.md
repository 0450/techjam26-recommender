# techjam26-recommender

If you have trouble extracting the KuaiRand-Pure.tar.gz, I have uploaded the extracted folder to `https://files.catbox.moe/ej9pmm.7z`. Just extract internal folder and drop in into .\kuairand-starter-kit

## Optional Gemini research advisor

The existing training and evaluation scripts remain unchanged. To let Gemini
act as the text-only ML engineer between iterations, install the optional
dependencies and put your key in `.env`:

```powershell
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `GEMINI_API_KEY` in `.env` (and optionally `GEMINI_MODEL`). The advisor is
available from `kuairand-starter-kit/gemini_agent.py` and accepts the current
pipeline code, recent iteration metrics and hypotheses, baseline gap, EDA, and
tracebacks. It returns suggestions or code as text only; it never runs or
applies the response.

Run the connected research loop from the repository root:

```powershell
.\.venv\Scripts\python.exe research_agent.py --iterations 1 --model fm
```

Each iteration asks Gemini for a hypothesis, runs the existing PyTorch
heterogeneous blend experiment for up to 50 epochs, asks Gemini whether to stop,
and writes `research_history.json`. Increase
`--iterations` after applying a suggested code change to `baseline.py` or the
other experiment files. The runner does not auto-apply generated code.

To run the PyTorch trainer directly with the same limit:

```powershell
cd kuairand-starter-kit
..\.venv\Scripts\python.exe train_heterogeneous_blend.py --epochs 50 --patience 50
```