import subprocess, shutil, uuid, os, json

def run_attempt(files: dict[str, str], base_version_dir: str) -> dict:
    """
    Copies base_version_dir to a fresh scratch dir, overwrites with `files`,
    executes the pipeline, returns a result dict — never raises.
    """
    attempt_dir = f"pipeline/attempts/{uuid.uuid4().hex[:8]}"
    shutil.copytree(base_version_dir, attempt_dir)
    for fname, content in files.items():
        with open(os.path.join(attempt_dir, fname), "w") as f:
            f.write(content)

    try:
        proc = subprocess.run(
            ["python3", "run_and_score.py", "--split", "valid"],
            cwd=attempt_dir, capture_output=True, timeout=900, text=True,
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "dir": attempt_dir}

    if proc.returncode != 0:
        return {"status": "error", "traceback": proc.stderr, "dir": attempt_dir}

    try:
        metrics = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return {"status": "bad_output", "stdout": proc.stdout, "dir": attempt_dir}

    if any(m != m for m in metrics.values()):  # NaN check
        return {"status": "nan", "metrics": metrics, "dir": attempt_dir}

    return {"status": "ok", "metrics": metrics, "dir": attempt_dir}