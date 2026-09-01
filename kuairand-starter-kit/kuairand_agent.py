import os
import sys
import csv
import json
import argparse
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[Env] Loaded environment variables from .env file.")
except ImportError:
    print("[Env Warning] 'python-dotenv' not installed. Reading system environment directly.")

from utils import save_json, clear_memory
from autotrainer import train_and_eval, run_full_ensemble_and_blend

try:
    from google import genai
    from google.genai import types
    google_genai_available = True
except ImportError:
    google_genai_available = False


class GeminiClient:
    """Manages hyperparameter optimization proposals using Google GenAI SDK."""
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if self.api_key and google_genai_available:
            print("[Agent] Gemini API Key detected. Initializing Gemini Client for autonomous optimization.")
            self.client = genai.Client(api_key=self.api_key)
        else:
            print("[WARNING] No valid GEMINI_API_KEY environment variable or SDK detected!")
            print("[WARNING] Autonomous Agent will fall back to static parameter space search.")
            self.client = None

        self.fallback_idx = 0
        self.static_fallbacks = [
            {"model_type": "senet", "lr": 0.0004, "embed_dim": 16, "hidden_dims": [128, 64], "dropout": 0.15, "batch_size": 8192},
            {"model_type": "senet", "lr": 0.0002, "embed_dim": 16, "hidden_dims": [256, 128], "dropout": 0.20, "batch_size": 8192},
            {"model_type": "lowrank_dcn", "lr": 0.0002, "embed_dim": 16, "hidden_dims": [128, 64], "dropout": 0.10, "batch_size": 8192},
            {"model_type": "lowrank_dcn", "lr": 0.0001, "embed_dim": 16, "hidden_dims": [256, 128], "dropout": 0.20, "batch_size": 8192},
        ]

    def propose_next_config(self, history: list) -> dict:
        if not self.client:
            cfg = self.static_fallbacks[self.fallback_idx % len(self.static_fallbacks)]
            self.fallback_idx += 1
            return cfg

        prompt = f"""You are an expert Recommendation Systems ML Engineer tuning models for the KuaiRand dataset.
Your goal is to maximize the official primary score = (GAUC + nDCG@5) / 2.0.

Past Experiment History (JSON):
{json.dumps(history, indent=2)}

Propose the next set of hyperparameter values as a valid JSON object ONLY.
Required JSON keys:
- "hypothesis": string (explanation of what is being tested and why)
- "model_type": string ("senet" or "lowrank_dcn")
- "lr": float (between 0.00005 and 0.001)
- "embed_dim": int (16 or 32)
- "hidden_dims": list of ints (e.g. [128, 64] or [256, 128])
- "dropout": float (between 0.05 and 0.3)
- "batch_size": int (4096 or 8192)
"""
        try:
            response = self.client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    response_mime_type="application/json"
                )
            )
            res = json.loads(response.text.strip())
            if "hypothesis" not in res:
                res["hypothesis"] = f"Exploring {res.get('model_type')} with lr={res.get('lr')}."
            return res
        except Exception as e:
            print(f"[Gemini API Call Failed] Error: {e}. Using fallback config.")
            cfg = self.static_fallbacks[self.fallback_idx % len(self.static_fallbacks)]
            cfg["hypothesis"] = "Fallback configuration due to API limits/error."
            self.fallback_idx += 1
            return cfg


class AutonomousResearchAgent:
    """Manages Stage 1 LLM exploration with convergence checking & Stage 2 ensemble execution."""
    def __init__(self, data_dir: str, output_dir: str, max_trials: int = 10, epsilon: float = 0.002):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.max_trials = max_trials
        self.epsilon = epsilon
        self.gemini = GeminiClient()
        self.history = []
        self.test_rows = []
        os.makedirs(output_dir, exist_ok=True)

    def load_dataset(self):
        """Loads KuaiRand data adhering strictly to official date splits and long_view label."""
        print(f"[Data Engine] Loading KuaiRand dataset from {self.data_dir}...")

        SPLITS = {
            'train': (20220408, 20220421),
            'valid': (20220422, 20220428),
            'test':  (20220429, 20220508)
        }

        # Load optional auxiliary features into dictionaries to avoid key errors & pandas duplicate suffixes
        u_ext = {}
        user_feat_path = os.path.join(self.data_dir, "user_features_pure.csv")
        USER_FE = ['follow_user_num_range', 'register_days_range', 'fans_user_num_range', 'friend_user_num_range', 'user_active_degree']
        if os.path.exists(user_feat_path):
            with open(user_feat_path, encoding='utf-8') as fh:
                for r in csv.DictReader(fh):
                    u_ext[r['user_id']] = [r.get(k, 'UNK') for k in USER_FE]

        v_ext = {}
        video_basic_path = os.path.join(self.data_dir, "video_features_basic_pure.csv")
        VID_FE = ['author_id', 'video_type', 'upload_type']
        if os.path.exists(video_basic_path):
            with open(video_basic_path, encoding='utf-8') as fh:
                for r in csv.DictReader(fh):
                    v_ext[r['video_id']] = [r.get(k, 'UNK') for k in VID_FE]

        rows = []
        for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
            path = os.path.join(self.data_dir, f)
            if os.path.exists(path):
                with open(path, encoding='utf-8') as fh:
                    for r in csv.DictReader(fh):
                        # Strict long_view label parsing per competition rules
                        label = 1.0 if r.get('long_view', '0') != '0' else 0.0
                        rows.append((int(r['date']), r['user_id'], r['video_id'], r.get('tab', 'UNK'),
                                     float(r.get('duration_ms', 0.0)), label))

        splits = {n: [x for x in rows if lo <= x[0] <= hi] for n, (lo, hi) in SPLITS.items()}
        self.test_rows = splits['test']

        print(f"[Data Engine] Loaded row counts: train={len(splits['train'])}, valid={len(splits['valid'])}, test={len(splits['test'])}")

        # Continuous duration binning calculated on train split ONLY
        train_durations = [x[4] for x in splits['train']]
        edges = np.quantile(train_durations, np.linspace(0, 1, 11)[1:-1])

        UNKU = ['UNK'] * len(USER_FE)
        UNKV = ['UNK'] * len(VID_FE)

        def raw_features(x):
            ue = u_ext.get(x[1], UNKU)
            ve = v_ext.get(x[2], UNKV)
            dur_bin = str(int(np.searchsorted(edges, x[4])))
            # Combined field list: user_id, video_id, tab, dur_bin, user_features, video_features
            return [x[1], x[2], x[3], dur_bin] + ue + ve

        n_fields = len(raw_features(splits['train'][0]))

        # Fit vocabularies on train split ONLY
        vocabs = [dict() for _ in range(n_fields)]
        for x in splits['train']:
            for i, v in enumerate(raw_features(x)):
                if v not in vocabs[i]:
                    vocabs[i][v] = len(vocabs[i])

        unk_slots = [len(v) for v in vocabs]
        field_dims = [len(v) + 1 for v in vocabs]
        offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)

        data_enc = {}
        for name, rws in splits.items():
            X = np.empty((len(rws), n_fields), dtype=np.int32)
            y = np.empty(len(rws), dtype=np.float32)
            users = []
            for j, x in enumerate(rws):
                for i, v in enumerate(raw_features(x)):
                    X[j, i] = vocabs[i].get(v, unk_slots[i]) + offsets[i]
                y[j] = x[5]
                users.append(x[1])
            data_enc[f"X_{name}"] = X
            data_enc[f"y_{name}"] = y
            data_enc[f"users_{name}"] = users

        # Fix train keys alias mapping for standard autotrainer dictionary
        data_enc["X_val"] = data_enc["X_valid"]
        data_enc["y_val"] = data_enc["y_valid"]
        data_enc["users_val"] = data_enc["users_valid"]

        num_features = int(sum(field_dims))
        print(f"[Data Engine] Preprocessed {n_fields} fields. Total Vocabulary Dimension: {num_features}")

        return data_enc, num_features

    def run(self):
        data_enc, num_features = self.load_dataset()

        print("\n=======================================================")
        print(f"  STARTING STAGE 1: AUTONOMOUS EXPLORATION (Max {self.max_trials} Trials)")
        print(f"  Convergence Threshold (epsilon): {self.epsilon}")
        print("=======================================================")

        best_agent_score = -float("inf")
        no_improvement_count = 0

        for trial in range(1, self.max_trials + 1):
            cfg = self.gemini.propose_next_config(self.history)
            cfg["seed"] = 42
            cfg["quick_epochs"] = 10

            hypothesis = cfg.get("hypothesis", "Exploratory hyperparameter adjustment.")
            print(f"\n--- Stage 1 Iteration {trial}/{self.max_trials}: Model={cfg.get('model_type')} | LR={cfg.get('lr')} ---")
            print(f"    Hypothesis: {hypothesis}")

            try:
                metrics = train_and_eval(cfg, data_enc, num_features)
                current_score = metrics["primary"]
                error_event = None
                print(f"--> Result: Val GAUC={metrics['val_auc']:.4f} | Val nDCG@5={metrics['val_ndcg']:.4f} | Primary Score={current_score:.4f}")
            except Exception as e:
                print(f"[Trial {trial} Error] Exception encountered: {e}. Handled gracefully.")
                metrics = {"val_auc": 0.5, "val_ndcg": 0.0, "primary": 0.25}
                current_score = 0.25
                error_event = str(e)

            # Record per-iteration run log requirement
            self.history.append({
                "iteration": trial,
                "hypothesis": hypothesis,
                "code_diff": f"Updated config parameters: model={cfg.get('model_type')}, lr={cfg.get('lr')}, embed_dim={cfg.get('embed_dim')}",
                "config": cfg,
                "metrics": metrics,
                "error_recovery": error_event,
                "manual_interventions": 0
            })

            if current_score > (best_agent_score + self.epsilon):
                print(f"[Convergence Tracker] Score improved: {best_agent_score:.4f} -> {current_score:.4f}")
                best_agent_score = current_score
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                print(f"[Convergence Tracker] Insufficient improvement (< {self.epsilon}). Stagnant trials: {no_improvement_count}/3")

            if no_improvement_count >= 3:
                print(f"\n[Agent Converged] Primary score stagnant for 3 consecutive trials. Stopping Stage 1.")
                break

        save_json(self.history, os.path.join(self.output_dir, "exploration_history.json"))

        sorted_history = sorted(self.history, key=lambda x: x["metrics"]["primary"], reverse=True)
        top_configs = [item["config"] for item in sorted_history[:2]]

        ensemble_results = run_full_ensemble_and_blend(
            best_configs=top_configs,
            data_enc=data_enc,
            num_features=num_features,
            num_seeds=3
        )

        # Write official submission file matching submit.py schema: row_id,user_id,video_id,score
        if "test_preds" in ensemble_results:
            sub_path = os.path.join(self.output_dir, "submission.csv")
            with open(sub_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["row_id", "user_id", "video_id", "score"])
                for i, (row_data, score) in enumerate(zip(self.test_rows, ensemble_results["test_preds"])):
                    # row_data tuple: (date, user_id, video_id, tab, duration, label)
                    writer.writerow([i, row_data[1], row_data[2], f"{float(score):.6g}"])
            print(f"[Submission] Saved predictions matching official schema to '{sub_path}'")

        save_json({
            "stage1_best_score": best_agent_score,
            "stage2_ensemble_metrics": {
                "val_auc": ensemble_results["val_auc"],
                "val_ndcg": ensemble_results["val_ndcg"],
                "primary": ensemble_results["primary"]
            },
            "manual_interventions": 0
        }, os.path.join(self.output_dir, "final_results.json"))

        print(f"\n[Pipeline Complete] Artifacts generated in '{self.output_dir}/'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KuaiRand Autonomous Agent Pipeline")
    parser.add_argument("--data-dir", type=str, default="./KuaiRand-Pure/data", help="Path to data directory")
    parser.add_argument("--output-dir", type=str, default="artifacts_pipeline", help="Output directory")
    parser.add_argument("--max-trials", type=int, default=10, help="Maximum Stage 1 trials")
    parser.add_argument("--epsilon", type=float, default=0.002, help="Convergence threshold")

    args = parser.parse_args()

    agent = AutonomousResearchAgent(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_trials=args.max_trials,
        epsilon=args.epsilon
    )
    agent.run()