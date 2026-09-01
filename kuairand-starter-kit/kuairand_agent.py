import os
import sys
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
Your goal is to maximize the primary score = AUC - 0.5 * LogLoss.

Past Experiment History (JSON):
{json.dumps(history, indent=2)}

Propose the next set of hyperparameter values as a valid JSON object ONLY.
Required JSON keys:
- "model_type": string ("senet" or "lowrank_dcn")
- "lr": float (between 0.00005 and 0.001)
- "embed_dim": int (16 or 32)
- "hidden_dims": list of ints (e.g. [128, 64] or [256, 128])
- "dropout": float (between 0.05 and 0.3)
- "batch_size": int (4096 or 8192)
"""
        try:
            response = self.client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    response_mime_type="application/json"
                )
            )
            return json.loads(response.text.strip())
        except Exception as e:
            print(f"[Gemini API Call Failed] Error: {e}. Using fallback config.")
            cfg = self.static_fallbacks[self.fallback_idx % len(self.static_fallbacks)]
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
        self.raw_test_df = None
        os.makedirs(output_dir, exist_ok=True)

    def load_dataset(self):
        print(f"[Data Engine] Loading KuaiRand dataset from {self.data_dir}...")
        
        npz_path = os.path.join(self.data_dir, "encoded_data.npz")
        if os.path.exists(npz_path):
            data = np.load(npz_path)
            data_enc = {
                "X_train": data["X_train"], "y_train": data["y_train"],
                "X_val": data["X_val"], "y_val": data["y_val"],
                "X_test": data["X_test"], "y_test": data["y_test"]
            }
            num_features = int(data.get("num_features", np.max(data["X_train"]) + 1))
            return data_enc, num_features

        raw_train = os.path.join(self.data_dir, "log_standard_4_08_to_4_21_pure.csv")
        raw_val = os.path.join(self.data_dir, "log_standard_4_22_to_5_08_pure.csv")
        raw_test = os.path.join(self.data_dir, "log_random_4_22_to_5_08_pure.csv")

        if os.path.exists(raw_train) and os.path.exists(raw_val) and os.path.exists(raw_test):
            import pandas as pd
            from sklearn.preprocessing import OrdinalEncoder

            print("[Data Engine] Parsing interaction logs & feature tables (including statistical features)...")

            df_train = pd.read_csv(raw_train)
            df_val = pd.read_csv(raw_val)
            df_test = pd.read_csv(raw_test)
            self.raw_test_df = df_test.copy()

            target_col = "is_click" if "is_click" in df_train.columns else ("click" if "click" in df_train.columns else df_train.columns[-1])

            user_feat_path = os.path.join(self.data_dir, "user_features_pure.csv")
            video_basic_path = os.path.join(self.data_dir, "video_features_basic_pure.csv")
            video_stat_path = os.path.join(self.data_dir, "video_features_statistic_pure.csv")

            feature_cols = [c for c in ["user_id", "video_id"] if c in df_train.columns]

            if os.path.exists(user_feat_path):
                df_user = pd.read_csv(user_feat_path)
                u_cols = [c for c in df_user.columns if c != "user_id"]
                df_train = df_train.merge(df_user, on="user_id", how="left")
                df_val = df_val.merge(df_user, on="user_id", how="left")
                df_test = df_test.merge(df_user, on="user_id", how="left")
                feature_cols.extend(u_cols)

            if os.path.exists(video_basic_path):
                df_video_basic = pd.read_csv(video_basic_path)
                vb_cols = [c for c in df_video_basic.columns if c != "video_id"]
                df_train = df_train.merge(df_video_basic, on="video_id", how="left")
                df_val = df_val.merge(df_video_basic, on="video_id", how="left")
                df_test = df_test.merge(df_video_basic, on="video_id", how="left")
                feature_cols.extend(vb_cols)

            if os.path.exists(video_stat_path):
                df_video_stat = pd.read_csv(video_stat_path)
                vs_cols = [c for c in df_video_stat.columns if c != "video_id"]
                df_train = df_train.merge(df_video_stat, on="video_id", how="left")
                df_val = df_val.merge(df_video_stat, on="video_id", how="left")
                df_test = df_test.merge(df_video_stat, on="video_id", how="left")
                feature_cols.extend(vs_cols)

            feature_cols = list(dict.fromkeys(feature_cols))

            print(f"[Data Engine] Preprocessing {len(feature_cols)} feature columns (Quantile Binning continuous statistics)...")
            
            combined_features = pd.concat([df_train[feature_cols], df_val[feature_cols], df_test[feature_cols]], axis=0)

            # Quantile binning for numerical/continuous columns
            for col in combined_features.select_dtypes(include=[np.number]).columns:
                if col not in ["user_id", "video_id"]:
                    combined_features[col] = pd.qcut(combined_features[col], q=10, labels=False, duplicates="drop")

            combined_str = combined_features.astype(str).fillna("-1")

            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            encoder.fit(combined_str)

            n_train = len(df_train)
            n_val = len(df_val)

            X_train = encoder.transform(combined_str.iloc[:n_train]) + 1
            X_val = encoder.transform(combined_str.iloc[n_train:n_train + n_val]) + 1
            X_test = encoder.transform(combined_str.iloc[n_train + n_val:]) + 1

            X_train[X_train < 0] = 0
            X_val[X_val < 0] = 0
            X_test[X_test < 0] = 0

            y_train = df_train[target_col].astype(np.float32).values
            y_val = df_val[target_col].astype(np.float32).values
            y_test = df_test[target_col].astype(np.float32).values

            num_features = int(max(X_train.max(), X_val.max(), X_test.max())) + 1

            data_enc = {
                "X_train": X_train.astype(np.int64), "y_train": y_train,
                "X_val": X_val.astype(np.int64), "y_val": y_val,
                "X_test": X_test.astype(np.int64), "y_test": y_test
            }

            print(f"[Data Engine] Dataset ready! Features: {len(feature_cols)} | Total Vocabulary: {num_features}")
            return data_enc, num_features

        raise FileNotFoundError(f"Could not find valid dataset files in {self.data_dir}.")

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

            print(f"\n--- Stage 1 Iteration {trial}/{self.max_trials}: Model={cfg.get('model_type')} | LR={cfg.get('lr')} ---")
            metrics = train_and_eval(cfg, data_enc, num_features)
            current_score = metrics["primary"]

            print(f"--> Result: Val AUC={metrics['val_auc']:.4f} | Val Loss={metrics['val_loss']:.4f} | Primary Score={current_score:.4f}")

            self.history.append({"trial": trial, "config": cfg, "metrics": metrics})

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

        # Save submission probabilities
        if "test_preds" in ensemble_results:
            import pandas as pd
            sub_df = pd.DataFrame()
            if self.raw_test_df is not None and "user_id" in self.raw_test_df.columns:
                sub_df["user_id"] = self.raw_test_df["user_id"]
                sub_df["video_id"] = self.raw_test_df["video_id"]
            sub_df["pred_prob"] = ensemble_results["test_preds"]
            sub_path = os.path.join(self.output_dir, "submission.csv")
            sub_df.to_csv(sub_path, index=False)
            print(f"[Submission] Saved test predictions to '{sub_path}'")

        save_json({
            "stage1_best_score": best_agent_score,
            "stage2_ensemble_metrics": {
                "val_auc": ensemble_results["val_auc"],
                "val_loss": ensemble_results["val_loss"],
                "primary": ensemble_results["primary"]
            }
        }, os.path.join(self.output_dir, "final_results.json"))

        print(f"\n[Pipeline Complete] Output files ready in '{self.output_dir}/'")


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