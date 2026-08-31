"""
utils.py
--------
Helper components, heterogeneous model definitions (SENet DeepFM & Low-Rank DCNv2),
multi-task loss calculation, power-rank normalization, and blending utilities.
"""

import os
import json
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from data import load, encode
from evaluate import evaluate


def mkdir(path):
    os.makedirs(path, exist_ok=True)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, cls=NpEncoder)

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def power_rank_normalize(scores: np.ndarray, power: float = 1.0) -> np.ndarray:
    ranks = rankdata(scores) / len(scores)
    return np.power(ranks, power)


def safety_checks(scores):
    if len(scores) == 0:
        raise ValueError("Prediction array is empty.")
    if np.isnan(scores).any():
        raise ValueError("Predictions contain NaN values.")
    if np.std(scores) < 1e-6:
        raise ValueError("Prediction variance is too low (constant output).")


# ==========================================
# Neural Network Layer Modules
# ==========================================
class SENetLayer(nn.Module):
    def __init__(self, num_fields, reduction=2):
        super().__init__()
        reduced_dim = max(1, num_fields // reduction)
        self.fc = nn.Sequential(
            nn.Linear(num_fields, reduced_dim, bias=False),
            nn.ReLU(),
            nn.Linear(reduced_dim, num_fields, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        weights = x.mean(dim=-1)
        weights = self.fc(weights).unsqueeze(-1)
        return x * weights


class LowRankDCNv2Layer(nn.Module):
    def __init__(self, in_features, rank=32, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.U = nn.ParameterList([nn.Parameter(torch.randn(in_features, rank) * 0.01) for _ in range(num_layers)])
        self.V = nn.ParameterList([nn.Parameter(torch.randn(rank, in_features) * 0.01) for _ in range(num_layers)])
        self.b = nn.ParameterList([nn.Parameter(torch.zeros(in_features)) for _ in range(num_layers)])

    def forward(self, x0):
        xl = x0
        for i in range(self.num_layers):
            vt_x = torch.matmul(xl, self.V[i].T)
            gate = torch.matmul(vt_x, self.U[i].T) + self.b[i]
            xl = x0 * gate + xl
        return xl


# ==========================================
# Heterogeneous Model Architectures
# ==========================================
class MultiTaskResidualSENetDeepFM(nn.Module):
    def __init__(self, num_features, num_fields=5, embedding_dim=48, dropout_rate=0.2):
        super().__init__()
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        
        self.embedding = nn.Embedding(num_features, embedding_dim)
        self.linear = nn.Embedding(num_features, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        
        self.senet = SENetLayer(num_fields)
        in_dim = num_fields * embedding_dim
        
        self.fc1 = nn.Linear(in_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        self.drop = nn.Dropout(dropout_rate)
        
        self.head_long_view = nn.Linear(128, 1)
        self.head_click = nn.Linear(128, 1)
        self.head_watch_ratio = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        linear_part = self.linear(x).sum(dim=1).squeeze(-1) + self.bias
        emb = self.embedding(x)
        emb_se = self.senet(emb)
        
        sum_emb = emb.sum(dim=1)
        sum_sq_emb = (emb ** 2).sum(dim=1)
        sq_sum_emb = sum_emb ** 2
        second_order = 0.5 * (sq_sum_emb - sum_sq_emb).sum(dim=-1)
        
        flat = emb_se.view(emb_se.size(0), -1)
        h1 = F.relu(self.ln1(self.fc1(flat)))
        h1 = self.drop(h1)
        h2 = F.relu(self.ln2(self.fc2(h1)))
        h2 = self.drop(h2)
        
        logits_long = linear_part + second_order + self.head_long_view(h2).squeeze(-1)
        logits_click = self.head_click(h2).squeeze(-1)
        pred_watch_ratio = self.head_watch_ratio(h2).squeeze(-1)
        
        return logits_long, logits_click, pred_watch_ratio


class MultiTaskLowRankDCNv2DeepFM(nn.Module):
    def __init__(self, num_features, num_fields=5, embedding_dim=64, rank=32, dropout_rate=0.2, emb_dropout=0.10):
        super().__init__()
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        
        self.embedding = nn.Embedding(num_features, embedding_dim)
        self.emb_drop = nn.Dropout(emb_dropout)
        self.linear = nn.Embedding(num_features, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        
        self.senet = SENetLayer(num_fields)
        in_dim = num_fields * embedding_dim
        
        self.cross_net = LowRankDCNv2Layer(in_features=in_dim, rank=rank, num_layers=2)
        
        self.fc1 = nn.Linear(in_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        self.drop = nn.Dropout(dropout_rate)
        
        combined_dim = in_dim + 128
        self.head_long_view = nn.Linear(combined_dim, 1)
        self.head_click = nn.Linear(combined_dim, 1)
        self.head_watch_ratio = nn.Sequential(
            nn.Linear(combined_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        linear_part = self.linear(x).sum(dim=1).squeeze(-1) + self.bias
        emb = self.embedding(x)
        emb_dropped = self.emb_drop(emb)
        emb_se = self.senet(emb_dropped)
        
        sum_emb = emb_dropped.sum(dim=1)
        sum_sq_emb = (emb_dropped ** 2).sum(dim=1)
        sq_sum_emb = sum_emb ** 2
        second_order = 0.5 * (sq_sum_emb - sum_sq_emb).sum(dim=-1)
        
        flat = emb_se.view(emb_se.size(0), -1)
        cross_out = self.cross_net(flat)
        
        h1 = F.relu(self.ln1(self.fc1(flat)))
        h1 = self.drop(h1)
        h2 = F.relu(self.ln2(self.fc2(h1)))
        h2 = self.drop(h2)
        
        h_combined = torch.cat([cross_out, h2], dim=-1)
        
        logits_long = linear_part + second_order + self.head_long_view(h_combined).squeeze(-1)
        logits_click = self.head_click(h_combined).squeeze(-1)
        pred_watch_ratio = self.head_watch_ratio(h_combined).squeeze(-1)
        
        return logits_long, logits_click, pred_watch_ratio


# ==========================================
# Data & Training Support Functions
# ==========================================
def compute_mtl_loss(logits_long, logits_click, pred_watch, y_long, y_click, y_watch, alpha=0.2, beta=0.3):
    loss_long = F.binary_cross_entropy_with_logits(logits_long, y_long)
    loss_click = F.binary_cross_entropy_with_logits(logits_click, y_click)
    loss_watch = F.mse_loss(pred_watch, y_watch)
    return loss_long + (alpha * loss_click) + (beta * loss_watch)


def load_auxiliary_targets(data_dir, splits):
    aux_targets = {}
    for split_name in ['train', 'valid', 'test']:
        csv_path = os.path.join(data_dir, f"{split_name}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, usecols=lambda c: c in ['is_click', 'play_time_ms', 'duration_ms'])
            click = df['is_click'].values.astype(np.float32) if 'is_click' in df.columns else np.zeros(len(df), dtype=np.float32)
            if 'play_time_ms' in df.columns and 'duration_ms' in df.columns:
                dur = np.maximum(df['duration_ms'].values, 1.0)
                watch_ratio = np.clip(df['play_time_ms'].values / dur, 0.0, 1.0).astype(np.float32)
            else:
                watch_ratio = np.zeros(len(df), dtype=np.float32)
            aux_targets[split_name] = (click, watch_ratio)
        else:
            n_samples = len(splits[split_name])
            aux_targets[split_name] = (np.zeros(n_samples, dtype=np.float32), np.zeros(n_samples, dtype=np.float32))
            
    return aux_targets


def train_architecture(model_type, seeds, enc, aux_targets, num_features, device, epochs=12, batch_size=8192, patience=3, lr=4e-4):
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']
    Xte, yte, ute = enc['test']
    click_tr, watch_tr = aux_targets['train']
    num_fields = Xtr.shape[1]

    train_dataset = TensorDataset(
        torch.tensor(Xtr, dtype=torch.long),
        torch.tensor(ytr, dtype=torch.float32),
        torch.tensor(click_tr, dtype=torch.float32),
        torch.tensor(watch_tr, dtype=torch.float32)
    )

    Xva_tensor = torch.tensor(Xva, dtype=torch.long).to(device)
    Xte_tensor = torch.tensor(Xte, dtype=torch.long).to(device)

    val_seed_preds = []
    test_seed_preds = []

    print(f"\n==================================================")
    print(f"=== TRAINING ARCHITECTURE: {model_type.upper()} ===")
    print(f"==================================================")

    for idx, seed in enumerate(seeds, 1):
        seed_everything(seed)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if model_type == 'senet':
            model = MultiTaskResidualSENetDeepFM(num_features, num_fields, embedding_dim=48, dropout_rate=0.2).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
            loss_alpha, loss_beta = 0.2, 0.3
        else:
            model = MultiTaskLowRankDCNv2DeepFM(num_features, num_fields, embedding_dim=64, rank=32, dropout_rate=0.2, emb_dropout=0.10).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-3)
            loss_alpha, loss_beta = 0.1, 0.1

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_primary = -1.0
        best_val = None
        best_test = None
        patience_counter = 0

        print(f"\n--- Model: {model_type} | Seed {idx}/{len(seeds)} (Seed ID: {seed}) ---")
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            model.train()
            running_loss = 0.0

            for bx, by_long, by_click, by_watch in train_loader:
                bx, by_long = bx.to(device), by_long.to(device)
                by_click, by_watch = by_click.to(device), by_watch.to(device)

                optimizer.zero_grad()
                logits_long, logits_click, pred_watch = model(bx)
                loss = compute_mtl_loss(logits_long, logits_click, pred_watch, by_long, by_click, by_watch, alpha=loss_alpha, beta=loss_beta)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            scheduler.step()
            epoch_loss = running_loss / len(train_loader)

            model.eval()
            with torch.no_grad():
                val_logits, _, _ = model(Xva_tensor)
                val_preds = torch.sigmoid(val_logits).cpu().numpy()

                test_logits, _, _ = model(Xte_tensor)
                test_preds = torch.sigmoid(test_logits).cpu().numpy()

            val_res = evaluate(uva, yva, val_preds)
            dt = time.time() - t0

            if val_res['primary'] > best_val_primary:
                best_val_primary = val_res['primary']
                best_val = val_preds.copy()
                best_test = test_preds.copy()
                patience_counter = 0
                flag = "[BEST]"
            else:
                patience_counter += 1
                flag = f"[Patience {patience_counter}/{patience}]"

            print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {epoch_loss:.4f} | Val GAUC: {val_res['GAUC']:.4f} | Val Primary: {val_res['primary']:.4f} | Time: {dt:.1f}s {flag}")

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch:02d}.")
                break

        val_seed_preds.append(best_val)
        test_seed_preds.append(best_test)

    model_val_ensemble = np.mean([power_rank_normalize(v, 2.0) for v in val_seed_preds], axis=0)
    model_test_ensemble = np.mean([power_rank_normalize(t, 2.0) for t in test_seed_preds], axis=0)

    val_res = evaluate(uva, yva, model_val_ensemble)
    print(f"\n--> {model_type.upper()} {len(seeds)}-Seed Ensemble finished | Val Primary: {val_res['primary']:.4f} | Val GAUC: {val_res['GAUC']:.4f}")

    return model_val_ensemble, model_test_ensemble, val_res


def optimize_blend(val_senet, val_dcn, test_senet, test_dcn, uva, yva):
    """Grid searches for the optimal power-rank exponent and blend weights on validation data."""
    best_primary = -1.0
    best_w1, best_w2 = 0.5, 0.5
    best_power = 2.0
    best_val_res = None

    power_candidates = [1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5]
    w1_candidates = np.linspace(0.1, 0.9, 17)

    for p in power_candidates:
        r_senet = power_rank_normalize(val_senet, power=p)
        r_dcn = power_rank_normalize(val_dcn, power=p)

        for w1 in w1_candidates:
            w2 = 1.0 - w1
            blended_val = w1 * r_senet + w2 * r_dcn
            res = evaluate(uva, yva, blended_val)

            if res['primary'] > best_primary:
                best_primary = res['primary']
                best_w1, best_w2 = w1, w2
                best_power = p
                best_val_res = res

    r_senet_test = power_rank_normalize(test_senet, power=best_power)
    r_dcn_test = power_rank_normalize(test_dcn, power=best_power)
    blended_test_preds = best_w1 * r_senet_test + best_w2 * r_dcn_test

    return blended_test_preds, best_val_res, best_w1, best_w2, best_power


def grid_proposals():
    return [
        {"model_type": "senet", "lr": 4e-4},
        {"model_type": "senet", "lr": 2e-4},
        {"model_type": "lowrank_dcn", "lr": 2e-4},
        {"model_type": "lowrank_dcn", "lr": 1e-4}
    ]