"""
train_pytorch_hybrid.py
------------------------
Multi-Task Low-Rank DCNv2 SENet DeepFM Model.
Updates:
  - Added Embedding Dropout (0.10) to prevent early sparse ID memorization.
  - Adjusted LR to 2.0e-4 & Weight Decay to 2.0e-3 for smoother convergence.
  - Streamlined Deep MLP layers (256 -> 128) matching Low-Rank DCN capacity.
  - Low-Rank DCNv2 Cross Network (rank=32).

Usage:
    python3 train_pytorch_hybrid.py
"""

import os
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
from submit import write_submission, read_submission


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SENetLayer(nn.Module):
    """Squeeze-and-Excitation field-level feature re-weighting"""
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
    """Low-Rank Matrix DCNv2 Layer: x_{l+1} = x_0 * (U @ (V^T @ x_l) + b) + x_l"""
    def __init__(self, in_features, rank=32, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.U = nn.ParameterList([
            nn.Parameter(torch.randn(in_features, rank) * 0.01) for _ in range(num_layers)
        ])
        self.V = nn.ParameterList([
            nn.Parameter(torch.randn(rank, in_features) * 0.01) for _ in range(num_layers)
        ])
        self.b = nn.ParameterList([
            nn.Parameter(torch.zeros(in_features)) for _ in range(num_layers)
        ])

    def forward(self, x0):
        xl = x0
        for i in range(self.num_layers):
            vt_x = torch.matmul(xl, self.V[i].T)
            gate = torch.matmul(vt_x, self.U[i].T) + self.b[i]
            xl = x0 * gate + xl
        return xl


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
        
        # Fast Low-Rank DCNv2 Cross Network
        self.cross_net = LowRankDCNv2Layer(in_features=in_dim, rank=rank, num_layers=2)
        
        # Streamlined Deep MLP
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
        
        # 2nd Order FM interaction
        sum_emb = emb_dropped.sum(dim=1)
        sum_sq_emb = (emb_dropped ** 2).sum(dim=1)
        sq_sum_emb = sum_emb ** 2
        second_order = 0.5 * (sq_sum_emb - sum_sq_emb).sum(dim=-1)
        
        flat = emb_se.view(emb_se.size(0), -1)
        
        # Explicit Low-Rank Cross Features
        cross_out = self.cross_net(flat)
        
        # Implicit Deep MLP Features
        h1 = F.relu(self.ln1(self.fc1(flat)))
        h1 = self.drop(h1)
        h2 = F.relu(self.ln2(self.fc2(h1)))
        h2 = self.drop(h2)
        
        h_combined = torch.cat([cross_out, h2], dim=-1)
        
        logits_long = linear_part + second_order + self.head_long_view(h_combined).squeeze(-1)
        logits_click = self.head_click(h_combined).squeeze(-1)
        pred_watch_ratio = self.head_watch_ratio(h_combined).squeeze(-1)
        
        return logits_long, logits_click, pred_watch_ratio


def compute_mtl_loss(logits_long, logits_click, pred_watch, y_long, y_click, y_watch, alpha=0.10, beta=0.10):
    loss_long = F.binary_cross_entropy_with_logits(logits_long, y_long)
    loss_click = F.binary_cross_entropy_with_logits(logits_click, y_click)
    loss_watch = F.mse_loss(pred_watch, y_watch)
    return loss_long + (alpha * loss_click) + (beta * loss_watch)


def load_auxiliary_targets(splits):
    aux_targets = {}
    data_dir = './KuaiRand-Pure/data'
    
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


def power_rank_normalize(scores, power=1.2):
    ranks = rankdata(scores) / len(scores)
    return np.power(ranks, power)


def train_single_seed(seed, enc_data, aux_targets, num_features, device, epochs=12, batch_size=8192, patience=3):
    seed_everything(seed)
    
    Xtr, ytr, utr = enc_data['train']
    Xva, yva, uva = enc_data['valid']
    Xte, yte, ute = enc_data['test']
    
    click_tr, watch_tr = aux_targets['train']
    num_fields = Xtr.shape[1]
    
    train_dataset = TensorDataset(
        torch.tensor(Xtr, dtype=torch.long),
        torch.tensor(ytr, dtype=torch.float32),
        torch.tensor(click_tr, dtype=torch.float32),
        torch.tensor(watch_tr, dtype=torch.float32)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = MultiTaskLowRankDCNv2DeepFM(
        num_features=num_features,
        num_fields=num_fields,
        embedding_dim=64,
        rank=32,
        dropout_rate=0.2,
        emb_dropout=0.10
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-4, weight_decay=2.0e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_primary = -1.0
    best_val_preds = None
    best_test_preds = None
    best_epoch = 0
    patience_counter = 0
    
    Xva_tensor = torch.tensor(Xva, dtype=torch.long).to(device)
    Xte_tensor = torch.tensor(Xte, dtype=torch.long).to(device)
    
    print(f"\n--- Training Seed {seed} ---")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        
        for bx, by_long, by_click, by_watch in train_loader:
            bx, by_long = bx.to(device), by_long.to(device)
            by_click, by_watch = by_click.to(device), by_watch.to(device)
            
            optimizer.zero_grad()
            logits_long, logits_click, pred_watch = model(bx)
            loss = compute_mtl_loss(logits_long, logits_click, pred_watch, by_long, by_click, by_watch)
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
            best_val_preds = val_preds.copy()
            best_test_preds = test_preds.copy()
            best_epoch = epoch
            patience_counter = 0
            flag = "[BEST]"
        else:
            patience_counter += 1
            flag = f"[Patience {patience_counter}/{patience}]"
            
        print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {epoch_loss:.4f} | Val GAUC: {val_res['GAUC']:.4f} | Val Primary: {val_res['primary']:.4f} | Time: {dt:.1f}s {flag}")
        
        if patience_counter >= patience:
            print(f"Early stopping triggered for seed {seed} at epoch {epoch:02d}.")
            break

    print(f"--> Using Best Single Checkpoint for Seed {seed}: Epoch {best_epoch:02d} (Val Primary: {best_val_primary:.4f})")
    return best_val_preds, best_test_preds, best_val_primary


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using compute device: {device}")
    if device.type == 'cpu':
        print("Warning: Running on CPU. Batch size increased to 8192 for speed.")

    data_dir = './KuaiRand-Pure/data'
    splits = load(data_dir)
    enc, num_features = encode(splits)
    
    aux_targets = load_auxiliary_targets(splits)
    
    seeds = [42, 1024, 2026, 7, 999]
    raw_val_seed_preds = []
    raw_test_seed_preds = []
    
    for idx, seed in enumerate(seeds, 1):
        print(f"\n==================== SEED {idx}/5 (Seed ID: {seed}) ====================")
        val_preds, test_preds, best_val_score = train_single_seed(
            seed, enc, aux_targets, num_features, device, epochs=12, patience=3
        )
        
        test_res = evaluate(enc['test'][2], enc['test'][1], test_preds)
        print(f"--> Seed {seed} Finished | Best Val Primary: {best_val_score:.4f} | Test Primary: {test_res['primary']:.4f}")
        
        raw_val_seed_preds.append(val_preds)
        raw_test_seed_preds.append(test_preds)

    print("\n--- Searching Optimal Power-Rank Exponent on Validation Ensemble ---")
    best_power = 2.0
    best_val_primary = -1.0
    ensemble_val_preds = None

    power_candidates = [1.0, 1.2, 1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0, 3.5, 4.0, 5.0]
    for p in power_candidates:
        val_ranked_list = [power_rank_normalize(v, power=p) for v in raw_val_seed_preds]
        val_ensemble_candidate = np.mean(val_ranked_list, axis=0)
        res = evaluate(enc['valid'][2], enc['valid'][1], val_ensemble_candidate)
        print(f"Power Exponent {p:.1f} -> Val Primary: {res['primary']:.4f} (GAUC: {res['GAUC']:.4f}, nDCG@5: {res['nDCG@5']:.4f})")
        
        if res['primary'] > best_val_primary:
            best_val_primary = res['primary']
            best_power = p
            ensemble_val_preds = val_ensemble_candidate

    print(f"\nOptimal Power Exponent selected: {best_power:.1f}")
    
    test_ranked_list = [power_rank_normalize(t, power=best_power) for t in raw_test_seed_preds]
    ensemble_test_preds = np.mean(test_ranked_list, axis=0)

    val_final_res = evaluate(enc['valid'][2], enc['valid'][1], ensemble_val_preds)
    test_final_res = evaluate(enc['test'][2], enc['test'][1], ensemble_test_preds)
    
    print("\n" + "="*60)
    print("=== FINAL 5-SEED POWER-RANKED ENSEMBLE RESULTS ===")
    print("="*60)
    print(f"Val GAUC     : {val_final_res['GAUC']:.4f} | Val nDCG@5  : {val_final_res['nDCG@5']:.4f} | Val Primary : {val_final_res['primary']:.4f}")
    print(f"Test GAUC    : {test_final_res['GAUC']:.4f} | Test nDCG@5 : {test_final_res['nDCG@5']:.4f} | Test Primary: {test_final_res['primary']:.4f}")
    print("="*60)
    
    np.save('pytorch_val_preds.npy', ensemble_val_preds)
    np.save('pytorch_test_preds.npy', ensemble_test_preds)
    
    write_submission('submission_hybrid.csv', splits['test'], ensemble_test_preds)
    read_submission('submission_hybrid.csv', splits['test'])
    print("\nSaved predictions to submission_hybrid.csv, pytorch_val_preds.npy, and pytorch_test_preds.npy!")


if __name__ == '__main__':
    main()