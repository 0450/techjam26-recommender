import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss

from utils import clear_memory

# --------------------------------------------------
# Neural Model Architecture Components
# --------------------------------------------------

class SENetLayer(nn.Module):
    """Squeeze-and-Excitation Layer for Deep Recommendation Features."""
    def __init__(self, num_fields: int, reduction_ratio: int = 3):
        super(SENetLayer, self).__init__()
        reduced_dim = max(1, num_fields // reduction_ratio)
        self.fc = nn.Sequential(
            nn.Linear(num_fields, reduced_dim, bias=False),
            nn.ReLU(),
            nn.Linear(reduced_dim, num_fields, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = torch.mean(x, dim=-1)
        w = self.fc(squeeze).unsqueeze(-1)
        return x * w


class LowRankDCNLayer(nn.Module):
    """Low-Rank Deep & Cross Network (DCN-v2) Layer."""
    def __init__(self, input_dim: int, rank: int = 16):
        super(LowRankDCNLayer, self).__init__()
        self.u = nn.Linear(input_dim, rank, bias=False)
        self.v = nn.Linear(rank, input_dim, bias=True)

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        out = self.v(self.u(x))
        return x0 * out + x


class RecommenderModel(nn.Module):
    """Unified Recommender Architecture supporting SENet and Low-Rank DCN."""
    def __init__(self, num_features: int, num_fields: int, embed_dim: int, architecture: str, 
                 hidden_dims: list, dropout: float = 0.2, dcn_rank: int = 16):
        super(RecommenderModel, self).__init__()
        self.architecture = architecture.lower()
        self.embeddings = nn.Embedding(num_features + 1, embed_dim, padding_idx=0)
        
        total_embed_dim = num_fields * embed_dim
        
        if self.architecture == "senet":
            self.senet = SENetLayer(num_fields=num_fields)
        elif self.architecture == "lowrank_dcn":
            self.dcn = LowRankDCNLayer(input_dim=total_embed_dim, rank=dcn_rank)
            
        layers = []
        in_dim = total_embed_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        embeds = self.embeddings(x_cat)  # [batch, num_fields, embed_dim]
        
        if self.architecture == "senet":
            embeds = self.senet(embeds)
            
        flat_embeds = embeds.view(embeds.size(0), -1)
        
        if self.architecture == "lowrank_dcn":
            flat_embeds = self.dcn(flat_embeds, flat_embeds)
            
        logits = self.mlp(flat_embeds)
        return self.sigmoid(logits).squeeze(-1)

# --------------------------------------------------
# Training Engine with Restored Timers
# --------------------------------------------------

def train_single_seed(cfg: dict, data_enc: dict, num_features: int, seed: int = 42, 
                      max_epochs: int = 30, patience: int = 4):
    """Trains a single model instance up to full convergence with epoch timer logging."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_fields = data_enc["X_train"].shape[1]

    train_ds = TensorDataset(torch.tensor(data_enc["X_train"], dtype=torch.long), torch.tensor(data_enc["y_train"], dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(data_enc["X_val"], dtype=torch.long), torch.tensor(data_enc["y_val"], dtype=torch.float32))
    test_ds = TensorDataset(torch.tensor(data_enc["X_test"], dtype=torch.long), torch.tensor(data_enc["y_test"], dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=cfg.get("batch_size", 4096), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8192, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=8192, shuffle=False)

    model = RecommenderModel(
        num_features=num_features,
        num_fields=num_fields,
        embed_dim=cfg.get("embed_dim", 16),
        architecture=cfg.get("model_type", "senet"),
        hidden_dims=cfg.get("hidden_dims", [128, 64]),
        dropout=cfg.get("dropout", 0.2)
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.get("lr", 0.001), weight_decay=cfg.get("weight_decay", 1e-4))

    best_val_primary = -float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            preds = model(x_b)
            loss = criterion(preds, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y_b)

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b = x_b.to(device)
                preds = model(x_b)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(y_b.numpy())

        elapsed = time.time() - t0
        val_loss = log_loss(val_targets, val_preds)
        val_auc = roc_auc_score(val_targets, val_preds)
        val_primary = val_auc - (0.5 * val_loss)
        train_loss_avg = train_loss / len(train_ds)

        is_best = val_primary > best_val_primary
        best_tag = " [BEST]" if is_best else ""

        print(f"  Epoch {epoch:02d}/{max_epochs:02d} | Loss: {train_loss_avg:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Val Primary: {val_primary:.4f} | Time: {elapsed:.1f}s{best_tag}")

        if is_best:
            best_val_primary = val_primary
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  --> Early stopping triggered at epoch {epoch}.")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)

    model.eval()
    val_preds, test_preds = [], []
    with torch.no_grad():
        for x_b, _ in val_loader:
            val_preds.extend(model(x_b.to(device)).cpu().numpy())
        for x_b, _ in test_loader:
            test_preds.extend(model(x_b.to(device)).cpu().numpy())

    clear_memory()
    return model, np.array(val_preds), np.array(test_preds)


def train_and_eval(cfg: dict, data_enc: dict, num_features: int) -> dict:
    """Stage 1 trial evaluation helper running quick epochs for feature signal."""
    quick_epochs = cfg.get("quick_epochs", 10)
    _, val_preds, _ = train_single_seed(
        cfg=cfg,
        data_enc=data_enc,
        num_features=num_features,
        seed=42,
        max_epochs=quick_epochs,
        patience=3
    )
    val_targets = data_enc["y_val"]
    auc = roc_auc_score(val_targets, val_preds)
    loss = log_loss(val_targets, val_preds)
    primary = auc - (0.5 * loss)

    return {
        "val_auc": float(auc),
        "val_loss": float(loss),
        "primary": float(primary)
    }


def run_full_ensemble_and_blend(best_configs: list, data_enc: dict, num_features: int, num_seeds: int = 3) -> dict:
    """Stage 2: Runs multi-seed full training across top configurations and blends predictions."""
    print("\n=======================================================")
    print(f"  STARTING STAGE 2: FULL ENSEMBLE ({len(best_configs)} Configs x {num_seeds} Seeds)")
    print("=======================================================")

    val_preds_all = []
    test_preds_all = []
    seeds = [42, 1024, 2026][:num_seeds]

    for c_idx, cfg in enumerate(best_configs, 1):
        for s_idx, seed in enumerate(seeds, 1):
            print(f"\n--- Training Model Config {c_idx}/{len(best_configs)} ({cfg['model_type']}) | Seed {s_idx}/{num_seeds} (ID: {seed}) ---")
            _, v_p, t_p = train_single_seed(
                cfg=cfg,
                data_enc=data_enc,
                num_features=num_features,
                seed=seed,
                max_epochs=30,
                patience=4
            )
            val_preds_all.append(v_p)
            test_preds_all.append(t_p)

    final_val_preds = np.mean(val_preds_all, axis=0)
    final_test_preds = np.mean(test_preds_all, axis=0)

    val_targets = data_enc["y_val"]
    ensemble_auc = roc_auc_score(val_targets, final_val_preds)
    ensemble_loss = log_loss(val_targets, final_val_preds)
    ensemble_primary = ensemble_auc - (0.5 * ensemble_loss)

    print("\n=======================================================")
    print(f"  STAGE 2 FINAL ENSEMBLE RESULTS")
    print(f"  Ensemble Val AUC   : {ensemble_auc:.4f}")
    print(f"  Ensemble Val Loss  : {ensemble_loss:.4f}")
    print(f"  Ensemble Primary   : {ensemble_primary:.4f}")
    print("=======================================================")

    return {
        "val_auc": float(ensemble_auc),
        "val_loss": float(ensemble_loss),
        "primary": float(ensemble_primary),
        "val_preds": final_val_preds,
        "test_preds": final_test_preds
    }