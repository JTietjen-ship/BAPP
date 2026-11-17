# Baseline model (seq -> ESM2 -> NN -> predict DMS score)
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import spearmanr
import os
from datetime import datetime


####################################
####### 0: Experiment Config #######
####################################

RESULTS_CSV = "experiment_results.csv"
HISTORY_CSV = "epoch_history.csv"

# Experiment hyperparameters (logged to CSV)
# options: random, position, amino_acid_type
split_type = "position"
hidden_dims = [128, 64, 32]
dropout_rate = 0.2
lr = 1e-3
weight_decay = 1e-2
batch_size = 32
epochs = 100
seed = 42

# Experiment identifiers
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
arch_str = "-".join(map(str, hidden_dims))
config_label = f"{split_type} | arch={arch_str} | lr={lr} | do={dropout_rate}"


###############################
####### 1: Data Imports #######
###############################

data_path = "data/p53_esm2_t6_8M_embeddings.csv"
df = pd.read_csv(data_path)
embed_cols = [col for col in df.columns if col.startswith("emb_")]
X = df[embed_cols].to_numpy(dtype=np.float32)
y_scores = df["DMS_score"].to_numpy(dtype=np.float32)


##############################
####### 2: Data Splits #######
##############################

# Device setup for torch
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Running on device: {device}")

if split_type == "random":
    # Random Splits
    torch.manual_seed(42)
    indices = torch.randperm(len(X))
    split = int(0.8 * len(X))  # 80% train, 20% validation

    train_idx, val_idx = indices[:split], indices[split:]

    X_train, y_train = torch.tensor(X[train_idx]), torch.tensor(y_scores[train_idx]).unsqueeze(1)
    X_val, y_val = torch.tensor(X[val_idx]), torch.tensor(y_scores[val_idx]).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

elif split_type == "position":
    # Position-based splits
    df["position"] = df["mutant"].str.extract(r"^[A-Z](\d+)[A-Z]$").astype(int)
    torch.manual_seed(42)

    positions = df["position"].unique()
    positions = torch.tensor(positions)
    positions = positions[torch.randperm(len(positions))]

    split = int(0.8 * len(positions))

    train_positions = positions[:split].numpy()
    val_positions = positions[split:].numpy()

    # Create row-level masks
    train_mask = df["position"].isin(train_positions)
    val_mask = df["position"].isin(val_positions)

    # Extract embeddings and scores
    X_train = torch.tensor(
        df.loc[train_mask, embed_cols].to_numpy(dtype=np.float32)
    )
    y_train = torch.tensor(
        df.loc[train_mask, "DMS_score"].to_numpy(dtype=np.float32)
    ).unsqueeze(1)

    X_val = torch.tensor(
        df.loc[val_mask, embed_cols].to_numpy(dtype=np.float32)
    )
    y_val = torch.tensor(
        df.loc[val_mask, "DMS_score"].to_numpy(dtype=np.float32)
    ).unsqueeze(1)

    # DataLoader
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=32,
        shuffle=True
    )

elif split_type == "amino_acid_type":
    raise ValueError("Not Yet Implemented")


##############################
####### 3: Define MLP #######
##############################

class DMSPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float = 0.2):
        super().__init__()
        layers = [nn.LayerNorm(input_dim)]

        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_dim = h_dim
            
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

regressor = DMSPredictor(
    input_dim=X.shape[1], 
    hidden_dims=hidden_dims, 
    dropout=dropout_rate
).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(regressor.parameters(), lr=lr, weight_decay=weight_decay)


################################
####### 4: Training/Eval #######
################################

print(f"\nTraining DMS regressor | Split: {split_type} | Arch: {hidden_dims}...")

best_val_mse = float("inf")
best_val_spearman = -1.0
best_epoch = 0
epoch_records = []

for epoch in range(1, epochs + 1):
    regressor.train()
    train_loss = 0.0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        preds = regressor(batch_X)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * len(batch_X)

    train_loss /= len(X_train)

    # Evaluate every 5 epochs
    if epoch % 5 == 0 or epoch == 1:
        regressor.eval()
        with torch.no_grad():
            val_preds = regressor(X_val.to(device)).cpu().numpy().flatten()
            val_targets = y_val.numpy().flatten()
            val_mse = np.mean((val_preds - val_targets) ** 2)
            rho, _ = spearmanr(val_targets, val_preds)
            rho = float(rho)

        if rho > best_val_spearman:
                    best_val_spearman = rho
                    best_val_mse = val_mse
                    best_epoch = epoch

        epoch_records.append({
            "run_id": run_id,
            "config_label": config_label,
            "split_type": split_type,
            "architecture": arch_str,
            "lr": lr,
            "dropout": dropout_rate,
            "weight_decay": weight_decay,
            "epoch": epoch,
            "train_mse": round(train_loss, 4),
            "val_mse": round(val_mse, 4),
            "val_spearman": round(rho, 4),
        })

        print(f"Epoch {epoch:02d}/{epochs} | Train MSE: {train_loss:.4f} | Val MSE: {val_mse:.4f} | Val Spearman: {rho:.4f}")

################################
####### 5: Save Results ########
################################

# 1. Save summary row
run_record = pd.DataFrame([{
    "run_id": run_id,
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "dataset": os.path.basename(data_path),
    "split_type": split_type,
    "depth": len(hidden_dims),
    "architecture": arch_str,
    "dropout": dropout_rate,
    "lr": lr,
    "weight_decay": weight_decay,
    "epochs": epochs,
    "best_epoch": best_epoch,
    "best_val_spearman": round(best_val_spearman, 4),
    "best_val_mse": round(best_val_mse, 4),
    "final_train_mse": round(train_loss, 4),
    "final_val_spearman": round(rho, 4),
    "final_val_mse": round(val_mse, 4),
}])
run_record.to_csv(RESULTS_CSV, mode="a", header=not os.path.exists(RESULTS_CSV), index=False)

# 2. Save per-epoch learning curves
history_df = pd.DataFrame(epoch_records)
history_df.to_csv(HISTORY_CSV, mode="a", header=not os.path.exists(HISTORY_CSV), index=False)

print(f"\nSaved summary to {RESULTS_CSV} and epoch history to {HISTORY_CSV}")