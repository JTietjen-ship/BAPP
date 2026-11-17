# Baseline model (seq -> ESM2 -> NN -> predict DMS score)
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import spearmanr


###############################
####### 1: Data Imports #######
###############################

df = pd.read_csv("data/p53_esm2_t6_8M_embeddings.csv")
embed_cols = [col for col in df.columns if col.startswith("emb_")]
X = df[embed_cols].to_numpy(dtype=np.float32)
y_scores = df["DMS_score"].to_numpy(dtype=np.float32)


##############################
####### 2: Data Splits #######
##############################

# random, position, amino_acid_type
split_type = "random"

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

# 128 -> GELU -> 64 -> GELU -> 1
# With dropout 0.2 to prevent neuron/pathway dependency
class DMSPredictor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),      # Stabilizes small differences between mutant vectors
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            # nn.Linear(128, 128),
            # nn.GELU(),
            # nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)              # Single continuous output for DMS_score
        )

    def forward(self, x):
        return self.net(x)

regressor = DMSPredictor(input_dim=X.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(regressor.parameters(), lr=1e-3, weight_decay=1e-2)


################################
####### 4: Training/Eval #######
################################

epochs = 100
print("\nTraining DMS regressor...")

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

        print(f"Epoch {epoch:02d}/{epochs} | Train MSE: {train_loss:.4f} | Val MSE: {val_mse:.4f} | Val Spearman: {rho:.4f}")

'''



SPLITS BY 
input->128->64->1
Training DMS regressor...
Epoch 01/100 | Train MSE: 0.9743 | Val MSE: 1.0461 | Val Spearman: 0.2878
Epoch 05/100 | Train MSE: 0.9640 | Val MSE: 1.0376 | Val Spearman: 0.3146
Epoch 10/100 | Train MSE: 0.8990 | Val MSE: 0.9218 | Val Spearman: 0.3733
Epoch 15/100 | Train MSE: 0.8558 | Val MSE: 0.8627 | Val Spearman: 0.4294
Epoch 20/100 | Train MSE: 0.8201 | Val MSE: 0.8420 | Val Spearman: 0.4582
Epoch 25/100 | Train MSE: 0.8001 | Val MSE: 0.8473 | Val Spearman: 0.4789
Epoch 30/100 | Train MSE: 0.7546 | Val MSE: 0.8196 | Val Spearman: 0.4892
Epoch 35/100 | Train MSE: 0.7221 | Val MSE: 0.8071 | Val Spearman: 0.5032
Epoch 40/100 | Train MSE: 0.7104 | Val MSE: 0.8018 | Val Spearman: 0.5022
Epoch 45/100 | Train MSE: 0.6975 | Val MSE: 0.7909 | Val Spearman: 0.4900
Epoch 50/100 | Train MSE: 0.6830 | Val MSE: 0.7785 | Val Spearman: 0.5079
Epoch 55/100 | Train MSE: 0.6662 | Val MSE: 0.7576 | Val Spearman: 0.5250
Epoch 60/100 | Train MSE: 0.6536 | Val MSE: 0.8016 | Val Spearman: 0.4819
Epoch 65/100 | Train MSE: 0.6533 | Val MSE: 0.7794 | Val Spearman: 0.5188
Epoch 70/100 | Train MSE: 0.6416 | Val MSE: 0.8150 | Val Spearman: 0.4997
Epoch 75/100 | Train MSE: 0.6356 | Val MSE: 0.7686 | Val Spearman: 0.5146
Epoch 80/100 | Train MSE: 0.6393 | Val MSE: 0.7235 | Val Spearman: 0.5454
Epoch 85/100 | Train MSE: 0.6248 | Val MSE: 0.7711 | Val Spearman: 0.5033
Epoch 90/100 | Train MSE: 0.6199 | Val MSE: 0.8154 | Val Spearman: 0.5165
Epoch 95/100 | Train MSE: 0.6070 | Val MSE: 0.7437 | Val Spearman: 0.5462
Epoch 100/100 | Train MSE: 0.6033 | Val MSE: 0.7363 | Val Spearman: 0.5450

input->128->128->64->1
Training DMS regressor...
Epoch 01/100 | Train MSE: 0.9688 | Val MSE: 1.0384 | Val Spearman: 0.1894
Epoch 05/100 | Train MSE: 0.9670 | Val MSE: 1.0347 | Val Spearman: 0.3169
Epoch 10/100 | Train MSE: 0.9322 | Val MSE: 0.9627 | Val Spearman: 0.3166
Epoch 15/100 | Train MSE: 0.8637 | Val MSE: 0.9250 | Val Spearman: 0.4008
Epoch 20/100 | Train MSE: 0.7801 | Val MSE: 0.8279 | Val Spearman: 0.4551
Epoch 25/100 | Train MSE: 0.7217 | Val MSE: 0.8913 | Val Spearman: 0.4971
Epoch 30/100 | Train MSE: 0.6983 | Val MSE: 0.7683 | Val Spearman: 0.5110
Epoch 35/100 | Train MSE: 0.6597 | Val MSE: 0.7640 | Val Spearman: 0.5207
Epoch 40/100 | Train MSE: 0.6503 | Val MSE: 0.7604 | Val Spearman: 0.5088
Epoch 45/100 | Train MSE: 0.6434 | Val MSE: 0.7873 | Val Spearman: 0.4942
Epoch 50/100 | Train MSE: 0.6369 | Val MSE: 0.7457 | Val Spearman: 0.5339
Epoch 55/100 | Train MSE: 0.6266 | Val MSE: 0.7480 | Val Spearman: 0.5358
Epoch 60/100 | Train MSE: 0.6082 | Val MSE: 0.7546 | Val Spearman: 0.5203
Epoch 65/100 | Train MSE: 0.5994 | Val MSE: 0.8075 | Val Spearman: 0.5050
Epoch 70/100 | Train MSE: 0.5885 | Val MSE: 0.7917 | Val Spearman: 0.4741
Epoch 75/100 | Train MSE: 0.5980 | Val MSE: 0.7334 | Val Spearman: 0.5381
Epoch 80/100 | Train MSE: 0.5862 | Val MSE: 0.7480 | Val Spearman: 0.5257
Epoch 85/100 | Train MSE: 0.5797 | Val MSE: 0.7555 | Val Spearman: 0.5355
Epoch 90/100 | Train MSE: 0.5758 | Val MSE: 0.7911 | Val Spearman: 0.4818
Epoch 95/100 | Train MSE: 0.5688 | Val MSE: 0.7497 | Val Spearman: 0.5203
Epoch 100/100 | Train MSE: 0.5603 | Val MSE: 0.7331 | Val Spearman: 0.5287

input->128->64->32->1
Training DMS regressor...
Epoch 01/100 | Train MSE: 0.9687 | Val MSE: 1.0374 | Val Spearman: 0.2246
Epoch 05/100 | Train MSE: 0.9668 | Val MSE: 1.0374 | Val Spearman: 0.2937
Epoch 10/100 | Train MSE: 0.9381 | Val MSE: 0.9646 | Val Spearman: 0.3232
Epoch 15/100 | Train MSE: 0.8481 | Val MSE: 0.8760 | Val Spearman: 0.4337
Epoch 20/100 | Train MSE: 0.7809 | Val MSE: 0.9508 | Val Spearman: 0.3965
Epoch 25/100 | Train MSE: 0.7442 | Val MSE: 0.8341 | Val Spearman: 0.4848
Epoch 30/100 | Train MSE: 0.7107 | Val MSE: 0.8138 | Val Spearman: 0.4905
Epoch 35/100 | Train MSE: 0.6769 | Val MSE: 0.8390 | Val Spearman: 0.4385
Epoch 40/100 | Train MSE: 0.6552 | Val MSE: 0.7624 | Val Spearman: 0.5167
Epoch 45/100 | Train MSE: 0.6576 | Val MSE: 0.8012 | Val Spearman: 0.4998
Epoch 50/100 | Train MSE: 0.6500 | Val MSE: 0.7688 | Val Spearman: 0.5128
Epoch 55/100 | Train MSE: 0.6415 | Val MSE: 0.7476 | Val Spearman: 0.5326
Epoch 60/100 | Train MSE: 0.6198 | Val MSE: 0.7806 | Val Spearman: 0.5204
Epoch 65/100 | Train MSE: 0.6304 | Val MSE: 0.7911 | Val Spearman: 0.4929
Epoch 70/100 | Train MSE: 0.6107 | Val MSE: 0.7625 | Val Spearman: 0.5217
Epoch 75/100 | Train MSE: 0.6287 | Val MSE: 0.7431 | Val Spearman: 0.5405
Epoch 80/100 | Train MSE: 0.5859 | Val MSE: 0.7635 | Val Spearman: 0.5192
Epoch 85/100 | Train MSE: 0.5875 | Val MSE: 0.8146 | Val Spearman: 0.4908
Epoch 90/100 | Train MSE: 0.5960 | Val MSE: 0.7359 | Val Spearman: 0.5507
Epoch 95/100 | Train MSE: 0.5931 | Val MSE: 0.7537 | Val Spearman: 0.5312
Epoch 100/100 | Train MSE: 0.5767 | Val MSE: 0.7847 | Val Spearman: 0.5478

input->200->100->50->1
Training DMS regressor...
Epoch 01/100 | Train MSE: 0.9695 | Val MSE: 1.0347 | Val Spearman: 0.2148
Epoch 05/100 | Train MSE: 0.9664 | Val MSE: 1.0376 | Val Spearman: 0.3176
Epoch 10/100 | Train MSE: 0.9383 | Val MSE: 0.9668 | Val Spearman: 0.3057
Epoch 15/100 | Train MSE: 0.8357 | Val MSE: 0.8393 | Val Spearman: 0.4473
Epoch 20/100 | Train MSE: 0.7635 | Val MSE: 0.8009 | Val Spearman: 0.4873
Epoch 25/100 | Train MSE: 0.7072 | Val MSE: 0.8097 | Val Spearman: 0.5078
Epoch 30/100 | Train MSE: 0.6637 | Val MSE: 0.7908 | Val Spearman: 0.5147
Epoch 35/100 | Train MSE: 0.6511 | Val MSE: 0.7772 | Val Spearman: 0.5244
Epoch 40/100 | Train MSE: 0.6403 | Val MSE: 0.8703 | Val Spearman: 0.3923
Epoch 45/100 | Train MSE: 0.6247 | Val MSE: 0.7472 | Val Spearman: 0.5366
Epoch 50/100 | Train MSE: 0.6147 | Val MSE: 0.7625 | Val Spearman: 0.5295
Epoch 55/100 | Train MSE: 0.6008 | Val MSE: 0.7486 | Val Spearman: 0.5386
Epoch 60/100 | Train MSE: 0.5876 | Val MSE: 0.7952 | Val Spearman: 0.4705
Epoch 65/100 | Train MSE: 0.5961 | Val MSE: 0.7544 | Val Spearman: 0.5189
Epoch 70/100 | Train MSE: 0.5810 | Val MSE: 0.7441 | Val Spearman: 0.5356
Epoch 75/100 | Train MSE: 0.5786 | Val MSE: 0.7758 | Val Spearman: 0.5019
Epoch 80/100 | Train MSE: 0.5691 | Val MSE: 0.7596 | Val Spearman: 0.5288
Epoch 85/100 | Train MSE: 0.5681 | Val MSE: 0.8526 | Val Spearman: 0.4387
Epoch 90/100 | Train MSE: 0.5673 | Val MSE: 0.7806 | Val Spearman: 0.5195
Epoch 95/100 | Train MSE: 0.5630 | Val MSE: 0.6987 | Val Spearman: 0.5659
Epoch 100/100 | Train MSE: 0.5456 | Val MSE: 0.7597 | Val Spearman: 0.5557
'''