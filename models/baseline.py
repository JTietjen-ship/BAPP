# Baseline model (seq -> ESM2 -> NN -> predict DMS score)
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import spearmanr


#####################################
####### 1: Data/Model Imports #######
#####################################

df = pd.read_csv("data/P53_HUMAN_Giacomelli_2018_Null_Nutlin.csv")
df = df.dropna(subset=["mutated_sequence", "DMS_score"]).reset_index(drop=True)
# print(df.head())
sequences = df["mutated_sequence"].tolist()
mutants = df["mutant"].tolist()
y_scores = df["DMS_score"].values.astype(np.float32)

# facebook/esm2_t12_35M_UR50D
# facebook/esm2_t30_150M_UR50D
# facebook/esm2_t33_650M_UR50D
# facebook/esm2_t36_3B_UR50D
model_name = "facebook/esm2_t6_8M_UR50D"

# Detect Apple Silicon GPU (MPS) or fall back to CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Running on device: {device}")


#####################################
####### 2: Extract Embeddings #######
#####################################

tokenizer = AutoTokenizer.from_pretrained(model_name)
esm_model = AutoModel.from_pretrained(model_name).to(device)
esm_model.eval()

batch_size_esm = 16
all_embeddings = []

print("Extracting ESM-2 embeddings...")

with torch.no_grad():
    for start_idx in range(0, len(sequences), batch_size_esm):
        batch_seqs = sequences[start_idx : start_idx + batch_size_esm]
        inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True).to(device)
        outputs = esm_model(**inputs)
        token_reps = outputs.last_hidden_state

        for i, seq in enumerate(batch_seqs):
            seq_len = len(seq)
            # Average across amino acids (excluding <cls>, <eos>, and <pad>)
            # shape: [hidden_dim], averages hidden dimensions (column-wise)
            seq_rep = token_reps[i, 1 : seq_len + 1].mean(dim=0)
            all_embeddings.append(seq_rep.cpu().numpy())

X = np.vstack(all_embeddings).astype(np.float32)  # Shape: (N, 320)
print(f"Extracted embeddings shape: {X.shape}")

# Optional: Pair mutants with their embeddings or attach back to a subset DataFrame
# subset_df = df.head(5).copy()
# subset_df["esm2_embedding"] = sequence_embeddings
# print(subset_df[["mutant", "DMS_score", "esm2_embedding"]])


##############################
####### 3: Data Splits #######
##############################

torch.manual_seed(42)
indices = torch.randperm(len(X))
split = int(0.8 * len(X))  # 80% train, 20% validation

train_idx, val_idx = indices[:split], indices[split:]

X_train, y_train = torch.tensor(X[train_idx]), torch.tensor(y_scores[train_idx]).unsqueeze(1)
X_val, y_val = torch.tensor(X[val_idx]), torch.tensor(y_scores[val_idx]).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)


##############################
####### 4: Define MLP #######
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
####### 5: Training/Eval #######
################################

epochs = 60
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

        print(f"Epoch {epoch:02d}/{epochs} | Train MSE: {train_loss:.4f} | Val MSE: {val_mse:.4f} | Val Spearman ρ: {rho:.4f}")

'''

Epochs = 30, ran under 2 min:
Epoch 01/30 | Train MSE: 0.9827 | Val MSE: 0.9984 | Val Spearman ρ: 0.2723
Epoch 05/30 | Train MSE: 0.9794 | Val MSE: 0.9826 | Val Spearman ρ: 0.2699
Epoch 10/30 | Train MSE: 0.8880 | Val MSE: 0.8898 | Val Spearman ρ: 0.3208
Epoch 15/30 | Train MSE: 0.8432 | Val MSE: 0.8572 | Val Spearman ρ: 0.3559
Epoch 20/30 | Train MSE: 0.8152 | Val MSE: 0.8360 | Val Spearman ρ: 0.3848
Epoch 25/30 | Train MSE: 0.7903 | Val MSE: 0.8133 | Val Spearman ρ: 0.4211
Epoch 30/30 | Train MSE: 0.7580 | Val MSE: 0.7990 | Val Spearman ρ: 0.4321

Epochs = 60, ran under 4 min, but my laptop got pretty hot:
Epoch 01/60 | Train MSE: 0.9827 | Val MSE: 0.9984 | Val Spearman ρ: 0.2723
Epoch 05/60 | Train MSE: 0.9794 | Val MSE: 0.9826 | Val Spearman ρ: 0.2699
Epoch 10/60 | Train MSE: 0.8880 | Val MSE: 0.8898 | Val Spearman ρ: 0.3208
Epoch 15/60 | Train MSE: 0.8432 | Val MSE: 0.8572 | Val Spearman ρ: 0.3559
Epoch 20/60 | Train MSE: 0.8152 | Val MSE: 0.8360 | Val Spearman ρ: 0.3848
Epoch 25/60 | Train MSE: 0.7903 | Val MSE: 0.8133 | Val Spearman ρ: 0.4211
Epoch 30/60 | Train MSE: 0.7580 | Val MSE: 0.7990 | Val Spearman ρ: 0.4321
Epoch 35/60 | Train MSE: 0.7359 | Val MSE: 0.8225 | Val Spearman ρ: 0.4483
Epoch 40/60 | Train MSE: 0.7234 | Val MSE: 0.7751 | Val Spearman ρ: 0.4983
Epoch 45/60 | Train MSE: 0.7082 | Val MSE: 0.7424 | Val Spearman ρ: 0.5026
Epoch 50/60 | Train MSE: 0.7044 | Val MSE: 0.7418 | Val Spearman ρ: 0.5140
Epoch 55/60 | Train MSE: 0.6865 | Val MSE: 0.7284 | Val Spearman ρ: 0.5302
Epoch 60/60 | Train MSE: 0.6677 | Val MSE: 0.6939 | Val Spearman ρ: 0.5512

'''