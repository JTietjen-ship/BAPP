import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


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

################################
####### 3: Store in File #######
################################

# Create column names for the 320 embedding dimensions
embed_cols = [f"emb_{i}" for i in range(X.shape[1])]
embed_df = pd.DataFrame(X, columns=embed_cols)

# Combine with your original metadata/targets
out_df = pd.concat([df[["mutant", "mutated_sequence", "DMS_score"]], embed_df], axis=1)

# Save to CSV (index=False prevents adding an unnamed row-number column)
out_df.to_csv("p53_esm2_t6_8M_embeddings.csv", index=False)