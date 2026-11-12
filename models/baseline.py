# Baseline model (seq -> ESM2 -> NN -> predict DMS score)
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

df = pd.read_csv("data/P53_HUMAN_Giacomelli_2018_Null_Nutlin.csv")
# print(df.head())
sequences = df["mutated_sequence"].head(5).tolist()
mutants = df["mutant"].head(5).tolist()

# facebook/esm2_t12_35M_UR50D
# facebook/esm2_t30_150M_UR50D
# facebook/esm2_t33_650M_UR50D
# facebook/esm2_t36_3B_UR50D
model_name = "facebook/esm2_t6_8M_UR50D"

# Detect Apple Silicon GPU (MPS) or fall back to CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Running on device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()

# tokenize the batch, padding to handle frameshifts
inputs = tokenizer(sequences, return_tensors="pt", padding=True).to(device)

with torch.no_grad():
    outputs = model(**inputs)

# Extract per-residue representations
# Shape: [batch_size, max_tokens_in_batch, hidden_dim]
token_representations = outputs.last_hidden_state

# Extract clean mean-pooled embeddings per sequence (ignoring <cls>, <eos>, and <pad>)
sequence_embeddings = []
for i, seq in enumerate(sequences):
    seq_len = len(seq)

    # Index 0 is <cls>, indices 1 to seq_len+1 are the actual amino acids
    # shape: [hidden_dim], averages hidden dimensions (column-wise)
    seq_rep = token_representations[i, 1 : seq_len + 1].mean(dim=0) 
    sequence_embeddings.append(seq_rep.cpu().numpy())

# Stack into a single matrix of shape (5, hidden_dim)
embeddings_matrix = np.vstack(sequence_embeddings)

print(f"Batch token representations shape: {token_representations.shape}")
print(f"Final sequence embeddings matrix shape: {embeddings_matrix.shape}")

# Optional: Pair mutants with their embeddings or attach back to a subset DataFrame
# subset_df = df.head(5).copy()
# subset_df["esm2_embedding"] = sequence_embeddings
# print(subset_df[["mutant", "DMS_score", "esm2_embedding"]])