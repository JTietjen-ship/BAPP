import pandas as pd

# Code just to convert .tsv into .fasta for alphafold:
'''
df = pd.read_csv("data/metadata_AA_seq_allmutations.tsv", sep="\t")
with open("alphafold_input.fasta", "w") as f:
    for _, row in df.iterrows():
        # Uses the HGVS mutation notation (e.g., p.E2K) as a descriptive ID
        header_id = f"{row['HGVS_protein']}"
        sequence = row["AA_sequence"]

        # Format and write to the FASTA file
        f.write(f">{header_id}\n{sequence}\n")
'''

# TODO: run this .fasta on alphafold to obtain PDB data