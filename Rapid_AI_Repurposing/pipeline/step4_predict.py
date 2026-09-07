"""
step4_predict.py
================
Generates ranked novel drug-repurposing candidates from the trained GraphSAGE model.

Phase 1 — Cosine-Normalized Scoring:
    Raw dot-product logits scale with ||z_u|| * ||z_v||, causing high-norm "hub" disease
    nodes (e.g., tibia fracture with ||z|| = 22.1) to dominate every top-ranked pair
    irrespective of actual biological similarity. To remove this magnitude bias, we
    L2-normalize all embeddings before scoring, reducing the decoder to a cosine similarity:

        sim(u, v) = z_u_hat.dot(z_v_hat)  in [-1, 1]

    A temperature parameter T scales logits before sigmoid so the output probability
    range is realistic and differentiates confident from borderline predictions.

Phase 2 — Per-Disease Diversity Cap:
    Even with cosine scoring, a single high-degree hub disease can still monopolize
    the global top-N list if many drugs align to it. We enforce a maximum of MAX_PER_DISEASE
    candidates per disease in the final top_50 list, ensuring the prioritized output
    covers a clinically diverse spectrum of therapeutic indications.
"""

import sys
import os
import torch
import torch.nn.functional as F
import pandas as pd
import json
from torch_geometric.nn import SAGEConv

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
DATA_DIR      = os.path.join(PROJECT_DIR, "preprocessed_data")
DATAVERSE_DIR = os.path.abspath(os.path.join(PROJECT_DIR, "..", "dataverse_files"))

# ── Scoring Hyperparameters ────────────────────────────────────────────────────
TEMPERATURE    = 10.0   # logit = cos_sim * T before sigmoid; keeps output in [0,1]
MAX_PER_DISEASE = 2     # maximum candidates per disease in the top-50 diverse list
TOP_N_GLOBAL   = 50     # number of candidates in the final output

print("--- STARTING STEP 4: Link Prediction & Top 50 Drug Repurposing Candidates ---")
print(f"   Scoring mode    : Cosine-normalized (temperature={TEMPERATURE})")
print(f"   Diversity cap   : max {MAX_PER_DISEASE} candidates per disease in top {TOP_N_GLOBAL}")

# ── Load Tensors & Mappings ────────────────────────────────────────────────────
x          = torch.load(os.path.join(DATA_DIR, "x.pt"))
edge_index = torch.load(os.path.join(DATA_DIR, "edge_index.pt"))
node_type  = torch.load(os.path.join(DATA_DIR, "node_type.pt"))

with open(os.path.join(DATA_DIR, "node_map.json"), "r") as f:
    node_map = json.load(f)
rev_node_map = {v: int(k) for k, v in node_map.items()}

nodes_df       = pd.read_csv(os.path.join(DATAVERSE_DIR, "nodes_subset.csv"))
node_name_dict = dict(zip(nodes_df["node_index"], nodes_df["node_name"]))

# ── Model Definition (must mirror step3_train.py exactly) ─────────────────────
class LinkPredictorSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        """Produces raw 32-dim GraphSAGE embeddings z for each node."""
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode_raw(self, z, edge_label_index):
        """Unnormalized dot-product decoder (kept for train/eval compatibility)."""
        src = edge_label_index[0]
        dst = edge_label_index[1]
        return (z[src] * z[dst]).sum(dim=-1)

    def decode_cosine(self, z_hat, edge_label_index, temperature):
        """
        Cosine-normalized decoder for candidate prioritization.
        z_hat must be L2-normalized before calling. Returns sigmoid probabilities.
        logit = dot(z_u_hat, z_v_hat) * temperature  (cos_sim in [-1,1] * T)
        """
        src = edge_label_index[0]
        dst = edge_label_index[1]
        cos_sim = (z_hat[src] * z_hat[dst]).sum(dim=-1)  # in [-1, 1]
        return torch.sigmoid(cos_sim * temperature)

model = LinkPredictorSAGE(in_channels=131, hidden_channels=64, out_channels=32)
model.load_state_dict(torch.load(os.path.join(DATA_DIR, "best_graphsage_model.pth")))
model.eval()

# ── Phase 1: Cosine-Normalized Embedding Generation ───────────────────────────
print("\n1. Generating Node Embeddings and Applying L2 Normalization...")
with torch.no_grad():
    z     = model.encode(x, edge_index)          # raw embeddings [N, 32]
    z_hat = F.normalize(z, p=2, dim=-1)           # L2-normalized, ||z_hat|| = 1

norms = torch.norm(z, dim=-1)
print(f"   Raw embedding norms  — min: {norms.min():.3f}  mean: {norms.mean():.3f}  max: {norms.max():.3f}")
print(f"   Cosine scores bounded to [-1, 1] * temperature ({TEMPERATURE}) before sigmoid.")

# ── Build Drug/Disease Node Sets ───────────────────────────────────────────────
print("\n2. Building Novel Drug-Disease Candidate Pairs...")
drug_seq_indices    = (node_type == 1).nonzero(as_tuple=True)[0]
disease_seq_indices = (node_type == 0).nonzero(as_tuple=True)[0]

global_edges = set(
    (u.item(), v.item()) for u, v in zip(edge_index[0], edge_index[1])
)
print(f"   Found {len(drug_seq_indices)} Drugs and {len(disease_seq_indices)} Diseases.")

# ── Phase 1: Score All Novel Pairs Using Cosine Similarity ────────────────────
print("\n3. Scoring All Novel (Unseen) Drug-Disease Pairs with Cosine Normalization...")
results = []

with torch.no_grad():
    for drug_idx in drug_seq_indices:
        drug_idx = drug_idx.item()
        novel_diseases = [
            d.item() for d in disease_seq_indices
            if (drug_idx, d.item()) not in global_edges
        ]
        if not novel_diseases:
            continue

        batch_diseases     = torch.tensor(novel_diseases, dtype=torch.long)
        batch_drugs        = torch.full((len(novel_diseases),), drug_idx, dtype=torch.long)
        edge_label_index   = torch.stack([batch_drugs, batch_diseases], dim=0)

        # Phase 1: cosine-normalized score — removes hub-norm bias
        cos_probs = model.decode_cosine(z_hat, edge_label_index, TEMPERATURE)

        orig_drug  = rev_node_map[drug_idx]
        drug_name  = node_name_dict.get(orig_drug, f"Drug_{orig_drug}")

        for disease_idx, prob in zip(novel_diseases, cos_probs.tolist()):
            orig_disease  = rev_node_map[disease_idx]
            disease_name  = node_name_dict.get(orig_disease, f"Disease_{orig_disease}")
            results.append({
                "drug_seq_idx":    drug_idx,
                "disease_seq_idx": disease_idx,
                "drug_name":       drug_name,
                "disease_name":    disease_name,
                "repurposing_score": round(prob, 6),
            })

# ── Phase 2: Per-Disease Diversity Cap ────────────────────────────────────────
print(f"\n4. Applying Per-Disease Diversity Cap (max {MAX_PER_DISEASE} per disease)...")
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("repurposing_score", ascending=False).reset_index(drop=True)

disease_counts: dict[str, int] = {}
diverse_rows = []
for _, row in results_df.iterrows():
    disease = row["disease_name"]
    count   = disease_counts.get(disease, 0)
    if count < MAX_PER_DISEASE:
        diverse_rows.append(row)
        disease_counts[disease] = count + 1
    if len(diverse_rows) >= TOP_N_GLOBAL:
        break

top50_df = pd.DataFrame(diverse_rows).reset_index(drop=True)
top50_df.index += 1  # rank starts from 1
top50_df.index.name = "rank"

# ── Save Outputs ───────────────────────────────────────────────────────────────
print("\n5. Saving Prediction Outputs...")

# Full ranked predictions (all pairs, cosine-scored)
full_out = os.path.join(PROJECT_DIR, "all_repurposing_predictions.csv")
results_df.index = results_df.index + 1
results_df.index.name = "rank"
results_df.to_csv(full_out)

# Diverse top-50 (Phase 2 filtered)
top50_out = os.path.join(PROJECT_DIR, "top_50_repurposing_predictions.csv")
top50_df.to_csv(top50_out)

# Also write to evaluation_outputs for reporting
eval_out = os.path.join(PROJECT_DIR, "evaluation_outputs", "top20_predictions.csv")
top50_df.head(20).to_csv(eval_out)

print("\n--- PREDICTION SUMMARY ---")
print(f"Total Unique Pairs Scored : {len(results_df):,}")
print(f"Diseases represented in top-{TOP_N_GLOBAL}: {top50_df['disease_name'].nunique()}")
print(f"\nTop 10 Diverse Candidates:")
print(top50_df[["drug_name", "disease_name", "repurposing_score"]].head(10).to_string())
print(f"\n--- STEP 4 COMPLETE ---")
print(f"   Full predictions : {full_out}")
print(f"   Top-{TOP_N_GLOBAL} diverse  : {top50_out}")
