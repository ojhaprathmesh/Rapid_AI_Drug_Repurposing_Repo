import sys
import os
import torch
import pandas as pd
import json
from torch_geometric.nn import SAGEConv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
BASE_DIR = PROJECT_DIR
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "preprocessed_data")
DATAVERSE_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "dataverse_files"))

print("--- STARTING STEP 4: Link Prediction & Top 50 Drug Repurposing Candidates ---")

# Load tensors
x = torch.load(os.path.join(DATA_DIR, "x.pt"))
edge_index = torch.load(os.path.join(DATA_DIR, "edge_index.pt"))
node_type = torch.load(os.path.join(DATA_DIR, "node_type.pt"))

# Load node name mapping from Step 1
with open(os.path.join(DATA_DIR, "node_map.json"), "r") as f:
    node_map = json.load(f)

# Reverse mapping: sequential_idx -> original_node_index
rev_node_map = {v: int(k) for k, v in node_map.items()}

# Load original nodes CSV to get human-readable names
nodes_df = pd.read_csv(os.path.join(DATAVERSE_DIR, "nodes_subset.csv"))
node_name_dict = dict(zip(nodes_df['node_index'], nodes_df['node_name']))
node_type_dict = dict(zip(nodes_df['node_index'], nodes_df['node_type']))

# Rebuild model architecture (must match Step 3 exactly)
class LinkPredictorSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        src = edge_label_index[0]
        dst = edge_label_index[1]
        return (z[src] * z[dst]).sum(dim=-1)

model = LinkPredictorSAGE(in_channels=131, hidden_channels=64, out_channels=32)
model.load_state_dict(torch.load(os.path.join(DATA_DIR, "best_graphsage_model.pth")))
model.eval()

print("1. Generating Node Embeddings from Trained Model...")
with torch.no_grad():
    z = model.encode(x, edge_index)

print("2. Building Full Drug-Disease Candidate Pairs...")
drug_seq_indices = (node_type == 1).nonzero(as_tuple=True)[0]
disease_seq_indices = (node_type == 0).nonzero(as_tuple=True)[0]

# Load global edges from Step 2 to exclude already-known links
global_edges = set([
    (u.item(), v.item())
    for u, v in zip(edge_index[0], edge_index[1])
])

print(f"   Found {len(drug_seq_indices)} Drugs and {len(disease_seq_indices)} Diseases.")

print("3. Scoring All Novel (Unseen) Drug-Disease Pairs...")
results = []

with torch.no_grad():
    for drug_idx in drug_seq_indices:
        drug_idx = drug_idx.item()
        # Only score pairs NOT already in the knowledge graph
        novel_diseases = [
            d.item() for d in disease_seq_indices
            if (drug_idx, d.item()) not in global_edges
        ]
        if not novel_diseases:
            continue

        batch_diseases = torch.tensor(novel_diseases, dtype=torch.long)
        batch_drugs = torch.full((len(novel_diseases),), drug_idx, dtype=torch.long)
        edge_label_index = torch.stack([batch_drugs, batch_diseases], dim=0)

        scores = model.decode(z, edge_label_index)  # Raw logits for proper ranking

        for disease_idx, score in zip(novel_diseases, scores.tolist()):
            orig_drug = rev_node_map[drug_idx]
            orig_disease = rev_node_map[disease_idx]
            drug_name = node_name_dict.get(orig_drug, f"Drug_{orig_drug}")
            disease_name = node_name_dict.get(orig_disease, f"Disease_{orig_disease}")
            results.append({
                "drug_seq_idx": drug_idx,
                "disease_seq_idx": disease_idx,
                "drug_name": drug_name,
                "disease_name": disease_name,
                "repurposing_score": round(torch.sigmoid(torch.tensor(score)).item(), 6),
                "raw_logit": round(score, 6)
            })

print("4. Finalizing and Saving All Predictions...")
results_df = pd.DataFrame(results)
# Sort by score but keep all rows
results_df = results_df.sort_values("raw_logit", ascending=False)
results_df = results_df.reset_index(drop=True)
results_df.index += 1  # Rank starts from 1

output_path = os.path.join(PROJECT_DIR, "all_repurposing_predictions.csv")
results_df.to_csv(output_path, index_label="rank")

print(f"\n--- PREDICTION SUMMARY ---")
print(f"Total Unique Pairs Scored: {len(results_df)}")
print(f"Top 10 Candidates:\n{results_df[['drug_name', 'disease_name', 'repurposing_score']].head(10).to_string()}")
print(f"\n--- STEP 4 COMPLETE: Full predictions saved to {output_path} ---")
