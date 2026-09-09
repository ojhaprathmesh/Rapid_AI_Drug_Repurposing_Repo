"""
step_baselines.py
=================
Implements and evaluates the four baseline architectures reported in Table II
of the manuscript on the same leakage-free train/val/test splits produced by
step2_build_graph.py.

Baselines:
  1. Common Neighbors  — topological heuristic (no training)
  2. MLP               — 131-dim node features only, no message passing
  3. GCN               — Kipf & Welling symmetric-Laplacian propagation
  4. GAT               — Veličković et al. attention-weighted aggregation

All four use:
  - Identical positive/negative supervision edges (edge_label_index)
  - The same leakage-free msg_edge_index built by step2_build_graph.py
    (GCN and GAT) or no graph edges (MLP) or the raw adjacency set (CN)

Results saved to:
  evaluation_outputs/baseline_metrics.csv
  evaluation_outputs/baseline_metrics.json
"""

import sys
import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

from torch_geometric.nn import GCNConv, GATConv, RGCNConv
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import warnings
warnings.filterwarnings("ignore")

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "preprocessed_data")
OUT_DIR  = os.path.join(PROJECT_DIR, "evaluation_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 65)
print("  Baseline Evaluation Suite — Table II Reproducibility")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load the leakage-free data splits (built by step2_build_graph.py)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/6] Loading leakage-free data splits …")
train_data = torch.load(os.path.join(DATA_DIR, "train_data.pt"), weights_only=False)
val_data   = torch.load(os.path.join(DATA_DIR, "val_data.pt"),   weights_only=False)
test_data  = torch.load(os.path.join(DATA_DIR, "test_data.pt"),  weights_only=False)

x              = test_data.x
node_type      = test_data.node_type
# msg_edge_index: leakage-free MP graph (test pos edges excised)
test_msg_ei    = test_data.msg_edge_index
train_msg_ei   = train_data.msg_edge_index
# edge_label supervision tensors
test_eli       = test_data.edge_label_index
test_labels    = test_data.edge_label.numpy().astype(int)
train_eli      = train_data.edge_label_index
train_labels   = train_data.edge_label
val_eli        = val_data.edge_label_index
val_labels     = val_data.edge_label

# 6 Biological relation types for Relational GCN (RGCN)
df_edges = pd.read_csv(os.path.join(PROJECT_DIR, "..", "dataverse_files", "edges_subset.csv"))
with open(os.path.join(DATA_DIR, "node_map.json")) as f:
    node_map = {int(k): v for k, v in json.load(f).items()}

REL_NAMES = ['indication', 'target', 'enzyme', 'transporter', 'carrier', 'associated with']
rel_to_id = {r: i for i, r in enumerate(REL_NAMES)}

edge_rel_dict = {}
for _, row in df_edges.iterrows():
    u = node_map[row['x_index']]
    v = node_map[row['y_index']]
    rel_id = rel_to_id[row['display_relation']]
    edge_rel_dict[(u, v)] = rel_id
    edge_rel_dict[(v, u)] = rel_id

def get_edge_types(msg_ei):
    types = [edge_rel_dict[(msg_ei[0, i].item(), msg_ei[1, i].item())] for i in range(msg_ei.shape[1])]
    return torch.tensor(types, dtype=torch.long)

train_et = get_edge_types(train_msg_ei)
val_et   = get_edge_types(val_data.msg_edge_index)
test_et  = get_edge_types(test_msg_ei)

print(f"   Test samples : {len(test_labels):,}  "
      f"(pos={test_labels.sum()}, neg={len(test_labels)-test_labels.sum()})")

EPOCHS = 100
LR     = 0.01
WD     = 1e-4
results = {}

# ─────────────────────────────────────────────────────────────────────────────
# Helper: compute all reported metrics from probability scores
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "AUC_ROC"   : round(float(roc_auc_score(y_true, y_prob)),                    4),
        "AUPRC"     : round(float(average_precision_score(y_true, y_prob)),           4),
        "Accuracy"  : round(float(accuracy_score(y_true, y_pred)),                   4),
        "Precision" : round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "Recall"    : round(float(recall_score(y_true, y_pred,    zero_division=0)), 4),
        "F1_Score"  : round(float(f1_score(y_true, y_pred,        zero_division=0)), 4),
    }

def print_metrics(name, m):
    print(f"\n   ── {name} ────────────────────────────")
    print(f"   AUC-ROC   : {m['AUC_ROC']:.4f}")
    print(f"   PR-AUC    : {m['AUPRC']:.4f}")
    print(f"   Accuracy  : {m['Accuracy']:.4f}")
    print(f"   Precision : {m['Precision']:.4f}")
    print(f"   Recall    : {m['Recall']:.4f}")
    print(f"   F1-Score  : {m['F1_Score']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Baseline 1: Common Neighbors (topological heuristic)
#    Score(u,v) = |N(u) ∩ N(v)| on the training message-passing graph.
#    Uses train_msg_ei adjacency (no val/test edges present).
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/6] Common Neighbors heuristic …")

# Build adjacency sets from train_msg_ei
adj = {}
for u, v in zip(train_msg_ei[0].tolist(), train_msg_ei[1].tolist()):
    adj.setdefault(u, set()).add(v)

cn_scores = []
for i in range(test_eli.shape[1]):
    u, v   = test_eli[0, i].item(), test_eli[1, i].item()
    nu, nv = adj.get(u, set()), adj.get(v, set())
    cn_scores.append(len(nu & nv))

cn_scores = np.array(cn_scores, dtype=float)
# Normalise to [0,1] so roc_auc_score can handle it
max_cn = cn_scores.max()
if max_cn > 0:
    cn_scores_norm = cn_scores / max_cn
else:
    cn_scores_norm = cn_scores

m_cn = compute_metrics(test_labels, cn_scores_norm)
print_metrics("Common Neighbors", m_cn)
results["Common Neighbors"] = m_cn

# ─────────────────────────────────────────────────────────────────────────────
# 3. Baseline 2: Feature-Only MLP (no graph topology)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/6] Feature-Only MLP (no graph message passing) …")

class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden, out_channels):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_channels, hidden)
        self.fc2 = torch.nn.Linear(hidden, out_channels)

    def encode(self, x, _edge_index=None):
        """_edge_index ignored — pure feature baseline."""
        return self.fc2(F.relu(self.fc1(x)))

    def decode(self, z, eli):
        return (z[eli[0]] * z[eli[1]]).sum(dim=-1)

mlp       = MLP(131, 64, 32)
opt_mlp   = torch.optim.Adam(mlp.parameters(), lr=LR, weight_decay=WD)
criterion = torch.nn.BCEWithLogitsLoss()
best_mlp_state, best_mlp_val = None, float("inf")

for epoch in range(1, EPOCHS + 1):
    mlp.train(); opt_mlp.zero_grad()
    z    = mlp.encode(train_data.x)
    loss = criterion(mlp.decode(z, train_eli), train_labels)
    loss.backward(); opt_mlp.step()

    mlp.eval()
    with torch.no_grad():
        z_v   = mlp.encode(val_data.x)
        vl    = criterion(mlp.decode(z_v, val_eli), val_labels).item()
    if vl < best_mlp_val:
        best_mlp_val   = vl
        best_mlp_state = {k: v.clone() for k, v in mlp.state_dict().items()}

mlp.load_state_dict(best_mlp_state)
mlp.eval()
with torch.no_grad():
    z_test   = mlp.encode(test_data.x)
    y_prob   = torch.sigmoid(mlp.decode(z_test, test_eli)).numpy()

m_mlp = compute_metrics(test_labels, y_prob)
print_metrics("Feature-Only MLP", m_mlp)
results["Feature-Only MLP"] = m_mlp
torch.save(best_mlp_state, os.path.join(DATA_DIR, "baseline_mlp.pth"))

# ─────────────────────────────────────────────────────────────────────────────
# 4. Baseline 3: GCN (Kipf & Welling, 2017)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/6] Graph Convolutional Network (GCN) …")

class GCNLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, eli):
        return (z[eli[0]] * z[eli[1]]).sum(dim=-1)

gcn       = GCNLinkPredictor(131, 64, 32)
opt_gcn   = torch.optim.Adam(gcn.parameters(), lr=LR, weight_decay=WD)
best_gcn_state, best_gcn_val = None, float("inf")

for epoch in range(1, EPOCHS + 1):
    gcn.train(); opt_gcn.zero_grad()
    z    = gcn.encode(train_data.x, train_msg_ei)
    loss = criterion(gcn.decode(z, train_eli), train_labels)
    loss.backward(); opt_gcn.step()

    gcn.eval()
    with torch.no_grad():
        z_v = gcn.encode(val_data.x, val_data.msg_edge_index)
        vl  = criterion(gcn.decode(z_v, val_eli), val_labels).item()
    if vl < best_gcn_val:
        best_gcn_val   = vl
        best_gcn_state = {k: v.clone() for k, v in gcn.state_dict().items()}

gcn.load_state_dict(best_gcn_state)
gcn.eval()
with torch.no_grad():
    z_test = gcn.encode(test_data.x, test_msg_ei)
    y_prob = torch.sigmoid(gcn.decode(z_test, test_eli)).numpy()

m_gcn = compute_metrics(test_labels, y_prob)
print_metrics("GCN", m_gcn)
results["GCN"] = m_gcn
torch.save(best_gcn_state, os.path.join(DATA_DIR, "baseline_gcn.pth"))

# ─────────────────────────────────────────────────────────────────────────────
# 5. Baseline 4: GAT (Veličković et al., 2018)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/6] Graph Attention Network (GAT) …")

class GATLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden, out_channels, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden, heads=heads, dropout=0.3)
        self.conv2 = GATConv(hidden * heads, out_channels, heads=1, concat=False, dropout=0.3)

    def encode(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

    def decode(self, z, eli):
        return (z[eli[0]] * z[eli[1]]).sum(dim=-1)

gat       = GATLinkPredictor(131, 16, 32, heads=4)
opt_gat   = torch.optim.Adam(gat.parameters(), lr=LR, weight_decay=WD)
best_gat_state, best_gat_val = None, float("inf")

for epoch in range(1, EPOCHS + 1):
    gat.train(); opt_gat.zero_grad()
    z    = gat.encode(train_data.x, train_msg_ei)
    loss = criterion(gat.decode(z, train_eli), train_labels)
    loss.backward(); opt_gat.step()

    gat.eval()
    with torch.no_grad():
        z_v = gat.encode(val_data.x, val_data.msg_edge_index)
        vl  = criterion(gat.decode(z_v, val_eli), val_labels).item()
    if vl < best_gat_val:
        best_gat_val   = vl
        best_gat_state = {k: v.clone() for k, v in gat.state_dict().items()}

gat.load_state_dict(best_gat_state)
gat.eval()
with torch.no_grad():
    z_test = gat.encode(test_data.x, test_msg_ei)
    y_prob = torch.sigmoid(gat.decode(z_test, test_eli)).numpy()

m_gat = compute_metrics(test_labels, y_prob)
print_metrics("GAT", m_gat)
results["GAT"] = m_gat
torch.save(best_gat_state, os.path.join(DATA_DIR, "baseline_gat.pth"))

# ─────────────────────────────────────────────────────────────────────────────
# 6. Baseline 5: RGCN (Schlichtkrull et al., 2018)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/7] Relational Graph Convolutional Network (RGCN) …")

class RGCNLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden, out_channels, num_relations=6):
        super().__init__()
        self.conv1 = RGCNConv(in_channels, hidden, num_relations=num_relations)
        self.conv2 = RGCNConv(hidden, out_channels, num_relations=num_relations)

    def encode(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type).relu()
        return self.conv2(x, edge_index, edge_type)

    def decode(self, z, eli):
        return (z[eli[0]] * z[eli[1]]).sum(dim=-1)

rgcn       = RGCNLinkPredictor(131, 64, 32, num_relations=6)
opt_rgcn   = torch.optim.Adam(rgcn.parameters(), lr=LR, weight_decay=WD)
best_rgcn_state, best_rgcn_val = None, float("inf")

for epoch in range(1, EPOCHS + 1):
    rgcn.train(); opt_rgcn.zero_grad()
    z    = rgcn.encode(train_data.x, train_msg_ei, train_et)
    loss = criterion(rgcn.decode(z, train_eli), train_labels)
    loss.backward(); opt_rgcn.step()

    rgcn.eval()
    with torch.no_grad():
        z_v = rgcn.encode(val_data.x, val_data.msg_edge_index, val_et)
        vl  = criterion(rgcn.decode(z_v, val_eli), val_labels).item()
    if vl < best_rgcn_val:
        best_rgcn_val   = vl
        best_rgcn_state = {k: v.clone() for k, v in rgcn.state_dict().items()}

rgcn.load_state_dict(best_rgcn_state)
rgcn.eval()
with torch.no_grad():
    z_test = rgcn.encode(test_data.x, test_msg_ei, test_et)
    y_prob = torch.sigmoid(rgcn.decode(z_test, test_eli)).numpy()

m_rgcn = compute_metrics(test_labels, y_prob)
print_metrics("RGCN", m_rgcn)
results["RGCN"] = m_rgcn
torch.save(best_rgcn_state, os.path.join(DATA_DIR, "baseline_rgcn.pth"))

# ─────────────────────────────────────────────────────────────────────────────
# 7. Save Results
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7/7] Saving baseline results …")

rows = []
for model_name, m in results.items():
    rows.append({"Model": model_name, **m})

df = pd.DataFrame(rows)
csv_path  = os.path.join(OUT_DIR, "baseline_metrics.csv")
json_path = os.path.join(OUT_DIR, "baseline_metrics.json")
df.to_csv(csv_path, index=False)
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n   Saved: {csv_path}")
print(f"   Saved: {json_path}")

print("\n" + "=" * 65)
print("  BASELINE EVALUATION COMPLETE")
print("=" * 65)
print(df.to_string(index=False))
print("=" * 65)
