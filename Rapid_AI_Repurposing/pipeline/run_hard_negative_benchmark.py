"""
run_hard_negative_benchmark.py
===============================
Req 14 — Hard Negative Sampling Benchmark

Evaluates all 5 architectures (Common Neighbors, MLP, GCN, GAT, GraphSAGE)
across three negative test regimes:

  Regime I   : Uniform Random Negatives   (standard baseline)
  Regime II  : Degree-Matched Negatives   (eliminates degree shortcuts)
  Regime III : 2-Hop Biological Near-Miss (drug->protein->disease, not indicated)

Key properties:
  - N = 940 positive test edges (held fixed across all regimes)
  - N = 940 negatives per regime (zero leakage: no pair in any indication split)
  - All models evaluated with their pre-trained weights (no retraining)
  - Results saved to evaluation_outputs/hard_negative_benchmark.{json,csv}
"""

import os, sys, json, random, math, warnings
import torch, torch.nn.functional as F
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch_geometric.nn import SAGEConv, GCNConv, GATConv
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score
)

warnings.filterwarnings("ignore")

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

DATA_DIR = os.path.join(PROJECT_DIR, "preprocessed_data")
OUT_DIR  = os.path.join(PROJECT_DIR, "evaluation_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

random.seed(42); np.random.seed(42); torch.manual_seed(42)

print("=" * 70)
print("  Hard Negative Sampling Benchmark — Req 14")
print("=" * 70)

# ── 1. Load Data ──────────────────────────────────────────────────────────────
print("\n[1/7] Loading leakage-free data splits ...")
train_data = torch.load(os.path.join(DATA_DIR, "train_data.pt"), weights_only=False)
val_data   = torch.load(os.path.join(DATA_DIR, "val_data.pt"),   weights_only=False)
test_data  = torch.load(os.path.join(DATA_DIR, "test_data.pt"),  weights_only=False)

x            = test_data.x
node_type    = test_data.node_type
edge_index   = test_data.edge_index
test_msg_ei  = test_data.msg_edge_index
train_msg_ei = train_data.msg_edge_index

pos_mask    = test_data.edge_label == 1
test_pos_ei = test_data.edge_label_index[:, pos_mask]
N_pos = test_pos_ei.shape[1]
print(f"   Test positives: {N_pos:,}")

# ── 2. Build Exclusion Set ────────────────────────────────────────────────────
print("\n[2/7] Building exclusion set ...")
exclusion_set = set()
for split in [train_data, val_data, test_data]:
    mask = split.edge_label == 1
    pos  = split.edge_label_index[:, mask]
    for i in range(pos.shape[1]):
        u, v = pos[0, i].item(), pos[1, i].item()
        exclusion_set.add((u, v)); exclusion_set.add((v, u))

global_edge_set = set()
for u, v in zip(edge_index[0].tolist(), edge_index[1].tolist()):
    global_edge_set.add((u, v))

forbidden = exclusion_set | global_edge_set
print(f"   Forbidden pairs: {len(forbidden):,}")

drug_idx_list    = (node_type == 1).nonzero(as_tuple=True)[0].tolist()
protein_idx_list = (node_type == 2).nonzero(as_tuple=True)[0].tolist()
disease_idx_list = (node_type == 0).nonzero(as_tuple=True)[0].tolist()
print(f"   Drugs={len(drug_idx_list):,}  Proteins={len(protein_idx_list):,}  Diseases={len(disease_idx_list):,}")

# ── 3. Regime I: Uniform Random (existing test negatives) ─────────────────────
print("\n[3/7] Regime I: Uniform random negatives ...")
neg_mask      = test_data.edge_label == 0
test_neg_unif = test_data.edge_label_index[:, neg_mask]
N_neg_unif    = test_neg_unif.shape[1]
leakage_unif  = sum(1 for i in range(N_neg_unif)
                    if (test_neg_unif[0,i].item(), test_neg_unif[1,i].item()) in exclusion_set)
print(f"   Negatives={N_neg_unif:,}  |  Leakage={leakage_unif}  (must be 0)")
assert leakage_unif == 0

# ── 4. Regime II: Degree-Matched ──────────────────────────────────────────────
print("\n[4/7] Regime II: Degree-matched negative sampling ...")
degree = {}
for u, v in zip(train_msg_ei[0].tolist(), train_msg_ei[1].tolist()):
    degree[u] = degree.get(u, 0) + 1

def log_bucket(d, n=10):
    if d == 0: return 0
    return min(int(math.log(d + 1) / math.log(200) * n), n - 1)

drug_buckets, disease_buckets = {}, {}
for idx in drug_idx_list:
    drug_buckets.setdefault(log_bucket(degree.get(idx, 0)), []).append(idx)
for idx in disease_idx_list:
    disease_buckets.setdefault(log_bucket(degree.get(idx, 0)), []).append(idx)

tdb  = [log_bucket(degree.get(test_pos_ei[0,i].item(), 0)) for i in range(N_pos)]
tdsb = [log_bucket(degree.get(test_pos_ei[1,i].item(), 0)) for i in range(N_pos)]

used_neg_dm, dm_neg_edges = set(), []
for i in range(N_pos):
    found = False
    for _ in range(200):
        for db_off in [0, 1, -1, 2, -2]:
            cands_d = drug_buckets.get(tdb[i] + db_off, [])
            if not cands_d: continue
            u = random.choice(cands_d)
            for dsb_off in [0, 1, -1, 2, -2]:
                cands_ds = disease_buckets.get(tdsb[i] + dsb_off, [])
                if not cands_ds: continue
                v = random.choice(cands_ds)
                key = (u, v)
                if key not in forbidden and key not in used_neg_dm:
                    dm_neg_edges.append([u, v])
                    used_neg_dm.add(key); used_neg_dm.add((v, u))
                    found = True; break
            if found: break
        if found: break
    if not found:
        for _ in range(10000):
            u = random.choice(drug_idx_list); v = random.choice(disease_idx_list)
            key = (u, v)
            if key not in forbidden and key not in used_neg_dm:
                dm_neg_edges.append([u, v])
                used_neg_dm.add(key); used_neg_dm.add((v, u)); break

test_neg_dm = torch.tensor(dm_neg_edges, dtype=torch.long).t()
N_neg_dm    = test_neg_dm.shape[1]
leakage_dm  = sum(1 for i in range(N_neg_dm)
                  if (test_neg_dm[0,i].item(), test_neg_dm[1,i].item()) in exclusion_set)
print(f"   Negatives={N_neg_dm:,}  |  Leakage={leakage_dm}  (must be 0)")
assert leakage_dm == 0

# ── 5. Regime III: 2-Hop Biological Near-Miss ─────────────────────────────────
print("\n[5/7] Regime III: 2-hop biological near-miss mining ...")
drug_to_proteins, protein_to_disease = {}, {}
for u, v in zip(edge_index[0].tolist(), edge_index[1].tolist()):
    ntu, ntv = node_type[u].item(), node_type[v].item()
    if ntu == 1 and ntv == 2:
        drug_to_proteins.setdefault(u, set()).add(v)
    if ntu == 2 and ntv == 0:
        protein_to_disease.setdefault(u, set()).add(v)

print(f"   Drugs with protein targets   : {len(drug_to_proteins):,}")
print(f"   Proteins with disease assoc. : {len(protein_to_disease):,}")
print("   Mining 2-hop near-miss candidates ...")

near_miss_set = set()
for drug, proteins in drug_to_proteins.items():
    for prot in proteins:
        if prot in protein_to_disease:
            for disease in protein_to_disease[prot]:
                pair = (drug, disease)
                if pair not in exclusion_set and pair not in global_edge_set:
                    near_miss_set.add(pair)

near_miss_unique = list(near_miss_set)
random.shuffle(near_miss_unique)
print(f"   Unique 2-hop near-miss candidates: {len(near_miss_unique):,}")

hard_neg_edges, used_hard = [], set()
for pair in near_miss_unique:
    u, v = pair
    key = (u, v)
    if key not in used_hard:
        hard_neg_edges.append([u, v])
        used_hard.add(key); used_hard.add((v, u))
        if len(hard_neg_edges) >= N_pos:
            break

if len(hard_neg_edges) < N_pos:
    raise RuntimeError(f"Not enough near-miss negatives: {len(hard_neg_edges)} < {N_pos}")

test_neg_hard = torch.tensor(hard_neg_edges[:N_pos], dtype=torch.long).t()
N_neg_hard    = test_neg_hard.shape[1]
leakage_hard  = sum(1 for i in range(N_neg_hard)
                    if (test_neg_hard[0,i].item(), test_neg_hard[1,i].item()) in exclusion_set)
print(f"   Negatives={N_neg_hard:,}  |  Leakage={leakage_hard}  (must be 0)")
assert leakage_hard == 0

# ── 6. Load Models ───────────────────────────────────────────────────────────
print("\n[6/7] Loading pre-trained model weights ...")

class LinkPredictorSAGE(torch.nn.Module):
    def __init__(self, in_c, hid, out_c):
        super().__init__()
        self.conv1 = SAGEConv(in_c, hid); self.conv2 = SAGEConv(hid, out_c)
    def encode(self, x, ei): return self.conv2(self.conv1(x, ei).relu(), ei)
    def decode(self, z, eli): return (z[eli[0]] * z[eli[1]]).sum(dim=-1)

class MLP(torch.nn.Module):
    def __init__(self, in_c, hid, out_c):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_c, hid); self.fc2 = torch.nn.Linear(hid, out_c)
    def encode(self, x, _=None): return self.fc2(F.relu(self.fc1(x)))
    def decode(self, z, eli): return (z[eli[0]] * z[eli[1]]).sum(dim=-1)

class GCNLinkPredictor(torch.nn.Module):
    def __init__(self, in_c, hid, out_c):
        super().__init__()
        self.conv1 = GCNConv(in_c, hid); self.conv2 = GCNConv(hid, out_c)
    def encode(self, x, ei): return self.conv2(self.conv1(x, ei).relu(), ei)
    def decode(self, z, eli): return (z[eli[0]] * z[eli[1]]).sum(dim=-1)

class GATLinkPredictor(torch.nn.Module):
    def __init__(self, in_c, hid, out_c, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_c, hid, heads=heads, dropout=0.3)
        self.conv2 = GATConv(hid * heads, out_c, heads=1, concat=False, dropout=0.3)
    def encode(self, x, ei):
        return self.conv2(F.elu(self.conv1(x, ei)), ei)
    def decode(self, z, eli): return (z[eli[0]] * z[eli[1]]).sum(dim=-1)

sage = LinkPredictorSAGE(131, 64, 32)
sage.load_state_dict(torch.load(os.path.join(DATA_DIR, "best_graphsage_model.pth"), weights_only=True))
sage.eval()

mlp_model = MLP(131, 64, 32)
mlp_model.load_state_dict(torch.load(os.path.join(DATA_DIR, "baseline_mlp.pth"), weights_only=True))
mlp_model.eval()

gcn_model = GCNLinkPredictor(131, 64, 32)
gcn_model.load_state_dict(torch.load(os.path.join(DATA_DIR, "baseline_gcn.pth"), weights_only=True))
gcn_model.eval()

gat_model = GATLinkPredictor(131, 16, 32, heads=4)
gat_model.load_state_dict(torch.load(os.path.join(DATA_DIR, "baseline_gat.pth"), weights_only=True))
gat_model.eval()

with torch.no_grad():
    z_sage = sage.encode(x, test_msg_ei)
    z_mlp  = mlp_model.encode(x)
    z_gcn  = gcn_model.encode(x, test_msg_ei)
    z_gat  = gat_model.encode(x, test_msg_ei)

cn_adj = {}
for u, v in zip(train_msg_ei[0].tolist(), train_msg_ei[1].tolist()):
    cn_adj.setdefault(u, set()).add(v)

print("   All embeddings computed.")

def compute_metrics(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    return {
        "AUC_ROC"   : round(float(roc_auc_score(y_true, y_prob)), 4),
        "AUPRC"     : round(float(average_precision_score(y_true, y_prob)), 4),
        "Accuracy"  : round(float(accuracy_score(y_true, y_pred)), 4),
        "Precision" : round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "Recall"    : round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "F1_Score"  : round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }

def evaluate_regime(pos_ei, neg_ei):
    eli    = torch.cat([pos_ei, neg_ei], dim=1)
    y_true = np.array([1]*pos_ei.shape[1] + [0]*neg_ei.shape[1], dtype=int)
    res    = {}
    # Common Neighbors
    cn_s = np.array([len(cn_adj.get(eli[0,i].item(),set()) & cn_adj.get(eli[1,i].item(),set()))
                     for i in range(eli.shape[1])], dtype=float)
    mx = cn_s.max(); cn_s = cn_s / mx if mx > 0 else cn_s
    res["Common Neighbors"] = compute_metrics(y_true, cn_s)
    # Parametric models
    with torch.no_grad():
        res["MLP"]       = compute_metrics(y_true, torch.sigmoid(mlp_model.decode(z_mlp, eli)).numpy())
        res["GCN"]       = compute_metrics(y_true, torch.sigmoid(gcn_model.decode(z_gcn, eli)).numpy())
        res["GAT"]       = compute_metrics(y_true, torch.sigmoid(gat_model.decode(z_gat, eli)).numpy())
        res["GraphSAGE"] = compute_metrics(y_true, torch.sigmoid(sage.decode(z_sage, eli)).numpy())
    return res

# ── 7. Evaluate ───────────────────────────────────────────────────────────────
print("\n[7/7] Running three-regime evaluation ...")
REGIME_LABELS = [
    "Regime I: Uniform Random",
    "Regime II: Degree-Matched",
    "Regime III: 2-Hop Hard (Bio Near-Miss)",
]
REGIME_NEGS = [test_neg_unif, test_neg_dm, test_neg_hard]
MODEL_NAMES  = ["Common Neighbors", "MLP", "GCN", "GAT", "GraphSAGE"]

all_results = {}
for label, neg_ei in zip(REGIME_LABELS, REGIME_NEGS):
    print(f"\n  -- {label}")
    results = evaluate_regime(test_pos_ei, neg_ei)
    all_results[label] = results
    for model, m in results.items():
        print(f"     {model:<22s}  AUC={m['AUC_ROC']:.4f}  AUPRC={m['AUPRC']:.4f}  F1={m['F1_Score']:.4f}")

# ── Save JSON & CSV ───────────────────────────────────────────────────────────
json_path = os.path.join(OUT_DIR, "hard_negative_benchmark.json")
with open(json_path, "w") as f: json.dump(all_results, f, indent=2)
print(f"\n  Saved: {json_path}")

csv_rows = []
for regime, models in all_results.items():
    for model, m in models.items():
        csv_rows.append({"Regime": regime, "Model": model, **m})
csv_df = pd.DataFrame(csv_rows)
csv_path = os.path.join(OUT_DIR, "hard_negative_benchmark.csv")
csv_df.to_csv(csv_path, index=False)
print(f"  Saved: {csv_path}")

# ── Pivot (Table VII) ─────────────────────────────────────────────────────────
pivot = csv_df.pivot_table(index="Model", columns="Regime", values="AUC_ROC").round(4)
pivot = pivot.reindex(MODEL_NAMES)
print("\n" + "=" * 70)
print("  Table VII: AUC-ROC by Model x Negative Sampling Regime")
print("=" * 70)
print(pivot.to_string())
print("=" * 70)

# ── Bar Chart ─────────────────────────────────────────────────────────────────
REGIME_SHORT = ["Uniform Random", "Degree-Matched", "2-Hop Hard Bio"]
MODEL_COLORS = {
    "Common Neighbors": "#E05C5C",
    "MLP"             : "#F4A15D",
    "GCN"             : "#5DA8F4",
    "GAT"             : "#7BC67A",
    "GraphSAGE"       : "#9D6BE0",
}
x_pos = np.arange(len(REGIME_SHORT))
width = 0.15
fig, ax = plt.subplots(figsize=(13, 6.5), dpi=300)
for i, model in enumerate(MODEL_NAMES):
    aucs   = [all_results[rl][model]["AUC_ROC"] for rl in REGIME_LABELS]
    offset = (i - len(MODEL_NAMES)/2 + 0.5) * width
    bars   = ax.bar(x_pos + offset, aucs, width*0.9,
                    label=model, color=MODEL_COLORS[model],
                    edgecolor="#222222", linewidth=0.8, alpha=0.92)
    for bar, val in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color="#000000")

ax.set_xticks(x_pos)
ax.set_xticklabels(REGIME_SHORT, fontsize=13, fontweight="bold", color="#000000")
ax.set_ylabel("AUC-ROC", fontsize=14, fontweight="bold", labelpad=10, color="#000000")
ax.set_xlabel("Negative Sampling Regime", fontsize=14, fontweight="bold", labelpad=10, color="#000000")
ax.set_title("Model Robustness Across Negative Sampling Regimes\n"
             "(Regime I -> II -> III: Increasing Biological Difficulty)",
             fontsize=15, fontweight="bold", pad=14, color="#000000")
ax.set_ylim(0.0, 1.25)
ax.set_yticks(np.arange(0.0, 1.21, 0.2))
ax.axhline(0.5, color="#555555", linestyle="--", linewidth=1.4, alpha=0.8, label="Random Baseline")
ax.legend(fontsize=10.5, loc="upper right", framealpha=0.95, edgecolor="#cbd5e1", ncol=3)
ax.tick_params(axis="both", labelsize=12, labelcolor="#000000")
for tick in ax.get_xticklabels():
    tick.set_fontweight("bold")
for tick in ax.get_yticklabels():
    tick.set_fontweight("bold")

for spine in ax.spines.values():
    spine.set_edgecolor("#000000")
    spine.set_linewidth(1.3)

ax.grid(axis="y", alpha=0.35, linestyle=":", color="#94a3b8")
fig.tight_layout()
plot_path = os.path.join(OUT_DIR, "hard_negative_regime_bar.png")
fig.savefig(plot_path, dpi=300, bbox_inches="tight")
paper_img_path = os.path.join(PROJECT_DIR, "..", "Rapid_AI_Drug_Repurposing__A_GraphSAGE_based_Clinical_Discovery_Lab_for_Drug_Repurposing_on_PrimeKG", "images", "hard_negative_regime_bar.png")
if os.path.exists(os.path.dirname(paper_img_path)):
    fig.savefig(paper_img_path, dpi=300, bbox_inches="tight")
    print(f"  Plot synced to paper: {paper_img_path}")
plt.close(fig)
print(f"  Plot saved: {plot_path}")

# ── Summary ───────────────────────────────────────────────────────────────────
sage_unif = all_results["Regime I: Uniform Random"]["GraphSAGE"]["AUC_ROC"]
sage_dm_v = all_results["Regime II: Degree-Matched"]["GraphSAGE"]["AUC_ROC"]
sage_hard_v = all_results["Regime III: 2-Hop Hard (Bio Near-Miss)"]["GraphSAGE"]["AUC_ROC"]
cn_hard_v   = all_results["Regime III: 2-Hop Hard (Bio Near-Miss)"]["Common Neighbors"]["AUC_ROC"]

print("\n" + "=" * 70)
print("  BENCHMARK SUMMARY")
print("=" * 70)
print(f"  2-Hop near-miss pool           : {len(near_miss_unique):,}")
print(f"  Pairs per regime               : {N_pos:,}")
print(f"  GraphSAGE AUC Uniform          : {sage_unif:.4f} ({sage_unif*100:.2f}%)")
print(f"  GraphSAGE AUC Degree-Matched   : {sage_dm_v:.4f} ({sage_dm_v*100:.2f}%)")
print(f"  GraphSAGE AUC 2-Hop Hard       : {sage_hard_v:.4f} ({sage_hard_v*100:.2f}%)")
print(f"  AUC drop (Uniform -> Hard)     : {(sage_unif - sage_hard_v)*100:.2f}%")
print(f"  Common Neighbors AUC 2-Hop     : {cn_hard_v:.4f} ({cn_hard_v*100:.2f}%)")
print("=" * 70)
