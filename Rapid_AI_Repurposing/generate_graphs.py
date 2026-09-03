"""
generate_graphs.py - Publication-quality visualizations for the GraphSAGE project.
Generates: ROC Curve, Loss vs Epoch, Accuracy vs Epoch, Graph Stats, System Architecture
"""
import os, json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = BASE_DIR
DATAVERSE_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "dataverse_files"))
DATA_DIR = os.path.join(PROJECT_DIR, "preprocessed_data")
EVAL_DIR = os.path.join(PROJECT_DIR, "evaluation_outputs")
GRAPH_DIR = os.path.join(PROJECT_DIR, "publication_graphs")
os.makedirs(GRAPH_DIR, exist_ok=True)

# ── Color palette ────────────────────────────
C_BG      = "#0F0F1A"
C_CARD    = "#1A1A2E"
C_ACCENT1 = "#00D4FF"   # cyan
C_ACCENT2 = "#FF6B6B"   # coral
C_ACCENT3 = "#51CF66"   # green
C_ACCENT4 = "#FFD43B"   # gold
C_ACCENT5 = "#CC5DE8"   # purple
C_GRID    = "#2A2A3E"
C_TEXT    = "#E8E8F0"
C_MUTED   = "#8888AA"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "font.size": 12,
    "text.color": C_TEXT,
    "axes.labelcolor": C_TEXT,
    "xtick.color": C_MUTED,
    "ytick.color": C_MUTED,
})

# ── Load data ────────────────────────────────
fpr   = np.load(os.path.join(EVAL_DIR, "roc_fpr.npy"))
tpr   = np.load(os.path.join(EVAL_DIR, "roc_tpr.npy"))
tl    = np.load(os.path.join(EVAL_DIR, "train_losses.npy"))
vl    = np.load(os.path.join(EVAL_DIR, "val_losses.npy"))
y_true = np.load(os.path.join(EVAL_DIR, "y_true.npy"))
y_prob = np.load(os.path.join(EVAL_DIR, "y_prob.npy"))
metrics_df = pd.read_csv(os.path.join(EVAL_DIR, "scalar_metrics.csv"))
auc_val = metrics_df["AUC_ROC"].values[0]

# ── Compute per-epoch accuracy (from stored losses via re-eval approximation) ──
# Use training data to compute per-epoch train & val accuracy
train_data = torch.load(os.path.join(DATA_DIR, "train_data.pt"), weights_only=False)
val_data   = torch.load(os.path.join(DATA_DIR, "val_data.pt"), weights_only=False)

from torch_geometric.nn import SAGEConv

class LinkPredictorSAGE(torch.nn.Module):
    def __init__(self, ic, hc, oc):
        super().__init__()
        self.conv1 = SAGEConv(ic, hc)
        self.conv2 = SAGEConv(hc, oc)
    def encode(self, x, ei):
        return self.conv2(self.conv1(x, ei).relu(), ei)
    def decode(self, z, eli):
        return (z[eli[0]] * z[eli[1]]).sum(dim=-1)

model = LinkPredictorSAGE(131, 64, 32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
criterion = torch.nn.BCEWithLogitsLoss()
msg_ei = train_data.edge_index

train_accs, val_accs = [], []
train_losses2, val_losses2 = [], []

print("Re-training 100 epochs for accuracy curve...")
for epoch in range(1, 101):
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, msg_ei)
    pred = model.decode(z, train_data.edge_label_index)
    loss = criterion(pred, train_data.edge_label)
    loss.backward()
    optimizer.step()
    train_losses2.append(loss.item())

    with torch.no_grad():
        model.eval()
        # train acc
        p_tr = torch.sigmoid(pred).numpy()
        tr_acc = ((p_tr >= 0.5).astype(int) == train_data.edge_label.numpy().astype(int)).mean()
        train_accs.append(tr_acc)
        # val acc
        zv = model.encode(val_data.x, msg_ei)
        pv = model.decode(zv, val_data.edge_label_index)
        vl_loss = criterion(pv, val_data.edge_label)
        val_losses2.append(vl_loss.item())
        p_vl = torch.sigmoid(pv).numpy()
        vl_acc = ((p_vl >= 0.5).astype(int) == val_data.edge_label.numpy().astype(int)).mean()
        val_accs.append(vl_acc)

    if epoch % 20 == 0:
        print(f"  Epoch {epoch}: train_acc={tr_acc:.4f}, val_acc={vl_acc:.4f}")

epochs = np.arange(1, 101)

# =====================================================================
# GRAPH 1 : ROC Curve
# =====================================================================
print("\n[1/5] ROC Curve...")
fig, ax = plt.subplots(figsize=(8, 7))
fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)

# fill under curve
ax.fill_between(fpr, tpr, alpha=0.15, color=C_ACCENT1)
ax.plot(fpr, tpr, color=C_ACCENT1, lw=3, label=f"GraphSAGE  AUC = {auc_val:.4f}",
        path_effects=[pe.Stroke(linewidth=5, foreground=C_ACCENT1+"40"), pe.Normal()])
ax.plot([0, 1], [0, 1], "--", color=C_MUTED, lw=1.5, label="Random Baseline (AUC = 0.50)")

# annotate AUC
ax.annotate(f"AUC = {auc_val:.4f}",
            xy=(0.45, 0.55), fontsize=28, fontweight="bold",
            color=C_ACCENT1, alpha=0.25, ha="center")

ax.set_xlabel("False Positive Rate (FPR)", fontsize=14, fontweight="bold")
ax.set_ylabel("True Positive Rate (TPR)", fontsize=14, fontweight="bold")
ax.set_title("ROC Curve - Drug-Disease Link Prediction", fontsize=16, fontweight="bold", pad=15)
ax.legend(loc="lower right", fontsize=12, facecolor=C_CARD, edgecolor=C_GRID, framealpha=0.9)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.grid(True, alpha=0.15, color=C_GRID)
for spine in ax.spines.values():
    spine.set_color(C_GRID)
fig.tight_layout()
fig.savefig(os.path.join(GRAPH_DIR, "1_roc_curve.png"), dpi=200, facecolor=C_BG)
plt.close(fig)
print("  Saved 1_roc_curve.png")

# =====================================================================
# GRAPH 2 : Loss vs Epoch
# =====================================================================
print("[2/5] Loss vs Epoch...")
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)

ax.plot(epochs, train_losses2, color=C_ACCENT2, lw=2.5, label="Training Loss",
        path_effects=[pe.Stroke(linewidth=4, foreground=C_ACCENT2+"30"), pe.Normal()])
ax.plot(epochs, val_losses2, color=C_ACCENT1, lw=2.5, label="Validation Loss",
        path_effects=[pe.Stroke(linewidth=4, foreground=C_ACCENT1+"30"), pe.Normal()])
ax.fill_between(epochs, train_losses2, alpha=0.08, color=C_ACCENT2)
ax.fill_between(epochs, val_losses2, alpha=0.08, color=C_ACCENT1)

# mark best val loss
best_idx = np.argmin(val_losses2)
ax.scatter(best_idx+1, val_losses2[best_idx], s=120, color=C_ACCENT3, zorder=5, edgecolors="white", linewidths=1.5)
ax.annotate(f"Best: {val_losses2[best_idx]:.4f}\nEpoch {best_idx+1}",
            xy=(best_idx+1, val_losses2[best_idx]),
            xytext=(best_idx+20, val_losses2[best_idx]+0.08),
            fontsize=11, color=C_ACCENT3, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C_ACCENT3, lw=1.5))

ax.set_xlabel("Epoch", fontsize=14, fontweight="bold")
ax.set_ylabel("BCE Loss", fontsize=14, fontweight="bold")
ax.set_title("Training & Validation Loss per Epoch", fontsize=16, fontweight="bold", pad=15)
ax.legend(fontsize=12, facecolor=C_CARD, edgecolor=C_GRID, framealpha=0.9)
ax.grid(True, alpha=0.15, color=C_GRID)
for spine in ax.spines.values():
    spine.set_color(C_GRID)
fig.tight_layout()
fig.savefig(os.path.join(GRAPH_DIR, "2_loss_vs_epoch.png"), dpi=200, facecolor=C_BG)
plt.close(fig)
print("  Saved 2_loss_vs_epoch.png")

# =====================================================================
# GRAPH 3 : Accuracy vs Epoch
# =====================================================================
print("[3/5] Accuracy vs Epoch...")
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)

ax.plot(epochs, np.array(train_accs)*100, color=C_ACCENT4, lw=2.5, label="Training Accuracy",
        path_effects=[pe.Stroke(linewidth=4, foreground=C_ACCENT4+"30"), pe.Normal()])
ax.plot(epochs, np.array(val_accs)*100, color=C_ACCENT5, lw=2.5, label="Validation Accuracy",
        path_effects=[pe.Stroke(linewidth=4, foreground=C_ACCENT5+"30"), pe.Normal()])
ax.fill_between(epochs, np.array(train_accs)*100, alpha=0.08, color=C_ACCENT4)
ax.fill_between(epochs, np.array(val_accs)*100, alpha=0.08, color=C_ACCENT5)

best_va_idx = np.argmax(val_accs)
ax.scatter(best_va_idx+1, val_accs[best_va_idx]*100, s=120, color=C_ACCENT3, zorder=5, edgecolors="white", linewidths=1.5)
ax.annotate(f"Peak: {val_accs[best_va_idx]*100:.1f}%\nEpoch {best_va_idx+1}",
            xy=(best_va_idx+1, val_accs[best_va_idx]*100),
            xytext=(best_va_idx+15, val_accs[best_va_idx]*100-5),
            fontsize=11, color=C_ACCENT3, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C_ACCENT3, lw=1.5))

ax.set_xlabel("Epoch", fontsize=14, fontweight="bold")
ax.set_ylabel("Accuracy (%)", fontsize=14, fontweight="bold")
ax.set_title("Training & Validation Accuracy per Epoch", fontsize=16, fontweight="bold", pad=15)
ax.legend(fontsize=12, facecolor=C_CARD, edgecolor=C_GRID, framealpha=0.9, loc="lower right")
ax.grid(True, alpha=0.15, color=C_GRID)
ax.set_ylim(45, 102)
for spine in ax.spines.values():
    spine.set_color(C_GRID)
fig.tight_layout()
fig.savefig(os.path.join(GRAPH_DIR, "3_accuracy_vs_epoch.png"), dpi=200, facecolor=C_BG)
plt.close(fig)
print("  Saved 3_accuracy_vs_epoch.png")

# =====================================================================
# GRAPH 4 : Knowledge Graph Statistics (Nodes/Edges Before & After)
# =====================================================================
print("[4/5] Graph Statistics...")

# Load raw data for "before" stats
raw_nodes = pd.read_csv(os.path.join(DATAVERSE_DIR, "nodes_subset.csv"))
raw_edges = pd.read_csv(os.path.join(DATAVERSE_DIR, "edges_subset.csv"))

node_type_tensor = torch.load(os.path.join(DATA_DIR, "node_type.pt"), weights_only=False)
edge_index_tensor = torch.load(os.path.join(DATA_DIR, "edge_index.pt"), weights_only=False)

before_nodes = len(raw_nodes)
before_edges = len(raw_edges)
after_nodes  = int(node_type_tensor.shape[0])
after_edges  = int(edge_index_tensor.shape[1])

# Type breakdown
disease_count = int((node_type_tensor == 0).sum())
drug_count    = int((node_type_tensor == 1).sum())
protein_count = int((node_type_tensor == 2).sum())

fig = plt.figure(figsize=(16, 8))
fig.patch.set_facecolor(C_BG)
gs = GridSpec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.3)

# -- Panel A: Before vs After bar chart --
ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor(C_BG)

categories = ["Nodes", "Edges"]
before_vals = [before_nodes, before_edges]
after_vals  = [after_nodes, after_edges]
x_pos = np.arange(len(categories))
w = 0.32

b1 = ax1.bar(x_pos - w/2, before_vals, w, color=C_ACCENT2, alpha=0.85, label="Raw CSV", edgecolor="white", linewidth=0.5)
b2 = ax1.bar(x_pos + w/2, after_vals, w, color=C_ACCENT1, alpha=0.85, label="After Processing", edgecolor="white", linewidth=0.5)

for bar, val in zip(b1, before_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
             f"{val:,}", ha="center", va="bottom", fontsize=11, fontweight="bold", color=C_ACCENT2)
for bar, val in zip(b2, after_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
             f"{val:,}", ha="center", va="bottom", fontsize=11, fontweight="bold", color=C_ACCENT1)

ax1.set_xticks(x_pos)
ax1.set_xticklabels(categories, fontsize=13, fontweight="bold")
ax1.set_ylabel("Count", fontsize=13, fontweight="bold")
ax1.set_title("Nodes & Edges\n(Before vs After)", fontsize=14, fontweight="bold", pad=10)
ax1.legend(fontsize=10, facecolor=C_CARD, edgecolor=C_GRID, framealpha=0.9)
ax1.grid(axis="y", alpha=0.15, color=C_GRID)
for spine in ax1.spines.values():
    spine.set_color(C_GRID)

# -- Panel B: Node type distribution (donut) --
ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor(C_BG)

sizes = [disease_count, drug_count, protein_count]
labels_d = [f"Disease\n({disease_count})", f"Drug\n({drug_count})", f"Protein\n({protein_count})"]
colors_d = [C_ACCENT2, C_ACCENT1, C_ACCENT4]
wedges, texts, autotexts = ax2.pie(
    sizes, labels=labels_d, colors=colors_d, autopct="%1.1f%%",
    startangle=90, pctdistance=0.78,
    wedgeprops=dict(width=0.45, edgecolor=C_BG, linewidth=2),
    textprops=dict(color=C_TEXT, fontsize=10, fontweight="bold")
)
for at in autotexts:
    at.set_color("white")
    at.set_fontsize(10)
ax2.set_title("Node Type Distribution\n(Processed Graph)", fontsize=14, fontweight="bold", pad=10)

# -- Panel C: Split statistics table --
ax3 = fig.add_subplot(gs[2])
ax3.set_facecolor(C_BG)
ax3.axis("off")

train_pos = int((train_data.edge_label == 1).sum())
train_neg = int((train_data.edge_label == 0).sum())
val_pos   = int((val_data.edge_label == 1).sum())
val_neg   = int((val_data.edge_label == 0).sum())
test_data_t = torch.load(os.path.join(DATA_DIR, "test_data.pt"), weights_only=False)
test_pos  = int((test_data_t.edge_label == 1).sum())
test_neg  = int((test_data_t.edge_label == 0).sum())

table_data = [
    ["Split", "Positive", "Negative", "Total"],
    ["Train (80%)", f"{train_pos:,}", f"{train_neg:,}", f"{train_pos+train_neg:,}"],
    ["Val (10%)",   f"{val_pos:,}",   f"{val_neg:,}",   f"{val_pos+val_neg:,}"],
    ["Test (10%)",  f"{test_pos:,}",  f"{test_neg:,}",  f"{test_pos+test_neg:,}"],
]

tbl = ax3.table(cellText=table_data, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1.0, 1.8)

for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor(C_GRID)
    if r == 0:
        cell.set_facecolor(C_ACCENT1 + "40")
        cell.set_text_props(fontweight="bold", color="white")
    else:
        cell.set_facecolor(C_CARD)
        cell.set_text_props(color=C_TEXT)

ax3.set_title("Data Split Summary", fontsize=14, fontweight="bold", color=C_TEXT, pad=20)

fig.suptitle("Knowledge Graph - Dataset Statistics", fontsize=18, fontweight="bold",
             color=C_TEXT, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(GRAPH_DIR, "4_graph_statistics.png"), dpi=200, facecolor=C_BG)
plt.close(fig)
print("  Saved 4_graph_statistics.png")

# =====================================================================
# GRAPH 5 : System Architecture Diagram
# =====================================================================
print("[5/5] System Architecture...")

fig, ax = plt.subplots(figsize=(18, 10))
fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)
ax.set_xlim(0, 18)
ax.set_ylim(0, 10)
ax.axis("off")

def draw_box(ax, x, y, w, h, label, sublabel="", color=C_ACCENT1, alpha=0.2):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                          facecolor=color, alpha=alpha, edgecolor=color, linewidth=2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2 + (0.15 if sublabel else 0), label,
            ha="center", va="center", fontsize=11, fontweight="bold", color="white")
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.22, sublabel,
                ha="center", va="center", fontsize=8, color=C_MUTED, style="italic")

def draw_arrow(ax, x1, y1, x2, y2, color=C_MUTED, label=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2, mutation_scale=18))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my + 0.2, label, ha="center", va="bottom", fontsize=8, color=color, fontweight="bold")

# Title
ax.text(9, 9.5, "System Architecture - Drug Repurposing Pipeline", ha="center",
        fontsize=20, fontweight="bold", color=C_TEXT)
ax.text(9, 9.1, "GraphSAGE-based Link Prediction with LLM Rationalization", ha="center",
        fontsize=12, color=C_MUTED)

# --- Layer 1: Data Sources ---
ax.text(2.5, 8.3, "DATA LAYER", fontsize=10, fontweight="bold", color=C_ACCENT4, ha="center")
draw_box(ax, 0.3, 7.2, 2.0, 0.9, "DRKG Dataset", "nodes_subset.csv", C_ACCENT4, 0.25)
draw_box(ax, 2.6, 7.2, 2.2, 0.9, "Drug Features", "drug_features.csv", C_ACCENT4, 0.25)

# --- Layer 2: Preprocessing ---
ax.text(9, 8.3, "PREPROCESSING LAYER", fontsize=10, fontweight="bold", color=C_ACCENT5, ha="center")
draw_box(ax, 6.0, 7.2, 2.2, 0.9, "TF-IDF (128d)", "Text Vectorization", C_ACCENT5, 0.25)
draw_box(ax, 8.5, 7.2, 2.4, 0.9, "StandardScaler", "3 Numeric Features", C_ACCENT5, 0.25)
draw_box(ax, 11.2, 7.2, 2.3, 0.9, "Feature Concat", "131-dim Vectors", C_ACCENT5, 0.25)

# --- Layer 3: Graph Construction ---
ax.text(9, 6.0, "GRAPH CONSTRUCTION LAYER", fontsize=10, fontweight="bold", color=C_ACCENT3, ha="center")
draw_box(ax, 5.5, 4.8, 3.0, 0.9, "PyG Data Object", "Undirected Graph + Splits", C_ACCENT3, 0.25)
draw_box(ax, 9.0, 4.8, 3.5, 0.9, "Neg. Sampling", "Collision-Free Disjoint", C_ACCENT3, 0.25)

# --- Layer 4: Model ---
ax.text(9, 3.8, "MODEL LAYER", fontsize=10, fontweight="bold", color=C_ACCENT1, ha="center")
draw_box(ax, 3.5, 2.6, 2.5, 0.9, "SAGEConv L1", "131 -> 64", C_ACCENT1, 0.3)
draw_box(ax, 6.5, 2.6, 2.5, 0.9, "SAGEConv L2", "64 -> 32", C_ACCENT1, 0.3)
draw_box(ax, 9.5, 2.6, 2.5, 0.9, "Dot-Product", "Link Decoder", C_ACCENT1, 0.3)
draw_box(ax, 12.5, 2.6, 2.5, 0.9, "BCE Loss", "Binary Classification", C_ACCENT1, 0.3)

# --- Layer 5: Output ---
ax.text(9, 1.6, "OUTPUT LAYER", fontsize=10, fontweight="bold", color=C_ACCENT2, ha="center")
draw_box(ax, 2.5, 0.4, 3.0, 0.9, "Top-K Predictions", "Drug-Disease Ranking", C_ACCENT2, 0.25)
draw_box(ax, 6.0, 0.4, 3.0, 0.9, "Evaluation Suite", "AUC / F1 / ROC", C_ACCENT2, 0.25)
draw_box(ax, 9.5, 0.4, 3.0, 0.9, "LLM Rationale", "Ollama / LLaMA 3.2", C_ACCENT2, 0.25)
draw_box(ax, 13.0, 0.4, 2.8, 0.9, "Streamlit UI", "Web Dashboard", C_ACCENT2, 0.25)

# --- Arrows ---
# Data -> Preprocessing
draw_arrow(ax, 2.3, 7.2, 6.0, 7.65, C_ACCENT4)
draw_arrow(ax, 4.8, 7.65, 6.0, 7.65, C_ACCENT4)
draw_arrow(ax, 8.2, 7.65, 8.5, 7.65, C_ACCENT5)
draw_arrow(ax, 10.9, 7.65, 11.2, 7.65, C_ACCENT5)

# Preprocessing -> Graph
draw_arrow(ax, 12.3, 7.2, 8.5, 5.7, C_ACCENT5)

# Graph -> Model
draw_arrow(ax, 7.0, 4.8, 4.75, 3.5, C_ACCENT3)
draw_arrow(ax, 10.75, 4.8, 10.75, 3.5, C_ACCENT3)

# Model chain
draw_arrow(ax, 6.0, 3.05, 6.5, 3.05, C_ACCENT1)
draw_arrow(ax, 9.0, 3.05, 9.5, 3.05, C_ACCENT1)
draw_arrow(ax, 12.0, 3.05, 12.5, 3.05, C_ACCENT1)

# Model -> Output
draw_arrow(ax, 4.75, 2.6, 4.0, 1.3, C_ACCENT1)
draw_arrow(ax, 7.75, 2.6, 7.5, 1.3, C_ACCENT1)
draw_arrow(ax, 10.75, 2.6, 11.0, 1.3, C_ACCENT2)
draw_arrow(ax, 13.0, 2.6, 14.4, 1.3, C_ACCENT2)

fig.tight_layout()
fig.savefig(os.path.join(GRAPH_DIR, "5_system_architecture.png"), dpi=200, facecolor=C_BG,
            bbox_inches="tight")
plt.close(fig)
print("  Saved 5_system_architecture.png")

print(f"\nAll 5 graphs saved to: {GRAPH_DIR}")
