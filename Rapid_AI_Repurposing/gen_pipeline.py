"""
Pipeline methodology diagram for the DRKG Drug Repurposing project.
Multi-stage infographic showing the full system from data to evaluation.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, ArrowStyle
import matplotlib.patheffects as pe

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "publication_graphs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Helper functions ─────────────────────────
def rounded_box(ax, x, y, w, h, label, color, fontsize=9, sublabel="",
                alpha=0.18, border_alpha=1.0, text_color=None, bold=True):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                          facecolor=color, alpha=alpha,
                          edgecolor=color, linewidth=2, linestyle="-",
                          clip_on=False, zorder=2)
    box.set_edgecolor((*matplotlib.colors.to_rgb(color), border_alpha))
    ax.add_patch(box)
    tc = text_color or color
    fw = "bold" if bold else "normal"
    if sublabel:
        ax.text(x + w/2, y + h/2 + 0.25, label, ha="center", va="center",
                fontsize=fontsize, fontweight=fw, color=tc, zorder=3,
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        ax.text(x + w/2, y + h/2 - 0.3, sublabel, ha="center", va="center",
                fontsize=fontsize - 2, color="#666666", zorder=3, style="italic")
    else:
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=fontsize, fontweight=fw, color=tc, zorder=3,
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])

def stage_label(ax, y, label, color, fontsize=11):
    """Vertical stage label on the left side with colored background"""
    box = FancyBboxPatch((-1.8, y - 1.5), 1.6, 3.0, boxstyle="round,pad=0.15",
                          facecolor=color, alpha=0.85, edgecolor="white",
                          linewidth=1.5, clip_on=False, zorder=4)
    ax.add_patch(box)
    ax.text(-1.0, y, label, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color="white", rotation=90, zorder=5)

def arrow_down(ax, x, y1, y2, color="#888888", lw=2.5):
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=20), zorder=1)

def arrow_right(ax, x1, x2, y, color="#888888", lw=2):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=16), zorder=1)

def arrow_custom(ax, x1, y1, x2, y2, color="#888888", lw=2, style="-|>"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                mutation_scale=16, connectionstyle="arc3,rad=0.1"),
                zorder=1)

# ── Create figure ────────────────────────────
fig, ax = plt.subplots(figsize=(16, 28))
fig.patch.set_facecolor("#FEFEFE")
ax.set_facecolor("#FEFEFE")
ax.set_xlim(-2.5, 18)
ax.set_ylim(-2, 42)
ax.axis("off")

# ── Title ────────────────────────────────────
ax.text(8, 41.5, "Drug Repurposing Pipeline using GraphSAGE on DRKG",
        ha="center", va="center", fontsize=20, fontweight="bold", color="#1A1A2E")
ax.text(8, 40.8, "End-to-End Methodology: From Biomedical Data to Novel Drug-Disease Predictions",
        ha="center", va="center", fontsize=12, color="#666666", style="italic")

# =============================================================
# STAGE 1: Knowledge Graph & Data Sources  (y ~ 36-40)
# =============================================================
stage_label(ax, 38, "Data Sources &\nKnowledge Graph", "#E65100")

# Data source boxes (left side)
rounded_box(ax, 0.5, 38.5, 3.0, 1.2, "DRKG", "#E65100", fontsize=13,
            sublabel="Drug Repurposing KG", alpha=0.2)
rounded_box(ax, 0.5, 37.0, 3.0, 1.2, "DrugBank", "#D32F2F", fontsize=11,
            sublabel="Drug Features & Targets")
rounded_box(ax, 0.5, 35.5, 3.0, 1.2, "Disease Ontology", "#7B1FA2", fontsize=10,
            sublabel="MONDO + UMLS Desc.")

# Arrow from sources to graph
arrow_right(ax, 3.5, 5.0, 38.1, "#E65100", 2.5)
arrow_right(ax, 3.5, 5.0, 37.6, "#D32F2F", 2.5)
arrow_right(ax, 3.5, 5.0, 36.1, "#7B1FA2", 2.5)

# Knowledge Graph visualization (right side) - mini network
kg_cx, kg_cy = 10.5, 37.5
kg_box = FancyBboxPatch((5.2, 35.2), 10.5, 4.8, boxstyle="round,pad=0.3",
                          facecolor="#FFF8E1", alpha=0.7, edgecolor="#FFB300",
                          linewidth=2, clip_on=False, zorder=1)
ax.add_patch(kg_box)
ax.text(10.5, 39.6, "Biomedical Knowledge Graph", ha="center", fontsize=12,
        fontweight="bold", color="#E65100")

# Graph nodes
drug_nodes = [(6.5, 37.8), (7.0, 36.5), (6.2, 36.0)]
protein_nodes = [(9.5, 38.2), (10.5, 37.0), (9.0, 36.5), (11.0, 38.5)]
disease_nodes = [(13.0, 37.8), (14.0, 36.8), (13.5, 35.8)]

for (nx, ny) in drug_nodes:
    c = Circle((nx, ny), 0.38, facecolor="#2196F3", edgecolor="white", lw=2, zorder=3)
    ax.add_patch(c)
    ax.text(nx, ny, "D", ha="center", va="center", fontsize=9, fontweight="bold", color="white", zorder=4)

for (nx, ny) in protein_nodes:
    c = Circle((nx, ny), 0.38, facecolor="#4CAF50", edgecolor="white", lw=2, zorder=3)
    ax.add_patch(c)
    ax.text(nx, ny, "P", ha="center", va="center", fontsize=9, fontweight="bold", color="white", zorder=4)

for (nx, ny) in disease_nodes:
    c = Circle((nx, ny), 0.38, facecolor="#FF5722", edgecolor="white", lw=2, zorder=3)
    ax.add_patch(c)
    ax.text(nx, ny, "Dis", ha="center", va="center", fontsize=8, fontweight="bold", color="white", zorder=4)

# Edges in the mini graph
edges = [
    (drug_nodes[0], protein_nodes[0]), (drug_nodes[0], protein_nodes[1]),
    (drug_nodes[1], protein_nodes[2]), (drug_nodes[2], protein_nodes[1]),
    (protein_nodes[0], disease_nodes[0]), (protein_nodes[1], disease_nodes[1]),
    (protein_nodes[2], disease_nodes[2]), (protein_nodes[3], disease_nodes[0]),
    (drug_nodes[0], disease_nodes[0]), (drug_nodes[1], disease_nodes[1]),
    (protein_nodes[0], protein_nodes[1]), (protein_nodes[1], protein_nodes[2]),
]
for (n1, n2) in edges:
    ax.plot([n1[0], n2[0]], [n1[1], n2[1]], color="#AAAAAA", lw=1.2, alpha=0.6, zorder=2)

# Legend for graph nodes
ax.plot([], [], 'o', color="#2196F3", markersize=10, label="Drug (1,801)")
ax.plot([], [], 'o', color="#4CAF50", markersize=10, label="Protein (7,433)")
ax.plot([], [], 'o', color="#FF5722", markersize=10, label="Disease (1,363)")
leg = ax.legend(loc="upper right", fontsize=9, frameon=True, facecolor="white",
                edgecolor="#E0E0E0", bbox_to_anchor=(1.0, 1.0),
                bbox_transform=ax.transAxes)

# Stats text
ax.text(10.5, 35.5, "10,597 Nodes  |  80,816 Edges  |  3 Relation Types",
        ha="center", fontsize=9, color="#888888", style="italic")

# =============================================================
# STAGE 2: Feature Engineering & Preprocessing  (y ~ 29-34)
# =============================================================
arrow_down(ax, 8, 35.2, 34.0, "#FF6F00", 3)
ax.text(8.5, 34.5, "Preprocessing", fontsize=10, fontweight="bold", color="#FF6F00")

stage_label(ax, 32, "Feature\nEngineering", "#FF6F00")

# Feature boxes
rounded_box(ax, 0.5, 31.5, 3.2, 1.8, "TF-IDF\nVectorization", "#1565C0", fontsize=11,
            sublabel="128 text features", alpha=0.15)
rounded_box(ax, 4.2, 31.5, 3.2, 1.8, "StandardScaler", "#2E7D32", fontsize=11,
            sublabel="3 numeric features", alpha=0.15)
rounded_box(ax, 8.0, 31.5, 3.5, 1.8, "Feature\nConcatenation", "#6A1B9A", fontsize=11,
            sublabel="131-dim vectors", alpha=0.15)

# Processing detail boxes
rounded_box(ax, 12.0, 32.5, 3.8, 1.0, "Drug: MW, TPSA, cLogP", "#2E7D32",
            fontsize=9, alpha=0.1)
rounded_box(ax, 12.0, 31.2, 3.8, 1.0, "Text: Names + Descriptions", "#1565C0",
            fontsize=9, alpha=0.1)
rounded_box(ax, 12.0, 29.9, 3.8, 1.0, "Output: x.pt (10,597 x 131)", "#6A1B9A",
            fontsize=9, alpha=0.1)

# Arrows between feature boxes
arrow_right(ax, 3.7, 4.2, 32.4, "#555555")
arrow_right(ax, 7.4, 8.0, 32.4, "#555555")
arrow_right(ax, 11.5, 12.0, 33.0, "#555555")
arrow_right(ax, 11.5, 12.0, 31.7, "#555555")
arrow_right(ax, 11.5, 12.0, 30.4, "#555555")

# =============================================================
# STAGE 3: Graph Construction  (y ~ 24-29)
# =============================================================
arrow_down(ax, 8, 31.5, 30.0, "#00897B", 3)
ax.text(8.5, 30.7, "Graph Construction", fontsize=10, fontweight="bold", color="#00897B")

stage_label(ax, 27.5, "Graph\nConstruction", "#00897B")

rounded_box(ax, 0.5, 27.0, 3.5, 2.2, "PyG Data Object", "#00897B", fontsize=11,
            sublabel="Undirected Graph", alpha=0.15)
rounded_box(ax, 4.5, 27.0, 3.5, 2.2, "Disjoint Splitting", "#00897B", fontsize=11,
            sublabel="80/10/10 Split", alpha=0.15)
rounded_box(ax, 8.8, 27.0, 3.5, 2.2, "Negative Sampling", "#00897B", fontsize=11,
            sublabel="Collision-Free", alpha=0.15)

# Split detail boxes
rounded_box(ax, 13.0, 28.3, 3.0, 0.8, "Train: 15,020", "#4F46E5", fontsize=9, alpha=0.15)
rounded_box(ax, 13.0, 27.3, 3.0, 0.8, "Val: 1,876", "#DC2626", fontsize=9, alpha=0.15)
rounded_box(ax, 13.0, 26.3, 3.0, 0.8, "Test: 1,880", "#16A34A", fontsize=9, alpha=0.15)

arrow_right(ax, 4.0, 4.5, 28.1, "#555555")
arrow_right(ax, 8.0, 8.8, 28.1, "#555555")
arrow_right(ax, 12.3, 13.0, 28.7, "#555555")
arrow_right(ax, 12.3, 13.0, 27.7, "#555555")
arrow_right(ax, 12.3, 13.0, 26.7, "#555555")

# =============================================================
# STAGE 4: GraphSAGE Model Architecture  (y ~ 19-25)
# =============================================================
arrow_down(ax, 6, 27.0, 25.5, "#1565C0", 3)
ax.text(6.5, 26.2, "Model Training", fontsize=10, fontweight="bold", color="#1565C0")

stage_label(ax, 22.5, "GraphSAGE\nModel", "#1565C0")

# Model architecture flow
rounded_box(ax, 0.5, 22.0, 3.0, 2.0, "Input Features\nx.pt", "#37474F", fontsize=10,
            sublabel="131-dim per node", alpha=0.15)
rounded_box(ax, 4.0, 22.0, 3.0, 2.0, "SAGEConv\nLayer 1", "#1565C0", fontsize=11,
            sublabel="131 -> 64", alpha=0.2)
rounded_box(ax, 7.5, 22.0, 3.0, 2.0, "SAGEConv\nLayer 2", "#1565C0", fontsize=11,
            sublabel="64 -> 32", alpha=0.2)
rounded_box(ax, 11.0, 22.0, 3.0, 2.0, "Dot-Product\nDecoder", "#6A1B9A", fontsize=11,
            sublabel="Link Prediction", alpha=0.2)
rounded_box(ax, 14.5, 22.0, 2.5, 2.0, "BCE Loss", "#C62828", fontsize=11,
            sublabel="Binary CE", alpha=0.2)

arrow_right(ax, 3.5, 4.0, 23.0, "#1565C0", 2.5)
arrow_right(ax, 7.0, 7.5, 23.0, "#1565C0", 2.5)
arrow_right(ax, 10.5, 11.0, 23.0, "#6A1B9A", 2.5)
arrow_right(ax, 14.0, 14.5, 23.0, "#C62828", 2.5)

# ReLU + Message Passing annotations
ax.text(5.5, 21.7, "ReLU + Message\nPassing", ha="center", fontsize=8,
        color="#1565C0", style="italic")
ax.text(9.0, 21.7, "Node\nEmbeddings", ha="center", fontsize=8,
        color="#1565C0", style="italic")

# Training details
rounded_box(ax, 4.0, 19.5, 4.0, 1.5, "Optimizer: Adam (lr=0.01)", "#37474F",
            fontsize=9, alpha=0.08)
rounded_box(ax, 8.5, 19.5, 4.0, 1.5, "100 Epochs | Weight Decay: 1e-4", "#37474F",
            fontsize=9, alpha=0.08)

# =============================================================
# STAGE 5: Predictions & Output  (y ~ 13-18)
# =============================================================
arrow_down(ax, 8, 19.5, 18.0, "#6A1B9A", 3)
ax.text(8.5, 18.7, "Prediction & Ranking", fontsize=10, fontweight="bold", color="#6A1B9A")

stage_label(ax, 16, "Predictions &\nRationalization", "#6A1B9A")

rounded_box(ax, 0.5, 15.0, 3.5, 2.2, "Trained Model\nEmbeddings", "#1565C0", fontsize=11,
            sublabel="32-dim per node", alpha=0.15)
rounded_box(ax, 4.5, 15.0, 3.5, 2.2, "Score All\nNovel Pairs", "#6A1B9A", fontsize=11,
            sublabel="Drug x Disease", alpha=0.15)
rounded_box(ax, 8.8, 15.0, 3.5, 2.2, "Top-K Drug\nRepurposing", "#E65100", fontsize=11,
            sublabel="Sigmoid Ranking", alpha=0.15)
rounded_box(ax, 13.0, 15.0, 3.5, 2.2, "LLM Clinical\nRationale", "#2E7D32", fontsize=11,
            sublabel="LLaMA 3.2 via Ollama", alpha=0.15)

arrow_right(ax, 4.0, 4.5, 16.1, "#555555")
arrow_right(ax, 8.0, 8.8, 16.1, "#555555")
arrow_right(ax, 12.3, 13.0, 16.1, "#555555")

# =============================================================
# STAGE 6: Evaluation Metrics Bar Chart  (y ~ 3-12)
# =============================================================
arrow_down(ax, 8, 15.0, 13.5, "#C62828", 3)
ax.text(8.5, 14.2, "Model Evaluation", fontsize=10, fontweight="bold", color="#C62828")

stage_label(ax, 9, "Evaluation\nMetrics", "#C62828")

# Create a bar chart as an inset axes
ax_bar = fig.add_axes([0.15, 0.04, 0.75, 0.22])  # [left, bottom, width, height]
ax_bar.set_facecolor("#FAFAFA")

metrics = ["Accuracy", "Precision\n(Pos)", "Precision\n(Neg)", "Recall\n(Pos)",
           "Recall\n(Neg)", "F1-Score\n(Pos)", "F1-Score\n(Neg)",
           "Macro\nPrecision", "Macro\nRecall", "Macro\nF1", "AUC-ROC"]
values  = [0.9197, 0.9071, 0.9327, 0.9351, 0.9043,
           0.9209, 0.9183, 0.9199, 0.9197, 0.9196, 0.9734]
colors  = ["#8B5CF6", "#3B82F6", "#06B6D4", "#10B981", "#22C55E",
           "#EAB308", "#F59E0B", "#F97316", "#EF4444", "#EC4899", "#6366F1"]

bars = ax_bar.bar(range(len(metrics)), [v * 100 for v in values], color=colors,
                  edgecolor="white", linewidth=1.5, width=0.75, alpha=0.88)

# Value labels on top of bars
for bar, val in zip(bars, values):
    ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{val*100:.1f}%", ha="center", va="bottom", fontsize=9,
                fontweight="bold", color="#333333")

ax_bar.set_xticks(range(len(metrics)))
ax_bar.set_xticklabels(metrics, fontsize=8, rotation=45, ha="right")
ax_bar.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
ax_bar.set_ylim(80, 102)
ax_bar.set_title("Model Performance Metrics on Test Set",
                 fontsize=14, fontweight="bold", color="#1A1A2E", pad=10)
ax_bar.grid(axis="y", alpha=0.2, color="#CCCCCC")
ax_bar.spines["top"].set_visible(False)
ax_bar.spines["right"].set_visible(False)
ax_bar.spines["left"].set_color("#DDDDDD")
ax_bar.spines["bottom"].set_color("#DDDDDD")

# =============================================================
# Connecting dotted lines on left side (stage flow)
# =============================================================
stage_ys = [38, 32, 27.5, 22.5, 16, 9]
for i in range(len(stage_ys) - 1):
    ax.plot([-1.0, -1.0], [stage_ys[i] - 1.5, stage_ys[i+1] + 1.5],
            color="#CCCCCC", lw=1.5, ls="--", zorder=0)

# Save
out_path = os.path.join(OUT_DIR, "8_full_pipeline_methodology.png")
fig.savefig(out_path, dpi=180, facecolor="#FEFEFE", bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")
