"""
Sunburst-style circular chart showing DRKG relation types
with Train / Validation / Test distribution.
Redesigned for maximum clarity and readability.
"""
import os, json, torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, ConnectionPatch
import matplotlib.patheffects as pe
from collections import Counter

# ── Paths ────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = BASE_DIR
DATA_DIR    = os.path.join(PROJECT_DIR, "preprocessed_data")
OUT_DIR     = os.path.join(PROJECT_DIR, "publication_graphs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Colors ───────────────────────────────────
BG      = "#FFFFFF"
TRAIN_C = "#4F46E5"   # deep indigo
VAL_C   = "#DC2626"   # vivid red
TEST_C  = "#16A34A"   # rich green

# Only 3 canonical relation types (forward direction) for cleaner display
REL_COLORS = {
    "disease_protein":  "#FF6D00",
    "indication":       "#AB47BC",
    "drug_protein":     "#1E88E5",
}
REL_LABELS = {
    "disease_protein":  "Disease - Protein",
    "indication":       "Drug - Disease\n(Indication)",
    "drug_protein":     "Drug - Protein",
}

print("Loading data...")

with open(os.path.join(DATA_DIR, "edge_rel_map.json"), "r") as f:
    rel_map = json.load(f)
rev_rel_map = {v: k for k, v in rel_map.items()}

edge_type  = torch.load(os.path.join(DATA_DIR, "edge_type.pt"),  weights_only=False)
train_data = torch.load(os.path.join(DATA_DIR, "train_data.pt"), weights_only=False)
val_data   = torch.load(os.path.join(DATA_DIR, "val_data.pt"),   weights_only=False)
test_data  = torch.load(os.path.join(DATA_DIR, "test_data.pt"),  weights_only=False)

# Merge forward + reverse into canonical types
rel_counts = Counter()
for et in edge_type.numpy():
    name = rev_rel_map[et].replace("_rev", "")
    rel_counts[name] += 1

sorted_rels = sorted(rel_counts.keys(), key=lambda x: rel_counts[x], reverse=True)
total_edges = sum(rel_counts.values())

train_total = int(train_data.edge_label.shape[0])
val_total   = int(val_data.edge_label.shape[0])
test_total  = int(test_data.edge_label.shape[0])
grand_total = train_total + val_total + test_total
train_ratio = train_total / grand_total
val_ratio   = val_total / grand_total
test_ratio  = test_total / grand_total

print("Building chart...")

# ── Figure setup ─────────────────────────────
fig = plt.figure(figsize=(14, 14))
fig.patch.set_facecolor(BG)
ax = fig.add_subplot(111, projection="polar")
ax.set_facecolor(BG)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.axis("off")

# ─────────────────────────────────────────────
# RING 1 (Inner): Relation types
# ─────────────────────────────────────────────
inner_bottom = 0.30
inner_width  = 0.25
gap          = 0.015  # gap between segments in radians

segments = []
start = 0.0
for rel in sorted_rels:
    count = rel_counts[rel]
    sweep = (count / total_edges) * 2 * np.pi - gap
    mid   = start + sweep / 2 + gap / 2

    color = REL_COLORS[rel]
    ax.bar(start + sweep/2 + gap/2, inner_width, width=sweep, bottom=inner_bottom,
           color=color, edgecolor="white", linewidth=2.5, alpha=0.92)

    segments.append((start + gap/2, sweep, count, rel, mid))
    start += sweep + gap

# ─────────────────────────────────────────────
# RING 2 (Outer): Train / Val / Test per relation
# ─────────────────────────────────────────────
outer_bottom = inner_bottom + inner_width + 0.04
outer_width  = 0.28

for (start_a, sweep, count, rel, mid) in segments:
    tr_sweep  = sweep * train_ratio
    val_sweep = sweep * val_ratio
    tst_sweep = sweep * test_ratio

    ax.bar(start_a + tr_sweep/2, outer_width, width=tr_sweep, bottom=outer_bottom,
           color=TRAIN_C, edgecolor="white", linewidth=1.0, alpha=0.85)
    ax.bar(start_a + tr_sweep + val_sweep/2, outer_width, width=val_sweep, bottom=outer_bottom,
           color=VAL_C, edgecolor="white", linewidth=1.0, alpha=0.85)
    ax.bar(start_a + tr_sweep + val_sweep + tst_sweep/2, outer_width, width=tst_sweep, bottom=outer_bottom,
           color=TEST_C, edgecolor="white", linewidth=1.0, alpha=0.85)

# ─────────────────────────────────────────────
# LABELS: Outside the rings with leader lines
# ─────────────────────────────────────────────
label_radius = outer_bottom + outer_width + 0.12

for (start_a, sweep, count, rel, mid) in segments:
    color = REL_COLORS[rel]
    label = REL_LABELS[rel]
    pct   = count / total_edges * 100

    angle_deg = np.degrees(mid)

    # Determine horizontal alignment based on angle
    if angle_deg < 10 or angle_deg > 350:
        ha, offset = "center", 0
    elif angle_deg <= 180:
        ha, offset = "left", 0.02
    else:
        ha, offset = "right", -0.02

    # Leader line from ring edge to label
    line_start = outer_bottom + outer_width + 0.02
    line_end   = label_radius - 0.04

    ax.plot([mid, mid], [line_start, line_end],
            color=color, lw=1.8, alpha=0.7)
    ax.scatter([mid], [line_end + 0.01], s=25, color=color, zorder=5)

    # Label text
    ax.text(mid + offset, label_radius + 0.02,
            f"{label}\n{count:,} edges ({pct:.1f}%)",
            ha=ha, va="center", fontsize=11, fontweight="bold",
            color=color, linespacing=1.4,
            path_effects=[pe.withStroke(linewidth=3, foreground="white")])

# ─────────────────────────────────────────────
# CENTER
# ─────────────────────────────────────────────
ax.text(0, 0.05, "DRKG", ha="center", va="center", fontsize=30,
        fontweight="bold", color="#1A1A2E",
        path_effects=[pe.withStroke(linewidth=4, foreground="white")])
ax.text(0, -0.08, f"{total_edges:,}\nedges (incl. reverse)", ha="center", va="center",
        fontsize=12, color="#666666", linespacing=1.5)

# ─────────────────────────────────────────────
# LEGENDS
# ─────────────────────────────────────────────
# Split legend (top)
split_legend = [
    Patch(facecolor=TRAIN_C, edgecolor="white", lw=1.5,
          label=f"  Train Set  -  {train_total:,}  ({train_ratio*100:.0f}%)"),
    Patch(facecolor=VAL_C, edgecolor="white", lw=1.5,
          label=f"  Validation Set  -  {val_total:,}  ({val_ratio*100:.0f}%)"),
    Patch(facecolor=TEST_C, edgecolor="white", lw=1.5,
          label=f"  Test Set  -  {test_total:,}  ({test_ratio*100:.0f}%)"),
]
leg1 = fig.legend(handles=split_legend, loc="upper center",
                  fontsize=13, ncol=3, frameon=True, handlelength=2.5,
                  facecolor="white", edgecolor="#E0E0E0",
                  bbox_to_anchor=(0.5, 0.96))

# Ring legend (bottom)
ring_desc = [
    Patch(facecolor="#CCCCCC", edgecolor="white",
          label="Inner Ring:  Relation Types"),
    Patch(facecolor="#AAAAAA", edgecolor="white",
          label="Outer Ring:  Train / Val / Test Split"),
]
leg2 = fig.legend(handles=ring_desc, loc="lower center",
                  fontsize=11, ncol=2, frameon=True, handlelength=2,
                  facecolor="white", edgecolor="#E0E0E0",
                  bbox_to_anchor=(0.5, 0.03))

fig.suptitle("Distribution of DRKG Relations in Train, Test, and Validation Sets",
             fontsize=17, fontweight="bold", color="#1A1A2E", y=0.99)

out_path = os.path.join(OUT_DIR, "7_sunburst_relation_split.png")
fig.savefig(out_path, dpi=200, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")
