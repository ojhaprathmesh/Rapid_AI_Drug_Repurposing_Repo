import os, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "preprocessed_data")
OUT = os.path.join(BASE_DIR, "publication_graphs")

train_data = torch.load(os.path.join(DATA_DIR, "train_data.pt"), weights_only=False)
val_data   = torch.load(os.path.join(DATA_DIR, "val_data.pt"),   weights_only=False)
test_data  = torch.load(os.path.join(DATA_DIR, "test_data.pt"),  weights_only=False)

train_pos = int((train_data.edge_label == 1).sum())
train_neg = int((train_data.edge_label == 0).sum())
val_pos   = int((val_data.edge_label == 1).sum())
val_neg   = int((val_data.edge_label == 0).sum())
test_pos  = int((test_data.edge_label == 1).sum())
test_neg  = int((test_data.edge_label == 0).sum())
train_tot = train_pos + train_neg
val_tot   = val_pos + val_neg
test_tot  = test_pos + test_neg
grand     = train_tot + val_tot + test_tot

BG   = "#0F0F1A"
CARD = "#1A1A2E"
GRID = "#2A2A3E"
TXT  = "#E8E8F0"
MUT  = "#8888AA"
C1   = "#00D4FF"
C2   = "#FF6B6B"
C3   = "#51CF66"
C4   = "#FFD43B"
C5   = "#CC5DE8"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "font.size": 12,
    "text.color": TXT,
})

fig = plt.figure(figsize=(14, 8))
fig.patch.set_facecolor(BG)

# ── LEFT: Nested donut chart ──
ax1 = fig.add_axes([0.02, 0.08, 0.52, 0.82])
ax1.set_facecolor(BG)

# Outer ring: Train / Val / Test totals
outer_sizes  = [train_tot, val_tot, test_tot]
outer_labels = [
    f"Training\n{train_tot:,} ({train_tot/grand*100:.1f}%)",
    f"Validation\n{val_tot:,} ({val_tot/grand*100:.1f}%)",
    f"Test\n{test_tot:,} ({test_tot/grand*100:.1f}%)",
]
outer_colors = [C1, C5, C4]

w1, t1 = ax1.pie(
    outer_sizes, labels=outer_labels, colors=outer_colors,
    explode=(0.03, 0.03, 0.03), startangle=90,
    wedgeprops=dict(width=0.38, edgecolor=BG, linewidth=3),
    textprops=dict(color=TXT, fontsize=11, fontweight="bold"),
)

# Inner ring: Positive / Negative breakdown per split
inner_sizes  = [train_pos, train_neg, val_pos, val_neg, test_pos, test_neg]
inner_colors = [
    C1 + "CC", C1 + "55",
    C5 + "CC", C5 + "55",
    C4 + "CC", C4 + "55",
]

w2, t2 = ax1.pie(
    inner_sizes, colors=inner_colors, labels=[""] * 6,
    radius=0.62, startangle=90,
    wedgeprops=dict(width=0.25, edgecolor=BG, linewidth=2),
)

# Center text
ax1.text(0, 0, f"{grand:,}\nTotal\nSamples", ha="center", va="center",
         fontsize=16, fontweight="bold", color=TXT)

ax1.set_title("Dataset Split Distribution", fontsize=16, fontweight="bold",
              color=TXT, pad=18)

# Legend for inner ring
legend_elements = [
    Patch(facecolor=C1 + "CC", edgecolor=BG, label=f"Train Positive ({train_pos:,})"),
    Patch(facecolor=C1 + "55", edgecolor=BG, label=f"Train Negative ({train_neg:,})"),
    Patch(facecolor=C5 + "CC", edgecolor=BG, label=f"Val Positive ({val_pos:,})"),
    Patch(facecolor=C5 + "55", edgecolor=BG, label=f"Val Negative ({val_neg:,})"),
    Patch(facecolor=C4 + "CC", edgecolor=BG, label=f"Test Positive ({test_pos:,})"),
    Patch(facecolor=C4 + "55", edgecolor=BG, label=f"Test Negative ({test_neg:,})"),
]
ax1.legend(handles=legend_elements, loc="lower center", fontsize=9,
           facecolor=CARD, edgecolor=GRID, labelcolor=TXT,
           ncol=3, framealpha=0.95, bbox_to_anchor=(0.5, -0.08))

# ── RIGHT: Summary cards ──
ax2 = fig.add_axes([0.58, 0.08, 0.40, 0.82])
ax2.set_facecolor(BG)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis("off")

ax2.text(5, 9.5, "Split Summary", ha="center", fontsize=16,
         fontweight="bold", color=TXT)

rows = [
    ("Training",   train_tot, train_pos, train_neg, "80%", C1),
    ("Validation", val_tot,   val_pos,   val_neg,   "10%", C5),
    ("Test",       test_tot,  test_pos,  test_neg,  "10%", C4),
]

for i, (name, tot, pos, neg, pct, color) in enumerate(rows):
    y = 7.5 - i * 2.8
    box = FancyBboxPatch((0.3, y - 0.5), 9.2, 2.2, boxstyle="round,pad=0.2",
                          facecolor=color, alpha=0.12, edgecolor=color, linewidth=2)
    ax2.add_patch(box)
    ax2.text(1.0, y + 1.1, f"{name}  ({pct})", fontsize=14, fontweight="bold", color=color)
    ax2.text(1.0, y + 0.3, f"Total: {tot:,}", fontsize=12, color=TXT)
    ax2.text(1.0, y - 0.25, f"Positive: {pos:,}   |   Negative: {neg:,}",
             fontsize=11, color=MUT)

out_path = os.path.join(OUT, "6_dataset_split_circular.png")
fig.savefig(out_path, dpi=200, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")
