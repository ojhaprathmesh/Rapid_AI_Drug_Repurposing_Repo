"""
Regenerates the Rapid AI Clinical Interaction Workflow Sequence Diagram
as a clean, print-ready PNG using matplotlib only.

Design decisions for print readability:
  - 300 DPI, A3-landscape proportions (16.5 x 9 inches)
  - Title placed above actor boxes with generous padding so it never overlaps
  - White background, black outlines — looks great in B&W and colour print
  - All fonts are at least 9 pt; labels on arrows are placed in a white-filled
    rounded rectangle so they are always legible on top of life-line dashes
  - Solid arrows for synchronous calls, dashed for returns/async
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ── Canvas ────────────────────────────────────────────────────────────────────
FIG_W, FIG_H = 17, 9.0         # inches  — calibrated for 11-step diagram
DPI          = 300
FONT_FAMILY  = "DejaVu Sans"

# ── Layout constants ─────────────────────────────────────────────────────────
TITLE_Y      = 9.12            # centre of title text
BOX_TOP      = 8.72            # top edge of actor boxes
BOX_H        = 0.76            # actor box height (slightly taller)
BOX_W        = 1.90            # actor box width
BOX_BOTTOM   = BOX_TOP - BOX_H # = 7.96
LIFELINE_BOT = 1.15            # ends just below step 11

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
ax.set_xlim(0, FIG_W)
ax.set_ylim(LIFELINE_BOT - 0.08, FIG_H)  # crop bottom whitespace
ax.axis("off")
fig.patch.set_facecolor("white")

# ── Actor definitions  ────────────────────────────────────────────────────────
# Each actor: (centre_x, label_lines, box_edge_colour, label_colour, bg_colour)
actors = [
    (1.50,  ["Researcher", "(Clinician)"],          "#9B59B6", "#9B59B6", "#F5EEF8"),
    (4.30,  ["Streamlit", "Dashboard"],             "#2E86C1", "#2E86C1", "#EBF5FB"),
    (7.10,  ["Discovery", "Engine"],                "#1A8A6F", "#1A8A6F", "#E8F8F5"),
    (9.90,  ["GraphSAGE", "Model (PyG)"],           "#27AE60", "#27AE60", "#EAFAF1"),
    (12.70, ["LLaMA 3.2", "(Local Ollama)"],        "#E67E22", "#E67E22", "#FEF9E7"),
    (15.50, ["External APIs", "(PubChem/Trials)"],  "#E74C3C", "#E74C3C", "#FDEDEC"),
]

ACTOR_XS = [a[0] for a in actors]

# ── Draw title ────────────────────────────────────────────────────────────────
ax.text(
    FIG_W / 2, TITLE_Y,
    "Rapid AI Clinical Interaction Workflow Sequence Diagram",
    ha="center", va="center",
    fontsize=14, fontweight="bold", fontfamily=FONT_FAMILY,
    color="#1C1C1C",
)

# ── Draw actor boxes & lifelines ──────────────────────────────────────────────
for cx, lines, edge_col, txt_col, bg_col in actors:
    # Box
    rect = FancyBboxPatch(
        (cx - BOX_W/2, BOX_BOTTOM),
        BOX_W, BOX_H,
        boxstyle="round,pad=0.06",
        linewidth=2.8, edgecolor=edge_col, facecolor=bg_col, zorder=3,
    )
    ax.add_patch(rect)

    # Label (supports 1 or 2 lines)
    label = "\n".join(lines)
    ax.text(
        cx, BOX_BOTTOM + BOX_H/2, label,
        ha="center", va="center",
        fontsize=10.0, fontweight="bold", fontfamily=FONT_FAMILY, color=txt_col,
        multialignment="center", zorder=4,
    )

    # Dashed lifeline
    ax.plot(
        [cx, cx], [BOX_BOTTOM, LIFELINE_BOT],
        color="#AAAAAA", linewidth=1.2, linestyle=(0, (6, 4)), zorder=1,
    )

# ── Helper: draw one message arrow ───────────────────────────────────────────
def arrow(ax, y, src_idx, dst_idx, label, color, dashed=False, lw=1.6):
    """Draw a labelled arrow between two actor lifelines at height y."""
    x0 = ACTOR_XS[src_idx]
    x1 = ACTOR_XS[dst_idx]
    dx = x1 - x0
    
    # Slight inset so arrowhead lands just at the lifeline, not past it
    pad = 0.06 * (1 if dx > 0 else -1)

    ls  = (0, (5, 3)) if dashed else "solid"
    arrowprops = dict(
        arrowstyle="->" if not dashed else "->",
        color=color,
        lw=lw,
        linestyle=ls,
        connectionstyle="arc3,rad=0",
    )
    ax.annotate(
        "",
        xy=(x1 - pad, y),
        xytext=(x0 + pad, y),
        arrowprops=arrowprops,
        zorder=2,
    )

    # Label in a white rounded box centred above the arrow
    mid_x = (x0 + x1) / 2
    # Push label up by a small fixed offset so it doesn't sit on the line
    label_y = y + 0.12          # more vertical gap between arrow and label box
    ax.text(
        mid_x, label_y, label,
        ha="center", va="bottom",
        fontsize=8.8, fontweight="semibold", fontfamily=FONT_FAMILY, color="#1C1C1C",
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white", edgecolor=color,
            linewidth=1.6, alpha=0.97,
        ),
        zorder=5,
    )

# ── Message steps ─────────────────────────────────────────────────────────────
# Y positions, descending from just below actor boxes
Y_STEPS = [
    7.18,  # 1  — lowered to clear actor box bottom (7.96)
    6.60,  # 2
    6.02,  # 3
    5.50,  # 4
    4.94,  # 5
    4.38,  # 6
    3.82,  # 7
    3.18,  # 8
    2.58,  # 9
    2.00,  # 10
    1.34,  # 11
]

# (y_idx, src, dst, label, color, dashed)
messages = [
    (0,  0, 1, "1. Select Target Disease & Threshold Regime",   "#2E86C1", False),
    (1,  1, 2, "2. Request Link Inference Task",                "#2E86C1", False),
    (2,  2, 3, "3. Extract 2-Hop Subgraph & Node Embeddings",   "#27AE60", False),
    (3,  3, 2, "4. Return Link Probability Logits (AUC = 0.9734)","#27AE60", True),
    (4,  2, 1, "5. Rank Candidates & Filter by Operational Threshold","#2E86C1", True),
    (5,  1, 0, "6. Display Interactive Bedside Radar Candidates","#9B59B6", True),
    (6,  0, 1, "7. Select Lead Molecule for Clinical Briefing",  "#9B59B6", False),
    (7,  1, 4, "8. Forward Degree-Penalized Paths to LLM",      "#E67E22", False),
    (8,  1, 5, "9. Fetch PubChem & PubMed Validation Context",  "#E74C3C", False),
    (9,  5, 1, "10. Stream Grounded Pharmacological Rationale", "#E67E22", True),
    (10, 1, 0, "11. Render Unified Bedside Repurposing Dossier", "#2E86C1", True),
]

for (yi, src, dst, lbl, col, dsh) in messages:
    arrow(ax, Y_STEPS[yi], src, dst, lbl, col, dashed=dsh)

# ── Horizontal step number tick marks on the left margin ─────────────────────
# (optional — subtle visual guide for the reader)
for i, y in enumerate(Y_STEPS):
    ax.plot([0.08, 0.20], [y, y], color="#CCCCCC", lw=0.7, zorder=0)

# ── Save ──────────────────────────────────────────────────────────────────────
import os
OUTPUTS = [
    "images/interaction_workflow.png",
    "../../../docs/report_figures/interaction_workflow.png",
]
for OUT in OUTPUTS:
    os.makedirs(os.path.dirname(OUT) if os.path.dirname(OUT) else ".", exist_ok=True)
    fig.savefig(
        OUT,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.18,
    )
    print(f"Saved \u2192 {OUT}")
plt.close(fig)
