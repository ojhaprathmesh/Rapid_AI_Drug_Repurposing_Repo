"""
legacy_plots.py
================
Supplementary research diagrams: Sunburst relation distribution,
detailed 11-metric evaluation bar chart, and system pipeline infographic.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

VIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(VIS_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(PROJECT_DIR, ".."))
REPORT_FIG_DIR = os.path.join(REPO_ROOT, "docs", "report_figures")

def gen_detailed_metrics_bar():
    """Generates 11-metric comprehensive performance breakdown."""
    metrics = ["Accuracy", "Precision\n(Pos)", "Precision\n(Neg)", "Recall\n(Pos)",
               "Recall\n(Neg)", "F1-Score\n(Pos)", "F1-Score\n(Neg)",
               "Macro\nPrecision", "Macro\nRecall", "Macro\nF1", "AUC-ROC"]
    values  = [0.9197, 0.9071, 0.9327, 0.9351, 0.9043,
               0.9209, 0.9183, 0.9199, 0.9197, 0.9196, 0.9734]
    colors  = ["#8B5CF6", "#3B82F6", "#06B6D4", "#10B981", "#22C55E",
               "#EAB308", "#F59E0B", "#F97316", "#EF4444", "#EC4899", "#6366F1"]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    bars = ax.bar(range(len(metrics)), [v * 100 for v in values], color=colors,
                  edgecolor="white", linewidth=1.5, width=0.75, alpha=0.9)

    for bar, val in zip(bars, values):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, h + 1.2, f"{val*100:.1f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color="#1E293B")

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, fontsize=10, fontweight="bold", color="#334155")
    ax.set_ylim(0, 115)
    ax.set_ylabel("Score (%)", fontsize=12, fontweight="bold", color="#0F172A")
    ax.set_title("Comprehensive Multi-Metric Evaluation (PrimeKG Benchmark)", fontsize=14, fontweight="bold", pad=15)
    ax.grid(axis='y', linestyle=':', alpha=0.5)

    out = os.path.join(REPORT_FIG_DIR, "detailed_metrics_bar.png")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] Generated: {out}")

if __name__ == "__main__":
    gen_detailed_metrics_bar()
