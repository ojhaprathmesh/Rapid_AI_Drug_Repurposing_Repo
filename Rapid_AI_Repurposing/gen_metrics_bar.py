import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "publication_graphs")
os.makedirs(OUT_DIR, exist_ok=True)

metrics = ["Accuracy", "Precision\n(Pos)", "Precision\n(Neg)", "Recall\n(Pos)",
           "Recall\n(Neg)", "F1-Score\n(Pos)", "F1-Score\n(Neg)",
           "Macro\nPrecision", "Macro\nRecall", "Macro\nF1", "AUC-ROC"]
values  = [0.9197, 0.9071, 0.9327, 0.9351, 0.9043,
           0.9209, 0.9183, 0.9199, 0.9197, 0.9196, 0.9734]
colors  = ["#8B5CF6", "#3B82F6", "#06B6D4", "#10B981", "#22C55E",
           "#EAB308", "#F59E0B", "#F97316", "#EF4444", "#EC4899", "#6366F1"]

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("#FFFFFF")
ax.set_facecolor("#FFFFFF")

bars = ax.bar(range(len(metrics)), [v * 100 for v in values], color=colors,
              edgecolor="white", linewidth=1.5, width=0.75, alpha=0.9)

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f"{val*100:.1f}%", ha="center", va="bottom", fontsize=10,
            fontweight="bold", color="#333333")

ax.set_xticks(range(len(metrics)))
ax.set_xticklabels(metrics, fontsize=10, rotation=45, ha="right")
ax.set_ylabel("Score (%)", fontsize=12, fontweight="bold")
ax.set_ylim(80, 102)
ax.set_title("Model Performance Metrics on Test Set",
             fontsize=16, fontweight="bold", color="#1A1A2E", pad=15)
ax.grid(axis="y", alpha=0.3, color="#CCCCCC", linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#DDDDDD")
ax.spines["bottom"].set_color("#DDDDDD")

out_path = os.path.join(OUT_DIR, "9_performance_metrics_bar.png")
fig.savefig(out_path, dpi=200, facecolor="#FFFFFF", bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")
