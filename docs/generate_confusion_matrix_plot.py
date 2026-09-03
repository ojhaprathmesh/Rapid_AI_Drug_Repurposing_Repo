import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-white' if 'seaborn-v0_8-white' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

# Confusion matrix values
# Rows: Actual (Negative: 0, Positive: 1)
# Cols: Predicted (Negative: 0, Positive: 1)
cm = np.array([
    [850, 90],
    [61, 879]
])
cm_percent = cm / np.sum(cm) * 100

fig, ax = plt.subplots(figsize=(7.2, 5.8), dpi=300)
fig.patch.set_facecolor('#FFFFFF')
ax.set_facecolor('#FFFFFF')

# Lively, modern multi-hue colormap: Soft ice cream -> Vibrant Sky/Cyan -> Deep Vibrant Royal Indigo
colors_list = ["#F0F4FA", "#D7E3FC", "#ABC4FF", "#7286D3", "#4361EE", "#3A0CA3"]
custom_cmap = mcolors.LinearSegmentedColormap.from_list("lively_indigo_cyan", colors_list, N=256)

# Heatmap display
im = ax.imshow(cm, interpolation='nearest', cmap=custom_cmap)

# Add stylized colorbar
cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=9.5, color="#2B2D42")
cbar.outline.set_edgecolor('#CBD5E1')
cbar.outline.set_linewidth(0.8)

# Format annotations with high readability and lively contrast
labels = [
    [f"True Negative (TN)\n\n{cm[0,0]:,}\n({cm_percent[0,0]:.1f}%)", 
     f"False Positive (FP)\n\n{cm[0,1]:,}\n({cm_percent[0,1]:.1f}%)"],
    [f"False Negative (FN)\n\n{cm[1,0]:,}\n({cm_percent[1,0]:.1f}%)", 
     f"True Positive (TP)\n\n{cm[1,1]:,}\n({cm_percent[1,1]:.1f}%)"]
]

thresh = (cm.max() + cm.min()) / 2.0
for i in range(2):
    for j in range(2):
        text_color = "#FFFFFF" if cm[i, j] > thresh else "#0F172A"
        sub_color = "#E0E7FF" if cm[i, j] > thresh else "#475569"
        
        parts = labels[i][j].split("\n\n")
        title_str = parts[0]
        val_str = parts[1]
        
        ax.text(j, i - 0.12, title_str, ha="center", va="center",
                color=sub_color, fontsize=10.5, fontweight="bold")
        ax.text(j, i + 0.12, val_str, ha="center", va="center",
                color=text_color, fontsize=13, fontweight="bold")

# Add clean cell dividers and multi-line ticks for clean spacing
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Predicted\nNon-Interaction", "Predicted\nIndication"], fontsize=10.5, fontweight="bold", color="#1E293B")
ax.set_yticklabels(["True\nNon-Interaction", "True\nIndication"], fontsize=10.5, fontweight="bold", color="#1E293B")

ax.set_xticks(np.arange(2) - 0.5, minor=True)
ax.set_yticks(np.arange(2) - 0.5, minor=True)
ax.grid(which="minor", color="white", linestyle='-', linewidth=4)
ax.tick_params(which="minor", bottom=False, left=False)

ax.set_title("Empirical Confusion Matrix on Held-Out Test Set\n(N = 1,880 PrimeKG Drug-Disease Pairs)", 
             fontsize=13, fontweight='bold', pad=16, color="#0F172A")
ax.set_xlabel("Predicted Class", fontsize=11, fontweight='bold', labelpad=12, color="#334155")
ax.set_ylabel("Actual Biological Ground Truth", fontsize=11, fontweight='bold', labelpad=12, color="#334155")

# Summary banner box with clean single line text
summary_text = "Sensitivity (Recall): 93.51%  •  Specificity: 90.43%  •  Precision: 90.71%  •  F1-Score: 92.09%"
plt.figtext(0.5, -0.05, summary_text, wrap=False, horizontalalignment='center', 
            fontsize=9.2, fontweight='bold', color='#1E293B',
            bbox=dict(boxstyle='round,pad=0.55', facecolor='#F0F4FF', edgecolor='#4361EE', linewidth=1.2))

plt.tight_layout()

# Save to target locations
out1 = "Rapid_AI_Drug_Repurposing__A_GraphSAGE_based_Clinical_Discovery_Lab_for_Drug_Repurposing_on_PrimeKG/confusion_matrix.png"
out2 = "Rapid_AI_Repurposing/evaluation_outputs/confusion_matrix.png"
out3 = "docs/report_figures/confusion_matrix.png"

plt.savefig(out1, dpi=300, bbox_inches='tight')
plt.savefig(out2, dpi=300, bbox_inches='tight')
plt.savefig(out3, dpi=300, bbox_inches='tight')

print("Lively confusion matrix successfully generated at:")
print(f"  {out1}")
print(f"  {out2}")
print(f"  {out3}")
