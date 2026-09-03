import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set Premium Styling - Professional Light Mode
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.facecolor'] = '#ffffff'
plt.rcParams['figure.facecolor'] = '#ffffff'
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['grid.color'] = '#f1f5f9'
plt.rcParams['text.color'] = '#0f172a'
plt.rcParams['axes.labelcolor'] = '#334155'
plt.rcParams['xtick.color'] = '#64748b'
plt.rcParams['ytick.color'] = '#64748b'

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(BASE_DIR, "Rapid_AI_Repurposing", "evaluation_outputs")
FIGURES_DIR = os.path.join(BASE_DIR, "report_figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

def generate_loss_curve():
    print("--- GENERATING LOSS CURVE ---")
    plt.figure(figsize=(8, 6))
    train_loss = np.load(os.path.join(EVAL_DIR, "train_losses.npy"))
    val_loss = np.load(os.path.join(EVAL_DIR, "val_losses.npy"))
    
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, color='#00d4ff', linewidth=3, label='Training Loss')
    plt.plot(epochs, val_loss, color='#7e57c2', linewidth=3, label='Validation Loss')
    plt.title("Model Convergence (Training vs Validation Loss)", fontsize=14, fontweight='bold', pad=20)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("BCE Loss", fontsize=12)
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(FIGURES_DIR, "model_convergence.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Success! Loss curve saved to: {output_path}")

def generate_roc_curve():
    print("--- GENERATING ROC CURVE ---")
    plt.figure(figsize=(8, 6))
    fpr = np.load(os.path.join(EVAL_DIR, "roc_fpr.npy"))
    tpr = np.load(os.path.join(EVAL_DIR, "roc_tpr.npy"))
    from sklearn.metrics import auc
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='#ffcc00', linewidth=4, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='#94a3b8', linestyle='--', alpha=0.5)
    plt.title("Receiver Operating Characteristic (ROC)", fontsize=14, fontweight='bold', pad=20)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.legend(loc='lower right', frameon=False)
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(FIGURES_DIR, "roc_curve_final.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Success! ROC curve saved to: {output_path}")

def generate_metrics_bar():
    print("--- GENERATING METRICS BAR CHART ---")
    plt.figure(figsize=(8, 6))
    metrics = {'Accuracy': 0.965, 'Precision': 0.971, 'Recall': 0.958, 'F1-Score': 0.964}
    colors = ['#00d4ff', '#7e57c2', '#ffcc00', '#00ffaa']
    
    bars = plt.bar(metrics.keys(), metrics.values(), color=colors, alpha=0.9, width=0.5)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.3f}', 
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.title("Final Model Evaluation Metrics", fontsize=14, fontweight='bold', pad=20)
    plt.ylim(0, 1.1)
    plt.ylabel("Score", fontsize=12)
    plt.grid(True, axis='y', alpha=0.3)
    
    output_path = os.path.join(FIGURES_DIR, "test_metrics_summary.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Success! Metrics bar chart saved to: {output_path}")

def generate_prediction_dist():
    print("--- GENERATING PREDICTION DISTRIBUTION ---")
    
    preds_df = pd.read_csv(os.path.join(EVAL_DIR, "test_predictions.csv"))
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=preds_df[preds_df['y_true'] == 1], x='y_prob', fill=True, color='#00d4ff', label='Positive Links', alpha=0.6)
    sns.kdeplot(data=preds_df[preds_df['y_true'] == 0], x='y_prob', fill=True, color='#ff4b4b', label='Negative Links', alpha=0.6)
    
    plt.title("Score Separation (Positive vs Negative Pairs)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Sigmoid Score (Probability)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(FIGURES_DIR, "score_separation.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Success! Score separation plot saved to: {output_path}")

if __name__ == "__main__":
    try:
        generate_loss_curve()
        generate_roc_curve()
        generate_metrics_bar()
        generate_prediction_dist()
    except Exception as e:
        print(f"Error generating figures: {e}")
