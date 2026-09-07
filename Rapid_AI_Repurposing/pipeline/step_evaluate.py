import sys
"""
step_evaluate.py
================
Comprehensive evaluation of the trained GraphSAGE Link Prediction model.

Outputs:
  1. y_true  : actual test labels
  2. y_pred  : predicted binary labels (threshold = 0.5)
  3. y_prob  : sigmoid probabilities
  4. AUC-ROC score
  5. Precision, Recall, F1-score, Accuracy
  6. ROC curve values (FPR, TPR, thresholds)
  7. Training and Validation loss per epoch  (model is re-trained from scratch so
     every epoch's loss is captured; best weights are re-saved at the end)
  8. Top 20 predicted drug-disease pairs with highest probabilities

All results are saved to:
  Rapid_AI_Repurposing/evaluation_outputs/
"""

import os, json, time
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless – no display required
import matplotlib.pyplot as plt

from torch_geometric.nn import SAGEConv
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, accuracy_score, roc_curve, classification_report,
    average_precision_score, confusion_matrix
)

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
BASE_DIR = PROJECT_DIR
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
DATAVERSE_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "dataverse_files"))
DATA_DIR     = os.path.join(PROJECT_DIR, "preprocessed_data")
OUT_DIR      = os.path.join(PROJECT_DIR, "evaluation_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 65)
print("  GraphSAGE Link Predictor — Full Evaluation Suite")
print("=" * 65)

# ─────────────────────────────────────────────
# 1. Load Data Splits
# ─────────────────────────────────────────────
print("\n[1/5] Loading pre-built data splits …")
train_data = torch.load(os.path.join(DATA_DIR, "train_data.pt"), weights_only=False)
val_data   = torch.load(os.path.join(DATA_DIR, "val_data.pt"),   weights_only=False)
test_data  = torch.load(os.path.join(DATA_DIR, "test_data.pt"),  weights_only=False)

print(f"      Train labels : {train_data.edge_label.shape[0]:,}")
print(f"      Val   labels : {val_data.edge_label.shape[0]:,}")
print(f"      Test  labels : {test_data.edge_label.shape[0]:,}")

# ─────────────────────────────────────────────
# 2. Model Definition  (must match step3_train.py exactly)
# ─────────────────────────────────────────────
class LinkPredictorSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        src = edge_label_index[0]
        dst = edge_label_index[1]
        return (z[src] * z[dst]).sum(dim=-1)

criterion = torch.nn.BCEWithLogitsLoss()
"""
── Leakage-free message-passing adjacencies ─────────────────────────────────
Each Data split carries `msg_edge_index`: the global graph with that split's
own positive (supervision) edges excised.  Using `edge_index` (full global
graph) here would reintroduce message-passing topological label leakage.
"""
train_msg_ei = train_data.msg_edge_index   # global − val_pos − test_pos
val_msg_ei   = val_data.msg_edge_index     # same topology as train_msg_ei
test_msg_ei  = test_data.msg_edge_index    # global − test_pos

# ─────────────────────────────────────────────
# 3. Re-train from scratch — capturing every epoch's loss
#    (the saved checkpoint has only weights, not history)
# ─────────────────────────────────────────────
print("\n[2/5] Re-training model for 100 epochs (capturing per-epoch loss) …")

model     = LinkPredictorSAGE(in_channels=131, hidden_channels=64, out_channels=32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

train_losses, val_losses = [], []
best_val_loss = float("inf")
best_state    = None
EPOCHS        = 100

t0 = time.time()
for epoch in range(1, EPOCHS + 1):
    # ── Train step ──────────────────────────────
    model.train()
    optimizer.zero_grad()
    # Encode with the train-split MP graph (val+test pos edges excised)
    z    = model.encode(train_data.x, train_msg_ei)
    pred = model.decode(z, train_data.edge_label_index)
    tr_loss = criterion(pred, train_data.edge_label)
    tr_loss.backward()
    optimizer.step()

    # ── Val step (no_grad) ──────────────────────
    model.eval()
    with torch.no_grad():
        # Encode with val MP graph (val+test pos edges excised)
        z_v      = model.encode(val_data.x, val_msg_ei)
        pred_v   = model.decode(z_v, val_data.edge_label_index)
        vl_loss  = criterion(pred_v, val_data.edge_label)

    train_losses.append(tr_loss.item())
    val_losses.append(vl_loss.item())

    if vl_loss.item() < best_val_loss:
        best_val_loss = vl_loss.item()
        best_state    = {k: v.clone() for k, v in model.state_dict().items()}

    if epoch % 10 == 0:
        elapsed = time.time() - t0
        print(f"   Epoch {epoch:3d}/{EPOCHS} | "
              f"Train Loss: {tr_loss.item():.4f} | "
              f"Val Loss: {vl_loss.item():.4f}  ({elapsed:.1f}s)")

# Restore best weights
model.load_state_dict(best_state)
torch.save(best_state, os.path.join(DATA_DIR, "best_graphsage_model.pth"))
print(f"\n   >> Best val-loss: {best_val_loss:.4f}  -- weights re-saved.")

# ─────────────────────────────────────────────
# 4. Evaluate on TEST set
# ─────────────────────────────────────────────
print("\n[3/5] Evaluating on test set …")
model.eval()
with torch.no_grad():
    # Encode with test MP graph (test pos edges excised — no leakage)
    z_test   = model.encode(test_data.x, test_msg_ei)
    logits   = model.decode(z_test, test_data.edge_label_index)
    y_prob   = torch.sigmoid(logits).cpu().numpy()
    y_true   = test_data.edge_label.cpu().numpy().astype(int)

y_pred = (y_prob >= 0.5).astype(int)

# ── Metric calculations ──────────────────────
auc       = roc_auc_score(y_true, y_prob)
auprc     = average_precision_score(y_true, y_prob)
precision = precision_score(y_true, y_pred, zero_division=0)
recall    = recall_score(y_true, y_pred, zero_division=0)
f1        = f1_score(y_true, y_pred, zero_division=0)
accuracy  = accuracy_score(y_true, y_pred)
fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
npv         = tn / (tn + fn) if (tn + fn) > 0 else 0.0

print(f"\n   ── Test-Set Core Performance Metrics ──────────────────────")
print(f"   AUC-ROC          : {auc:.4f}")
print(f"   PR-AUC (AUPRC)   : {auprc:.4f}")
print(f"   Accuracy         : {accuracy:.4f}")
print(f"   Sensitivity (Rec): {recall:.4f}")
print(f"   Specificity      : {specificity:.4f}")
print(f"   Precision (PPV)  : {precision:.4f}")
print(f"   NPV              : {npv:.4f}")
print(f"   F1-Score         : {f1:.4f}")
print(f"\n   Confusion Matrix Breakdown:")
print(f"   True Positives (TP) : {tp:4d} ({tp/len(y_true)*100:5.2f}%)")
print(f"   True Negatives (TN) : {tn:4d} ({tn/len(y_true)*100:5.2f}%)")
print(f"   False Positives (FP): {fp:4d} ({fp/len(y_true)*100:5.2f}%)")
print(f"   False Negatives (FN): {fn:4d} ({fn/len(y_true)*100:5.2f}%)")
print(f"\n   Full Classification Report:")
print(classification_report(y_true, y_pred, target_names=["Negative", "Positive"]))

# Multi-threshold clinical sensitivity evaluation
thresholds_list = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
regime_map = {
    0.30: "High-Sensitivity Screening",
    0.35: "High-Sensitivity Screening",
    0.40: "High-Sensitivity Screening",
    0.45: "Balanced Prioritization",
    0.50: "Balanced Prioritization (Default)",
    0.55: "Balanced Prioritization",
    0.60: "High-Confidence Validation",
    0.65: "High-Confidence Validation",
    0.70: "High-Confidence Validation"
}

thresh_rows = []
for t_val in thresholds_list:
    t_pred = (y_prob >= t_val).astype(int)
    t_tp   = int(np.sum((y_true == 1) & (t_pred == 1)))
    t_tn   = int(np.sum((y_true == 0) & (t_pred == 0)))
    t_fp   = int(np.sum((y_true == 0) & (t_pred == 1)))
    t_fn   = int(np.sum((y_true == 1) & (t_pred == 0)))

    t_prec = t_tp / (t_tp + t_fp) if (t_tp + t_fp) > 0 else 0.0
    t_rec  = t_tp / (t_tp + t_fn) if (t_tp + t_fn) > 0 else 0.0
    t_spec = t_tn / (t_tn + t_fp) if (t_tn + t_fp) > 0 else 0.0
    t_f1   = 2 * t_prec * t_rec / (t_prec + t_rec) if (t_prec + t_rec) > 0 else 0.0
    t_acc  = (t_tp + t_tn) / len(y_true)

    thresh_rows.append({
        "threshold"          : t_val,
        "clinical_regime"    : regime_map.get(t_val, ""),
        "precision"          : round(t_prec, 4),
        "recall"             : round(t_rec, 4),
        "specificity"        : round(t_spec, 4),
        "f1_score"           : round(t_f1, 4),
        "accuracy"           : round(t_acc, 4),
        "true_positives"     : t_tp,
        "false_positives"    : t_fp,
        "false_negatives"    : t_fn,
        "true_negatives"     : t_tn
    })
thresh_df = pd.DataFrame(thresh_rows)

# ─────────────────────────────────────────────
# 5. Top-20 Drug-Disease Predictions
# ─────────────────────────────────────────────
print("\n[4/5] Generating Top-20 novel drug-disease predictions …")

x_all        = test_data.x
edge_index   = test_data.edge_index
node_type    = test_data.node_type

with open(os.path.join(DATA_DIR, "node_map.json"), "r") as f:
    node_map = json.load(f)
rev_node_map = {v: int(k) for k, v in node_map.items()}

nodes_csv = os.path.join(DATAVERSE_DIR, "nodes_subset.csv")
nodes_df  = pd.read_csv(nodes_csv)
node_name_dict = dict(zip(nodes_df["node_index"], nodes_df["node_name"]))
node_type_dict = dict(zip(nodes_df["node_index"], nodes_df["node_type"]))

drug_seq_indices    = (node_type == 1).nonzero(as_tuple=True)[0]
disease_seq_indices = (node_type == 0).nonzero(as_tuple=True)[0]

global_edges = set(
    (u.item(), v.item())
    for u, v in zip(edge_index[0], edge_index[1])
)

model.eval()
with torch.no_grad():
    """
    Novel-candidate discovery: we score ONLY (drug, disease) pairs that do NOT
    exist in the global graph (filtered by `global_edges` below), so using the
    full edge_index here does not introduce leakage for this specific task.
    """
    z_full = model.encode(x_all, edge_index)

results = []
with torch.no_grad():
    for drug_idx in drug_seq_indices:
        drug_idx = drug_idx.item()
        novel_diseases = [
            d.item() for d in disease_seq_indices
            if (drug_idx, d.item()) not in global_edges
        ]
        if not novel_diseases:
            continue

        batch_diseases     = torch.tensor(novel_diseases, dtype=torch.long)
        batch_drugs        = torch.full((len(novel_diseases),), drug_idx, dtype=torch.long)
        edge_label_index   = torch.stack([batch_drugs, batch_diseases], dim=0)
        scores             = model.decode(z_full, edge_label_index)
        probs              = torch.sigmoid(scores).cpu().numpy()

        for disease_idx, prob, raw in zip(novel_diseases, probs, scores.tolist()):
            orig_drug    = rev_node_map[drug_idx]
            orig_disease = rev_node_map[disease_idx]
            results.append({
                "drug_name"        : node_name_dict.get(orig_drug,    f"Drug_{orig_drug}"),
                "disease_name"     : node_name_dict.get(orig_disease, f"Disease_{orig_disease}"),
                "probability"      : round(float(prob), 6),
                "raw_logit"        : round(raw, 6),
            })

top20_df = (
    pd.DataFrame(results)
    .sort_values("probability", ascending=False)
    .head(20)
    .reset_index(drop=True)
)
top20_df.index += 1   # rank from 1

print("\n   Top-20 Drug-Disease Repurposing Candidates:")
print(top20_df[["drug_name", "disease_name", "probability"]].to_string())

# ─────────────────────────────────────────────
# 6. Save Everything to Disk
# ─────────────────────────────────────────────
print("\n[5/5] Saving all outputs …")

# ── NumPy arrays ────────────────────────────
np.save(os.path.join(OUT_DIR, "y_true.npy"),          y_true)
np.save(os.path.join(OUT_DIR, "y_pred.npy"),          y_pred)
np.save(os.path.join(OUT_DIR, "y_prob.npy"),          y_prob)
np.save(os.path.join(OUT_DIR, "roc_fpr.npy"),         fpr)
np.save(os.path.join(OUT_DIR, "roc_tpr.npy"),         tpr)
np.save(os.path.join(OUT_DIR, "roc_thresholds.npy"),  roc_thresholds)
np.save(os.path.join(OUT_DIR, "train_losses.npy"),    np.array(train_losses))
np.save(os.path.join(OUT_DIR, "val_losses.npy"),      np.array(val_losses))

# ── Scalar metrics CSV ──────────────────────
metrics_df = pd.DataFrame([{
    "AUC_ROC"            : round(float(auc),         4),
    "AUPRC"              : round(float(auprc),       4),
    "Accuracy"           : round(float(accuracy),    4),
    "Precision"          : round(float(precision),   4),
    "Recall"             : round(float(recall),      4),
    "Specificity"        : round(float(specificity), 4),
    "F1_Score"           : round(float(f1),          4),
    "NPV"                : round(float(npv),         4),
    "True_Positives"     : int(tp),
    "True_Negatives"     : int(tn),
    "False_Positives"    : int(fp),
    "False_Negatives"    : int(fn),
    "Total_Test_Samples" : int(len(y_true)),
}])
metrics_df.to_csv(os.path.join(OUT_DIR, "scalar_metrics.csv"), index=False)

# ── Threshold Sensitivity CSV ───────────────
thresh_df.to_csv(os.path.join(OUT_DIR, "threshold_sensitivity.csv"), index=False)

# ── Confusion Matrix CSV ────────────────────
cm_df = pd.DataFrame([
    {"Actual_Class": "Ground Truth Positive (Indication)",  "Predicted_Positive": int(tp), "Predicted_Negative": int(fn), "Total": int(tp + fn), "Class_Recall": f"{recall*100:.2f}%"},
    {"Actual_Class": "Ground Truth Negative (Non-Indication)", "Predicted_Positive": int(fp), "Predicted_Negative": int(tn), "Total": int(fp + tn), "Class_Recall": f"{specificity*100:.2f}%"}
])
cm_df.to_csv(os.path.join(OUT_DIR, "confusion_matrix.csv"), index=False)

# ── Per-epoch loss CSV ──────────────────────
loss_df = pd.DataFrame({
    "epoch"       : list(range(1, EPOCHS + 1)),
    "train_loss"  : train_losses,
    "val_loss"    : val_losses,
})
loss_df.to_csv(os.path.join(OUT_DIR, "loss_per_epoch.csv"), index=False)

# ── Top-20 CSV ──────────────────────────────
top20_df.to_csv(os.path.join(OUT_DIR, "top20_predictions.csv"), index_label="rank")

# ── Predictions detail CSV ──────────────────
pred_df = pd.DataFrame({
    "y_true" : y_true,
    "y_pred" : y_pred,
    "y_prob" : y_prob,
})
pred_df.to_csv(os.path.join(OUT_DIR, "test_predictions.csv"), index=False)

# ─────────────────────────────────────────────
# 7. Plots
# ─────────────────────────────────────────────

# ── ROC Curve ───────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color="#4F8EF7", lw=2.5,
        label=f"GraphSAGE (AUC = {auc:.4f})")
ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Classifier")
ax.set_xlabel("False Positive Rate", fontsize=13)
ax.set_ylabel("True Positive Rate",  fontsize=13)
ax.set_title("ROC Curve — Drug-Disease Link Prediction", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "roc_curve.png"), dpi=150)
plt.close(fig)

# ── Loss Curves ─────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
epochs_range = range(1, EPOCHS + 1)
ax.plot(epochs_range, train_losses, color="#F4845F", lw=2,   label="Train Loss")
ax.plot(epochs_range, val_losses,   color="#4F8EF7", lw=2,   label="Val Loss")
ax.set_xlabel("Epoch", fontsize=13)
ax.set_ylabel("BCE Loss", fontsize=13)
ax.set_title("Training & Validation Loss per Epoch", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "loss_curves.png"), dpi=150)
plt.close(fig)

# ─────────────────────────────────────────────
# 8. Final Summary Print
# ---------------------------------------------
print("\n" + "=" * 65)
print("  EVALUATION COMPLETE - Summary of All 8 Outputs")
print("=" * 65)

print(f"  1. y_true  : shape {y_true.shape}  | saved -> y_true.npy")
print(f"  2. y_pred  : shape {y_pred.shape}  | saved -> y_pred.npy")
print(f"  3. y_prob  : shape {y_prob.shape}  | saved -> y_prob.npy")
print(f"\n  4. AUC-ROC : {auc:.4f}")
print(f"\n  5. Metrics :")
print(f"       Accuracy  = {accuracy:.4f}")
print(f"       Precision = {precision:.4f}")
print(f"       Recall    = {recall:.4f}")
print(f"       F1-Score  = {f1:.4f}")
print(f"  6. ROC curve: FPR shape {fpr.shape}, TPR shape {tpr.shape}")
print(f"               saved -> roc_fpr.npy / roc_tpr.npy")
print(f"               plot  -> roc_curve.png")

print(f"\n  7. Loss history ({EPOCHS} epochs):")
print(f"       Train loss: epoch-1={train_losses[0]:.4f}  "
      f"-> epoch-{EPOCHS}={train_losses[-1]:.4f}")
print(f"       Val   loss: epoch-1={val_losses[0]:.4f}  "
      f"-> epoch-{EPOCHS}={val_losses[-1]:.4f}")
print(f"               saved -> loss_per_epoch.csv")
print(f"               plot  -> loss_curves.png")

print(f"\n  8. Top-20 predictions:")
for i, row in top20_df.iterrows():
    print(f"       #{i:2d}  {row['drug_name']:<35s} -> "
          f"{row['disease_name']:<35s}  p={row['probability']:.4f}")

print(f"\n  All files saved to: {OUT_DIR}")
print("=" * 65)
