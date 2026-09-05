import sys
import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
BASE_DIR = PROJECT_DIR
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "preprocessed_data")

print("--- STARTING STEP 3: GraphSAGE Neural Network Training (Full-Batch) ---")

train_data = torch.load(os.path.join(DATA_DIR, "train_data.pt"), weights_only=False)
val_data = torch.load(os.path.join(DATA_DIR, "val_data.pt"), weights_only=False)

class LinkPredictorSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # GraphSAGE aggregates local neighborhood features dynamically
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        # Passes text/chemistry inputs through structural convolutions
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        # Binary Classification Link Prediction via Dot Product constraints
        src = edge_label_index[0]
        dst = edge_label_index[1]
        return (z[src] * z[dst]).sum(dim=-1)

# In Step 1, x.pt mathematically finalized 131 dimensions (128 TF-IDF + 3 Scaled Features)
model = LinkPredictorSAGE(in_channels=131, hidden_channels=64, out_channels=32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

# Criterion forces model to map true drug-disease paths to 1.0, and fake paths to 0.0
criterion = torch.nn.BCEWithLogitsLoss()

# The original topology (connections between all nodes) drives message passing
msg_edge_index = train_data.edge_index

def train():
    model.train()
    optimizer.zero_grad()
    
    # 1. Forward Pass Embeddings
    z = model.encode(train_data.x, msg_edge_index)
    
    # 2. Score specifically our filtered Drug-Disease boundaries
    pred = model.decode(z, train_data.edge_label_index)
    
    # 3. Compute loss against 1s and 0s
    loss = criterion(pred, train_data.edge_label)
    
    # 4. Backpropagation
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, msg_edge_index)
    pred = model.decode(z, data.edge_label_index)
    
    # Apply sigmoid strictly to convert logit boundaries to 0-1 probabilities
    pred_prob = torch.sigmoid(pred).cpu().numpy()
    target = data.edge_label.cpu().numpy()
    
    auc = roc_auc_score(target, pred_prob)
    ap = average_precision_score(target, pred_prob)
    return auc, ap

print("Initialization complete! Commencing optimization...\n")

best_val_auc = 0.0

for epoch in range(1, 101):
    loss = train()
    if epoch % 10 == 0:
        val_auc, val_ap = test(val_data)
        print(f"Epoch: {epoch:03d} | Train Loss: {loss:.4f} | Validation AUC-ROC: {val_auc:.4f} | Validation AP: {val_ap:.4f}")
        
        # Save absolute best generalization model automatically
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), os.path.join(DATA_DIR, "best_graphsage_model.pth"))

print("\n--- STEP 3 COMPLETE: Model Optimized and Saved! ---")
