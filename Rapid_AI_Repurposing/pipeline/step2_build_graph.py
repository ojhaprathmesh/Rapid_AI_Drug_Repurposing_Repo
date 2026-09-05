import sys
import os
import torch
import random
from torch_geometric.data import Data

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
BASE_DIR = PROJECT_DIR
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "preprocessed_data")

print("--- STARTING STEP 2 (FINAL): Clean Disjoint Drug-Disease Split ---")

print("1. Loading Preprocessed Tensors...")
x          = torch.load(os.path.join(DATA_DIR, "x.pt"))
edge_index = torch.load(os.path.join(DATA_DIR, "edge_index.pt"))
edge_type  = torch.load(os.path.join(DATA_DIR, "edge_type.pt"))
node_type  = torch.load(os.path.join(DATA_DIR, "node_type.pt"))

data = Data(x=x, edge_index=edge_index, edge_type=edge_type, node_type=node_type)
print(f"   Loaded: {data.num_nodes} nodes, {data.num_edges} edges.")

print("2. Extracting ONLY Drug-Disease Positive Edges (Canonical Direction)...")
src = data.edge_index[0]
dst = data.edge_index[1]

# Drug(1)->Disease(0) only — keep canonical direction, drop Disease->Drug reverses
dd_mask = (node_type[src] == 1) & (node_type[dst] == 0)
dd_edges_all = data.edge_index[:, dd_mask]

# Deduplicate: since graph is undirected, (u,v) and (v,u) both exist. Keep only unique (u,v) pairs.
seen = set()
unique_cols = []
for i in range(dd_edges_all.shape[1]):
    key = (dd_edges_all[0, i].item(), dd_edges_all[1, i].item())
    if key not in seen:
        seen.add(key)
        unique_cols.append(i)
dd_edges = dd_edges_all[:, unique_cols]
print(f"   Found {dd_edges.shape[1]} unique Drug-Disease positive edges.")

# Build global edge set (both directions) for collision checks
global_edges = set()
for u, v in zip(data.edge_index[0], data.edge_index[1]):
    global_edges.add((u.item(), v.item()))
    global_edges.add((v.item(), u.item()))

print("3. Splitting Positive Edges into Disjoint Train/Val/Test (80/10/10)...")
n = dd_edges.shape[1]
perm = torch.randperm(n)
dd_edges = dd_edges[:, perm]  # Shuffle

n_train = int(0.8 * n)
n_val   = int(0.1 * n)

train_pos = dd_edges[:, :n_train]
val_pos   = dd_edges[:, n_train:n_train + n_val]
test_pos  = dd_edges[:, n_train + n_val:]

print(f"   Train: {train_pos.shape[1]}, Val: {val_pos.shape[1]}, Test: {test_pos.shape[1]}")

print("4. Strict Collision-Free Negative Sampling (No duplicates across splits)...")
# Pre-seed used_edges with ALL positive edges (both directions)
used_edges = set()
for pos in [train_pos, val_pos, test_pos]:
    for i in range(pos.shape[1]):
        u, v = pos[0, i].item(), pos[1, i].item()
        used_edges.add((u, v))
        used_edges.add((v, u))

drug_indices    = (node_type == 1).nonzero(as_tuple=True)[0]
disease_indices = (node_type == 0).nonzero(as_tuple=True)[0]

def sample_negatives(num_neg, used_edges):
    neg_edges = []
    while len(neg_edges) < num_neg:
        u = drug_indices[random.randint(0, len(drug_indices) - 1)].item()
        v = disease_indices[random.randint(0, len(disease_indices) - 1)].item()
        if (u, v) not in global_edges and (u, v) not in used_edges:
            neg_edges.append([u, v])
            used_edges.add((u, v))
    return torch.tensor(neg_edges, dtype=torch.long).t()

train_neg = sample_negatives(train_pos.shape[1], used_edges)
val_neg   = sample_negatives(val_pos.shape[1],   used_edges)
test_neg  = sample_negatives(test_pos.shape[1],  used_edges)

print("5. Building Final Data Objects...")
def build_data(pos_ei, neg_ei):
    edge_label_index = torch.cat([pos_ei, neg_ei], dim=1)
    edge_label = torch.cat([
        torch.ones(pos_ei.shape[1]),
        torch.zeros(neg_ei.shape[1])
    ])
    d = Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        node_type=node_type,
        edge_label_index=edge_label_index,
        edge_label=edge_label
    )
    return d

train_data = build_data(train_pos, train_neg)
val_data   = build_data(val_pos,   val_neg)
test_data  = build_data(test_pos,  test_neg)

print("6. Verifying Zero Positive Overlap...")
def get_pos_set(split):
    return set([
        (u.item(), v.item())
        for u, v, y in zip(
            split.edge_label_index[0],
            split.edge_label_index[1],
            split.edge_label
        )
        if y.item() == 1
    ])

train_pos_set = get_pos_set(train_data)
val_pos_set   = get_pos_set(val_data)
test_pos_set  = get_pos_set(test_data)

print(f"   Train & Val overlap:  {len(train_pos_set & val_pos_set)}")
print(f"   Train & Test overlap: {len(train_pos_set & test_pos_set)}")
print(f"   Val & Test overlap:   {len(val_pos_set & test_pos_set)}")

all_zero = (
    len(train_pos_set & val_pos_set) == 0 and
    len(train_pos_set & test_pos_set) == 0 and
    len(val_pos_set & test_pos_set) == 0
)
print("RESULT:", "CLEAN - Zero Positive Leakage!" if all_zero else "WARNING: Leakage detected!")

print("7. Saving...")
torch.save(train_data, os.path.join(DATA_DIR, "train_data.pt"))
torch.save(val_data,   os.path.join(DATA_DIR, "val_data.pt"))
torch.save(test_data,  os.path.join(DATA_DIR, "test_data.pt"))
print("--- STEP 2 FINAL COMPLETE ---")
