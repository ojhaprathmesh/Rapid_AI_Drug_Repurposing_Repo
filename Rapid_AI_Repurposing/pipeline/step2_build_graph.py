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

# ─────────────────────────────────────────────────────────────────────────────
# 5. Build Per-Split Message-Passing Graphs (DATA LEAKAGE PREVENTION)
# ─────────────────────────────────────────────────────────────────────────────
"""
The GNN encoder must NOT see any edge it is later asked to score.
We excise both the forward and reverse direction of every held-out positive
edge from the message-passing adjacency (edge_index).

Protocol (strict transductive):
  train_msg_ei  = global  |  val_pos  |  test_pos   (both directions)
  val_msg_ei    = global  |  val_pos  |  test_pos   (same; val scored during training)
  test_msg_ei   = global  |  test_pos               (both directions)
"""
print("5. Building Leakage-Free Per-Split Message-Passing Graphs...")

def pos_to_set_bidirectional(pos_ei):
    """Return the set of directed edges (u,v) AND (v,u) for a positive split."""
    s = set()
    for i in range(pos_ei.shape[1]):
        u, v = pos_ei[0, i].item(), pos_ei[1, i].item()
        s.add((u, v))
        s.add((v, u))
    return s

def excise_edges(base_ei, edges_to_remove):
    """
    Remove all edges in `edges_to_remove` (a set of (u,v) tuples) from
    `base_ei` (shape [2, E]) and return the filtered edge index.
    """
    src_list = base_ei[0].tolist()
    dst_list = base_ei[1].tolist()
    keep_mask = [
        (u, v) not in edges_to_remove
        for u, v in zip(src_list, dst_list)
    ]
    keep_idx = torch.tensor(keep_mask, dtype=torch.bool)
    return base_ei[:, keep_idx]

val_excise_set  = pos_to_set_bidirectional(val_pos)
test_excise_set = pos_to_set_bidirectional(test_pos)

# Train/Val encoder: remove val + test positives
train_val_msg_ei = excise_edges(edge_index, val_excise_set | test_excise_set)
# Test encoder: remove test positives only
test_msg_ei      = excise_edges(edge_index, test_excise_set)

print(f"   Global edge_index     : {edge_index.shape[1]:,} directed edges")
print(f"   Val  excision set     : {len(val_excise_set):,}  directed edges removed  (val pos × 2)")
print(f"   Test excision set     : {len(test_excise_set):,}  directed edges removed  (test pos × 2)")
print(f"   train_msg_ei / val_msg_ei : {train_val_msg_ei.shape[1]:,} edges  "
      f"(removed {edge_index.shape[1] - train_val_msg_ei.shape[1]:,})")
print(f"   test_msg_ei           : {test_msg_ei.shape[1]:,} edges  "
      f"(removed {edge_index.shape[1] - test_msg_ei.shape[1]:,})")

# ── Sanity checks ──────────────────────────────────────────────────────────
print("   Verifying excision correctness...")
train_msg_set = set(zip(train_val_msg_ei[0].tolist(), train_val_msg_ei[1].tolist()))
test_msg_set  = set(zip(test_msg_ei[0].tolist(),      test_msg_ei[1].tolist()))

val_leak  = sum(1 for e in val_excise_set  if e in train_msg_set)
test_leak = sum(1 for e in test_excise_set if e in test_msg_set)
print(f"   Val  pos edges remaining in train_msg_ei : {val_leak}  (must be 0)")
print(f"   Test pos edges remaining in test_msg_ei  : {test_leak}  (must be 0)")
assert val_leak  == 0, "FATAL: Val positive edges still present in train message-passing graph!"
assert test_leak == 0, "FATAL: Test positive edges still present in test message-passing graph!"
print("   PASSED — zero leakage confirmed.")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Build Final Data Objects (each carrying its own msg_edge_index)
# ─────────────────────────────────────────────────────────────────────────────
print("6. Building Final Data Objects...")

def build_data(pos_ei, neg_ei, msg_ei):
    """
    Construct a PyG Data object for one split.

    Args:
        pos_ei: Positive supervision edges for this split (shape [2, P]).
        neg_ei: Negative supervision edges for this split (shape [2, N]).
        msg_ei: Leakage-free message-passing adjacency — the global edge_index
                with this split's positive edges (both directions) excised.
                This is the ONLY adjacency that should be passed to
                model.encode(); passing edge_index instead would reintroduce
                message-passing topological label leakage.

    Note:
        edge_index (the full global graph) is retained on the object solely
        for reference and novel-candidate discovery (step4_predict.py).
        It must NOT be used as the message-passing adjacency during training
        or evaluation.
    """
    edge_label_index = torch.cat([pos_ei, neg_ei], dim=1)
    edge_label = torch.cat([
        torch.ones(pos_ei.shape[1]),
        torch.zeros(neg_ei.shape[1])
    ])
    d = Data(
        x=x,
        msg_edge_index=msg_ei,
        edge_index=edge_index,
        edge_type=edge_type,
        node_type=node_type,
        edge_label_index=edge_label_index,
        edge_label=edge_label
    )
    return d

train_data = build_data(train_pos, train_neg, train_val_msg_ei)
val_data   = build_data(val_pos,   val_neg,   train_val_msg_ei)  # same MP graph as train
test_data  = build_data(test_pos,  test_neg,  test_msg_ei)

print("7. Verifying Zero Positive Overlap...")
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

print("8. Saving...")
torch.save(train_data, os.path.join(DATA_DIR, "train_data.pt"))
torch.save(val_data,   os.path.join(DATA_DIR, "val_data.pt"))
torch.save(test_data,  os.path.join(DATA_DIR, "test_data.pt"))
print("--- STEP 2 FINAL COMPLETE ---")
