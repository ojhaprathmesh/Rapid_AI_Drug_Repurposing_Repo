import os
import torch
import numpy as np
import pandas as pd
from collections import Counter

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = BASE_DIR
DATA_DIR = os.path.join(PROJECT_DIR, "preprocessed_data")

print("--- INITIATING DATASET AUDIT ---")

# Load tensors
try:
    x = torch.load(os.path.join(DATA_DIR, "x.pt"))
    edge_index = torch.load(os.path.join(DATA_DIR, "edge_index.pt"))
    edge_type = torch.load(os.path.join(DATA_DIR, "edge_type.pt"))
    node_type = torch.load(os.path.join(DATA_DIR, "node_type.pt"))
except Exception as e:
    print(f"Failed to load tensors: {e}")
    exit(1)

print("\n[1] FEATURE SCALING & QUALITY")
tf_idf_features = x[:, :128]
num_features = x[:, 128:]

print(f"TF-IDF -> Min: {tf_idf_features.min().item():.4f}, Max: {tf_idf_features.max().item():.4f}, Mean: {tf_idf_features.mean().item():.4f}, STD: {tf_idf_features.std().item():.4f}")
print(f"Numeric -> Min: {num_features.min().item():.4f}, Max: {num_features.max().item():.4f}, Mean: {num_features.mean().item():.4f}, STD: {num_features.std().item():.4f}")

var_tfidf = torch.var(tf_idf_features, dim=0)
var_num = torch.var(num_features, dim=0)
near_zero_tfidf = (var_tfidf < 1e-5).sum().item()
near_zero_num = (var_num < 1e-5).sum().item()
print(f"Near-zero variance features (useless): {near_zero_tfidf} out of 128 TF-IDF, {near_zero_num} out of 3 Numeric")

print("\n[2] GRAPH TOPOLOGY QUALITY")
num_nodes = x.shape[0]
num_edges = edge_index.shape[1]
degrees = torch.zeros(num_nodes, dtype=torch.long)
degrees.scatter_add_(0, edge_index[0], torch.ones(num_edges, dtype=torch.long))
degrees.scatter_add_(0, edge_index[1], torch.ones(num_edges, dtype=torch.long))
isolated_nodes = (degrees == 0).sum().item()

# Very basic connected components check (undirected approximation)
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
adj = coo_matrix((np.ones(num_edges), (edge_index[0].numpy(), edge_index[1].numpy())), shape=(num_nodes, num_nodes))
n_components, labels = connected_components(csgraph=adj, directed=False, return_labels=True)

print(f"Total Nodes: {num_nodes}")
print(f"Total Edges: {num_edges}")
print(f"Isolated Nodes (Degree 0): {isolated_nodes} ({(isolated_nodes/num_nodes)*100:.2f}%)")
print(f"Connected Components: {n_components} (Ideal is 1 main component + isolated nodes)")

print("\n[3] EDGE DISTRIBUTION")
rel_counts = Counter(edge_type.numpy())
for rel, count in rel_counts.most_common(5):
    print(f"Relation ID {rel}: {count} edges ({(count/num_edges)*100:.2f}%)")

print("\n[4] DATA INTEGRITY")
has_nans = torch.isnan(x).any().item()
has_infs = torch.isinf(x).any().item()
invalid_edges_high = (edge_index >= num_nodes).any().item()
invalid_edges_low = (edge_index < 0).any().item()

print(f"Contains NaNs: {has_nans}")
print(f"Contains Infinite Values: {has_infs}")
print(f"Exceeds max node index bounds: {invalid_edges_high}")
print(f"Falls below 0 node index bounds: {invalid_edges_low}")
