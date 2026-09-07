import sys
import os
import json
import pandas as pd
import numpy as np
import torch
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
BASE_DIR = PROJECT_DIR
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "dataverse_files"))
PROJECT_DIR = BASE_DIR
OUTPUT_DIR = os.path.join(PROJECT_DIR, "preprocessed_data")

print(f"--- STARTING STEP 1 IN ISOLATED FOLDER: {PROJECT_DIR} ---")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("1. Loading subset nodes...")
nodes_subset = pd.read_csv(os.path.join(DATA_DIR, "nodes_subset.csv"))

unique_nodes = nodes_subset['node_index'].unique()
num_nodes = len(unique_nodes)
node_map = {int(node_idx): i for i, node_idx in enumerate(unique_nodes)}

with open(os.path.join(OUTPUT_DIR, "node_map.json"), "w") as f:
    json.dump(node_map, f)

print("2. Loading Drug and Disease features...")
drug_feat = pd.read_csv(os.path.join(DATA_DIR, "drug_features.csv"))
disease_feat = pd.read_csv(os.path.join(DATA_DIR, "disease_features.csv"))

drug_subset = drug_feat[drug_feat['node_index'].isin(unique_nodes)].copy()
disease_subset = disease_feat[disease_feat['node_index'].isin(unique_nodes)].copy()

print("3. Executing Text Processing & Cleaning via TF-IDF...")
# Text Cleaning Function
def clean_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text) # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()   # Normalize spacing
    return text

text_dict = {}
nodes_subset['node_name'] = nodes_subset['node_name'].fillna("")
for _, row in nodes_subset.iterrows():
    text_dict[row['node_index']] = clean_text(row['node_name'])

for _, row in drug_subset.iterrows():
    text = f"{row.get('description', '')} {row.get('indication', '')} {row.get('mechanism_of_action', '')}"
    if len(text.strip()) > 0:
        text_dict[row['node_index']] = clean_text(text)

for _, row in disease_subset.iterrows():
    text = f"{row.get('mondo_name', '')} {row.get('mondo_definition', '')} {row.get('umls_description', '')}"
    if len(text.strip()) > 0:
        text_dict[row['node_index']] = clean_text(text)

ordered_texts = [text_dict[idx] for idx in unique_nodes]
# Note: TF-IDF inherently applies L2 Normalization (norm='l2'). It scales its vectors natively.
vectorizer = TfidfVectorizer(max_features=128, stop_words='english', norm='l2')
text_features = vectorizer.fit_transform(ordered_texts).toarray()

print("4. Executing Numerical Processing (StandardScaler on purely numeric data)...")
num_features = np.zeros((num_nodes, 3))
scaler = StandardScaler()

drug_nums = drug_subset[['node_index', 'molecular_weight', 'tpsa', 'clogp']].copy()

def extract_number(val):
    if pd.isna(val): return 0.0
    match = re.search(r"([-+]?\d*\.\d+|\d+)", str(val))
    return float(match.group(1)) if match else 0.0

drug_nums['molecular_weight'] = drug_nums['molecular_weight'].apply(extract_number)
drug_nums['tpsa'] = drug_nums['tpsa'].apply(extract_number)
drug_nums['clogp'] = drug_nums['clogp'].apply(extract_number)

if not drug_nums.empty:
    scaled_nums = scaler.fit_transform(drug_nums[['molecular_weight', 'tpsa', 'clogp']])
    drug_nums['w'] = scaled_nums[:, 0]
    drug_nums['t'] = scaled_nums[:, 1]
    drug_nums['c'] = scaled_nums[:, 2]

    for _, row in drug_nums.iterrows():
        seq_idx = node_map[int(row['node_index'])]
        num_features[seq_idx] = [row['w'], row['t'], row['c']]

# Final Normalization Step: Merge independent representations (TF-IDF L2 norm + StandardScaler z-score)
final_node_features = np.concatenate([text_features, num_features], axis=1)
final_node_features = np.nan_to_num(final_node_features, nan=0.0, posinf=1.0, neginf=-1.0)
x_tensor = torch.tensor(final_node_features, dtype=torch.float)

print("5. Processing Edges and Graph Topology (Making Graph UNDIRECTED for Message Passing)...")
edges_subset = pd.read_csv(os.path.join(DATA_DIR, "edges_subset.csv"))
edge_index = []
edge_types = []

rel_map = {}
curr_rel_idx = 0
dropped_edges = 0

unique_edges = set()

# We make the graph UNDIRECTED by explicitly adding reverse edges.
# This is crucial for GraphSAGE message passing so that if a Drug -> Treats -> Disease,
# the target Disease can also pass its features back to the Drug.
for _, row in edges_subset.iterrows():
    u = row['x_index']
    v = row['y_index']
    rel = row['relation']
    
    if u in node_map and v in node_map and u != v:
        u_mapped = node_map[u]
        v_mapped = node_map[v]
        
        # Forward Edge
        fw_edge = (u_mapped, v_mapped, rel)
        if fw_edge not in unique_edges:
            unique_edges.add(fw_edge)
            if rel not in rel_map:
                rel_map[rel] = curr_rel_idx
                curr_rel_idx += 1
            edge_index.append([u_mapped, v_mapped])
            edge_types.append(rel_map[rel])
            
        # Reverse Edge (Undirected flow)
        rev_rel = rel + "_rev"
        bw_edge = (v_mapped, u_mapped, rev_rel)
        if bw_edge not in unique_edges:
            unique_edges.add(bw_edge)
            if rev_rel not in rel_map:
                rel_map[rev_rel] = curr_rel_idx
                curr_rel_idx += 1
            edge_index.append([v_mapped, u_mapped]) # Flip direction
            edge_types.append(rel_map[rev_rel])
    else:
        dropped_edges += 1

edge_index = torch.tensor(edge_index, dtype=torch.long).t()
edge_types = torch.tensor(edge_types, dtype=torch.long)

node_type_list = [0] * num_nodes
ntype_map = {"disease": 0, "drug": 1, "gene/protein": 2}
for _, row in nodes_subset.iterrows():
    seq_idx = node_map[row['node_index']]
    node_type_list[seq_idx] = ntype_map.get(row['node_type'], 3)
node_type_tensor = torch.tensor(node_type_list, dtype=torch.long)

print("6. Performing Final Strict Validation and DIMENSION CHECKS...")
print(f"   [Dimension] Text Features (TF-IDF): {text_features.shape[1]} dimensions")
print(f"   [Dimension] Numeric Features (MinMax): {num_features.shape[1]} dimensions")
print(f"   [Dimension] Final Combined Node Tensor: {x_tensor.shape[1]} total dimensions")
print(f"   [Validation] Dropped invalid/self-loop edges: {dropped_edges}")
print(f"   [Validation] Final Edge Count (x2 for Undirected): {edge_index.shape[1]}")

assert not torch.isnan(x_tensor).any(), "Validation FAILED: NaN found in node features!"
assert edge_index.max() < num_nodes, "Validation FAILED: Edge index exceeds number of nodes!"
assert edge_index.min() >= 0, "Validation FAILED: Negative edge index found!"

isolated_nodes = num_nodes - len(torch.unique(edge_index))
print(f"   [Validation] Isolated Nodes (degree 0): {isolated_nodes}")

print("7. Saving Tensors to disk for GraphSAGE...")
torch.save(x_tensor, os.path.join(OUTPUT_DIR, "x.pt"))
torch.save(edge_index, os.path.join(OUTPUT_DIR, "edge_index.pt"))
torch.save(edge_types, os.path.join(OUTPUT_DIR, "edge_type.pt"))
torch.save(node_type_tensor, os.path.join(OUTPUT_DIR, "node_type.pt"))

with open(os.path.join(OUTPUT_DIR, "edge_rel_map.json"), "w") as f:
    json.dump(rel_map, f)

print(f"--- STEP 1 COMPLETE! Validated files saved cleanly to: {OUTPUT_DIR} ---")
