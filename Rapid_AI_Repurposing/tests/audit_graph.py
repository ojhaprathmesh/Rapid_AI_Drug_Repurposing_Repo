import os, sys, json, torch

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_DIR, "preprocessed_data")

def audit():
    print("=== GRAPH TOPOLOGY AUDIT ===")
    x = torch.load(os.path.join(DATA_DIR, "x.pt"))
    edge_index = torch.load(os.path.join(DATA_DIR, "edge_index.pt"))
    edge_type = torch.load(os.path.join(DATA_DIR, "edge_type.pt"))
    node_type = torch.load(os.path.join(DATA_DIR, "node_type.pt"))

    print(f"Total Nodes: {x.size(0)}")
    print(f"Feature Dimension: {x.size(1)}")
    print(f"Total Edges (Bidirectional): {edge_index.size(1)}")
    print(f"Unique Relations: {torch.unique(edge_type).tolist()}")
    print("Topology audit complete.\n")

if __name__ == "__main__":
    audit()
