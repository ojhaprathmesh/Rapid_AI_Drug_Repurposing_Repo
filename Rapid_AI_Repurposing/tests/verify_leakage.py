import os
import sys
import torch

# Resilient unpickler for PyG Data objects when running on systems without torch_geometric
try:
    import torch_geometric
except ImportError:
    import types
    tg = types.ModuleType("torch_geometric")
    tg_data = types.ModuleType("torch_geometric.data")
    tg_data_data = types.ModuleType("torch_geometric.data.data")
    class MockData(dict):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.__dict__ = self
        def __getattr__(self, item):
            return self.get(item)
    tg_data.Data = MockData
    tg_data_data.Data = MockData
    sys.modules["torch_geometric"] = tg
    sys.modules["torch_geometric.data"] = tg_data
    sys.modules["torch_geometric.data.data"] = tg_data_data

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_DIR, "preprocessed_data")

train_data = torch.load(os.path.join(DATA_DIR, "train_data.pt"), weights_only=False)
val_data   = torch.load(os.path.join(DATA_DIR, "val_data.pt"),   weights_only=False)
test_data  = torch.load(os.path.join(DATA_DIR, "test_data.pt"),  weights_only=False)

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

train_pos = get_pos_set(train_data)
val_pos   = get_pos_set(val_data)
test_pos  = get_pos_set(test_data)

print("--- [1/2] POSITIVE SUPERVISION EDGE SPLIT OVERLAP ---")
print(f"Train & Val  : {len(train_pos & val_pos)}")
print(f"Train & Test : {len(train_pos & test_pos)}")
print(f"Val & Test   : {len(val_pos & test_pos)}")

overlaps = len(train_pos & val_pos) + len(train_pos & test_pos) + len(val_pos & test_pos)
if overlaps == 0:
    print("RESULT: PERFECT - Zero positive edge overlap across splits!\n")
else:
    print(f"RESULT: {overlaps} overlapping positive edges detected\n")

print("--- [2/2] MESSAGE-PASSING ADJACENCY LEAKAGE CHECK (msg_edge_index) ---")
def to_directed_set(ei):
    return set(zip(ei[0].tolist(), ei[1].tolist()))

def make_bidirectional(edge_set):
    return edge_set | set((v, u) for (u, v) in edge_set)

val_pos_bi  = make_bidirectional(val_pos)
test_pos_bi = make_bidirectional(test_pos)

train_msg_set = to_directed_set(train_data.msg_edge_index)
val_msg_set   = to_directed_set(val_data.msg_edge_index)
test_msg_set  = to_directed_set(test_data.msg_edge_index)

val_in_train  = len(val_pos_bi & train_msg_set)
test_in_train = len(test_pos_bi & train_msg_set)
val_in_val    = len(val_pos_bi & val_msg_set)
test_in_test  = len(test_pos_bi & test_msg_set)

print(f"Val  pos edges in train_msg_ei (both directions) : {val_in_train} (must be 0)")
print(f"Test pos edges in train_msg_ei (both directions) : {test_in_train} (must be 0)")
print(f"Val  pos edges in val_msg_ei   (both directions) : {val_in_val} (must be 0)")
print(f"Test pos edges in test_msg_ei  (both directions) : {test_in_test} (must be 0)")

mp_leaks = val_in_train + test_in_train + val_in_val + test_in_test
if mp_leaks == 0:
    print("\nRESULT: PERFECT - Zero message-passing edge leakage! GNN encoder is strictly leakage-free.\n")
else:
    print(f"\nFATAL: {mp_leaks} evaluation edges detected in message-passing graphs!\n")
    sys.exit(1)
