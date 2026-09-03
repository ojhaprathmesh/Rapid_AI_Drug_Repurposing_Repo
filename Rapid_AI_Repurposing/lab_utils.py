import os
import torch
import pandas as pd
import json
import requests
import difflib
from torch_geometric.nn import SAGEConv
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# CONSTANTS & PATHS
# ---------------------------------------------------------
# ---------------------------------------------------------
# CONSTANTS & PATHS
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

candidate_dataverse_dirs = [
    os.path.abspath(os.path.join(BASE_DIR, "..", "..", "dataverse_files")),
    os.path.abspath(os.path.join(BASE_DIR, "..", "dataverse_files")),
    os.path.join(BASE_DIR, "dataverse_files"),
]
DATAVERSE_DIR = next((d for d in candidate_dataverse_dirs if os.path.exists(d)), candidate_dataverse_dirs[0])

candidate_data_dirs = [
    os.path.join(BASE_DIR, "preprocessed_data"),
    os.path.abspath(os.path.join(BASE_DIR, "..", "..", "preprocessed_data")),
]
DATA_DIR = next((d for d in candidate_data_dirs if os.path.exists(d)), candidate_data_dirs[0])

NODES_PATH = os.path.join(DATAVERSE_DIR, "nodes_subset.csv")
MODEL_PATH = os.path.join(DATA_DIR, "best_graphsage_model.pth")


# ---------------------------------------------------------
# MODEL ARCHITECTURE (Exact match to Step 3)
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# DISCOVERY ENGINE
# ---------------------------------------------------------
class DiscoveryEngine:
    def __init__(self):
        self.device = torch.device('cpu')
        self.load_data()
        self.load_model()
        self.compute_all_embeddings()

    def load_data(self):
        # Load Tensors
        self.x = torch.load(os.path.join(DATA_DIR, "x.pt"), weights_only=False).to(self.device)
        self.edge_index = torch.load(os.path.join(DATA_DIR, "edge_index.pt"), weights_only=False).to(self.device)
        
        with open(os.path.join(DATA_DIR, "node_map.json"), "r") as f:
            self.node_map = json.load(f)  # original_idx -> graph_idx
        
        # Load Metadata for names/types
        self.nodes_df = pd.read_csv(NODES_PATH)
        # Type mapping: 1=drug, 0=disease in our preprocess logic
        # Actually our node_type.pt has the ground truth.
        self.node_types = torch.load(os.path.join(DATA_DIR, "node_type.pt"), weights_only=False).cpu().numpy()
        
        # Create helper lookup structures
        self.name_to_gid = {} # Name -> Graph Index
        self.gid_to_name = {} # Graph Index -> Name
        self.drug_names = []
        self.disease_names = []
        
        for _, row in self.nodes_df.iterrows():
            orig_idx = str(row['node_index'])
            if orig_idx in self.node_map:
                gid = self.node_map[orig_idx]
                name = row['node_name']
                ntype = row['node_type']
                
                self.name_to_gid[name.lower()] = (gid, ntype)
                self.gid_to_name[gid] = (name, ntype)
                
                if ntype == 'drug':
                    self.drug_names.append(name)
                elif ntype == 'disease':
                    self.disease_names.append(name)
        
        self.drug_names = sorted(list(set(self.drug_names)))
        self.disease_names = sorted(list(set(self.disease_names)))

    def load_model(self):
        self.model = LinkPredictorSAGE(131, 64, 32).to(self.device)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.eval()

    def compute_all_embeddings(self):
        with torch.no_grad():
            # EXACT match to Step 3 logic
            self.z = self.model.encode(self.x, self.edge_index)

    def fuzzy_search(self, query, top_n=3):
        all_names = list(self.name_to_gid.keys())
        matches = difflib.get_close_matches(query.lower(), all_names, n=top_n, cutoff=0.6)
        # Return capitalized versions from the lookup
        return [self.gid_to_name[self.name_to_gid[m][0]][0] for m in matches]

    def predict(self, node_name, target_type='disease', top_k=10):
        # 1. Map input to gid
        if node_name.lower() not in self.name_to_gid:
            return None
        
        query_gid, query_type = self.name_to_gid[node_name.lower()]
        query_vec = self.z[query_gid]
        
        # 2. Identify all targets of requested type
        # DRKG Standard in our Step 2: 1=Drug, 0=Disease
        target_val = 0 if target_type == 'disease' else 1
        target_indices = [i for i, t in enumerate(self.node_types) if t == target_val]
        
        target_vecs = self.z[target_indices]
        
        # 3. Batch prediction using dot product
        # Ensure target_vecs has shape [num_targets, Emb_dim]
        # query_vec has shape [Emb_dim]
        logits = (target_vecs * query_vec).sum(dim=-1)
        scores = torch.sigmoid(logits).cpu().numpy()
        
        # 4. Rank and format
        results = []
        for idx_in_subset, score in enumerate(scores):
            gid = target_indices[idx_in_subset]
            name, _ = self.gid_to_name[gid]
            results.append({
                "name": name,
                "score": float(score),
                "confidence": "High" if score > 0.85 else ("Medium" if score > 0.7 else "Low")
            })
            
        results = sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]
        return results

# ---------------------------------------------------------
# EXTERNAL API HELPERS
# ---------------------------------------------------------
def get_pubchem_info(drug_name):
    """Fetches SMILES and Image URL from PubChem with robust key scanning."""
    try:
        # Get SMILES - fetch all properties to be safe
        smiles_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name}/property/CanonicalSMILES,ConnectivitySMILES,IsomericSMILES/JSON"
        resp = requests.get(smiles_url, timeout=5)
        smiles = None
        if resp.status_code == 200:
            data = resp.json()
            props = data.get('PropertyTable', {}).get('Properties', [{}])[0]
            # Find the first key that contains 'SMILES'
            for k, v in props.items():
                if "SMILES" in k:
                    smiles = v
                    break
        
        # Image URL (direct link for Streamlit)
        img_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name}/PNG"
        
        return {"smiles": smiles, "img_url": img_url}
    except:
        return None

def get_clinical_trials_data(drug, disease):
    """Checks ClinicalTrials.gov and returns a list of study links/titles."""
    try:
        # Increased to 10 studies per the user request
        url = f"https://clinicaltrials.gov/api/v2/studies?query.term={drug}+{disease}&pageSize=10"
        resp = requests.get(url, timeout=5)
        studies_data = []
        if resp.status_code == 200:
            data = resp.json()
            studies = data.get('studies', [])
            for s in studies:
                try:
                    ident = s['protocolSection']['identificationModule']
                    studies_data.append({
                        "id": ident['nctId'],
                        "title": ident['briefTitle']
                    })
                except KeyError:
                    continue
        return studies_data
    except Exception as e:
        print(f"ClinicalTrials Error: {e}")
        return []

def check_ollama_status():
    """Checks if Ollama server is responsive on localhost."""
    try:
        resp = requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
        return resp.status_code == 200
    except:
        return False
