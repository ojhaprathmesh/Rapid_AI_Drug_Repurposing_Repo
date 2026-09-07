import os
import pandas as pd
import json
import difflib
import warnings
import urllib.request

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    requests = None
    REQUESTS_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    from torch_geometric.nn import SAGEConv
except ImportError:
    if TORCH_AVAILABLE:
        class SAGEConv(torch.nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.lin_l = torch.nn.Linear(in_channels, out_channels, bias=True)
                self.lin_r = torch.nn.Linear(in_channels, out_channels, bias=False)

            def forward(self, x, edge_index):
                row, col = edge_index[0], edge_index[1]
                num_nodes = x.size(0)
                deg = torch.zeros(num_nodes, dtype=torch.float, device=x.device)
                deg.scatter_add_(0, col, torch.ones_like(col, dtype=torch.float))
                deg.clamp_(min=1.0)
                out = torch.zeros(num_nodes, x.size(1), dtype=torch.float, device=x.device).scatter_add_(
                    0, col.unsqueeze(-1).expand(-1, x.size(1)), x[row]
                )
                out = out / deg.unsqueeze(-1)
                return self.lin_l(out) + self.lin_r(x)
    else:
        SAGEConv = object

warnings.filterwarnings('ignore')

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


_ModuleBase = torch.nn.Module if TORCH_AVAILABLE else object

class LinkPredictorSAGE(_ModuleBase):
    def __init__(self, in_channels=131, hidden_channels=64, out_channels=32):
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
            self.embeddings = self.z

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

    def explain_prediction(self, drug_name, disease_name, top_k=5):
        """Extracts hub-penalized 2-hop biological pathways bridging drug and disease."""
        if not hasattr(self, '_saliency_engine'):
            self._saliency_engine = PathSaliencyEngine()
        return self._saliency_engine.extract_paths(drug_name, disease_name, top_k=top_k)

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
        if REQUESTS_AVAILABLE:
            resp = requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
            return resp.status_code == 200
        else:
            req = urllib.request.Request("http://127.0.0.1:11434/api/tags")
            with urllib.request.urlopen(req, timeout=2) as resp:
                return resp.status == 200
    except:
        return False

# ---------------------------------------------------------
# HUB-PENALIZED 2-HOP PATH SALIENCY ENGINE (Section VI-G)
# ---------------------------------------------------------
class PathSaliencyEngine:
    """
    Implements degree-penalized path saliency S(u -> p -> v) to extract
    pathway-specific target proteins while down-weighting promiscuous hubs.
    Formalized in Section VI-G of the Rapid AI manuscript.
    """
    def __init__(self, embeddings=None):
        import numpy as np
        self.np = np
        self.embeddings = embeddings
        self.nodes_df = pd.read_csv(NODES_PATH)
        self.edges_path = os.path.join(DATAVERSE_DIR, "edges_subset.csv")
        self.edges_df = pd.read_csv(self.edges_path) if os.path.exists(self.edges_path) else pd.DataFrame()

        drug_feat_path = os.path.join(DATAVERSE_DIR, "drug_features.csv")
        disease_feat_path = os.path.join(DATAVERSE_DIR, "disease_features.csv")

        self.drug_feat_df = pd.read_csv(drug_feat_path).set_index('node_index') if os.path.exists(drug_feat_path) else None
        self.disease_feat_df = pd.read_csv(disease_feat_path).set_index('node_index') if os.path.exists(disease_feat_path) else None

        # Precompute degree counts for hub penalization: 1 / sqrt(deg(p))
        if not self.edges_df.empty:
            degree_counts = pd.concat([self.edges_df['x_index'], self.edges_df['y_index']]).value_counts()
            self.degrees = degree_counts.to_dict()
        else:
            self.degrees = {}

        # Precompute entity mappings
        self.name_to_node = {}
        self.idx_to_name = {}
        self.idx_to_type = {}
        for _, row in self.nodes_df.iterrows():
            n_idx = int(row['node_index'])
            n_name = str(row['node_name'])
            n_type = str(row['node_type'])
            self.name_to_node[n_name.lower()] = (n_idx, n_type, n_name)
            self.idx_to_name[n_idx] = n_name
            self.idx_to_type[n_idx] = n_type

        # Load node_map for sequential embedding indexing
        node_map_path = os.path.join(DATA_DIR, "node_map.json")
        if os.path.exists(node_map_path):
            with open(node_map_path, "r") as f:
                self.node_map = json.load(f)
        else:
            self.node_map = {}

        # Build adjacency for fast 2-hop lookup
        self.adj = {}
        if not self.edges_df.empty:
            for _, row in self.edges_df.iterrows():
                u = int(row['x_index'])
                v = int(row['y_index'])
                rel = str(row['display_relation'])
                if u not in self.adj: self.adj[u] = []
                if v not in self.adj: self.adj[v] = []
                self.adj[u].append((v, rel))
                self.adj[v].append((u, rel))

    def _compute_saliency(self, drug_idx, p_idx, dis_idx, deg):
        """
        Computes degree-penalized path saliency S(u -> p -> v):
        S = [sigma(z_u^T z_p) + sigma(z_p^T z_v)] / [2 * sqrt(deg(p))]
        Formalized in Equation (11) of the Rapid AI manuscript.
        """
        if self.embeddings is not None and TORCH_AVAILABLE and isinstance(self.embeddings, torch.Tensor):
            try:
                gid_u = self.node_map.get(str(drug_idx))
                gid_p = self.node_map.get(str(p_idx))
                gid_v = self.node_map.get(str(dis_idx))
                if gid_u is not None and gid_p is not None and gid_v is not None:
                    zu = self.embeddings[gid_u]
                    zp = self.embeddings[gid_p]
                    zv = self.embeddings[gid_v]
                    aff_up = float(torch.sigmoid((zu * zp).sum()))
                    aff_pv = float(torch.sigmoid((zp * zv).sum()))
                    return float(((aff_up + aff_pv) / 2.0) / self.np.sqrt(deg))
            except Exception:
                pass
        return float(1.0 / self.np.sqrt(deg))

    def extract_paths(self, drug_name, disease_name, top_k=5):
        """
        Extracts 2-hop biological bridges (Drug -> Protein -> Disease) penalized
        by inverse square-root structural degree: S(u -> p -> v) = [sigma(z_u^T z_p) + sigma(z_p^T z_v)] / [2 * sqrt(deg(p))].
        """
        drug_match = self.name_to_node.get(str(drug_name).lower())
        dis_match = self.name_to_node.get(str(disease_name).lower())

        if not drug_match or not dis_match:
            return {
                "paths": [],
                "physicochemical": {},
                "disease_summary": "",
                "has_direct_paths": False
            }

        drug_idx, _, drug_canonical = drug_match
        dis_idx, _, dis_canonical = dis_match

        # 1. Candidate proteins connected to drug
        drug_neighbors = self.adj.get(drug_idx, [])
        drug_targets = {}
        for tgt_idx, rel in drug_neighbors:
            if self.idx_to_type.get(tgt_idx) == "gene/protein":
                drug_targets[tgt_idx] = rel

        # 2. Candidate proteins connected to disease
        dis_neighbors = self.adj.get(dis_idx, [])
        dis_proteins = {}
        for tgt_idx, rel in dis_neighbors:
            if self.idx_to_type.get(tgt_idx) == "gene/protein":
                dis_proteins[tgt_idx] = rel

        # 3. Direct 2-hop bridging proteins
        shared_proteins = set(drug_targets.keys()).intersection(set(dis_proteins.keys()))

        scored_paths = []
        if shared_proteins:
            for p_idx in shared_proteins:
                deg = self.degrees.get(p_idx, 1)
                saliency = self._compute_saliency(drug_idx, p_idx, dis_idx, deg)
                scored_paths.append({
                    "protein_idx": p_idx,
                    "protein_name": self.idx_to_name.get(p_idx, f"Protein_{p_idx}"),
                    "drug_relation": drug_targets[p_idx],
                    "disease_relation": dis_proteins[p_idx],
                    "degree": deg,
                    "saliency": round(float(saliency), 4),
                    "is_direct_bridge": True
                })
            scored_paths = sorted(scored_paths, key=lambda x: x["saliency"], reverse=True)[:top_k]
            has_direct = True
        else:
            # Novel Repurposing Candidate: direct 2-hop unannotated in graph
            # Rank drug targets by degree specificity to identify primary mechanism
            for p_idx, rel in drug_targets.items():
                deg = self.degrees.get(p_idx, 1)
                saliency = self._compute_saliency(drug_idx, p_idx, dis_idx, deg)
                scored_paths.append({
                    "protein_idx": p_idx,
                    "protein_name": self.idx_to_name.get(p_idx, f"Protein_{p_idx}"),
                    "drug_relation": rel,
                    "disease_relation": "latent inductive proximity",
                    "degree": deg,
                    "saliency": round(float(saliency), 4),
                    "is_direct_bridge": False
                })
            scored_paths = sorted(scored_paths, key=lambda x: x["saliency"], reverse=True)[:top_k]
            has_direct = False

        # 4. Extract physicochemical properties
        physicochem = {}
        if self.drug_feat_df is not None and drug_idx in self.drug_feat_df.index:
            row = self.drug_feat_df.loc[drug_idx]
            physicochem = {
                "molecular_weight": str(row.get("molecular_weight", "N/A")),
                "tpsa": str(row.get("tpsa", "N/A")),
                "clogp": str(row.get("clogp", "N/A")),
                "half_life": str(row.get("half_life", "N/A")),
                "mechanism_of_action": str(row.get("mechanism_of_action", "N/A"))
            }

        # 5. Extract disease clinical phenotype
        dis_summary = ""
        if self.disease_feat_df is not None and dis_idx in self.disease_feat_df.index:
            d_row = self.disease_feat_df.loc[dis_idx]
            dis_summary = str(d_row.get("mondo_definition", d_row.get("umls_description", d_row.get("mayo_symptoms", ""))))

        return {
            "drug_name": drug_canonical,
            "disease_name": dis_canonical,
            "paths": scored_paths,
            "physicochemical": physicochem,
            "disease_summary": dis_summary,
            "has_direct_paths": has_direct
        }
