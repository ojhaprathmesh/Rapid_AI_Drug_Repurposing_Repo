# Rapid AI Drug Repurposing: A GraphSAGE-based Clinical Discovery Lab on PrimeKG

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.3+-orange.svg)](https://pyg.org/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper: IEEE Format](https://img.shields.io/badge/Paper-IEEE%20Format-brightgreen.svg)](Rapid_AI_Drug_Repurposing__A_GraphSAGE_based_Clinical_Discovery_Lab_for_Drug_Repurposing_on_PrimeKG/research_paper.tex)

> **Rapid AI** is an end-to-end computational drug repurposing and clinical decision-support platform. By combining inductive Graph Neural Networks (**GraphSAGE**) over the Precision Medicine Knowledge Graph (**PrimeKG**) with local, grounded **Explainable AI (XAI)** reasoning, Rapid AI predicts novel therapeutic associations and synthesizes evidence-based clinical rationales.

---

## 🌟 Key Highlights

- **Predictive Performance:** Achieves **0.9734 AUC-ROC**, **91.97% Accuracy**, and **92.09% F1-score** across 1,880 balanced held-out test links.
- **Principled Subgraph Curation:** Distills PrimeKG's 129k nodes and 8.1M edges into a targeted 10,597-node, 161,632-edge therapeutic triad (Drugs, Diseases, Target Proteins) to prevent oversmoothing while preserving biological fidelity.
- **Multimodal Node Embeddings:** Synthesizes 131-dimensional feature vectors integrating 128-dimensional TF-IDF textual semantic embeddings with 3 standardized physicochemical descriptors (Molecular Weight, TPSA, ClogP).
- **Grounded Explainability:** Queries 2-hop topological bridging paths ($\text{Drug} \to \text{Protein} \to \text{Disease}$) and routes them through a local Large Language Model (**Llama 3.2 via Ollama**) to generate hallucination-free mechanistic rationales under strict data privacy.
- **Interactive Discovery Lab:** Full-featured Streamlit dashboard for interactive graph navigation, candidate link ranking, subnetwork visualization, and real-time clinical hypothesis generation.

---

## 📊 Benchmark Evaluation

Evaluated against classical heuristics and neural architectures on the held-out PrimeKG benchmark (1,880 test pairs):

| Model Architecture | AUC-ROC | Accuracy | Precision | Recall | F1-Score | Paradigm |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Common Neighbors** | 82.60% | 75.32% | 74.80% | 73.63% | 74.21% | Pure Graph Heuristic |
| **Feature-Only MLP** | 78.40% | 72.13% | 71.50% | 69.57% | 70.52% | Non-Graph Baseline |
| **GCN (Kipf & Welling)** | 94.12% | 88.45% | 87.20% | 89.02% | 88.10% | Spectral GNN |
| **GAT (Veličković et al.)** | 95.80% | 90.15% | 89.60% | 91.26% | 90.42% | Attention GNN |
| **GraphSAGE (Proposed)** | **97.34%** | **91.97%** | **90.71%** | **93.51%** | **92.09%** | **Inductive Representation** |

---

## 🧪 Top Prioritized Repurposing Hypotheses

| Candidate Compound | Approved Indication | Predicted Indication | $\hat{y}_{uv}$ | Biological Mechanism & Literature Validation |
|---|---|---|:---:|---|
| **Niacin** (Vitamin $\text{B}_3$) | Dyslipidemia / pellagra | Ocular hypertension | **0.9999** | Restores depleted neuronal NAD$^+$, preventing retinal ganglion cell mitochondrial dysfunction (*Clin Exp Ophthalmol*). |
| **Vardenafil** | Erectile dysfunction | Hypertension | **0.9998** | Selective PDE5 inhibition elevating cGMP to drive vascular smooth muscle relaxation and reduce arterial resistance (*J Am Coll Cardiol*). |
| **Doxorubicin** | Solid tumors / sarcoma | T-cell leukemia | **0.9998** | Topoisomerase II$\alpha$ inhibition and DNA intercalation inducing apoptotic arrest in malignant T-lymphoblasts (*Blood*). |
| **Polaprezinc** (Zinc L-carnosine) | Gastric mucosal lesions | Drug-induced osteoporosis | **0.9997** | Dual regulation: stimulates osteoblastic collagen synthesis while inhibiting NF-$\kappa$B osteoclastogenesis (*Mol Cell Biochem*). |

---

## 📁 Repository Structure

```text
Rapid_AI_Drug_Repurposing/
├── .gitignore                          # Centralized rules (blocks >100MB files, keeps curated data)
├── README.md                           # Project overview & documentation
├── LICENSE                             # MIT open-source license
├── requirements.txt                    # Top-level Python dependencies
│
├── Rapid_AI_Drug_Repurposing__A_GraphSAGE_based_Clinical_Discovery_Lab_for_Drug_Repurposing_on_PrimeKG/
│   ├── research_paper.tex              # Peer-reviewed, IEEE-compliant LaTeX manuscript
│   ├── IEEEtran.cls                    # Official IEEE conference template
│   ├── system_arch.png                 # End-to-end framework architecture
│   ├── interaction_workflow.png        # Sequence interaction diagram
│   ├── dataset_split_distribution.png  # Reconciled 80/10/10 split distribution
│   ├── graph_entities_schema.png       # Knowledge graph schema
│   ├── graph_visualization_primekg.png # 10.5k-node network cluster visualization
│   ├── metrics.jpeg                    # Quantitative performance breakdown
│   ├── roc_curve_final.png             # Final ROC trajectory (AUC = 0.9734)
│   └── model_convergence.png           # Training dynamics & epoch 62 checkpoint
│
├── Rapid_AI_Repurposing/               # Discovery Lab Engine & Web Dashboard
│   ├── pipeline/                       # End-to-end ML pipeline (preprocess -> train -> evaluate)
│   │   ├── step1_preprocess.py         # Subgraph extraction & TF-IDF feature scaling
│   │   ├── step2_build_graph.py        # PyTorch Geometric heterogeneous graph construction
│   │   ├── step3_train.py              # Inductive GraphSAGE neural model training
│   │   ├── step4_predict.py            # Latent candidate scoring and ranking
│   │   ├── step5_rationalize.py        # Grounded LLM clinical reasoning
│   │   └── step_evaluate.py            # Benchmark evaluation & metrics generation
│   ├── visualization/                  # Unified publication & report asset generation
│   │   └── generate_all_figures.py     # Master 300 DPI figure generator for paper & docs
│   ├── tests/                          # Independent verification & audit suite
│   │   ├── verify_paper_claims.py      # Standalone replication suite (28/28 passed)
│   │   ├── verify_leakage.py           # Split leakage verification
│   │   ├── audit_graph.py              # Graph topology auditing
│   │   └── test_connectivity.py       # API & Ollama connectivity suite
│   ├── reports/                        # Clinical reports & research documentation
│   ├── web.py                          # Interactive Streamlit discovery lab
│   ├── lab_utils.py                    # Standalone GNN inference engine & 2-hop explorer
│   ├── preprocessed_data/              # All PyG tensors & trained weights (85 KB)
│   └── evaluation_outputs/             # Test evaluation curves, metrics, and CSVs
│
├── dataverse_files/                    # Curated datasets (all clone-and-run)
│   ├── cleaned_nodes.csv               # 10,597 curated nodes
│   ├── cleaned_edges.csv               # 161,632 curated edges
│   ├── nodes_subset.csv                # Entity lookup metadata
│   ├── edges_subset.csv                # Edge lookup metadata
│   └── top_20_drug_repurposing_predictions.csv
│
└── docs/                               # Technical architecture specs, reports & figures
    ├── PROJECT_DESCRIPTION_TECHNICAL.md # In-depth technical architecture document
    ├── SYSTEM_ARCHITECTURE.mmd         # Mermaid system architecture flowchart
    ├── PRJ-III Report.pdf              # Comprehensive project report
    └── report_figures/                 # High-resolution architectural diagrams
```

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/ojhaprathmesh/Rapid_AI_Drug_Repurposing_Repo.git
cd Rapid_AI_Drug_Repurposing_Repo
```

### 2. Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Launch Interactive Clinical Discovery Lab
Run the pre-trained interactive dashboard immediately (no retraining or massive dataset download needed):
```bash
cd Rapid_AI_Repurposing
streamlit run web.py
```

### 4. Retraining & Evaluation
To retrain or run the comprehensive evaluation suite:
```bash
cd Rapid_AI_Repurposing
python step3_train.py
python step_evaluate.py
```

---

## 📜 Citation

If you use Rapid AI or the curated PrimeKG benchmark in your research, please cite:

```bibtex
@article{ojha2026rapidai,
  title={Rapid AI Drug Repurposing: A GraphSAGE-based Clinical Discovery Lab for Drug Repurposing on PrimeKG},
  author={Ojha, Prathmesh and Bansal, Jhankar},
  journal={IEEE Transactions / Conference Proceedings},
  year={2026}
}
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
