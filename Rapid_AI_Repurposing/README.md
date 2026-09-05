# 🛡️ Rapid AI Drug Repurposing Discovery Lab

An end-to-end, high-performance drug repurposing pipeline using **Heterogeneous Graph Neural Networks (GraphSAGE)** and **LLaMA 3.2** for clinical rationalization.

## 🚀 Overview
This module contains the complete machine learning pipeline to predict novel therapeutic uses for existing drugs using the **Precision Medicine Knowledge Graph (PrimeKG)**. It features an interactive Streamlit dashboard for real-time hypothesis testing, subnetwork exploration, and biological evidence verification.

## 🛠️ Project Structure

```text
Rapid_AI_Repurposing/
├── pipeline/                     # Core ML pipeline stages
│   ├── step1_preprocess.py       # Extracts chemical features & TF-IDF encodings
│   ├── step2_build_graph.py      # Constructs Heterogeneous GNN topology (PyG)
│   ├── step3_train.py            # Optimizes GraphSAGE model (97.34% AUC on PrimeKG)
│   ├── step4_predict.py          # Generates ranked candidate repurposing predictions
│   ├── step5_rationalize.py      # Grounded LLM reasoning with degree-penalized path saliency
│   └── step_evaluate.py          # Comprehensive test evaluation & metrics export
│
├── visualization/                # Centralized publication & report asset generation
│   ├── generate_all_figures.py   # Master 300 DPI figure generator for paper & docs
│   └── legacy_plots.py           # Supplementary research diagrams
│
├── tests/                        # Audit & verification suite
│   ├── verify_paper_claims.py    # Standalone audit suite verifying all 28 paper claims (0.05s)
│   ├── verify_leakage.py         # Split leakage verification
│   ├── audit_graph.py            # Graph topology auditing
│   └── test_connectivity.py     # Diagnostics for Ollama (Llama 3.2) & external biomedical APIs
│
├── reports/                      # Clinical & research documentation
│   ├── clinical_rationalization_report.md
│   ├── humanized_research_paper.md
│   └── research_paper.md
│
├── preprocessed_data/            # Cached graph tensors & model weights (85 KB checkpoint)
├── evaluation_outputs/           # Test curves, metrics CSVs, prediction matrices
├── lab_utils.py                  # Standalone GNN inference engine & 2-hop clinical explorer
├── web.py                        # Interactive Streamlit Clinical Discovery Lab dashboard
├── requirements.txt              # Module dependencies
└── README.md
```

---

## 💻 How to Run

### 1. Requirements
Ensure you have **Python 3.10+** and optionally **Ollama** (with `llama3.2`) installed for AI rationalization.

### 2. Setup Environment
```bash
pip install -r requirements.txt
```

### 3. Launch the Discovery Dashboard
```bash
streamlit run web.py
```

### 4. Running the ML Pipeline
You can run any step directly from root or via the `pipeline/` module:
```bash
# Data Preprocessing & Feature Extraction
python step1_preprocess.py

# Heterogeneous Graph Construction
python step2_build_graph.py

# GraphSAGE Training
python step3_train.py

# Candidate Scoring & Ranking
python step4_predict.py

# Benchmark Evaluation
python step_evaluate.py
```

### 5. Running the Independent Verification Suite
Verify all 28 mathematical, architectural, and clinical claims in the research paper:
```bash
python verify_paper_claims.py
# Or directly from tests/:
python tests/verify_paper_claims.py
```

### 6. Generating All Paper & Report Figures
Generate 300 DPI publication-ready figures synchronized directly to both the LaTeX paper (`images/`) and technical reports (`docs/report_figures/`):
```bash
python -m visualization.generate_all_figures --target all
```

---

## 🔬 Technologies Used
- **ML Engine:** PyTorch, PyTorch Geometric (GraphSAGE)
- **Frontend:** Streamlit (Custom Clinical Theme)
- **LLM:** Ollama (LLaMA 3.2)
- **APIs:** PubChem PUG REST, ClinicalTrials.gov v2

## 📝 License
This project is for academic and research purposes.
