# 🛡️ Rapid AI Drug Repurposing Discovery Lab

An end-to-end, high-performance drug repurposing pipeline using **Heterogeneous Graph Neural Networks (GraphSAGE)** and **LLaMA 3.2** for clinical rationalization.

## 🚀 Overview
This repository contains a complete pipeline to predict novel therapeutic uses for existing drugs using the **Biomedical Knowledge Graph (DRKG)**. It features a modern Streamlit dashboard for real-time hypothesis testing and biological evidence verification.

## 🛠️ Project Structure
- `step1_preprocess.py`: Extracts chemical features and text encodings (TF-IDF).
- `step2_build_graph.py`: Constructs the Heterogeneous GNN topology.
- `step3_train.py`: Optimizes the GraphSAGE model (96.95% AUC).
- `step4_predict.py`: Generates the ranked prediction board.
- `web.py`: The interactive AI Discovery Dashboard.
- `lab_utils.py`: Backend engine for GNN inference and external API (PubChem/ClinicalTrials) integrations.

---

## 💻 How to Run in Visual Studio (VS Code)

Follow these steps to run the project locally on your machine:

### 1. Requirements
Ensure you have **Python 3.10+** and **Ollama** installed on your system.

### 2. Setup Environment
Open your folder in VS Code and open the **Integrated Terminal** (`Ctrl + ` `):
```bash
# Install dependencies
pip install -r requirements.txt
```

### 3. Launch the Dashboard
Run the following command in the VS Code terminal:
```bash
python -m streamlit run web.py
```

### 4. Running the ML Pipeline (Optional)
If you wish to re-run the training or preprocessing, execute the steps in order:
```bash
python step1_preprocess.py
python step2_build_graph.py
python step3_train.py
```

---

## 🔬 Technologies Used
- **ML Engine:** PyTorch, PyTorch Geometric (GraphSAGE)
- **Frontend:** Streamlit (Custom Medical Theme)
- **LLM:** Ollama (LLaMA 3.2)
- **APIs:** PubChem PUG REST, ClinicalTrials.gov v2

## 📝 License
This project is for academic and research purposes.
