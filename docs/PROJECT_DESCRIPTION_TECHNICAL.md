# 🛡️ Rapid AI Drug Repurposing Discovery Lab
### *Harnessing Heterogeneous Graph Neural Networks & LLM-Driven Rationalization*

## 🌟 Executive Summary
The **Rapid AI Drug Repurposing Discovery Lab** is a high-performance computational pharmacology platform. It integrates multi-modal data—molecular features, clinical text, and graph topology—to predict novel therapeutic uses for existing drugs. By combining **Heterogeneous Graph Neural Networks (GraphSAGE)** with **LLaMA 3.2**, the system provides a dual-layer discovery approach: high-precision neural inference and clinically-grounded biological rationalization.

---

## 🛠️ Technical Deep Dive: The Pipeline

> [!TIP]
> **[FIGURE 1: System Architecture Overview]**
> *Description: A high-level block diagram showing the flow from raw PrimeKG data input, through the 5-step preprocessing and training pipeline, ending with the Streamlit Dashboard and AI Rationalization output.*

### 1. Multi-Modal Feature Engineering (`step1`)
The system transforms heterogeneous biomedical raw data into a high-dimensional unified feature space ($\mathbb{R}^{131}$):

#### A. Semantic Textual Pipeline ($128$ dimensions)
- **Data Harvesting**:
  - **Drugs**: Concatenates `description`, `indication`, and `mechanism_of_action`.
  - **Diseases**: Merges `mondo_name`, `mondo_definition`, and `umls_description`.
  - **General Nodes**: Uses raw `node_name` as a fallback.
- **Preprocessing Pipeline**:
  - **Normalization**: Regex-based cleaning to remove non-alphanumeric characters and normalize white space.
  - **Tokenization**: Lowercase conversion and removal of English stop-words.
- **Vectorization**:
  - **Algorithm**: TF-IDF (Term Frequency-Inverse Document Frequency) with a max-feature cap of 128.
  - **Normalization**: Native **L2 Normalization** is applied to the resulting vectors to ensure unit length across the semantic subspace.

#### B. Biochemical Numerical Pipeline ($3$ dimensions)
- **Raw Feature Extraction**: Captures **Molecular Weight**, **TPSA** (Topological Polar Surface Area), and **ClogP** (Octanol-Water Partition Coefficient).
- **Data Cleaning**:
  - **Regex Extraction**: Since raw data often contains units (e.g., "500.1 Da"), the system uses a numeric extractor `([-+]?\d*\.\d+|\d+)` to isolate float values.
  - **Imputation**: Missing values (NaNs) are zero-filled prior to scaling.
- **Scaling**:
  - **Algorithm**: **StandardScaler (Z-Score Standardization)**.
  - **Mathematics**: $z = \frac{x - \mu}{\sigma}$, ensuring all biochemical features are centered at zero with a standard deviation of one.

#### C. Unified Tensor Assembly
- **Concatenation**: The text features ($\mathbb{R}^{128}$) and scaled numerical features ($\mathbb{R}^{3}$) are merged along the feature axis to form the final $131$-dimensional node tensor.
- **Validation**:
  - **Integrity Check**: Automated NaN/Infinity checks ensure the tensor is clean for GPU processing.
  - **Type Casting**: Final conversion to `torch.float32` for optimized message passing in PyTorch Geometric.
- **Persistence**: Mapping dictionaries and the final `x.pt` tensor are serialized to disk to maintain state across the pipeline.

> [!TIP]
> **[FIGURE 2: Feature Space Distribution]**
> *Description: A visualization showing the distribution of numerical features (Molecular Weight, TPSA, ClogP) before and after StandardScaler, along with a word cloud or bar chart representing the top TF-IDF terms extracted from the clinical text.*

### 2. Graph Topology Construction (`step2`)
The graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ is constructed with node types $\mathcal{T}_v \in \{\text{Drug, Disease, Gene}\}$ and relation types $\mathcal{T}_e$.
- **Undirected Expansion**: To facilitate message passing in the GNN, every directed edge $(u, v, r)$ is supplemented with a reverse edge $(v, u, r_{\text{rev}})$.
- **Disjoint Edge Splitting**: 
  - **Positive Edges**: Split 80/10/10 into Train/Val/Test sets.
  - **Negative Sampling**: Implements a strict "Collision-Free" sampler that selects Drug-Disease pairs $(u, v)$ such that $(u, v) \notin \mathcal{E}_{\text{positive}}$ and $(u, v) \notin \mathcal{E}_{\text{other\_splits}}$.

> [!TIP]
> **[FIGURE 3: Heterogeneous Graph Schema]**
> *Description: A node-link diagram illustrating the heterogeneous nature of the graph, showing different node types (Drugs, Diseases, Proteins) and the various relation types (Treats, Associates, etc.) between them.*

### 3. Neural Architecture: GraphSAGE Engine (`step3`)
The model utilizes a **Heterogeneous GraphSAGE** architecture for link prediction:
- **Encoder**: 
  - Two layers of `SAGEConv` (Sample and Aggregate).
  - **Layer 1**: $131 \to 64$ channels with ReLU activation.
  - **Layer 2**: $64 \to 32$ channels (Final Embedding Space $\mathcal{Z}$).
- **Decoder**:
  - Score Function: $S(u, v) = \sigma(\mathbf{z}_u \cdot \mathbf{z}_v)$
  - Methodology: Computes the dot product between the learned embeddings of a Drug and a Disease, passed through a Sigmoid activation for probability mapping.
- **Training**: Uses **BCEWithLogitsLoss** to optimize the link prediction task.

> [!TIP]
> **[FIGURE 4: Training Metrics & Convergence]**
> *Description: A plot showing the training and validation loss curves over epochs, along with the AUC-ROC and Precision-Recall metrics, demonstrating the model's learning progress and final performance.*

### 4. Knowledge Discovery & Scoring (`step4`)
- **Global Inference**: The trained encoder generates embeddings $\mathcal{Z}$ for all nodes.
- **Exclusion Logic**: The system iterates through all $\text{Drug} \times \text{Disease}$ pairs, excluding those already present in the source Knowledge Graph.
- **Ranking**: Novel pairs are ranked by their sigmoid score, with the top 50 persisted for clinical review.

---

## 🤖 Clinical Rationalization Engine
The system bridges the gap between AI predictions and clinical utility using **LLaMA 3.2** (via Ollama).

- **Prompt Engineering**: The LLM is queried using a zero-shot clinical prompt: *"Explain the therapeutic potential of {drug} for {disease} using clinical terminology."*
- **Reasoning Structure**: The AI synthesizes:
  1. **Pathway Analysis**: Identifying potential protein targets.
  2. **Structural Plausibility**: Based on the 131-dimensional feature similarity.
  3. **Mechanism Alignment**: Correlating drug action with disease pathophysiology.

> [!TIP]
> **[FIGURE 5: AI Rationalization Output Example]**
> *Description: A screenshot of the 'Clinical Research Briefing' section from the dashboard, showing a detailed LLaMA 3.2 generated report for a specific drug-disease prediction.*

---

## 📊 Interactive Discovery Lab (`web.py`)
A Streamlit-based interface designed for real-time hypothesis testing:

> [!TIP]
> **[FIGURE 6: Dashboard User Interface]**
> *Description: A wide-angle screenshot of the Streamlit dashboard showing the discovery controls, the ranked prediction table, and the integrated PubChem molecular viewer.*

- **API Orchestration**:
  - **PubChem PUG REST**: Real-time molecular structure visualization and SMILES retrieval.
  - **ClinicalTrials.gov v2**: Dynamic fetching of current clinical evidence to validate AI predictions.
- **Pipeline Monitoring**: A sidebar-integrated "Status Tracker" monitors the lifecycle of a prediction from "Node Matching" to "Scientific Rationale Generation."

---

## 🔬 System Stack
- **Languages**: Python 3.10+
- **Deep Learning**: PyTorch 2.1+, PyTorch Geometric 2.4+
- **Data Science**: Pandas, NumPy, Scikit-Learn
- **LLM**: Ollama API (Llama 3.2)
- **APIs**: PubChem REST, ClinicalTrials.gov JSON API
- **Styling**: Vanilla CSS, Google Fonts (Inter)

---

## ⚖️ Design Trade-offs & Engineering Decisions

Building a high-performance drug repurposing system requires balancing computational efficiency with predictive accuracy. Below are the core trade-offs made during development:

### 1. TF-IDF vs. Transformer-based Embeddings (e.g., SciBERT)
- **Decision**: Used **TF-IDF (128-dim)** for textual feature extraction.
- **Trade-off**: While Transformers capture deep semantic context, they are computationally expensive and require significant GPU memory. TF-IDF was chosen to allow for **instant preprocessing** on CPU-only environments without sacrificing the model's ability to differentiate between distinct medical terminologies.
- **Impact**: Reduced preprocessing time from hours to seconds while maintaining high AUC-ROC through structural graph connectivity.

### 2. Full-Batch vs. Mini-Batch (Neighbor Loading)
- **Decision**: Implemented **Full-Batch Training** for the GraphSAGE encoder.
- **Trade-off**: Full-batch training provides the most stable gradients as it considers the entire graph topology in every pass. However, it is strictly limited by the **GPU VRAM** (capacity cap). 
- **Impact**: Optimized for the current 18GB hardware environment. For larger datasets (e.g., full PrimeKG), the system would need to transition to `NeighborLoader` or `ClusterGCN`, which introduces sampling noise.

### 3. Static vs. Dynamic Negative Sampling
- **Decision**: **Static sampling** performed once in `step2`.
- **Trade-off**: Dynamic (online) negative sampling provides the model with new "hard negatives" in every epoch, improving robustness. However, it significantly increases the training time per epoch due to the overhead of collision-checking on a large graph.
- **Impact**: Chose static sampling to ensure **rapid model iteration** and deterministic results during the research phase.

### 4. Dot-Product vs. MLP Decoder
- **Decision**: Used a **Simple Dot-Product Decoder** for link prediction scores.
- **Trade-off**: An MLP (Multi-Layer Perceptron) decoder can learn non-linear interaction patterns between drug and disease embeddings. However, a dot product treats the embedding space as a latent metric space, ensuring that **similarity in the embedding space directly correlates to therapeutic potential**.
- **Impact**: Improved interpretability and reduced model parameters, preventing overfitting on the limited positive edge set.

### 5. Local LLM (Ollama) vs. Cloud APIs
- **Decision**: Integrated **Ollama (LLaMA 3.2)** for rationalization.
- **Trade-off**: Cloud APIs (like GPT-4) offer higher reasoning capabilities but introduce latency, cost, and data privacy concerns. 
- **Impact**: Prioritized **data sovereignty and offline capability**, allowing the system to run in secure lab environments without external dependencies.

---
*Technical Specification Document - Rapid AI PrimeKG Discovery Pipeline v3.1*
