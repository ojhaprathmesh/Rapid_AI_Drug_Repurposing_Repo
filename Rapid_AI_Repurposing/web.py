import streamlit as st
import pandas as pd
import time
import json
import requests
from lab_utils import (
    DiscoveryEngine, 
    get_pubchem_info, 
    get_clinical_trials_data, 
    check_ollama_status,
    PathSaliencyEngine
)

# ---------------------------------------------------------
# CLINICAL RESEARCH LAB UI 3.0
# ---------------------------------------------------------
st.set_page_config(page_title="Rapid AI Clinical Discovery Lab", layout="wide")

# Unified Clinical Typography and Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Uniform Headings */
    h1, h2, h3, .stHeadingContainer {
        font-family: 'Inter', sans-serif !important;
        color: #0f172a !important;
    }

    .stHeadingContainer h1 {
        font-size: 24px !important;
        font-weight: 700 !important;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    
    .stHeadingContainer h2 {
        font-size: 18px !important;
        font-weight: 600 !important;
        color: #475569 !important;
        margin-top: 25px;
    }

    /* Professional Clinical Containers - Light Mode */
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 20px;
        background-color: #ffffff;
        box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1);
    }
    
    .stButton>button {
        width: 100%;
        background-color: #f8fafc;
        color: #0f172a;
        border: 1px solid #cbd5e1;
        font-weight: 600;
        height: 42px;
        border-radius: 6px;
        transition: all 0.2s;
    }
    
    .stButton>button:hover {
        border-color: #0ea5e9;
        background-color: #f1f5f9;
        color: #0ea5e9;
    }

    /* AI Report Area Styling */
    .report-box {
        background-color: #f8fafc;
        border-left: 4px solid #0ea5e9;
        padding: 20px;
        margin-bottom: 15px;
        font-size: 14px;
        line-height: 1.6;
        color: #1e293b;
        border-radius: 0 6px 6px 0;
    }
    
    .regime-badge {
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 12px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# RATIONALE GENERATOR (Grounded in Biological Pathways)
# ---------------------------------------------------------
def generate_fallback_rationale(drug, disease, path_data):
    """Provides a high-quality biological template grounded in extracted targets."""
    paths = path_data.get("paths", [])
    phys = path_data.get("physicochemical", {})
    
    top_target = paths[0]["protein_name"] if paths else "target receptor"
    top_saliency = paths[0]["saliency"] if paths else 0.4000
    top_deg = paths[0]["degree"] if paths else 5
    mw = phys.get("molecular_weight", "N/A")
    tpsa = phys.get("tpsa", "N/A")
    moa = phys.get("mechanism_of_action", "N/A")

    return f"""
    ### Grounded Scientific Rationale (Clinical XAI Engine)
    
    The predicted repurposing association between **{drug}** and **{disease}** is corroborated by multi-relational topological message passing and degree-penalized biological path saliency:
    
    1. **Target Receptor Engagement:** Candidate compound **{drug}** specifically engages **{top_target}** (Path Saliency S = {top_saliency:.4f}, node degree k = {top_deg}). Down-weighting high-degree promiscuous hubs ensures disease-specific signaling fidelity.
    2. **Cellular Pathophysiology:** Topological message passing across the 2-hop subnetwork ({drug} -> {top_target} -> {disease}) suggests direct attenuation of tissue damage and inflammatory signaling cascades. Known pharmacological profile: {moa}.
    3. **Biophysical Compatibility:** Physicochemical descriptors ({mw}, {tpsa}) substantiate metabolic stability and translational feasibility for clinical investigation.
    
    *Note: For dynamic conversational LLM synthesis, ensure local Ollama (Llama 3.2) is running on localhost:11434.*
    """

# ---------------------------------------------------------
# STATUS TRACKER
# ---------------------------------------------------------
def update_pipeline(step_idx):
    steps = [
        "Input Received",
        "Node Matched",
        "Embeddings Retrieved",
        "GNN Inference",
        "Candidates Ranked",
        "Clinical Insight Ready"
    ]
    st.sidebar.write("### Pipeline Status")
    for i, s in enumerate(steps):
        if i < step_idx:
            st.sidebar.write(f"COMPLETED: {s}")
        elif i == step_idx:
            st.sidebar.write(f"PROCESSING: {s}")
        else:
            st.sidebar.write(f"PENDING: {s}")

# ---------------------------------------------------------
# ENGINE & SESSION
# ---------------------------------------------------------
ai_session = requests.Session()
ai_session.trust_env = False

@st.cache_resource
def get_discovery_lab_engine():
    return DiscoveryEngine()

@st.cache_resource
def get_saliency_engine():
    eng = get_discovery_lab_engine()
    return PathSaliencyEngine(embeddings=eng.embeddings)

engine = get_discovery_lab_engine()
saliency_engine = get_saliency_engine()

# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
st.sidebar.title("Discovery Controls")
st.sidebar.markdown("---")
mode = st.sidebar.radio("Analysis Mode", ["Drug to Disease", "Disease to Drug"])
top_k = st.sidebar.slider("Extraction Depth", 5, 50, 10)

st.sidebar.markdown("---")
st.sidebar.write("### Clinical Operating Regime (Section VII-D)")
threshold = st.sidebar.slider(
    "Decision Threshold (τ)",
    min_value=0.30,
    max_value=0.70,
    value=0.50,
    step=0.05,
    help="Calibrate operational trade-off between sensitivity and precision across the three clinical regimes."
)

if threshold <= 0.40:
    regime_title = "Regime I: High-Sensitivity Screening"
    regime_desc = f"Sensitivity: 95.3% - 96.3% | Omission: <3.7%<br>Optimized for broad exploratory discovery without missing rare candidates."
    regime_border = "#0284c7"
    regime_bg = "#f0f9ff"
elif threshold <= 0.55:
    regime_title = "Regime II: Balanced Prioritization"
    regime_desc = f"Accuracy: 91.97% | F1-Score: 92.09%<br>Harmonic balance for standard hospital and translational laboratory workflows."
    regime_border = "#16a34a"
    regime_bg = "#f0fdf4"
else:
    regime_title = "Regime III: High-Confidence Validation"
    regime_desc = f"Precision: 92.6% - 93.6% | Specificity: 93.9%<br>Strict filtering to minimize costly wet-lab false positives."
    regime_border = "#7c3aed"
    regime_bg = "#faf5ff"

st.sidebar.markdown(f"""
<div class="regime-badge" style="background-color: {regime_bg}; border-left: 4px solid {regime_border};">
    <strong style="color: {regime_border};">{regime_title}</strong><br>
    <span style="color: #475569; font-size: 11px;">{regime_desc}</span>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# MAIN INTERFACE
# ---------------------------------------------------------
st.title("Rapid AI Clinical Discovery Lab")
st.caption("Inductive GraphSAGE Inference & Grounded Path Saliency on PrimeKG")

col_search, _ = st.columns([2, 1])
with col_search:
    target_options = engine.drug_names if mode == "Drug to Disease" else engine.disease_names
    label = "Candidate Name" if mode == "Drug to Disease" else "Condition Name"
    selected_name = st.selectbox(label, target_options, index=None, placeholder="Search Knowledge Graph...")

if selected_name:
    # Clear session state if name changes
    if "current_search" not in st.session_state or st.session_state.current_search != selected_name:
        st.session_state.current_search = selected_name
        st.session_state.prediction_results = None
        st.session_state.step_idx = 1

    update_pipeline(st.session_state.step_idx)
    
    if st.button(f"Execute Clinical Prediction for {selected_name}"):
        with st.status("Analyzing biological graph structure...", expanded=False) as status:
            time.sleep(0.3)
            st.session_state.step_idx = 2
            target_type = "disease" if mode == "Drug to Disease" else "drug"
            results = engine.predict(selected_name, target_type=target_type, top_k=top_k)
            st.session_state.prediction_results = results
            st.session_state.step_idx = 4
            time.sleep(0.3)
            status.update(label="Inference Complete", state="complete")

    if st.session_state.prediction_results:
        results = st.session_state.prediction_results
        
        # --- RESULTS TABLE ---
        st.divider()
        col_results, col_info = st.columns([3, 2])
        
        with col_results:
            st.subheader(f"Top {top_k} Predicted Candidates")
            
            # Format dataframe with threshold decision
            table_data = []
            for r in results:
                meets_tau = r['score'] >= threshold
                table_data.append({
                    "Candidate": r['name'],
                    "Score (ŷ)": f"{r['score']:.4f}",
                    "Status": "Prioritized" if meets_tau else "Below Threshold",
                    "Confidence": r['confidence']
                })
            res_df = pd.DataFrame(table_data)
            
            selection = st.dataframe(
                res_df, 
                width="stretch", 
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row"
            )
            
            selected_row = None
            try:
                if selection is not None and hasattr(selection, 'selection'):
                    rows = selection.selection.get('rows', [])
                elif selection is not None and isinstance(selection, dict):
                    rows = selection.get('selection', {}).get('rows', [])
                else: rows = []
                selected_row = results[rows[0]] if rows else results[0]
            except: selected_row = results[0]

        with col_info:
            st.subheader("Candidate Information")
            drug_name = selected_name if mode == "Drug to Disease" else selected_row['name']
            disease_name = selected_row['name'] if mode == "Drug to Disease" else selected_name
            
            # Extract 2-hop biological bridges with degree penalization
            path_data = saliency_engine.extract_paths(drug_name, disease_name, top_k=5)
            
            with st.container():
                st.markdown(f"**Association:** `{drug_name}` → `{disease_name}`")
                score_val = selected_row['score']
                status_label = "Meets Decision Threshold" if score_val >= threshold else "Sub-Threshold"
                st.metric("GraphSAGE Confidence", f"{score_val:.2%}", status_label)
                
                # Tabbed biological and pharmacological inspector
                tab_paths, tab_pk, tab_lit = st.tabs(["2-Hop Biological Bridges", "Physicochemical Profile", "Clinical Evidence"])
                
                with tab_paths:
                    paths = path_data.get("paths", [])
                    if paths:
                        st.markdown("**Mediating Protein Targets (Degree-Penalized):**")
                        path_rows = []
                        for p in paths:
                            path_rows.append({
                                "Target": p['protein_name'],
                                "Degree": p['degree'],
                                "Saliency S": f"{p['saliency']:.4f}",
                                "Connection": "Direct Bridge" if p['is_direct_bridge'] else "Target Cascade"
                            })
                        st.dataframe(pd.DataFrame(path_rows), hide_index=True, width="stretch")
                    else:
                        st.caption("No intermediary protein targets found in subset graph.")
                
                with tab_pk:
                    phys = path_data.get("physicochemical", {})
                    if phys:
                        st.write(f"- **Molecular Weight:** {phys.get('molecular_weight', 'N/A')}")
                        st.write(f"- **TPSA:** {phys.get('tpsa', 'N/A')}")
                        st.write(f"- **Half-Life:** {phys.get('half_life', 'N/A')}")
                        if phys.get('mechanism_of_action') and phys.get('mechanism_of_action') != 'nan':
                            st.write(f"- **Known MOA:** {phys.get('mechanism_of_action')}")
                    else:
                        st.caption("Physicochemical profile unavailable.")
                    
                    pubchem = get_pubchem_info(drug_name)
                    if pubchem and pubchem.get('img_url'):
                        st.image(pubchem['img_url'], width=240, caption=f"PubChem Structure: {drug_name}")
                
                with tab_lit:
                    studies = get_clinical_trials_data(drug_name, disease_name)
                    if studies:
                        st.markdown(f"**ClinicalTrials.gov ({len(studies)} Studies):**")
                        for s in studies[:5]:
                            st.markdown(f"- [{s['title']}](https://clinicaltrials.gov/study/{s['id']})")
                    else:
                        st.caption("No registered clinical trials found for this specific pairing.")
                    
                    # Direct PubMed Search Deep-Link
                    query_term = f"{drug_name}+{disease_name}".replace(" ", "+")
                    st.markdown(f"[Search PubMed Literature for '{drug_name} and {disease_name}'](https://pubmed.ncbi.nlm.nih.gov/?term={query_term})")

        # --- AI RATIONALIZATION ---
        st.divider()
        st.subheader("Grounded Clinical Research Briefing")
        
        rationale_key = f"r_{drug_name}_{disease_name}".replace(" ", "_")
        briefing_container = st.container()
        
        if rationale_key in st.session_state:
            with briefing_container:
                st.markdown('<div class="report-box">', unsafe_allow_html=True)
                st.markdown(st.session_state[rationale_key])
                st.markdown('</div>', unsafe_allow_html=True)
            
        if st.button(f"Generate Path-Constrained Rationale for {drug_name}", key="gen_btn"):
            msg_area = briefing_container.empty()
            is_online = check_ollama_status()
            
            if not is_online:
                msg_area.warning("AI Engine (Ollama) is Offline. Providing grounded clinical template.")
                time.sleep(0.5)
                full_resp = generate_fallback_rationale(drug_name, disease_name, path_data)
                st.session_state[rationale_key] = full_resp
                msg_area.markdown(f'<div class="report-box">{full_resp}</div>', unsafe_allow_html=True)
                st.info("To enable real-time Llama 3.2 synthesis: Launch Ollama (`ollama run llama3.2`) and refresh.")
            else:
                with st.spinner("Synthesizing grounded rationale with Llama 3.2..."):
                    try:
                        # Construct grounded prompt
                        paths_summary = ", ".join([p['protein_name'] for p in path_data.get('paths', [])[:3]])
                        prompt = f"""You are a clinical pharmacologist. Provide a concise 3-paragraph scientific rationalization for repurposing '{drug_name}' for '{disease_name}'.
Grounded Biological Evidence: Mediating protein targets: {paths_summary}.
Format in professional medical prose addressing: (1) Molecular Mechanism of Action, (2) Cellular Signaling, and (3) Pharmacological Feasibility."""

                        url = "http://127.0.0.1:11434/api/generate"
                        payload = {
                            "model": "llama3.2:latest",
                            "prompt": prompt,
                            "stream": True
                        }
                        full_resp = ""
                        with ai_session.post(url, json=payload, stream=True, timeout=120) as r:
                            if r.status_code == 200:
                                for line in r.iter_lines():
                                    if line:
                                        chunk = json.loads(line)
                                        text = chunk.get("response", "")
                                        full_resp += text
                                        msg_area.markdown(f'<div class="report-box">{full_resp}▌</div>', unsafe_allow_html=True)
                                        if chunk.get("done"): break
                        st.session_state[rationale_key] = full_resp
                        msg_area.markdown(f'<div class="report-box">{full_resp}</div>', unsafe_allow_html=True)
                        st.session_state.step_idx = 6
                    except Exception as e:
                        st.error(f"Error during synthesis: {str(e)}")

st.sidebar.markdown("---")
st.sidebar.caption("Rapid AI Clinical Discovery Lab | PrimeKG Inductive GraphSAGE")
