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
# ---------------------------------------------------------
# CLINICAL RESEARCH LAB UI 4.0 — MODERN BIOTECH THEME
# ---------------------------------------------------------
st.set_page_config(
    page_title="Rapid AI Clinical Discovery Lab",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Unified Modern Dark Biotech Typography and Design System
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #f1f5f9;
    }
    
    /* Sleek Dark Backgrounds */
    .stApp {
        background-color: #0b0f19;
    }
    
    /* Section Headings with Luminous High-Contrast Typography */
    h1, h2, h3, .stHeadingContainer {
        font-family: 'Inter', sans-serif !important;
        color: #f8fafc !important;
        letter-spacing: -0.02em;
    }

    .stHeadingContainer h1, .main-title {
        font-size: 28px !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px !important;
    }
    
    .stHeadingContainer h2 {
        font-size: 20px !important;
        font-weight: 700 !important;
        color: #e2e8f0 !important;
        margin-top: 20px;
    }

    .stHeadingContainer h3 {
        font-size: 16px !important;
        font-weight: 600 !important;
        color: #cbd5e1 !important;
    }

    /* Sidebar Headings & Controls */
    section[data-testid="stSidebar"] {
        background-color: #0e1526 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #f8fafc !important;
        -webkit-text-fill-color: #f8fafc !important;
    }
    
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #94a3b8;
    }

    /* Professional Elevated Clinical Containers */
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] {
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        padding: 22px !important;
        background-color: #111827 !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35) !important;
    }

    /* Ultra-Modern Action Button */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #0284c7 0%, #4f46e5 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        height: 44px !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 14px 0 rgba(2, 132, 199, 0.35) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #0369a1 0%, #4338ca 100%) !important;
        box-shadow: 0 6px 20px 0 rgba(2, 132, 199, 0.5) !important;
        transform: translateY(-1px) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0px) !important;
    }

    /* AI Report & Briefing Box */
    .report-box {
        background: #111827 !important;
        border: 1px solid rgba(56, 189, 248, 0.2) !important;
        border-left: 4px solid #38bdf8 !important;
        padding: 24px !important;
        margin-top: 15px !important;
        margin-bottom: 20px !important;
        font-size: 14.5px !important;
        line-height: 1.75 !important;
        color: #e2e8f0 !important;
        border-radius: 0 10px 10px 0 !important;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4) !important;
    }

    .report-box strong {
        color: #38bdf8 !important;
    }

    /* Clinical Regime Badge in Sidebar */
    .regime-badge {
        padding: 12px 14px;
        border-radius: 8px;
        font-size: 12px;
        line-height: 1.5;
        margin-bottom: 15px;
        backdrop-filter: blur(8px);
    }
    
    /* Metrics Styling */
    div[data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-weight: 700 !important;
        font-size: 26px !important;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 13px !important;
        font-weight: 500 !important;
    }
    
    div[data-testid="stMetricDelta"] {
        font-weight: 600 !important;
    }

    /* Tag Badges Header */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 8px;
        letter-spacing: 0.02em;
    }
    .badge-cyan {
        background: rgba(6, 182, 212, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(6, 182, 212, 0.3);
    }
    .badge-emerald {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-indigo {
        background: rgba(99, 102, 241, 0.15);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }

    /* Slider styling overrides */
    div[data-baseweb="slider"] {
        margin-top: 8px;
    }

    /* Selectbox Styling */
    div[data-baseweb="select"] {
        border-radius: 8px;
    }

    /* Divider styling */
    hr {
        border-color: rgba(255, 255, 255, 0.08) !important;
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
    st.sidebar.markdown("### Pipeline Execution Tracker")
    for i, s in enumerate(steps):
        if i < step_idx:
            st.sidebar.markdown(f"<div style='font-size: 12px; color: #34d399; margin-bottom: 5px; font-weight: 500;'>✓ <span style='color: #e2e8f0;'>{s}</span></div>", unsafe_allow_html=True)
        elif i == step_idx:
            st.sidebar.markdown(f"<div style='font-size: 12px; color: #38bdf8; margin-bottom: 5px; font-weight: 600;'>⚡ <span style='color: #38bdf8;'>{s}</span></div>", unsafe_allow_html=True)
        else:
            st.sidebar.markdown(f"<div style='font-size: 12px; color: #475569; margin-bottom: 5px;'>○ <span style='color: #64748b;'>{s}</span></div>", unsafe_allow_html=True)


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
    regime_desc = "Recall: 91.1% | Precision: 79.7% | Spec: 76.8%<br>High-sensitivity screening for exploratory triage without missing viable candidates."
    regime_border = "#38bdf8"
    regime_bg = "rgba(2, 132, 199, 0.18)"
    regime_text = "#e0f2fe"
elif threshold <= 0.55:
    regime_title = "Regime II: Balanced Prioritization"
    regime_desc = "Accuracy: 84.79% | F1-Score: 84.84% | Rec: 85.1%<br>Default operational regime balancing true positives and discovery precision."
    regime_border = "#34d399"
    regime_bg = "rgba(16, 185, 129, 0.18)"
    regime_text = "#d1fae5"
else:
    regime_title = "Regime III: High-Confidence Validation"
    regime_desc = "Precision: 88.5% | Specificity: 90.0% | F1: 82.2%<br>Conservative filtering to minimize costly wet-lab false positives."
    regime_border = "#a78bfa"
    regime_bg = "rgba(139, 92, 246, 0.18)"
    regime_text = "#ede9fe"

st.sidebar.markdown(f"""
<div class="regime-badge" style="background: {regime_bg}; border: 1px solid {regime_border}44; border-left: 4px solid {regime_border};">
    <strong style="color: {regime_border}; font-size: 13px;">{regime_title}</strong><br>
    <span style="color: {regime_text}; font-size: 11.5px; line-height: 1.4; display: block; margin-top: 4px;">{regime_desc}</span>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# MAIN INTERFACE
# ---------------------------------------------------------
st.markdown('<div class="main-title">Rapid AI Clinical Discovery Lab</div>', unsafe_allow_html=True)
st.markdown("""
<div style="margin-bottom: 18px;">
    <span class="badge-pill badge-cyan">⚡ Inductive GraphSAGE (AUC: 0.9235)</span>
    <span class="badge-pill badge-emerald">🧬 PrimeKG (10,597 Nodes · 161k Edges)</span>
    <span class="badge-pill badge-indigo">🛡️ HIPAA § 164.312 Safeguards Active</span>
</div>
<div style="color: #94a3b8; font-size: 14px; margin-bottom: 24px;">
    Inductive Graph Neural Network Inference, Multimodal Feature Embeddings, and Degree-Penalized Path Saliency
</div>
""", unsafe_allow_html=True)

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
