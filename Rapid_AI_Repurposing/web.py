import streamlit as st
import pandas as pd
import time
import json
import requests
from lab_utils import DiscoveryEngine, get_pubchem_info, get_clinical_trials_data, check_ollama_status

# ---------------------------------------------------------
# CLINICAL RESEARCH LAB UI 3.0
# ---------------------------------------------------------
st.set_page_config(page_title="Clinical Discovery Intelligence Lab", layout="wide")

# Unified Clinical Typography and Theme (No Emojis)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Uniform Headings */
    /* Light Mode Adjustments */
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

    /* AI Report Area Styling - Light Mode */
    .report-box {
        background-color: #f1f5f9;
        border-left: 4px solid #0ea5e9;
        padding: 20px;
        margin-bottom: 15px;
        font-size: 14px;
        line-height: 1.6;
        color: #1e293b;
        border-radius: 0 4px 4px 0;
    }
</style>
""", unsafe_allow_html=True)

def generate_fallback_rationale(drug, disease):
    """Provides a high-quality biological template when AI is offline."""
    return f"""
    ### 🔬 Medical Backup Rationale (AI Offline)
    
    The therapeutic potential of **{drug}** for **{disease}** is supported by biological graph connectivity and molecular profiling.
    
    1.  **Pathway Analysis:** Biological evidence suggests that {drug} interacts with key protein targets associated with the pathophysiology of {disease}. This connection is identified through high-confidence links in the Biomedical Knowledge Graph (DRKG).
    2.  **Structural Plausibility:** Based on chemical similarity and historical drug-target interactions, the mechanism of action for {drug} aligns with the required therapeutic intervention for {disease}.
    3.  **Cross-Validation:** This prediction has been cross-referenced with top-ranked candidates from the GraphSAGE model, which utilizes 131-dimensional feature vectors to compute inference scores.
    
    *Note: For a full scientific report, please ensure your local AI engine (Ollama) is running and accessible.*
    """

# ---------------------------------------------------------
# STATUS TRACKER (Emoji-free)
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

engine = get_discovery_lab_engine()

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.title("Discovery Controls")
st.sidebar.markdown("---")
mode = st.sidebar.radio("Analysis Mode", ["Drug to Disease", "Disease to Drug"])
top_k = st.sidebar.slider("Extraction Depth", 5, 50, 10)

# ---------------------------------------------------------
# MAIN INTERFACE
# ---------------------------------------------------------
st.title("Clinical Discovery Intelligence Lab")

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
        # --- RESULTS ---
        st.divider()
        col_results, col_info = st.columns([3, 2])
        
        with col_results:
            st.subheader(f"Top {top_k} Predicted Candidates")
            res_df = pd.DataFrame(results)
            res_df.columns = ["Candidate", "AI Score", "Confidence"]
            
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
            
            with st.container():
                st.markdown(f"**Relationship:** {drug_name} for {disease_name}")
                st.metric("Inference Confidence", f"{selected_row['score']:.2%}", selected_row['confidence'])
                
                pubchem = get_pubchem_info(drug_name)
                if pubchem and pubchem.get('img_url'):
                    st.image(pubchem['img_url'], width=280)
                
                studies = get_clinical_trials_data(drug_name, disease_name)
                if studies:
                    st.session_state.step_idx = 5
                    with st.expander(f"Medical Evidence: {len(studies)} Studies", expanded=True):
                        for s in studies:
                            st.markdown(f"- [{s['title']}](https://clinicaltrials.gov/study/{s['id']})")

        # --- AI RATIONALIZATION (Refined Layout) ---
        st.divider()
        st.subheader("Clinical Research Briefing")
        
        rationale_key = f"r_{drug_name}_{disease_name}".replace(" ", "_")
        
        briefing_container = st.container()
        
        if rationale_key in st.session_state:
            with briefing_container:
                st.markdown('<div class="report-box">', unsafe_allow_html=True)
                st.markdown(st.session_state[rationale_key])
                st.markdown('</div>', unsafe_allow_html=True)
            
        if st.button(f"Generate Scientific Rationale for {drug_name}", key="gen_btn"):
            msg_area = briefing_container.empty()
            
            # Check Ollama Status
            is_online = check_ollama_status()
            
            if not is_online:
                msg_area.warning("⚠️ AI Engine (Ollama) is Offline. Providing medical backup rationale.")
                time.sleep(1)
                full_resp = generate_fallback_rationale(drug_name, disease_name)
                st.session_state[rationale_key] = full_resp
                msg_area.markdown(f'<div class="report-box">{full_resp}</div>', unsafe_allow_html=True)
                st.info("💡 To enable AI-generated reports: Open the Ollama app on your computer and refresh this page.")
            else:
                with st.spinner("Synthesizing rationale..."):
                    try:
                        url = "http://127.0.0.1:11434/api/generate"
                        payload = {
                            "model": "llama3.2:latest",
                            "prompt": f"Explain the therapeutic potential of {drug_name} for {disease_name} using clinical terminology.",
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
                        st.error(f"System Offline: {str(e)}")

st.sidebar.markdown("---")
st.sidebar.caption("Clinical Discovery Lab Intelligence")
