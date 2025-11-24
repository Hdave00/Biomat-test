# entry_app.py
import sys
import streamlit as st
import importlib

st.set_page_config(page_title="Biomaterial and Bioinformatics Platform", layout="wide")

# --- Custom CSS ---
st.markdown("""
<style>
/* --- Page background with subtle animated gradient --- */
body, .main-container {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    color: #f8fafc;
    font-family: 'Inter', sans-serif;
}

/* Animate background gradient */
@keyframes gradientBG {
    0%{background-position:0% 50%;}
    50%{background-position:100% 50%;}
    100%{background-position:0% 50%;}
}

/* --- Headings --- */
h1 { font-size: 3rem; font-weight: 800; color: #ffffff; text-shadow: 2px 2px 10px rgba(0,0,0,0.5); }
h2 { font-size: 2.5rem; font-weight: 700; color: #fefefe; text-shadow: 1px 1px 6px rgba(0,0,0,0.5); }
h3 { font-size: 2rem; font-weight: 600; color: #fefefe; }
h4 { font-size: 1.8rem; font-weight: 500; color: #fefefe; }

/* --- Tabs --- */
.stTabs [role="tab"] button {
    font-size: 1.2rem;
    font-weight: 600;
    color: #f8fafc;
    background: linear-gradient(135deg, #3b82f6, #6366f1);
    border-radius: 12px 12px 0 0;
    padding: 0.8rem 1.3rem;
    transition: all 0.3s ease;
}
.stTabs [role="tab"] button:hover {
    transform: scale(1.05);
}

/* --- Containers & sections --- */
.main-container {
    text-align: center;
    padding: 3rem 2rem;
}
.section {
    padding: 2.5rem;
    border-radius: 20px;
    background: rgba(255,255,255,0.05);
    margin-bottom: 2rem;
    box-shadow: 0px 8px 30px rgba(0,0,0,0.4);
}

/* --- Domain cards: glassmorphism style --- */
.domain-section {
    border-radius: 25px;
    padding: 2.5rem;
    margin: 1rem 0.5rem;
    text-align: center;
    box-shadow: 0px 8px 30px rgba(0,0,0,0.35);
    backdrop-filter: blur(10px);
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    transition: all 0.3s ease;
    color: #ffffff;
}

/* Module colors */
.domain-section:nth-child(1) { border-left: 5px solid #3b82f6; }
.domain-section:nth-child(2) { border-left: 5px solid #16a34a; }
.domain-section:nth-child(3) { border-left: 5px solid #8b5cf6; }
.domain-section:nth-child(4), .domain-section:nth-child(5), .domain-section:nth-child(6) { border-left: 5px solid #f97316; }

.domain-section:hover {
    transform: translateY(-8px) scale(1.03);
    box-shadow: 0px 12px 40px rgba(0,0,0,0.5);
    background: rgba(255,255,255,0.08);
}

/* Titles & descriptions */
.domain-title { font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem; }
.domain-desc { font-size: 1.3rem; line-height: 1.7; margin-bottom: 1.5rem; }

/* --- Buttons --- */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #6366f1);
    color: white;
    font-size: 1.2rem;
    padding: 1rem 2.5rem;
    border-radius: 18px;
    border: none;
    box-shadow: 0px 5px 20px rgba(0,0,0,0.4);
    transition: all 0.3s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #4f46e5);
    transform: translateY(-3px) scale(1.04);
}

/* --- Main header subtext --- */
.header-subtext {
    font-size: 1.5rem;      /* bigger than default */
    line-height: 1.8;       /* better spacing */
    color: #e5e7eb;         /* light grey for readability */
    max-width: 900px;       /* limit width for better readability */
    margin: 0.5rem auto 2rem auto;
}

/* --- Accordion --- */
.st-expanderHeader {
    font-weight: 700;
    font-size: 1.7rem;      /* bigger header */
    color: #fefefe;
    transition: all 0.3s ease;
}
.st-expanderHeader:hover {
    color: #60a5fa;
}

.st-expanderContent {
    background-color: rgba(255,255,255,0.05);
    padding: 1.5rem 1.8rem;
    border-radius: 15px;
    margin-bottom: 0.5rem;
    font-size: 1.4rem;      /* bigger content text */
    line-height: 1.8;       /* better spacing */
}

/* Info boxes inside accordion */
.stAlert > div {
    font-size: 1.35rem !important; /* increase font size */
    line-height: 1.8 !important;

/* --- Footer --- */
.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 1.05rem;
    margin-top: 3rem;
    padding-bottom: 1rem;
    border-top: 1px solid rgba(255,255,255,0.15);
}
</style>
""", unsafe_allow_html=True)

# --- Main Page ---
st.markdown("<div class='main-container'>", unsafe_allow_html=True)
st.title("Biomaterial and Bioinformatics Novel Platform")
st.markdown("<p class='header-subtext'>Machine Learning-driven discovery, prediction, and selection of advanced biomaterials, unified in one pragmatic platform.</p>", unsafe_allow_html=True)

# --- Domain Overview ---
col1, col2, col3 = st.columns(3)

# ML Module
with col1:
    st.markdown("<div class='domain-section'>", unsafe_allow_html=True)
    st.markdown("<div class='domain-title'>Machine Learning Module</div>", unsafe_allow_html=True)
    st.markdown("<div class='domain-desc'>Use pretrained ML models to predict key material properties, from mechanical strength to toxicity. Cut down in prototyping time.</div>", unsafe_allow_html=True)
    if st.button("Enter ML Module", use_container_width=True, key="ml_btn"):
        st.session_state.page = "ml"
    st.markdown("</div>", unsafe_allow_html=True)

# Material Selection Module
with col2:
    st.markdown("<div class='domain-section'>", unsafe_allow_html=True)
    st.markdown("<div class='domain-title'>Material Selection Module</div>", unsafe_allow_html=True)
    st.markdown("<div class='domain-desc'>Browse, compare, and analyze materials using integrated property datasets and visualization tools for selection and optimization.</div>", unsafe_allow_html=True)
    if st.button("Enter Selection Module", use_container_width=True, key="sel_btn"):
        st.session_state.page = "selection"
    st.markdown("</div>", unsafe_allow_html=True)

# Anatomy & Implant Visualizer
with col3:
    st.markdown("<div class='domain-section'>", unsafe_allow_html=True)
    st.markdown("<div class='domain-title'>Anatomy & Implant Visualizer</div>", unsafe_allow_html=True)
    st.markdown("<div class='domain-desc'>Explore anatomically accurate 3D organs and cross-referenced ASCT+B tables to understand implant placement and biological context.</div>", unsafe_allow_html=True)
    if st.button("Enter Anatomy Visualizer", use_container_width=True, key="hra_btn"):
        st.session_state.page = "hra"
    st.markdown("</div>", unsafe_allow_html=True)

# --- Coming Soon Section ---
st.markdown("<h2 style='text-align:center; margin-top:3rem;'>Coming Soon</h2>", unsafe_allow_html=True)
col3, col4, col5 = st.columns(3)

with col3:
    st.markdown("<div class='domain-section'><h4>Simulation Suite</h4><p>Run mechanical and biocompatibility simulations to evaluate candidate materials.</p></div>", unsafe_allow_html=True)
with col4:
    st.markdown("<div class='domain-section'><h4>Dataset Uploader</h4><p>Upload your own material datasets for custom ML model training.</p></div>", unsafe_allow_html=True)
with col5:
    st.markdown("<div class='domain-section'><h4>Protein-Ligand binding prediction</h4><p>Predict protein ligand binding ability.</p></div>", unsafe_allow_html=True)

# --- Features Overview Accordion ---
st.markdown("<h2 style='text-align:center; margin-top:3rem;'>Features Overview</h2>", unsafe_allow_html=True)

with st.container():
    feature_sections = {
        "Machine Learning Module": "Predict material properties such as tensile strength, Young's modulus, and cytotoxicity using pretrained AI models.",
        "Material Selection Module": "Browse, filter, and compare materials using integrated datasets and visualization tools.",
        "Anatomy & Implant Visualizer": "Explore 3D anatomical models and understand implant placement in biological context.",
        "Simulation Suite (Coming Soon)": "Run mechanical and biocompatibility simulations for candidate materials.",
        "Dataset Uploader (Coming Soon)": "Upload custom material datasets to train personalized AI models.",
        "Protein-Ligand Binding Predictor (Coming Soon)": "Predict binding affinity between proteins and ligands using structural features."
    }

    for feature_title, feature_desc in feature_sections.items():
        with st.expander(feature_title, expanded=False):
            st.write(feature_desc)
            st.info("More details will be added here soon.")

# --- Footer ---
st.markdown("""
<div class='footer'>
&copy; 2025 Biomat project | Working prototype for scientific biomaterial exploration
</div>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# --- Routing ---
page = st.session_state.get("page", "home")
if page == "ml":
    module = importlib.import_module("app.main_app")
    module.run_ml_app()
elif page == "selection":
    module = importlib.import_module("app.visualization")
    module.run_selection_app()
elif page == "hra":
    module = importlib.import_module("app.human_atlas.hra_visualizer")
    module.run_hra_visualizer()