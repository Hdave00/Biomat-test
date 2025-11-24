# app/visualization.py
"""
Streamlit UI for Material Selection + Materials Project integration.

Features implemented:
- Local-only search for Structural / Mechanical (as before) with optional free-text and optional pick-from-list
- Automatic fallback to Materials Project when local search returns no results *if* user enables fallback
- A separate "Materials Project Explorer" tab that provides a periodic-element grid, quick element-based searches
  and an advanced filter form that maps directly to mp_api's documented search() parameters.
- Material detail view shows summary fields, elasticity/thermo/bonds/etc (if available via mp_integration.query_materials_project)
  and attempts a py3Dmol rendering when a structure is present; otherwise falls back to showing CIF/text.

Notes:
- This file expects `app/mp_integration.py` to expose three functions:
    * query_materials_project(query: str | material_id) -> dict | None
    * get_mp_property_dataframe(mp_json: dict) -> pandas.DataFrame
    * query_mp_advanced_filters(...) -> list[dict] | None
  The advanced filter function should accept only *official* mp_api search() kwargs (e.g. elements, formula, band_gap, density, energy_above_hull, is_stable, etc.)
- MP API key should be loaded inside mp_integration (from .env). The UI does NOT ask the user for an API key.

"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from typing import Optional, Any, Iterable
import json

# optional visualization helper
try:
    import py3Dmol
    _HAS_PY3DMOL = True
except Exception:
    _HAS_PY3DMOL = False

from app.mp_integration import (
    query_materials_project,
    get_mp_property_dataframe,
    query_mp_advanced_filters,
    cached_query_mp_advanced_filters,
    cached_query_material,
)

# caching helpers
@st.cache_data(show_spinner=False)
def load_dataset(path):
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except FileNotFoundError:
        return None


# Minimal CSS tweaks
st.markdown(
    """
    <style>
    /* Base body text bigger */
    div.stApp * {
        font-size: 1.25rem !important;
    }

    /* Main title */
    .large-title {
        font-size: 3.5rem !important;
        font-weight: 900 !important;
        text-align: center;
        color: #f0f0f0 !important;
        margin-bottom: 1.5rem !important;
    }

    /* Subheaders */
    h2, h3, h4 {
        font-size: 2rem !important;
        font-weight: 700 !important;
    }

    /* Expander headers */
    .stExpanderHeader {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }

    /* Metric text */
    .stMetricValue {
        font-size: 2.5rem !important;
    }
    .stMetricLabel {
        font-size: 1.5rem !important;
    }

    /* Table text */
    div.stDataFrame table td, div.stDataFrame table th {
        font-size: 1.3rem !important;
    }

    /* Buttons */
    button {
        font-size: 1.4rem !important;
        padding: 0.6rem 1.2rem !important;
    }

    /* Result / MP cards */
    .result-card {
        font-size: 1.3rem !important;
        line-height: 1.6rem !important;
        padding: 1rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Session State Initialization
# These keys are shared between tabs and should persist
if "mp_selected_element" not in st.session_state:
    st.session_state.mp_selected_element = None  

if "mp_search_results" not in st.session_state:
    st.session_state.mp_search_results = None   

if "mp_selected_material_id" not in st.session_state:
    st.session_state.mp_selected_material_id = None 

if "mp_detailed_doc" not in st.session_state:
    st.session_state.mp_detailed_doc = None


# Session State for Local Search 
if "local_results" not in st.session_state:
    st.session_state.local_results = None

if "local_selected" not in st.session_state:
    st.session_state.local_selected = None


# Helpers
def display_results(df: pd.DataFrame, x=None, y=None, color=None):
    if df is None:
        st.error("Dataset missing.")
        return
    if df.empty:
        st.warning("No records found matching your criteria.")
        return

    st.success(f"Found {len(df)} matching records.")
    st.dataframe(df.head(40), use_container_width=True)

    if x in df.columns and y in df.columns:
        try:
            fig = px.scatter(df, x=x, y=y, color=color, hover_name=df.columns[0])
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass


def _render_structure_html_from_cif(cif_text: str, width: int = 700, height: int = 450) -> str:

    """
    Render CIF structure to embeddable HTML using py3Dmol.
    If py3Dmol fails or is unavailable, return a compact <pre> fallback.
    """

    if not cif_text:
        return "<pre>No CIF data provided.</pre>"

    if not _HAS_PY3DMOL:
        return f"<pre style='white-space:pre-wrap; max-height:{height}px; overflow:auto;'>CIF view unavailable (py3Dmol missing)</pre>"

    try:
        v = py3Dmol.view(width=width, height=height)
        v.addModel(cif_text, "cif")
        v.setStyle({"stick": {}})
        v.zoomTo()
        html = getattr(v, "get_html", getattr(v, "_make_html", None))
        if callable(html):
            return html()
        
        # fallback for older versions
        return v._make_html() if hasattr(v, "_make_html") else "<pre>Could not generate 3D view.</pre>"
    except Exception as e:
        return f"<pre style='white-space:pre-wrap; max-height:{height}px; overflow:auto;'>Structure render failed: {e}</pre>"


def safe_iter(x: Any):

    """Return an iterable if x is dict/list-like, else yield x itself for display."""

    if x is None:
        return []
    if isinstance(x, dict):
        return [x]
    if isinstance(x, (list, tuple)):
        return x
    # single scalar
    return [x]


def extract_cif_text(structure) -> Optional[str]:

    """
    Try several robust ways to get a CIF text string from a materials-project
    'structure' object. Return None if we cannot obtain a CIF string.
    """

    # first case, already a string, may be a CIF or representation
    if isinstance(structure, str):
        txt = structure.strip()

        # check heuristic: if it looks like a CIF or POSCAR, return it
        if (
            txt.startswith("data_")
            or txt.lower().startswith("data_")
            or "CELL_LENGTH" in txt.upper()
            or txt.lower().startswith("# cif")
            or txt.startswith("CRYST1")
            or "loop_" in txt
        ):
            return txt
        
        # even if not a CIF, return string (py3Dmol may parse CIF-like strings)
        return txt

    # then ideally pymatgen Structure-like: try .to(fmt='cif') or .to(fmt='poscar')
    try:
        if hasattr(structure, "to"):

            # many versions: try 'cif' then 'poscar'
            for fmt in ("cif", "poscar", "json"):
                try:
                    txt = structure.to(fmt=fmt)
                    if isinstance(txt, str) and txt.strip():
                        return txt
                except Exception:
                    continue
    except Exception:
        pass

    # then, some mp-api returns a dictionary or object with 'cif' or 'structure' keys
    if isinstance(structure, dict):
        for key in ("cif", "cif_string", "structure", "pretty_formula"):
            if key in structure and isinstance(structure[key], str) and structure[key].strip():
                return structure[key]

    # finally fallback: try str() if not empty
    try:
        txt = str(structure)
        if txt and len(txt) > 10:
            return txt
    except Exception:
        pass

    return None


def safe_table(data_dict, title=None):

    """Render a clean table that hides N/A, None, and empty columns."""

    if not isinstance(data_dict, dict) or not data_dict:
        st.caption(f"No {title or 'valid'} data available.")
        return

    df = pd.DataFrame(list(data_dict.items()), columns=["Property", "Value"])

    # drop N/A, None, or blank values
    df = df[
        df["Value"].notna() &
        (df["Value"].astype(str) != "N/A") &
        (df["Value"].astype(str) != "None") &
        (df["Value"].astype(str).str.strip() != "")
    ]
    if df.empty:
        st.caption(f"No {title or 'valid'} data available.")
    else:
        st.table(df)


def show_mp_card(mp_json: dict):

    """
    Display a Materials Project card in Streamlit with summary, elasticity,
    structure, thermo, bonding, magnetism, oxidation states, and robocrystallographer info.
    Cleanly skips empty data and avoids raw CIF dumps.
    """

    if not mp_json:
        st.info("No Materials Project data available.")
        return

    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    left_col, right_col = st.columns([2, 3])

    # Left column: core info 
    with left_col:
        st.subheader(mp_json.get("pretty_formula", mp_json.get("material_id", "Unknown")))
        st.caption(mp_json.get("material_id", ""))

        summary = mp_json.get("summary") or mp_json.get("remarks") or mp_json.get("description")
        if summary:
            st.write(summary)

        # Clean metrics (skip None)
        metrics = {
            "Density (g/cm³)": mp_json.get("density"),
            "Band gap (eV)": mp_json.get("band_gap"),
            "Energy above hull (eV/atom)": mp_json.get("e_above_hull"),
        }
        nonempty_metrics = {k: v for k, v in metrics.items() if v is not None}
        if nonempty_metrics:
            cols = st.columns(len(nonempty_metrics))
            for (label, val), c in zip(nonempty_metrics.items(), cols):
                c.metric(label, val)
        else:
            st.caption("No numeric metrics available.")

        # Chemical system / formula
        chemsys = (
            mp_json.get("chemsys")
            or mp_json.get("composition")
            or mp_json.get("formula_pretty")
        )
        if chemsys:
            st.write(f"**System / formula:** {chemsys}")

    # Right column: Elasticity + 3D Structure
    with right_col:
        elasticity = mp_json.get("elasticity") or {}

        # filter out None values
        elasticity = {k: v for k, v in elasticity.items() if v is not None}

        if elasticity:
            with st.expander("Elasticity data", expanded=False):
                for key, val in elasticity.items():
                    st.write(f"- **{key}**: {val}")
        else:
            st.caption("No elasticity data available.")

        # NOTE nested sructures to mimic the data from the api in json format
        # Structure rendering
        structure = mp_json.get("structure") or mp_json.get("cif")
        if structure:
            st.write("### 3D Structure")
            try:
                cif_text = extract_cif_text(structure)
                if cif_text:
                    html = _render_structure_html_from_cif(cif_text, width=700, height=450)
                    if isinstance(html, str):
                        st.components.v1.html(html, height=480, scrolling=False)
                    else:
                        st.caption("Structure available but could not be visualized.")
                else:
                    st.caption("Structure present, but no CIF text extractable.")
            except Exception as e:
                st.warning(f"Structure could not be displayed: {e}")
        else:
            st.caption("No structure data available.")

    # --- Nested properties (thermo, bonding, magnetism, oxidation, robocrystallography) ---
    nested_docs = {
        "Thermodynamic data": mp_json.get("thermo") or mp_json.get("thermodynamics"),
        "Bonding / coordination": mp_json.get("bonds") or mp_json.get("bonding") or mp_json.get("bond_data"),
        "Magnetism": mp_json.get("magnetism") or mp_json.get("magnetic"),
        "Oxidation states": mp_json.get("oxidation_states") or mp_json.get("oxidation_state") or mp_json.get("oxi_states"),
        "Robocrystallographer": mp_json.get("robocrys") or mp_json.get("robocrystallogapher") or mp_json.get("robocryst"),
    }

    for title, data in nested_docs.items():

        # Skip completely empty or None fields
        if not data:
            continue

        with st.expander(title, expanded=False):
            if isinstance(data, (dict, list)):
                cleaned = [d for d in safe_iter(data) if d]
                if not cleaned:
                    st.caption("No data available.")
                else:
                    for entry in cleaned:
                        if isinstance(entry, dict):
                            flat = flatten_dict(entry)
                            safe_table(flat, title=title)
                        elif isinstance(entry, list):
                            st.write(", ".join(map(str, entry)))
                        else:
                            st.write(entry)
            else:
                st.write(str(data))

    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# DOMAIN DATASET LOADER HELPERS
# -----------------------------

def load_cytotoxicity():
    return load_dataset("master_data/chemical_toxicity_measurements.csv")

def load_corrosion():
    return load_dataset("master_data/corr_lookup_database.csv")

def load_heas():
    return load_dataset("master_data/high_entropy_alloys_properties.csv")

def load_polymers():
    return load_dataset("master_data/polymer_lookup_data.csv")


# -----------------------------
# DOMAIN SEARCH RENDERING
# -----------------------------

def render_cytotoxicity(df):
    st.subheader("Cytotoxicity Database Search")

    col1, col2 = st.columns([2,1])

    with col1:
        # ---- Auto-suggestion lists ----
        name_list   = sorted(df["Name"].dropna().unique().tolist())
        cas_list    = sorted(df["CAS"].dropna().astype(str).unique().tolist())
        cell_list   = sorted(df["Cell_line"].dropna().unique().tolist())
        source_list = sorted(df["source_file"].dropna().unique().tolist()) \
                      if "source_file" in df.columns else []

        # ---- Primary unified suggestion field ----
        free_text = st.selectbox(
            "Chemical / CAS / Cell line (optional)",
            [""] + name_list + cas_list + cell_list
        )

        # ---- New: Source file selector ----
        source_file = st.selectbox(
            "Source File (optional)",
            [""] + source_list
        )

        # ---- Numeric filters ----
        max_cc50 = st.number_input("Max CC50 (mM)", 0.0, 10000.0, 10.0)
        max_ic50 = st.number_input("Max IC50 (mM)", 0.0, 10000.0, 10.0)
        max_exposure = st.number_input("Max exposure time (hrs)", 0.0, 1000.0, 48.0)

    with col2:
        do = st.button("Search Cytotoxicity")

    if do:
        mask = pd.Series([True] * len(df))

        # ---- text matching ----
        if free_text:
            mask &= (
                df["Name"].fillna("").str.contains(free_text, case=False, na=False) |
                df["CAS"].astype(str).str.contains(free_text, case=False, na=False) |
                df["Cell_line"].fillna("").str.contains(free_text, case=False, na=False)
            )

        # ---- source file filtering ----
        if source_file:
            mask &= df["source_file"].fillna("").str.contains(source_file, case=False, na=False)

        # ---- numeric filters ----
        mask &= pd.to_numeric(df["CC50_mM"], errors="coerce").fillna(99999) <= max_cc50
        mask &= pd.to_numeric(df.get("IC50_mM"), errors="coerce").fillna(99999) <= max_ic50
        mask &= pd.to_numeric(df.get("Exposure_hr"), errors="coerce").fillna(99999) <= max_exposure

        st.dataframe(df[mask], use_container_width=True)


def render_corrosion(df):
    st.subheader("Corrosion Lookup Search")

    col1, col2 = st.columns([2,1])

    with col1:
        env_list = sorted(df["Environment"].dropna().unique().tolist())
        mat_list = sorted(df["Material"].dropna().unique().tolist())
        mech_list = sorted(df["Mechanism"].dropna().unique().tolist()) if "Mechanism" in df else []

        env = st.selectbox("Environment (optional)", [""] + env_list)
        material = st.selectbox("Material / Family (optional)", [""] + mat_list)
        mechanism = st.selectbox("Mechanism (optional)", [""] + mech_list)

        max_rate = st.number_input("Max Corrosion Rate (mm/yr)", 0.0, 200.0, 20.0)
        max_temp = st.number_input("Max Temperature (°C)", -50.0, 1000.0, 100.0)
        max_ph = st.number_input("Max pH", 0.0, 14.0, 14.0)

    with col2:
        do = st.button("Search Corrosion")

    if do:
        mask = pd.Series([True] * len(df))

        if env:
            mask &= df["Environment"].fillna("").str.contains(env, case=False, na=False)
        if material:
            mask &= df["Material"].fillna("").str.contains(material, case=False, na=False)
        if mechanism:
            mask &= df["Mechanism"].fillna("").str.contains(mechanism, case=False, na=False)

        mask &= pd.to_numeric(df["Temperature_C"], errors="coerce").fillna(9999) <= max_temp
        mask &= pd.to_numeric(df["pH"], errors="coerce").fillna(9999) <= max_ph

        numeric_rates = pd.to_numeric(df["Rate (mm/yr) or Rating"], errors="coerce")
        mask &= numeric_rates.fillna(9999) <= max_rate

        st.dataframe(df[mask], use_container_width=True)


def render_heas(df):
    st.subheader("High Entropy Alloy Database Search")

    col1, col2 = st.columns([2,1])

    with col1:
        formula_list = sorted(df["FORMULA"].dropna().unique().tolist())

        formula = st.selectbox(
            "Formula / partial match (optional)",
            [""] + formula_list
        )

        min_uts = st.number_input("Min UTS (MPa)", 0.0, 3000.0, 200.0)
        min_yield = st.number_input("Min Yield Strength (MPa)", 0.0, 3000.0, 150.0)
        min_hardness = st.number_input("Min Hardness (HV)", 0.0, 2000.0, 200.0)
        min_elong = st.number_input("Min Elongation (%)", 0.0, 100.0, 5.0)

    with col2:
        do = st.button("Search HEAs")

    if do:
        mask = pd.Series([True] * len(df))

        if formula:
            mask &= df["FORMULA"].fillna("").str.contains(formula, case=False, na=False)

        mask &= pd.to_numeric(df["PROPERTY: UTS (MPa)"], errors="coerce").fillna(0) >= min_uts
        mask &= pd.to_numeric(df.get("YIELD_STRENGTH"), errors="coerce").fillna(0) >= min_yield
        mask &= pd.to_numeric(df.get("HARDNESS_HV"), errors="coerce").fillna(0) >= min_hardness
        mask &= pd.to_numeric(df.get("ELONGATION_%"), errors="coerce").fillna(0) >= min_elong

        st.dataframe(df[mask], use_container_width=True)


def render_polymers(df):
    st.subheader("Polymer Material Database Search")

    col1, col2 = st.columns([2,1])

    with col1:
        name_list = sorted(df["name"].dropna().unique().tolist())
        class_list = sorted(df["Polymer Class"].dropna().unique().tolist())
        smiles_list = sorted(df["SMILES"].dropna().unique().tolist())

        name = st.selectbox(
            "Name / class / SMILES (optional)",
            [""] + name_list + class_list + smiles_list
        )

        min_tg = st.number_input("Min Tg (°C)", -200.0, 800.0, 50.0)
        min_ts = st.number_input("Min Tensile Strength (MPa)", 0.0, 200.0, 20.0)
        min_modulus = st.number_input("Min Young’s Modulus (GPa)", 0.0, 50.0, 1.0)
        min_mw = st.number_input("Min Molecular Weight (g/mol)", 0.0, 1e7, 5000.0)

    with col2:
        do = st.button("Search Polymers")

    if do:
        mask = pd.Series([True] * len(df))

        if name:
            mask &= (
                df["name"].fillna("").str.contains(name, case=False, na=False) |
                df["Polymer Class"].fillna("").str.contains(name, case=False, na=False) |
                df["SMILES"].fillna("").str.contains(name, case=False, na=False)
            )

        mask &= pd.to_numeric(df["Tg"], errors="coerce").fillna(-9999) >= min_tg
        mask &= pd.to_numeric(df.get("Tensile_Strength_MPa"), errors="coerce").fillna(0) >= min_ts
        mask &= pd.to_numeric(df.get("Youngs_Modulus_GPa"), errors="coerce").fillna(0) >= min_modulus
        mask &= pd.to_numeric(df.get("Molecular_Weight"), errors="coerce").fillna(0) >= min_mw

        st.dataframe(df[mask], use_container_width=True)


def render_property_comparison(df_local: Optional[pd.DataFrame], mp_df: Optional[pd.DataFrame], selected_name: str):

    """
    Radar comparison between local properties (if present) and MP-derived properties (mp_df).
    """

    props = ['density', 'E_est_GPa', 'Tensile_Strength_MPa']
    local_vals, mp_vals = [], []

    # find the local row if possible (first match)
    if df_local is not None:
        row = df_local[df_local.iloc[:, 0].str.contains(selected_name, case=False, na=False)]
        local_row = row.iloc[0] if not row.empty else None
    else:
        local_row = None

    for p in props:
        if local_row is not None and p in df_local.columns:
            try:
                local_vals.append(float(local_row.get(p, 0) or 0))
            except Exception:
                local_vals.append(0)
        else:
            local_vals.append(0)

        if mp_df is not None and p in mp_df.columns:
            try:
                mp_vals.append(float(mp_df.iloc[0].get(p, 0) or 0))
            except Exception:
                mp_vals.append(0)
        else:
            mp_vals.append(0)

    # build plotly gigure
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=local_vals, theta=props, fill='toself', name='Local DB'))
    fig.add_trace(go.Scatterpolar(r=mp_vals, theta=props, fill='toself', name='Materials Project'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True, title=f'Property comparison for {selected_name}')
    st.plotly_chart(fig, use_container_width=True)


def flatten_dict(d, parent_key='', sep='.'):

    """Recursively flatten nested dictionaries for table display."""

    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# ---------- UI NOTE -> this is where it gets really convoluted because of how streamlit renders and queries
def run_selection_app():
    st.markdown("<div class='large-title'>Material Selection Module</div>", unsafe_allow_html=True)
    st.write('Find the best materials based on engineering, biological, or chemical requirements.')


    # Use radio to control active tab
    tab_choice = st.radio("Choose module", ['Local Search', 'Materials Project Explorer'], index=0)

    # Load local datasets lazily
    local_unified = load_dataset('master_data/unified_material_data.csv')


    # Local Search Section
    if tab_choice == 'Local Search':
        st.subheader('Local Material Databases')

        # Here
        # Correct indentation, domain radio belongs here
        domain = st.radio(
            "Choose database",
            ["Structural", "Cytotoxicity", "Corrosion", "High Entropy Alloys", "Polymers"],
            horizontal=True
        )

        # ----------------------------------------------------------------------
        # 1. STRUCTURAL  
        # ----------------------------------------------------------------------
        if domain == "Structural":
            
            # here
            st.subheader('Structural / Mechanical Local Search')

            if local_unified is None:
                st.error('Local materials dataset not found (master_data/unified_material_data.csv).')
            else:
                df = local_unified

                col_a, col_b = st.columns([2, 1])

                with col_a:
                    pick_from_list = st.selectbox(
                        'Pick material family (optional)',
                        [''] + sorted(df['Material_Type'].dropna().unique().tolist())
                    )

                    free_text = st.text_input('Or type a material name / formula (optional)', '')

                    min_youngs = st.number_input("Min Young's Modulus (GPa)", 0.0, 500.0, 100.0)
                    min_tensile = st.number_input('Min Tensile Strength (MPa)', 0.0, 3000.0, 200.0)

                    fallback_to_mp = st.checkbox(
                        'If no local matches, fallback to Materials Project?',
                        value=False
                    )

                with col_b:
                    st.write('')
                    st.write('')
                    st.write('Quick actions')
                    do_search = st.button('Search Local')

                if do_search:
                    term = free_text.strip() or pick_from_list or ''

                    mask = pd.Series([True] * len(df))

                    if term:
                        mask &= (
                            df['Material_Type'].str.contains(term, case=False, na=False) |
                            df[df.columns[0]].astype(str).str.contains(term, case=False, na=False)
                        )

                    mask &= (
                        df['Youngs_Modulus_GPa'] >= min_youngs
                    ) & (
                        df['Tensile_Strength_MPa'] >= min_tensile
                    )

                    results = df[mask]
                    st.session_state.local_results = results

                    display_results(
                        results,
                        x='Youngs_Modulus_GPa',
                        y='Tensile_Strength_MPa',
                        color='Material_Type'
                    )

                    if results is not None and not results.empty:
                        st.info('Pick a material from the local results to view details or fetch MP data:')
                        st.selectbox(
                            'Select from Local Results',
                            results.iloc[:, 0].astype(str).tolist(),
                            key='local_selected'
                        )

                    # fallback to Materials Project
                    if results.empty and fallback_to_mp:
                        st.info('No local results, attempting Materials Project fallback.')
                        mp_term = term or free_text or pick_from_list

                        if mp_term:
                            mp_json = query_materials_project(mp_term)
                            if mp_json:
                                st.success('Found matches in Materials Project (fallback).')
                                show_mp_card(mp_json)
                                mp_df = get_mp_property_dataframe(mp_json)
                                # render_property_comparison(...)
                            else:
                                st.warning('No fallback match found on Materials Project.')

                # Show MP details for selected local material
                if st.session_state.get('local_selected'):
                    if st.button('Show Materials Project details for selected material'):
                        picked = st.session_state.local_selected
                        mp_json = query_materials_project(picked)

                        if mp_json:
                            show_mp_card(mp_json)
                            mp_df = get_mp_property_dataframe(mp_json)
                            # render_property_comparison(...)
                        else:
                            st.warning('Materials Project has no entry for the selected local material.')

        # ----------------------------------------------------------------------
        # 2. CYTOTOXICITY
        # ----------------------------------------------------------------------
        elif domain == "Cytotoxicity":
            df = load_cytotoxicity()
            if df is None:
                st.error("Dataset not found: chemical_toxicity_measurements.csv")
            else:
                render_cytotoxicity(df)

        # ----------------------------------------------------------------------
        # 3. CORROSION
        # ----------------------------------------------------------------------
        elif domain == "Corrosion":
            df = load_corrosion()
            if df is None:
                st.error("Dataset not found: corr_lookup_database.csv")
            else:
                render_corrosion(df)

        # ----------------------------------------------------------------------
        # 4. HIGH ENTROPY ALLOYS
        # ----------------------------------------------------------------------
        elif domain == "High Entropy Alloys":
            df = load_heas()
            if df is None:
                st.error("Dataset not found: high_entropy_alloys_properties.csv")
            else:
                render_heas(df)

        # ----------------------------------------------------------------------
        # 5. POLYMERS
        # ----------------------------------------------------------------------
        elif domain == "Polymers":
            df = load_polymers()
            if df is None:
                st.error("Dataset not found: polymer_lookup_data.csv")
            else:
                render_polymers(df)

    # -------------------------
    # Materials Project Explorer Section
    # -------------------------
    elif tab_choice == 'Materials Project Explorer':
        st.subheader('Materials Project Explorer')
        st.write('Pick an element, or use the advanced filter panel to search MP summary data.')

        # Quick Element Grid ---
        st.write('### Quick Element Selection')
        elements = [
            'H','He','Li','Be','B','C','N','O','F','Ne',
            'Na','Mg','Al','Si','P','S','Cl','Ar',
            'K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
            'Ga','Ge','As','Se','Br','Kr',
            'Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd',
            'In','Sn','Sb','Te','I','Xe',
            'Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu',
            'Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg',
            'Tl','Pb','Bi','Po','At','Rn',
            'Fr','Ra'
        ]

        # set the columns per row in UI, the for each element in row, set the row as elements contiguiously after each other,
        # and the cols as columns as long as 10 
        cols_per_row = 10
        for i in range(0, len(elements), cols_per_row):
            row = elements[i:i + cols_per_row]
            cols = st.columns(len(row))

            # for each element in row, keep track and track which button in which row was clicked, set that as the current element, 
            # and DONT let it fall back to local database instead of Materials Project
            for j, el in enumerate(row):
                if cols[j].button(el):
                    st.session_state.mp_selected_element = el
                    st.session_state.mp_search_term = ""

        # Reset chosen material if element changed
        if st.session_state.get('last_selected_element') != st.session_state.get('mp_selected_element'):
            st.session_state.mp_chosen_material_id = None
            st.session_state.view_compound = False
        st.session_state.last_selected_element = st.session_state.get('mp_selected_element')

        # Show compounds for selected element ---
        if st.session_state.get('mp_selected_element'):
            st.info(f"Selected element: {st.session_state.mp_selected_element}")

            with st.spinner("Fetching Materials Project data… this may take a minute ⏳"):
                results = cached_query_mp_advanced_filters(elements=[st.session_state.mp_selected_element])

            if results:
                st.session_state.mp_search_results = results
                res_df = pd.DataFrame(results)

                # Dropdown: pick material
                sub_options = [
                    f"{row['pretty_formula']} ({row['material_id']})" for idx, row in res_df.iterrows()
                ]
                chosen = st.selectbox("Pick a material", sub_options)
                material_id = chosen.split("(")[-1].replace(")","").strip()

                # View detailed info button
                if st.button("View selected material"):

                    with st.spinner("Loading material details…"):
                        doc = cached_query_material(material_id)

                    if doc:
                        st.session_state.mp_detailed_doc = doc
                        show_mp_card(doc)

                        # Scalar overview table
                        overview_keys = [
                            "material_id", "pretty_formula", "density", "band_gap",
                            "energy_above_hull", "formation_energy_per_atom", "is_stable"
                        ]
                        overview = {k: str(doc.get(k, "N/A")) for k in overview_keys}
                        st.subheader("Material Overview")

                        # Add symmetry + magnetism manually
                        sym = doc.get("symmetry", {})
                        overview["crystal_system"] = sym.get("crystal_system", "N/A")
                        overview["space_group"] = sym.get("symbol", "N/A")

                        overview["magnetic_ordering"] = doc.get("ordering", "N/A")
                        overview["total_magnetization"] = doc.get("total_magnetization", "N/A")

                        st.table(pd.DataFrame(list(overview.items()), columns=["Property","Value"]))

                        # Atomic Sites / Wyckoff Positions
                        if "sites" in doc and isinstance(doc["sites"], list):
                            st.subheader("Atomic Sites / Wyckoff Positions")
                            st.dataframe(pd.DataFrame(doc["sites"]))

                        # Crystal parameters (from structure.lattice)
                        st.subheader("Crystal Parameters")

                        lattice = (
                            doc.get("structure", {}).get("lattice")
                            if isinstance(doc.get("structure"), dict)
                            else None
                        )

                        if lattice:
                            crystal_keys = ["a", "b", "c", "alpha", "beta", "gamma", "volume"]
                            crystal_params = {k: str(lattice.get(k, "N/A")) for k in crystal_keys}
                            st.table(pd.DataFrame(list(crystal_params.items()), columns=["Parameter", "Value"]))
                        else:
                            st.caption("No lattice data available.")

                        # Elasticity / Thermo / Magnetism / Oxidation / Bonding
                        for doc_type in ["elasticity","thermo","magnetism","oxidation_states","bonding"]:
                            data = doc.get(doc_type)
                            if isinstance(data, dict) and data:
                                st.subheader(f"{doc_type.replace('_',' ').capitalize()} Data")
                                flat = flatten_dict(data)
                                clean = {k: str(v)[:80] if v is not None else "N/A" for k, v in flat.items()}  # truncate long arrays
                                st.table(pd.DataFrame(list(clean.items()), columns=["Property","Value"]))

                        # Local comparison
                        if local_unified is not None:
                            mp_df = get_mp_property_dataframe(doc)

                            #render_property_comparison(
                                #df_local=local_unified,
                                #mp_df=mp_df,
                                #selected_name=doc.get('pretty_formula', material_id)
                            #)
            else:
                st.warning("No compounds found for this element.")
        st.write("---")

        # --- Advanced MP Filter Form ---
        with st.expander('Advanced MP filters'):
            col1, col2, col3 = st.columns(3)
            with col1:
                q_formula = st.text_input('Formula (e.g. Fe2O3 or Si*)')
                q_elements = st.text_input('Elements (comma separated, overrides formula if set)')
                q_bandgap_min = st.number_input('Min band gap (eV)', value=0.0)
                q_bandgap_max = st.number_input('Max band gap (eV)', value=10.0)
            with col2:
                q_density_min = st.number_input('Min density (g/cm³)', value=0.0)
                q_density_max = st.number_input('Max density (g/cm³)', value=50.0)
                q_ehull_max = st.number_input('Max energy above hull (eV/atom)', value=0.1)
            with col3:
                q_is_stable = st.checkbox('Only stable materials (is_stable)', value=False)
                q_num_results = st.number_input('Max results', min_value=1, max_value=200, value=50)

        # --- Direct MP search (persistent) ---
        mp_search_col1, mp_search_col2 = st.columns([3, 1])
        with mp_search_col1:
            mp_search_term = st.text_input(
                'Direct MP query (material id, formula or element). Overrides element selection.',
                value=st.session_state.get("mp_search_term", "")
            )
        with mp_search_col2:
            do_mp_search = st.button('Search MP')

        # Persist the direct search term
        if do_mp_search:
            st.session_state.mp_search_term = mp_search_term.strip()

        # Retrieve current effective search term
        current_search = st.session_state.get("mp_search_term", "").strip()

        if do_mp_search and current_search:
            # Direct MP search
            st.session_state.mp_detailed_doc = None
            st.session_state.mp_selected_material_id = None

            with st.spinner("Loading material details…"):
                doc = cached_query_material(current_search)

            if doc:
                st.session_state.mp_detailed_doc = doc
                show_mp_card(doc)
                mp_df = get_mp_property_dataframe(doc)
            else:
                st.warning("No MP document found for that term.")

        elif do_mp_search and not current_search and st.session_state.mp_selected_element:

            # Fallback: element selection search
            with st.spinner("Fetching Materials Project data… this may take a minute ⏳"):
                results = cached_query_mp_advanced_filters(elements=[st.session_state.mp_selected_element])

            if results:
                st.session_state.mp_search_results = results
                res_df = pd.DataFrame(results)
                st.dataframe(
                    res_df[['material_id', 'pretty_formula', 'density', 'band_gap', 'e_above_hull']].head(q_num_results),
                    use_container_width=True,
                )

                st.selectbox(
                    'Choose a material from results',
                    res_df['material_id'].tolist(),
                    key='mp_chosen_material_id'
                )

                if st.session_state.get('mp_chosen_material_id'):
                    chosen_id = st.session_state.mp_chosen_material_id

                    if st.button('View selected material (element results)'):

                        with st.spinner("Loading material details…"):
                            doc = cached_query_material(chosen_id)

                        if doc:
                            st.session_state.mp_detailed_doc = doc
                            show_mp_card(doc)
                        else:
                            st.warning('Could not fetch full MP details for chosen material.')
            else:
                st.warning('No results found for that element.')

        elif do_mp_search:
            # --- Advanced filter search ---
            # Persist filter values
            filter_keys = ["q_formula", "q_elements", "q_bandgap_min", "q_bandgap_max",
                           "q_density_min", "q_density_max", "q_ehull_max", "q_is_stable", "q_num_results"]
            for k in filter_keys:
                if k in locals():
                    st.session_state[k] = locals()[k]

            # Use persisted or current form values
            q_formula = st.session_state.get("q_formula", q_formula)
            q_elements = st.session_state.get("q_elements", q_elements)
            q_bandgap_min = st.session_state.get("q_bandgap_min", q_bandgap_min)
            q_bandgap_max = st.session_state.get("q_bandgap_max", q_bandgap_max)
            q_density_min = st.session_state.get("q_density_min", q_density_min)
            q_density_max = st.session_state.get("q_density_max", q_density_max)
            q_ehull_max = st.session_state.get("q_ehull_max", q_ehull_max)
            q_is_stable = st.session_state.get("q_is_stable", q_is_stable)
            q_num_results = st.session_state.get("q_num_results", q_num_results)

            kw = {}
            if q_formula:
                kw['formula'] = q_formula
            if q_elements:
                kw['elements'] = [e.strip() for e in q_elements.split(',') if e.strip()]
            if q_bandgap_min is not None or q_bandgap_max is not None:
                kw['band_gap'] = (q_bandgap_min, q_bandgap_max)
            if q_density_min is not None or q_density_max is not None:
                kw['density'] = (q_density_min, q_density_max)
            if q_ehull_max is not None:
                kw['energy_above_hull'] = (0.0, q_ehull_max)
            if q_is_stable:
                kw['is_stable'] = True

            with st.spinner("Fetching Materials Project data… this may take a minute ⏳"):
                results = cached_query_mp_advanced_filters(**kw)

            if results:
                st.session_state.mp_search_results = results
                res_df = pd.DataFrame(results)
                st.dataframe(
                    res_df[['material_id', 'pretty_formula', 'density', 'band_gap', 'e_above_hull']].head(q_num_results),
                    use_container_width=True,
                )

                st.selectbox(
                    'Choose a material from results',
                    res_df['material_id'].tolist(),
                    key='mp_chosen_material_id'
                )

                if st.session_state.get('mp_chosen_material_id'):
                    chosen_id = st.session_state.mp_chosen_material_id

                    if st.button('View selected material (advanced results)'):

                        with st.spinner("Loading material details…"):
                            doc = cached_query_material(chosen_id)

                        if doc:
                            st.session_state.mp_detailed_doc = doc
                            show_mp_card(doc)
                        else:
                            st.warning('Could not fetch full MP details for chosen material.')
            else:
                st.warning('No results found for these filter criteria.')

    st.divider()
    if st.button('Back to Home', use_container_width=True):
        st.session_state.page = 'home'
        st.rerun()


# Run when module invoked directly 
if __name__ == '__main__':
    run_selection_app()