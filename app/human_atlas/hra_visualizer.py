"""
NOTE -> Complete visualization not yet implemented due to lack of visualization experience and time.
"""

# hra_visualizer.py
import streamlit as st
import json

def run_hra_visualizer():
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.header("Human Reference Atlas Explorer")

    # Organ selection
    organ = st.selectbox(
        "Select an anatomical region:",
        ["Hip", "Knee", "Kidney (Left)", "Kidney (Right)", "Pelvis", "Heart"]
    )

    sex = st.radio(
        "Reference model sex:",
        ["male", "female"],
        horizontal=True
    )

    side = "left" if "Left" in organ else "right" if "Right" in organ else "none"

    # Map organ name -> UBERON IDs
    organ_iri_map = {
        "Hip": "http://purl.obolibrary.org/obo/UBERON_0001270",
        "Knee": "http://purl.obolibrary.org/obo/FMA_24978",
        "Kidney (Left)": "http://purl.obolibrary.org/obo/UBERON_0004538",
        "Kidney (Right)": "http://purl.obolibrary.org/obo/UBERON_0004539",
        "Pelvis": "http://purl.obolibrary.org/obo/UBERON_0001270",
        "Heart": "http://purl.obolibrary.org/obo/UBERON_0000948",
    }
    organ_iri = organ_iri_map.get(organ)

    tabs = st.tabs(["3D Viewer", "RUI Registration", "ASCT+B Reporter"])


    # Tab 1: 3D Viewer (EUI)
    with tabs[0]:
        st.info("CCF Exploration User Interface (3D Viewer)")
        st.components.v1.html(
            f"""
            <div style="width:100%; height:90vh; overflow:auto; border:1px solid #ddd;">
            <base href="https://cdn.humanatlas.io/ui/ccf-eui/">

            <link href="https://cdn.humanatlas.io/ui/ccf-eui/styles.css" rel="stylesheet" />
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500&display=swap" rel="stylesheet">
            <link href="https://fonts.googleapis.com/icon?family=Material+Icons|Material+Icons+Outlined" rel="stylesheet" />

            <script type="module" src="https://cdn.humanatlas.io/ui/ccf-eui/wc.js"></script>

            <ccf-eui
                id="eui"
                remote-api-endpoint="https://apps.humanatlas.io/api"
                data-sources='{json.dumps(["https://purl.humanatlas.io/collection/ds-graphs"])}'
                selected-organs='{json.dumps([organ])}'
                filter='{json.dumps({"sex": sex.capitalize()})}'
                header="true"
                style="width:100%; height:100%;"
            >
            </ccf-eui>
            </div>
            """,
            height=900,
        )


    # Tab 2: RUI Registration
    with tabs[1]:
        st.info("Registration User Interface (RUI)")
        st.components.v1.html(
            f"""
            <div style="width:100%; height:90vh; overflow:auto; border:1px solid #ddd;">
            <base href="https://cdn.humanatlas.io/ui/ccf-rui/">

            <link href="https://cdn.humanatlas.io/ui/ccf-rui/styles.css" rel="stylesheet" />
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500&display=swap" rel="stylesheet">
            <link href="https://fonts.googleapis.com/icon?family=Material+Icons|Material+Icons+Outlined" rel="stylesheet" />

            <script type="module" src="https://cdn.humanatlas.io/ui/ccf-rui/wc.js"></script>

            <ccf-rui
                base-href="https://cdn.humanatlas.io/ui/ccf-rui/"
                user='{{"firstName": "demo", "lastName": "demo"}}'
                organ='{{"name": "{organ}", "ontologyId": "{organ_iri}", "side": "{side}", "sex": "{sex}"}}'
                header="false"
                style="width:100%; height:100%;"
            >
            </ccf-rui>
            </div>
            """,
            height=900,
        )


    # Tab 3: ASCT+B Reporter
    with tabs[2]:
        st.info("ASCT+B Reporter")
        st.components.v1.html(
            f"""
            <div style="width:100%; height:90vh; overflow:auto; border:1px solid #ddd;">
            <base href="https://cdn.humanatlas.io/ui/ccf-organ-info/">

            <link href="https://cdn.humanatlas.io/ui/ccf-organ-info/styles.css" rel="stylesheet" />
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500&display=swap" rel="stylesheet">
            <link href="https://fonts.googleapis.com/icon?family=Material+Icons|Material+Icons+Outlined" rel="stylesheet" />

            <script type="module" src="https://cdn.humanatlas.io/ui/ccf-organ-info/wc.js"></script>

            <ccf-organ-info
                base-href="https://cdn.humanatlas.io/ui/ccf-organ-info/"
                remote-api-endpoint="https://apps.humanatlas.io/api"
                organ-iri="{organ_iri}"
                data-sources='{json.dumps(["https://purl.humanatlas.io/collection/ds-graphs"])}'
                style="width:100%; height:100%;"
            >
            </ccf-organ-info>
            </div>
            """,
            height=900,
        )

    st.markdown("</div>", unsafe_allow_html=True)