"""
This is where most of the machine learning UI section of the app lives. The whole tab is wrapped in a function that is called at the end of the script.

"""

def run_ml_app():

    # app/main_app.py
    import streamlit as st
    import pandas as pd
    import sys
    import os
    import numpy as np

    # Ensure src is accessible ---
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

    # Load models from src/inference/model_interface.py
    from src.inference.model_interface import (
        load_polymer_tg_model,
        load_youngs_modulus_model,
        load_qsar_model,
        load_oligomeric_state_model,
        # load_binding_model,
        predict_polymer_tg,
        predict_youngs_modulus,
        predict_qsar_toxicity,
        # predict_binding,
        predict_oligomeric_state,
    )


    # --- Load models once at startup ---
    @st.cache_resource
    def get_models():

        return (
            load_polymer_tg_model(),
            load_youngs_modulus_model(),
            load_qsar_model(),
            # load_binding_model(),
        )

    # Set the models that are loaded and in use, as a callable function
    polymer_model, youngs_model, qsar_model = get_models()

    # This model was acting weird with the session_st caching so im loading it separately for now
    oligomer_model = load_oligomeric_state_model()

    # NOTE This is for UI debugging
    #st.text("=== Oligomeric state model bundle ===")
    #st.text(f"Keys: {list(oligomer_model.keys())}")
    #st.text(f"Scaler: {oligomer_model.get('scaler')}")
    #st.text(f"Encoder: {oligomer_model.get('encoder')}")


    # Styling CSS
    st.markdown("""
    <style>
    /* --- Base typography --- */
    body, p, span, div {
        font-family: 'Inter', sans-serif;
        color: #f8fafc;
        font-size: 1.2rem;
        line-height: 1.6;
        letter-spacing: 0.5px;
    }
    h1 { font-size: 3rem; font-weight: 800; color: #ffffff; text-shadow: 2px 2px 10px rgba(0,0,0,0.5);}
    h2 { font-size: 2.5rem; font-weight: 700; color: #fefefe; text-shadow: 1px 1px 6px rgba(0,0,0,0.3);}
    h3 { font-size: 2rem; font-weight: 600; color: #fefefe; }
    h4 { font-size: 1.8rem; font-weight: 500; color: #fefefe; }

    /* --- Tabs --- */
    .stTabs [role="tab"] button {
        font-size: 1.25rem;
        font-weight: 700;
        color: #f8fafc;
        background: linear-gradient(135deg, #3b82f6, #6366f1);
        border-radius: 12px 12px 0 0;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
    }
    .stTabs [role="tab"] button:hover {
        transform: scale(1.05);
        background: linear-gradient(135deg, #2563eb, #4f46e5);
    }

    /* --- Page container --- */
    .main-container {
        text-align: center;
        padding: 3rem 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }

    /* --- Section blocks --- */
    .section {
        padding: 2.5rem;
        border-radius: 20px;
        background: rgba(255,255,255,0.05);
        margin-bottom: 2rem;
        box-shadow: 0px 8px 30px rgba(0,0,0,0.35);
        transition: all 0.3s ease, box-shadow 0.5s ease;
    }

   /* Home page containers */
    .home-container {
        text-align: center;
        padding: 3rem 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }

    /* Domain cards */
    .home-container .domain-section {
        border-radius: 25px;
        padding: 2.5rem;
        margin: 1rem 0.5rem;
        text-align: center;
        box-shadow: 0px 10px 25px rgba(0,0,0,0.4);
        transition: all 0.3s ease;
        background: linear-gradient(135deg, rgba(59,130,246,0.8), rgba(99,102,241,0.8));
    }
    .home-container .domain-section:hover {
        transform: translateY(-8px) scale(1.03);
        box-shadow: 0px 15px 35px rgba(0,0,0,0.5);
    }   

    /* Titles & descriptions */
    .domain-title { font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem; }
    .domain-desc { font-size: 1.25rem; line-height: 1.7; margin-bottom: 1.5rem; color: #e5e7eb; }

    /* --- Buttons --- */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #6366f1);
        color: white;
        font-size: 1.2rem;
        padding: 1rem 3rem;
        border-radius: 15px;
        border: none;
        box-shadow: 0px 5px 20px rgba(0,0,0,0.35);
        transition: all 0.3s ease, box-shadow 0.5s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb, #4f46e5);
        transform: translateY(-3px) scale(1.05);
    }

    /* --- Accordion styling --- */
    .st-expanderHeader {
        font-weight: 700;
        font-size: 1.6rem;
        color: #ffffff;
        text-shadow: 1px 1px 5px rgba(0,0,0,0.3);
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
        font-size: 1.35rem;
        line-height: 1.8;
    }

    /* --- Info boxes --- */
    .stAlert > div {
        font-size: 1.35rem !important;
        line-height: 1.8 !important;
    }

    /* --- Footer --- */
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-top: 3rem;
        padding-bottom: 1rem;
        border-top: 1px solid rgba(255,255,255,0.1);
    }

    /* --- Charts and tables --- */
    .stDataFrame, .stMetric {
        font-size: 1.2rem !important;
    }
    </style>
    """, unsafe_allow_html=True)


    # Page config
    st.set_page_config(page_title="Biomaterial ML Suite", layout="wide")
    st.title("Material and Protein ML Prediction Suite")
    st.markdown("Choose a domain to begin exploring predictions and material insights.")

    # --- Navigation ---
    tabs = st.tabs([
        "Welcome",
        "Polymer Tg Predictor",
        "Young's Modulus Predictor",
        "QSAR Toxicity Predictor",
        "Oligomeric State Predictor",
        "Protein–Ligand Binding Predictor"
    ])


    # HOME TAB --- 
    with tabs[0]:
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.header("Welcome")
        st.write(
        """
        Explore trained models for:
        - Polymer Tg estimation
        - Mechanical property prediction
        - QSAR toxicity assessment
        - Oligometric state prediction
        - Protien-Ligand biding prediction -> Coming soon
        """
        )
        st.markdown("</div>", unsafe_allow_html=True)



    # --- POLYMER Tg ONLY ---
    with tabs[1]:
        st.subheader("Polymer Tg Prediction")

        st.markdown("""
        <p style='font-size:1rem;'>
        This model predicts the <b>glass transition temperature (Tg)</b> of a polymer
        directly from its <b>SMILES</b> string.<br>
        <b>Example:</b> C=CC(=O)OCC
        </p>
        """, unsafe_allow_html=True)

        # Since the user has to predict the Tg of a polymer using smiles, let them input an acceptable sequence
        smiles_val = st.text_input(
            "Enter Polymer SMILES:",
            value="C=CC(=O)OCC",
            key="polymer_smiles"
        )

        # When th predict button is pressed, show a spinnder animation for wait time, call the predict_polymer_tg function from model_interface.py
        # and output the prediction below all the input fields, in its own metric box
        if st.button("Predict Tg", use_container_width=True, key="btn_poly_tg"):
            try:
                with st.spinner("Calculating Tg..."):
                    # Uses the cached Tg-only model
                    tg_pred = predict_polymer_tg(polymer_model, smiles_val)

                st.success("Prediction complete!")

                # Pretty result box
                st.metric(
                    label="Predicted Tg (°C)",
                    value=f"{tg_pred['Tg_pred']:.2f}",
                )

                # Optional details expanded, incase the user wants to see detailed output
                with st.expander("Show Prediction Details"):
                    st.json({
                        "SMILES": smiles_val,
                        "Predicted_Tg_C": float(tg_pred["Tg_pred"]),
                        "Model_version": tg_pred.get("model_version", "v1"),
                        "Feature_vector_size": tg_pred.get("feature_dim", "N/A")
                    })

            except Exception as e:
                st.error(f"Prediction failed: {e}")


    # --- YOUNG'S MODULUS ---
    with tabs[2]:
        st.subheader("Young’s Modulus Predictor")

        st.markdown("""
        <p style='font-size:1rem;'>
        Predict Young’s Modulus (GPa) based on known <b>mechanical and thermal properties</b> of the material.
        These features are derived from the unified materials dataset used during model training.
        </p>
        """, unsafe_allow_html=True)

        # We need more fields for calculating the Youngs modulus of a material
        with st.form("youngs_modulus_form"):
            col1, col2 = st.columns(2)

            with col1:
                density = st.number_input("Density (g/cm³)", min_value=0.0, max_value=25.0, value=7.8, step=0.1)
                hardness = st.number_input("Hardness (BHN)", min_value=0.0, max_value=500.0, value=200.0, step=1.0)
                tensile = st.number_input("Tensile Strength (MPa)", min_value=0.0, max_value=2500.0, value=400.0, step=10.0)
                yield_strength = st.number_input("Yield Strength (MPa)", min_value=0.0, max_value=2000.0, value=250.0, step=10.0)
                elongation = st.number_input("Elongation (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.5)

            with col2:
                poisson = st.number_input("Poisson’s Ratio", min_value=0.0, max_value=0.6, value=0.3, step=0.01)
                hardness_hv = st.number_input("Hardness (HV)", min_value=0.0, max_value=2000.0, value=150.0)
                shear_modulus = st.number_input("Shear Modulus (GPa)", min_value=0.0, max_value=1000.0, value=80.0)

                # ----NOTE---- 
                # These fields do NOT go into the prediction, maybe with a future dataset we might be able to make use of these example fields
                    # fracture_toughness = st.number_input("Fracture Toughness (MPa√m)", min_value=0.0, max_value=200.0, value=50.0, step=1.0)
                    # melting_point = st.number_input("Melting Point (°C)", min_value=0.0, max_value=4000.0, value=1500.0, step=10.0)
                    # thermal_cond = st.number_input("Thermal Conductivity (W/mK)", min_value=0.0, max_value=1000.0, value=100.0, step=1.0)
                    # specific_heat = st.number_input("Specific Heat (J/g·K)", min_value=0.0, max_value=10.0, value=0.5, step=0.01)

            # NOTE ---- same goes for this feild, i do not currently have data to predict based on one-hot encoding for different types pf materials and 
            # their youngs modulus. Maybe with a dataset in the future
            # st.markdown("#### Material Type")
            # material_type = st.selectbox("Select Material Type", ["Metal", "Ceramic", "Polymer", "Composite"])

            submitted = st.form_submit_button("Predict Young’s Modulus")

        if submitted:

            # --- Only include features that exist in the training model ---
            input_dict = {
                "Density_gcm3": density,
                "Hardness_BHN": hardness,
                "Hardness_HV": hardness_hv,
                "Tensile_Strength_MPa": tensile,
                "Yield_Strength_MPa": yield_strength,
                "Elongation_percent": elongation,
                "Poisson_Ratio": poisson,
                "Shear_Modulus_GPa": shear_modulus,
            }

            result = predict_youngs_modulus(youngs_model, input_dict)

            st.success(f"### Predicted Young’s Modulus: **{result:.2f} GPa**")

            # --- Summary table ---
            st.markdown("#### Input Summary")
            st.dataframe(pd.DataFrame(input_dict.items(), columns=["Feature", "Value"]), use_container_width=True)

            # --- Comparison chart ---
            st.markdown("#### Comparison with Common Materials")
            comparison_data = pd.DataFrame({
                "Material": ["Predicted", "Steel", "Aluminum", "Titanium", "Copper", "Polymer"],
                "Young’s Modulus (GPa)": [result, 200, 70, 116, 110, 3]
            })
            st.bar_chart(comparison_data.set_index("Material"), height=300)

            # --- Interpretation ---
            st.markdown("#### Interpretation")
            if result < 10:
                st.warning("This suggests a **soft polymeric or elastomeric material**.")
            elif result < 70:
                st.info("This modulus is typical for **light alloys or composites.**")
            elif result < 200:
                st.success("This range indicates a **strong metallic alloy** (e.g., steel or titanium).")
            else:
                st.info("This modulus suggests a **ceramic or high-stiffness material.**")


    # --- QSAR TOXICITY (CYTOTOXICITY) ---
    with tabs[3]:
        st.subheader("QSAR Cytotoxicity Predictor")

        st.markdown("""
        <p style='font-size:1rem;'>
        This model predicts **cytotoxicity (CC50/IC50/EC50)** and **toxic probability**  
        from molecular <b>SMILES</b> strings using QSAR features and fingerprints.
        </p>
        """, unsafe_allow_html=True)

        smiles_val = st.text_input(
            "Enter compound SMILES:",
            value="CCCCCCCCCCCCCCCCCC[n+]1cn(Cc2ccccc2)c2ccccc21.[Br-]",
            key="qsar_smiles_input"
        )

        col1, col2 = st.columns([1, 3])
        with col1:
            predict_btn = st.button("Predict Toxicity", use_container_width=True)
        with col2:
            st.info("Example: Ionic liquids, benzimidazolium, imidazolium, etc.")

        if predict_btn:
            try:
                with st.spinner("Analyzing molecular features and predicting toxicity..."):

                # Use the preloaded qsar_model from get_models(), don't reload from disk cause slower
                    preds = predict_qsar_toxicity(qsar_model, smiles_val)

                st.success("Prediction complete!")

                # --- Display Results ---
                st.markdown("### QSAR Prediction Results")
                st.metric("Predicted log(CC50)", f"{preds['log_CC50_pred']:.3f}")
                st.metric("Predicted CC50 (mM)", f"{preds['CC50_mM_pred']:.4f}")
                st.metric("Toxicity Probability", f"{preds['prob_toxic']*100:.1f}%")
                st.metric("Predicted Class", "Toxic" if preds["toxic_label_pred"] else "Non-Toxic")

                # --- Descriptor Table ---
                desc_df = pd.DataFrame({
                    "Descriptor": ["MolWt_calc", "LogP_calc", "TPSA_calc"],
                    "Value": [preds["MolWt_calc"], preds["LogP_calc"], preds["TPSA_calc"]],
                })
                st.markdown("### Molecular Descriptors")
                st.dataframe(desc_df, use_container_width=True)

                # --- Visualization ---
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                ax.bar(["Toxic", "Non-Toxic"], [preds["prob_toxic"], 1 - preds["prob_toxic"]])
                ax.set_ylabel("Probability")
                ax.set_ylim(0, 1)
                ax.set_title("Toxicity Classification Probability")
                st.pyplot(fig)

                # --- Interpretation ---
                st.markdown("### Interpretation")
                if preds["toxic_label_pred"]:
                    st.error("This compound is likely **cytotoxic** (CC50 < 1 mM).")
                else:
                    st.success("This compound is likely **non-toxic** (CC50 ≥ 1 mM).")

            except Exception as e:
                st.error(f"Prediction failed: {e}")


    # Oligomeric State predictor ----
    with tabs[4]:
        st.subheader("Protein Oligomeric State Predictor")
        st.markdown(
            "Predict whether a protein forms monomers, dimers, or higher-order oligomers.\n\n"
            "**Important:** Inputs must be in the same units as used during training."
        )

        # --- Load model bundle ---
        model_bundle = oligomer_model  # dict returned by load_oligomeric_state_model()

        if model_bundle is None or "model" not in model_bundle:
            st.error("Oligomeric state model not loaded properly.")
        else:
            scaler_obj = model_bundle.get("scaler", None)
            encoder_obj = model_bundle.get("encoder", None)
            model_obj = model_bundle["model"]

            if not scaler_obj or not encoder_obj:
                st.error("Scaler or encoder not loaded properly.")
            else:
                # --- Feature names and defaults from scaler ---
                feature_names = list(scaler_obj.feature_names_in_)
                mean_values = scaler_obj.mean_
                std_values = scaler_obj.scale_

                feature_dict = {}

                # --- Generate input fields ---
                for i, f in enumerate(feature_names):
                    if f in ["Number of Chains", "Stoichiometry"]:
                        min_val = max(1, int(round(mean_values[i] - 3*std_values[i])))
                        max_val = int(round(mean_values[i] + 3*std_values[i]))
                        value = int(round(mean_values[i]))
                        feature_dict[f] = st.number_input(
                            f, min_value=min_val, max_value=max_val, value=value, step=1
                        )
                    else:
                        min_val = max(0.0, mean_values[i] - 3*std_values[i])
                        max_val = mean_values[i] + 3*std_values[i]
                        value = float(mean_values[i])
                        feature_dict[f] = st.number_input(
                            f, min_value=min_val, max_value=max_val, value=value
                        )

                # --- Predict button ---
                if st.button("Predict Oligomeric State"):
                    with st.spinner("Running model..."):
                        # Predict using helper function
                        result = predict_oligomeric_state(model_bundle, feature_dict)

                    # --- Display results ---
                    st.success(f"Predicted State: **{result['Predicted_state']}**")
                    if result["Predicted_count"] is not None:
                        st.metric("Predicted Oligomeric Count", result["Predicted_count"])
                    st.metric("Confidence", f"{result['State_confidence']*100:.1f}%")

                    # NOTE --- Optional debug: show scaled features ---
                    # X_df = pd.DataFrame([{f: feature_dict[f] for f in feature_names}])
                    #X_scaled = scaler_obj.transform(X_df)
                    #st.text("Scaled features fed to model (debug):")
                    #st.text(X_scaled)


    # NOTE-- Section not implemented yet!
    with tabs[5]:
        st.subheader("Protein–Ligand Binding Predictor")
        st.markdown("Estimate whether a given ligand binds to a protein target structure.")
        st.markdown("Feature coming soon, not yet implemented.")

        """
        with st.form("binding_form"):
            # Only ask for the features your model expects
            n_residues = st.number_input("Number of Residues", min_value=50, max_value=1000, value=100)
            n_chains = st.number_input("Number of Chains", min_value=1, max_value=12, value=1)
            helix = st.number_input("Helix Count", min_value=0, max_value=1500, value=0)
            sheet = st.number_input("Sheet Count", min_value=0, max_value=500, value=0)
            coil = st.number_input("Coil Count", min_value=0, max_value=2000, value=0)
            mol_weight = st.number_input("Molecular Weight", min_value=5000.0, max_value=200_000.0, value=50_000.0, step=1000.0)
            submitted = st.form_submit_button("Predict Binding")

        if submitted:
            input_features = {
                "Number of Residues": n_residues,
                "Number of Chains": n_chains,
                "Helix": helix,
                "Sheet": sheet,
                "Coil": coil,
                "Molecular Weight per Deposited Model": mol_weight,
            }

            with st.spinner("Predicting binding probability..."):
                result = predict_binding(binding_model, input_features)

            st.success(f"Predicted Binding: **{result['Binding_label']}**")
            st.metric("Binding Probability", f"{result['Binding_probability']*100:.1f}%")
        """


if __name__ == "__main__":
    run_ml_app()