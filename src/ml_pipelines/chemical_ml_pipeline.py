# NOTE --- This script is depricated, because the idea of a unified biocompatibility dataset with labelled values for machine learing is too vague!
# As a resultm it does not, yet, have a basis in reality to even remotely calculate "bio-compatibility" in a reliable manner.

import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def run_chemical_ml_pipeline(
        input_path="master_data/chemical/chemical_features.csv",
        output_path="master_data/chemical/chemical_ml_ready.csv"
    ):
    logging.info("Starting chemical ML pipeline...")

    df = pd.read_csv(input_path)
    logging.info(f"Loaded {len(df)} chemical records with columns: {list(df.columns)}")


    # Basic cleanup
    # Drop name and text fields (not used for numeric ML features)
    drop_cols = ["Name", "num_sources"]
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Drop rows missing essential SMILES or label
    df = df.dropna(subset=["Canonical_SMILES", "biocompatibility_label"])

    # Define features vs target
    target = "biocompatibility_label"
    features = [
        "MolWt", "LogP", "TPSA", "HBD", "HBA", "AromaticRings"
    ]

    X = df[features]
    y = df[target]

 
    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)


    # Combine into ML-ready dataset
    ml_ready = pd.concat([df["Canonical_SMILES"], X_scaled, y], axis=1)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ml_ready.to_csv(output_path, index=False)
    logging.info(f"Chemical ML dataset saved -> {output_path} ({len(ml_ready)} rows)")

    return ml_ready


if __name__ == "__main__":
    run_chemical_ml_pipeline()