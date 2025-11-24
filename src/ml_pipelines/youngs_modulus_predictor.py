# utils/ml_pipelines/youngs_modulus_predictor.py
"""
Pipeline for predicting Young's modulus (or related mechanical properties)
from alloy elemental compositions.

Dataset: Located under /data/alloy_properties.csv

Steps:
1. Load dataset
2. Clean and rename columns (standardize to element symbols)
3. Handle missing values, normalize features
4. Train both KNN and RandomForest regressors
5. Evaluate models (R² and MAE)
6. Save best-performing model + scaler
7. Provide predict_youngs_modulus(composition_dict) for inference

Then to use it, for example:

from utils.ml_pipelines.youngs_modulus_predictor import predict_youngs_modulus

example_alloy = {
    "Fe": 85.0,
    "Cr": 13.0,
    "C": 0.1,
    "Si": 0.7,
    "Mn": 0.5,
    "Ni": 0.5,
    # rest default to 0 if missing
}

pred = predict_youngs_modulus(example_alloy)
print(f"Predicted Young's modulus (proxy UTS) ≈ {pred:.2f} psi")

"""

import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import logging
import os
import json


# Logging setup
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# Utility functions
def extract_symbol(col_name):
    """Extract element symbol from strings like 'Cerium (Ce)Ce' or leave unchanged."""
    m = re.search(r"\(([^)]+)\)", col_name)
    return m.group(1) if m else col_name.strip()


def rename_columns(df):
    """Standardize columns to element symbols and simpler property names."""
    rename_map = {}
    for col in df.columns:
        if 'Ultimate' in col:
            rename_map[col] = 'UTS'
        elif 'Liquidus' in col:
            rename_map[col] = 'Liquidus'
        else:
            rename_map[col] = extract_symbol(col)
    return df.rename(columns=rename_map)


# Load + Preprocess Dataset
def load_and_prepare_data(csv_path):
    """
    Load unified materials dataset, clean columns, handle duplicates,
    and prepare numeric features for Youngs modulus prediction.
    """

    df = pd.read_csv(csv_path)

    # Drop empty columns and rows with missing target
    df = df.dropna(axis=1, how='all').dropna(subset=['Youngs_Modulus_GPa'], how='any')

    # Automatically merge duplicate numeric columns by averaging
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    base_map = {}

    for col in numeric_cols:
        base = col.split(".")[0]
        base_map.setdefault(base, []).append(col)

    for base, cols in base_map.items():
        if len(cols) > 1:
            df[base] = df[cols].mean(axis=1)
            df.drop(columns=cols, inplace=True)
        else:
            if cols[0] != base:
                df.rename(columns={cols[0]: base}, inplace=True)

    rename_map = {
        'Youngs_Modulus_GPa': 'Youngs_Modulus_GPa',
        'Elastic_Modulus_GPa': 'Youngs_Modulus_GPa',
        'E': 'Youngs_Modulus_GPa',
        'UTS': 'Tensile_Strength_MPa',
        'Tensile_Strength_MPa_1': 'Tensile_Strength_MPa',
        'Tensile_Strength_MPa_2': 'Tensile_Strength_MPa',
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    target_col = 'Youngs_Modulus_GPa'
    if target_col not in df.columns:
        raise ValueError("No Youngs modulus column found in dataset!")

    numeric_df = df.select_dtypes(include=[np.number])
    feature_cols = [c for c in numeric_df.columns if c != target_col]

    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(numeric_df[feature_cols])
    y = numeric_df[target_col]

    X = pd.DataFrame(X, columns=feature_cols)

    log.info(f"Loaded {len(df)} materials, {len(feature_cols)} numeric features.")
    log.info(f"Predicting target: {target_col}")

    print("TRAINED FEATURE COLUMNS:")
    for f in feature_cols:
        print(f)

    return X, y, feature_cols


# Model Training + Evaluation
def train_models(X, y):
    """Train KNN and RandomForest regressors, return best one."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn = KNeighborsRegressor(n_neighbors=7)
    knn.fit(X_train_scaled, y_train)
    knn_preds = knn.predict(X_test_scaled)
    knn_r2 = r2_score(y_test, knn_preds)
    knn_mae = mean_absolute_error(y_test, knn_preds)
    log.info(f"KNN Regressor: R² = {knn_r2:.3f}, MAE = {knn_mae:.3f}")

    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_r2 = r2_score(y_test, rf_preds)
    rf_mae = mean_absolute_error(y_test, rf_preds)
    log.info(f"RandomForest: R² = {rf_r2:.3f}, MAE = {rf_mae:.3f}")

    if rf_r2 > knn_r2:
        log.info("Using RandomForestRegressor as final model.")
        best_model = rf
        use_scaler = False
    else:
        log.info("Using KNeighborsRegressor as final model.")
        best_model = knn
        use_scaler = True

    return best_model, scaler if use_scaler else None


# Save + Predict Interface
MODEL_PATH = "models/youngs_modulus_model.pkl"
SCALER_PATH = "models/youngs_modulus_scaler.pkl"
FEATURE_PATH = "models/youngs_modulus_features.json"


def save_model(model, scaler, feature_cols):
    """Save model, scaler, and feature list to /models directory."""
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    if scaler is not None:
        joblib.dump(scaler, SCALER_PATH)
    with open(FEATURE_PATH, "w") as f:
        json.dump(feature_cols, f)
    log.info("Model and artifacts saved successfully.")


# Main entrypoint for training
if __name__ == "__main__":
    data_path = "master_data/unified_material_data.csv"
    X, y, feature_cols = load_and_prepare_data(data_path)
    model, scaler = train_models(X, y)
    save_model(model, scaler, feature_cols)
    log.info("Training complete and model ready for inference.")