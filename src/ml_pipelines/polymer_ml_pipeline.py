#!/usr/bin/env python3
"""
polymer_tg_predictor.py

Predict polymer glass transition temperature (Tg)
from SMILES structures using molecular fingerprints
and a single-output TensorFlow regression model.

Inputs:
    - master_data/unified_polymer_data.csv

Outputs:
    - models/polymer_tg_predictor_v1.keras
    - models/polymer_tg_scaler_mean_v1.npy
    - models/polymer_tg_scaler_scale_v1.npy
    - master_data/polymer_tg/polymer_tg_dataset_v1.csv
    - master_data/polymer_tg/polymer_tg_metadata_v1.json
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from shared_utils import load_smiles_to_fingerprints


BASE = Path(__file__).resolve().parents[2]
DATA_FILE = BASE / "master_data" / "unified_polymer_data.csv"
MASTER_DIR = BASE / "master_data" / "polymer_tg"
MODEL_DIR = BASE / "models"

MASTER_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_tg_data():
    df = pd.read_csv(DATA_FILE, dtype=str)

    df["Tg"] = pd.to_numeric(df["Tg"], errors="coerce")

    # valid rows only
    df = df.dropna(subset=["SMILES", "Tg"])
    df["SMILES"] = df["SMILES"].str.strip()

    return df[["SMILES", "Tg"]]


def preprocess(df, n_bits=2048):
    print(f"[INFO] Generating fingerprints for {len(df)} polymers...")

    X = load_smiles_to_fingerprints(df["SMILES"], n_bits=n_bits)
    y = df["Tg"].values.reshape(-1, 1)

    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)

    print(f"[INFO] Feature matrix shape: {X.shape}")
    print(f"[INFO] Target shape: {y_scaled.shape}")

    return X, y_scaled, scaler_y


def build_tg_model(input_dim):
    inputs = tf.keras.layers.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(512, activation="relu")(inputs)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)

    outputs = tf.keras.layers.Dense(1)(x)   # SINGLE output
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss="mse",
        metrics=["mae"]
    )
    return model


def train_model():
    df = load_tg_data()
    X, y_scaled, scaler_y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_scaled, test_size=0.2, random_state=42
    )

    model = build_tg_model(X.shape[1])

    print("[INFO] Training Tg-only model...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=120,
        batch_size=32,
        verbose=1
    )

    loss, mae = model.evaluate(X_test, y_test)
    print(f"Test loss: {loss:.4f}, MAE (scaled): {mae:.4f}")

    # Save Tg-only model
    model_path = MODEL_DIR / "polymer_tg_predictor_v1.keras"
    mean_path = MODEL_DIR / "polymer_tg_scaler_mean_v1.npy"
    scale_path = MODEL_DIR / "polymer_tg_scaler_scale_v1.npy"
    csv_path = MASTER_DIR / "polymer_tg_dataset_v1.csv"
    json_path = MASTER_DIR / "polymer_tg_metadata_v1.json"

    model.save(model_path)
    np.save(mean_path, scaler_y.mean_)
    np.save(scale_path, scaler_y.scale_)

    df.to_csv(csv_path, index=False)

    metadata = {
        "model_name": "polymer_tg_predictor_v1",
        "date_trained": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_samples": len(df),
        "num_features": X.shape[1],
        "test_loss": float(loss),
        "test_mae_scaled": float(mae),
        "paths": {
            "model": str(model_path),
            "scaler_mean": str(mean_path),
            "scaler_scale": str(scale_path),
            "dataset": str(csv_path)
        }
    }
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"[INFO] Saved: {json_path}")

    return model, scaler_y


if __name__ == "__main__":
    model, scaler = train_model()