"""
Oligomeric State Prediction Pipeline

Predicts oligomeric state (monomer, dimer, trimer, etc.)
and oligomeric count based on protein structural features.

Model outputs:
- Categorical: Oligomeric State
- Continuous: Oligomeric Count (regression)
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.utils.multiclass import unique_labels
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, Input
from keras.callbacks import EarlyStopping
import joblib

# === Paths ===
DATA_PATH = "master_data/rcsb_pdb/RCSB_PDB_Macromolecular_Structure_Dataset_with_Structural_Features.csv"
MODEL_OUT = "models/oligomeric_state_model.keras"
SCALER_OUT = "models/oligomeric_state_scaler.pkl"
ENCODER_OUT = "models/oligomeric_state_encoder.pkl"

# === Load and prepare data ===
df = pd.read_csv(DATA_PATH)
print(f"Loaded dataset: {df.shape[0]} rows")

feature_cols = [
    "Number of Chains",
    "Helix",
    "Sheet",
    "Coil",
    "Molecular Weight per Deposited Model",
    "Stoichiometry",
]

# Drop missing
df = df.dropna(subset=feature_cols + ["Oligomeric State", "Oligomeric Count"])

# Encode categorical features if needed
if "Stoichiometry" in df:
    df["Stoichiometry"] = df["Stoichiometry"].astype("category").cat.codes

X = df[feature_cols].astype(float)

# Targets
y_state = df["Oligomeric State"].astype(str)
y_count = df["Oligomeric Count"].astype(float)

# Label encode the state
encoder = LabelEncoder()
y_state_encoded = encoder.fit_transform(y_state)

# === Split ===
X_train, X_test, y_state_train, y_state_test, y_count_train, y_count_test = train_test_split(
    X, y_state_encoded, y_count, test_size=0.2, random_state=42
)

# === Scale ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Define dual-output model ===
inp = Input(shape=(X_train_scaled.shape[1],))
x = Dense(128, activation="relu")(inp)
x = Dropout(0.3)(x)
x = Dense(64, activation="relu")(x)
x = Dense(32, activation="relu")(x)

# Outputs
out_state = Dense(len(np.unique(y_state_encoded)), activation="softmax", name="state_output")(x)
out_count = Dense(1, activation="linear", name="count_output")(x)

model = Model(inputs=inp, outputs=[out_state, out_count])
model.compile(
    optimizer="adam",
    loss={
        "state_output": "sparse_categorical_crossentropy",
        "count_output": "mse",
    },
    metrics={
        "state_output": "accuracy",
        "count_output": "mae",
    },
)

# === Train ===
history = model.fit(
    X_train_scaled,
    {"state_output": y_state_train, "count_output": y_count_train},
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    verbose=2
)

# === Evaluate ===
preds = model.predict(X_test_scaled)
pred_state = np.argmax(preds[0], axis=1)
pred_count = preds[1].flatten()

print("\nClassification performance:")

# Dynamically restrict to only the labels present in the test set + predictions
labels_present = unique_labels(y_state_test, pred_state)
class_names_present = encoder.inverse_transform(labels_present)

print(classification_report(
    y_state_test,
    pred_state,
    labels=labels_present,
    target_names=class_names_present
))

mae = mean_absolute_error(y_count_test, pred_count)
print(f"Mean Absolute Error for Oligomeric Count: {mae:.3f}")

# === Save artifacts ===
os.makedirs("models", exist_ok=True)
model.save(MODEL_OUT)
joblib.dump(scaler, SCALER_OUT)
joblib.dump(encoder, ENCODER_OUT)

print(f"Saved model -> {MODEL_OUT}")
print(f"Saved scaler -> {SCALER_OUT}")
print(f"Saved encoder -> {ENCODER_OUT}")