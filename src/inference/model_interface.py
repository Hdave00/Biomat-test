# src/inference/model_interface.py
import os
import json
import numpy as np
import pandas as pd
import pickle
import joblib
import tensorflow as tf
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from pathlib import Path
from src.ml_pipelines.shared_utils import generate_binding_features

try:
    # adjust this import if your package layout differs
    from src.ml_pipelines.shared_utils import load_smiles_to_fingerprints
except Exception:
    # fallback if running as script from other cwd
    from ml_pipelines.shared_utils import load_smiles_to_fingerprints



# fail gracefully if RDKit isn’t installed
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    Chem = AllChem = None

MODEL_REGISTRY = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))

REGISTRY_PATH = os.path.join(MODEL_REGISTRY, 'registry.json')

with open(REGISTRY_PATH, 'r') as f:
    MODEL_MAP = json.load(f)

def _load_npy(path):
    return np.load(path, allow_pickle=True)

def _load_joblib_or_pickle(path):
    import joblib, pickle
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found")
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

def _load_keras(path):
    return tf.keras.models.load_model(path)


def load_model_by_name(filename):
    """Unified loader by filename (basename or full path). Returns object + type."""
    path = filename if os.path.isabs(filename) else os.path.join(MODEL_REGISTRY, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    lower = path.lower()

    # --- Explicit file type detection ---
    if lower.endswith(('.keras', '.h5', '.tf')):
        return _load_keras(path), 'keras'

    if lower.endswith('.joblib'):
        return joblib.load(path), 'joblib'

    if lower.endswith('.pkl'):
        # try joblib first, fallback to pickle
        try:
            return joblib.load(path), 'joblib'
        except Exception as e:
            print(f"Joblib failed for {filename}, trying pickle. Reason: {e}")
            with open(path, 'rb') as f:
                return pickle.load(f), 'pickle'

    if lower.endswith('.npy'):
        return _load_npy(path), 'npy'

    # Default fallback: joblib first
    try:
        return joblib.load(path), 'joblib'
    except Exception:
        with open(path, 'rb') as f:
            return pickle.load(f), 'pickle'


# Example loader for polymer Tg predictor (keras)
def load_polymer_tg_model():
    model_obj, typ = load_model_by_name('polymer_tg_predictor_v1.keras')
    mean = np.load(os.path.join(MODEL_REGISTRY, 'polymer_tg_scaler_mean_v1.npy'))
    scale = np.load(os.path.join(MODEL_REGISTRY, 'polymer_tg_scaler_scale_v1.npy'))

    mean = np.asarray(mean).reshape(-1)
    scale = np.asarray(scale).reshape(-1)

    return {
        'model': model_obj,
        'target_mean': mean,
        'target_scale': scale,
        'meta': {'format': typ}
    }



def predict_polymer_tg(model_bundle, smiles_str):
    if Chem is None:
        raise ImportError("RDKit must be installed.")

    model = model_bundle['model']
    mean = model_bundle['target_mean']
    scale = model_bundle['target_scale']

    fps = load_smiles_to_fingerprints([smiles_str], n_bits=2048, save_bad_log=False)
    X = fps.reshape(1, -1).astype(np.float32)

    y_pred_scaled = model.predict(X, verbose=0)[0][0]

    Tg = (y_pred_scaled * scale[0]) + mean[0]

    return {"Tg_pred": float(Tg)}
    

# YOUNG'S MODULUS MODEL LOADER 
def load_youngs_modulus_model():
    """
    Load the trained model, scaler, and feature list for predicting Young's modulus (GPa).
    """
    model_path = os.path.join(MODEL_REGISTRY, "youngs_modulus_model.pkl")
    scaler_path = os.path.join(MODEL_REGISTRY, "youngs_modulus_scaler.pkl")
    feature_path = os.path.join(MODEL_REGISTRY, "youngs_modulus_features.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError("Young's modulus model not found in models/")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    with open(feature_path) as f:
        feature_cols = json.load(f)

    return {"model": model, "scaler": scaler, "features": feature_cols}


#  YOUNG'S MODULUS PREDICTOR 
def predict_youngs_modulus(model_bundle, input_dict):

    """Predict Young’s modulus or related property given alloy composition (%)."""

    model = model_bundle['model']
    scaler = model_bundle.get('scaler')
    feature_cols = model_bundle.get('features', [])

    import pandas as pd
    # Build DataFrame preserving column names
    X_input = pd.DataFrame([[input_dict.get(col, 0.0) for col in feature_cols]], columns=feature_cols)

    if scaler is not None:
        X_input = pd.DataFrame(scaler.transform(X_input), columns=feature_cols)

    prediction = model.predict(X_input)
    return float(prediction[0])


# Load the models from the path, into a functional dict
QSAR_MODELS = {
    "regressor": Path("models/qsar_regressor_rf.pkl"),
    "classifier": Path("models/qsar_classifier_rf.pkl"),
    "scaler": Path("models/qsar_scaler.pkl"),
    "features": Path("models/qsar_feature_list.json"),
}

# set a global bits input for thr SMILES to fingerprint function below, cause training on strings of SMILES wont be that effectivem just used rdkit
N_BITS = 2048

def safe_smiles_to_fingerprint(smiles, n_bits=N_BITS):

    """Convert SMILES to binary Morgan fingerprint."""

    if not isinstance(smiles, str) or not smiles.strip():
        return np.zeros(n_bits, dtype=int)
    
    s = smiles.strip().split(';')[0].split('.')[0]
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return np.zeros(n_bits, dtype=int)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
    return np.array(fp, dtype=int)


# QSAR LOAD 
def load_qsar_model():

    """Load QSAR artifacts (regressor, classifier, scaler, feature list)."""

    try:
        reg = joblib.load(QSAR_MODELS["regressor"])
        clf = joblib.load(QSAR_MODELS["classifier"])
        scaler = joblib.load(QSAR_MODELS["scaler"]) if QSAR_MODELS["scaler"].exists() else None

        with open(QSAR_MODELS["features"]) as f:
            feature_cols = json.load(f)
        return {"regressor": reg, "classifier": clf, "scaler": scaler, "features": feature_cols}
    
    except Exception as e:
        raise RuntimeError(f"Failed to load QSAR models: {e}")


# QSAR PREDICT
def predict_qsar_toxicity(qsar_model, smiles: str):

    """Predict log(CC50), CC50 (mM), and toxicity probability from SMILES."""
    
    reg = qsar_model["regressor"]
    clf = qsar_model["classifier"]
    scaler = qsar_model["scaler"]
    feature_cols = qsar_model["features"]

    fp = safe_smiles_to_fingerprint(smiles, N_BITS).reshape(1, -1)

    mol = Chem.MolFromSmiles(smiles)
    molwt = Descriptors.MolWt(mol) if mol else 0.0
    logp = Descriptors.MolLogP(mol) if mol else 0.0
    tpsa = Descriptors.TPSA(mol) if mol else 0.0

    desc_vals = []
    for c in feature_cols[fp.shape[1]:]:
        if c == "MolWt_calc":
            desc_vals.append(molwt)
        elif c == "LogP_calc":
            desc_vals.append(logp)
        elif c == "TPSA_calc":
            desc_vals.append(tpsa)
        else:
            desc_vals.append(0.0)

    if desc_vals:
        desc_vals = np.array([desc_vals])
        if scaler is not None:
            desc_vals = scaler.transform(desc_vals)
        X_full = np.hstack([fp, desc_vals])
    else:
        X_full = fp

    log_cc50_pred = float(reg.predict(X_full)[0])
    cc50_mM_pred = 10 ** log_cc50_pred
    prob_toxic = float(clf.predict_proba(X_full)[0, 1])
    toxic_label = int(prob_toxic > 0.5)

    return {
        "SMILES": smiles,
        "log_CC50_pred": log_cc50_pred,
        "CC50_mM_pred": cc50_mM_pred,
        "prob_toxic": prob_toxic,
        "toxic_label_pred": toxic_label,
        "MolWt_calc": molwt,
        "LogP_calc": logp,
        "TPSA_calc": tpsa,
    }


# OLIGOMERIC STATE
def load_oligomeric_state_model():
    """Load trained Keras model + scaler + encoder for oligomeric state prediction."""
    model_path = os.path.join(MODEL_REGISTRY, "oligomeric_state_model.keras")
    scaler_path = os.path.join(MODEL_REGISTRY, "oligomeric_state_scaler.pkl")
    encoder_path = os.path.join(MODEL_REGISTRY, "oligomeric_state_encoder.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError("Oligomeric state model not found in models/")

    model = tf.keras.models.load_model(model_path)
    scaler = _load_joblib_or_pickle(scaler_path)
    encoder = _load_joblib_or_pickle(encoder_path)

    return {"model": model, "scaler": scaler, "encoder": encoder}


def predict_oligomeric_state(model_bundle, feature_dict):
    import pandas as pd
    import numpy as np

    model = model_bundle["model"]
    scaler = model_bundle.get("scaler", None)
    encoder = model_bundle.get("encoder", None)

    # use scalers exact feature names and order 
    if scaler:
        feature_names = list(scaler.feature_names_in_)
        X = pd.DataFrame([{f: feature_dict.get(f, scaler.mean_[i]) 
                           for i, f in enumerate(feature_names)}])
        X_scaled = scaler.transform(X)
    else:
        feature_names = list(feature_dict.keys())
        X_scaled = pd.DataFrame([list(feature_dict.values())], columns=feature_names)

    # --- Predict ---
    preds = model.predict(X_scaled, verbose=0)
    
    # --- Multi-output handling ---
    if isinstance(preds, list):
        pred_state_array = preds[0]
        pred_count_array = preds[1] if len(preds) > 1 else None
    else:
        pred_state_array = preds
        pred_count_array = None

    # --- Categorical prediction ---
    if pred_state_array.ndim == 1:
        pred_state_array = pred_state_array.reshape(1, -1)
    state_idx = int(np.argmax(pred_state_array[0]))
    state_conf = float(np.max(pred_state_array[0]))
    state_pred = encoder.inverse_transform([state_idx])[0] if encoder else str(state_idx)

    # --- Count prediction ---
    count_pred = float(np.round(pred_count_array.flatten()[0], 0)) if pred_count_array is not None else None

    return {
        "Predicted_state": state_pred,
        "Predicted_count": count_pred,
        "State_confidence": state_conf
    }


# -------------------------------
# Protein-Ligand Binding Model NOTE -----> This is not yet implemented/ Its not work / skill issue, couldnt get it to work
# -------------------------------
'''
def load_binding_model():
    """Load Keras model + scaler for protein-ligand binding prediction."""
    model_path = os.path.join(MODEL_REGISTRY, "protein_ligand_binding_model.keras")
    scaler_path = os.path.join(MODEL_REGISTRY, "protein_ligand_binding_scaler.pkl")

    model = tf.keras.models.load_model(model_path)
    scaler = _load_joblib_or_pickle(scaler_path)

    return {"model": model, "scaler": scaler}


TRAINING_RANGES = {
    "Number of Residues": (50, 1000),
    "Number of Chains": (1, 12),
    "Helix": (0, 1500),
    "Sheet": (0, 500),
    "Coil": (0, 2000),
    "Molecular Weight per Deposited Model": (5000, 200000),
}

def predict_binding(model_bundle, input_features):
    FEATURE_COLUMNS = [
        "Number of Residues",
        "Number of Chains",
        "Helix",
        "Sheet",
        "Coil",
        "Molecular Weight per Deposited Model",
    ]
    
    model = model_bundle["model"]
    scaler = model_bundle.get("scaler", None)

    # Ensure all features exist and enforce order
    X = pd.DataFrame([[float(input_features.get(col, 0.0)) for col in FEATURE_COLUMNS]],
                     columns=FEATURE_COLUMNS)

    # Warn if out of range
    for col in FEATURE_COLUMNS:
        val = X[col][0]
        min_val, max_val = TRAINING_RANGES[col]
        if val < min_val or val > max_val:
            print(f"Warning: {col}={val} is outside training range ({min_val}-{max_val})")

    # Scale
    X_scaled = scaler.transform(X) if scaler else X

    # Predict
    prob = float(model.predict(X_scaled, verbose=0)[0][0])

    return {
        "Binding_probability": prob,
        "Binding_label": "Binds" if prob >= 0.5 else "No Binding"
    }
'''