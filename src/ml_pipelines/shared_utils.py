"""
And also utilities for:

    -Normalizing units

    -Label encoding categorical features

    -Handling missing values

    -Scaling numeric features
"""

from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator
import os
import numpy as np
import logging
import pandas as pd
import joblib
import numpy as np
import json


def validate_smiles_dataset(path_in, path_out=None, smiles_col="SMILES", n_bits=2048):

    """
    Validates and cleans a CSV dataset containing SMILES strings.

    - Removes rows with invalid or unparseable SMILES.
    - Generates Morgan fingerprints for validation purposes (optional check).
    - Saves cleaned dataset to path_out if provided.
    - Logs invalid entries to logs/invalid_smiles_in_dataset.json.

    Args:
        path_in (str or Path): Path to input CSV file.
        path_out (str or Path, optional): Path to save cleaned dataset.
        smiles_col (str): Column name containing SMILES strings.
        n_bits (int): Fingerprint size for quick validity checks.

    Returns:
        pd.DataFrame: Cleaned dataframe (invalid SMILES removed).

    This way we can use it in any training script:
    from shared_utils import validate_smiles_dataset

    validate_smiles_dataset(
        path_in="data/polymer_tg_density/raw_polymer_data.csv",
        path_out="data/polymer_tg_density/cleaned_polymer_data.csv"
    )
    """

    log = logging.getLogger(__name__)
    df = pd.read_csv(path_in)

    if smiles_col not in df.columns:
        raise ValueError(f"Column '{smiles_col}' not found in {path_in}")

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    valid_idx, invalid_smiles = [], []

    for i, smi in enumerate(df[smiles_col]):
        if not isinstance(smi, str) or not smi.strip():
            invalid_smiles.append(smi)
            continue
        
        cleaned = smi.strip().replace("*", "").replace("()", "")
        mol = Chem.MolFromSmiles(cleaned)

        if mol is None:
            invalid_smiles.append(cleaned)
        else:
            try:
                _ = gen.GetFingerprint(mol)  # sanity check
                valid_idx.append(i)
            except Exception:
                invalid_smiles.append(cleaned)

    cleaned_df = df.iloc[valid_idx].reset_index(drop=True)

    os.makedirs("logs", exist_ok=True)
    log_path = "logs/invalid_smiles_in_dataset.json"
    with open(log_path, "w") as f:
        json.dump(invalid_smiles, f, indent=2)

    log.info(f"Found {len(invalid_smiles)} invalid SMILES (saved to {log_path})")
    log.info(f"Retained {len(cleaned_df)} valid entries from {len(df)} total")

    if path_out:
        os.makedirs(os.path.dirname(path_out), exist_ok=True)
        cleaned_df.to_csv(path_out, index=False)
        log.info(f"Cleaned dataset saved -> {path_out}")

    return cleaned_df


def load_smiles_to_fingerprints(smiles_series, n_bits=2048, save_bad_log=True):
    """
    Converts a pandas Series of SMILES strings into RDKit Morgan fingerprints
    using the modern rdFingerprintGenerator API.

    Improvements:
    - Replaces polymer placeholders '*' with 'C' (to retain valid structure).
    - Guarantees consistent (n_samples, n_bits) output shape.
    - Skips and logs all-zero fingerprints (invalid/fragmentary SMILES).
    - Prevents mean/scale shape mismatch and tf retracing warnings.
    """

    log = logging.getLogger(__name__)
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    bad_smiles, zero_fp_smiles, fps = [], [], []

    for smi in smiles_series:
        # Validate type and non-empty string
        if not isinstance(smi, str) or not smi.strip():
            fps.append(np.zeros(n_bits))
            bad_smiles.append(str(smi))
            continue

        # Sanitize: replace polymer placeholders with 'C' for structural validity
        cleaned = smi.strip().replace("*", "C").replace("()", "")
        try:
            mol = Chem.MolFromSmiles(cleaned)
            if mol is not None:
                fp = np.array(gen.GetFingerprint(mol))
                if np.all(fp == 0):
                    zero_fp_smiles.append(cleaned)
                fps.append(fp)
            else:
                fps.append(np.zeros(n_bits))
                bad_smiles.append(cleaned)
        except Exception:
            fps.append(np.zeros(n_bits))
            bad_smiles.append(cleaned)

    # Logging for invalid/zero fingerprints
    if bad_smiles or zero_fp_smiles:
        os.makedirs("logs", exist_ok=True)

        if bad_smiles:
            log.warning(f"Example invalid SMILES: {bad_smiles[:5]}")
            if save_bad_log:
                with open("logs/invalid_smiles_log.json", "w") as f:
                    json.dump(bad_smiles, f, indent=2)

        if zero_fp_smiles:
            log.warning(f"Detected {len(zero_fp_smiles)} all-zero fingerprints.")
            if save_bad_log:
                with open("logs/zero_fingerprint_smiles.json", "w") as f:
                    json.dump(zero_fp_smiles, f, indent=2)

    # Convert to consistent numpy shape (n_samples, n_bits)
    fps = np.array(fps)
    if fps.ndim == 1:
        fps = fps.reshape(1, -1)
    elif fps.ndim == 0:
        fps = np.zeros((1, n_bits))

    # Prevent normalization mismatches downstream
    fps = fps.astype(np.float32)

    return fps


# For model paths
MODEL_PATH = "models/crystal_structure_model.pkl"
SCALER_PATH = "models/crystal_scaler.pkl"
ENCODER_PATH = "models/crystal_label_encoder.pkl"

_model = None
_scaler = None
_encoder = None

def _load_predict_objects():
    global _model, _scaler, _encoder
    if _model is None or _scaler is None or _encoder is None:
        if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(ENCODER_PATH)):
            raise FileNotFoundError("Model, scaler, or encoder files are missing in 'models/'.")
        _model = joblib.load(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
        _encoder = joblib.load(ENCODER_PATH)
    return _model, _scaler, _encoder


def predict_single(input_dict):
    """
    Predicts crystalline structure from numeric descriptors.

    input_dict keys expected:
        v(A), v(B), r(AXII)(Å), r(AVI)(Å), r(BVI)(Å),
        EN(A), EN(B), l(A-O)(Å), l(B-O)(Å),
        ΔENR, tG, τ, μ
    """
    model, scaler, encoder = _load_predict_objects()

    cols = ["v(A)","v(B)","r(AXII)(Å)","r(AVI)(Å)","r(BVI)(Å)",
            "EN(A)","EN(B)","l(A-O)(Å)","l(B-O)(Å)",
            "ΔENR","tG","τ","μ"]

    row = [input_dict.get(c, np.nan) for c in cols]
    df_row = pd.DataFrame([row], columns=cols)
    df_row = df_row.apply(pd.to_numeric, errors="coerce")

    # later for streamlit web interfacce: fill missing with medians if stored
    # with open("models/crystal_feature_medians.json") as f:
    #     TRAIN_MEDIANS = json.load(f)
    # for c in cols:
    #     df_row[c].fillna(TRAIN_MEDIANS.get(c, 0), inplace=True)

    X_scaled = scaler.transform(df_row)
    pred_encoded = model.predict(X_scaled)
    try:
        pred_label = encoder.inverse_transform(pred_encoded.astype(int))
    except Exception:
        pred_label = pred_encoded
    return pred_label[0]


# but we also want it to be able to make the predictor usable for CSV uploads or API batch calls.
def predict_batch(input_dicts):
    model, scaler, encoder = _load_predict_objects()
    cols = ["v(A)","v(B)","r(AXII)(Å)","r(AVI)(Å)","r(BVI)(Å)",
            "EN(A)","EN(B)","l(A-O)(Å)","l(B-O)(Å)",
            "ΔENR","tG","τ","μ"]
    df = pd.DataFrame(input_dicts)[cols]
    df = df.apply(pd.to_numeric, errors="coerce")
    X_scaled = scaler.transform(df)
    preds = model.predict(X_scaled)
    try:
        return encoder.inverse_transform(preds.astype(int))
    except Exception:
        return preds
    

def structure_features_dict(num_residues, num_chains, helix, sheet, coil, mol_weight):
    return {
        "Number of Residues": num_residues,
        "Number of Chains": num_chains,
        "Helix": helix,
        "Sheet": sheet,
        "Coil": coil,
        "Molecular Weight per Deposited Model": mol_weight
    }


# NOTE THIS SECTION IS ALSO NOT YET IMPLEMENTED
# --- Protein-Ligand Feature Generator ---
def generate_binding_features(protein_seq: str, ligand_smiles: str, fp_bits: int = 2048):
    """
    Convert a protein sequence and a ligand SMILES string into numeric features
    suitable for ML input.
    
    Args:
        protein_seq (str): Protein sequence (e.g., "MKT...").
        ligand_smiles (str): Ligand SMILES string.
        fp_bits (int): Size of fingerprint vector for ligand (default 2048).

    Returns:
        np.ndarray: 2D array of shape (1, n_features) ready for ML input.
    """

    # first Ligand Fingerprint ---
    if not ligand_smiles or not isinstance(ligand_smiles, str):
        ligand_fp = np.zeros(fp_bits, dtype=float)
    else:
        try:
            mol = Chem.MolFromSmiles(ligand_smiles.strip())
            if mol:
                ligand_fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=fp_bits), dtype=float)
            else:
                ligand_fp = np.zeros(fp_bits, dtype=float)
        except Exception:
            ligand_fp = np.zeros(fp_bits, dtype=float)

    # second Protein Sequence Features ---
    # Simple encoding: amino acid composition (20 standard AAs)
    aa_list = "ACDEFGHIKLMNPQRSTVWY"
    seq = protein_seq.upper() if protein_seq else ""
    aa_counts = np.array([seq.count(aa) for aa in aa_list], dtype=float)
    
    # Normalize by sequence length
    seq_len = max(len(seq), 1)
    aa_freq = aa_counts / seq_len

    # finally, combine features 
    features = np.concatenate([ligand_fp, aa_freq]).reshape(1, -1)

    return features.astype(np.float32)