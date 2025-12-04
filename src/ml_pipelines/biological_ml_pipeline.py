# src/ml_pipelines/biological_qsar_pipeline.py
"""
QSAR pipeline for ionic-liquid cytotoxicity (from multiple CSVs).

- Produces fingerprint columns fp_0 .. fp_2047 and descriptors (MolWt_calc, LogP_calc, TPSA_calc).
- Creates regression target: log_CC50 (log10 of CC50/IC50/EC50 in mM).
- Creates classification target: toxic_label (1 if CC50 < threshold_mM, else 0).
- Trains RandomForestRegressor and RandomForestClassifier.
- Saves artifacts to models/ with new names to avoid overwriting:
    - models/qsar_regressor_rf.pkl
    - models/qsar_classifier_rf.pkl
    - models/qsar_scaler.pkl   (if scaling used)
    - models/qsar_feature_list.json
    - master_data/biological/biological_qsar_ml_ready.csv

    NOTE --- This dataset is a unification of a lot of other CSV files and datasets, not only for the use of Machine Learning prediction/regression but also
                a database than can be used for lookup, in one place like a general dataset.

"""

from pathlib import Path
import pandas as pd
import numpy as np
import logging
import json
import joblib
import os
import glob
import re

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdFingerprintGenerator, DataStructs

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, roc_auc_score


# Config
LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

N_BITS = 2048
FP_PREFIX = "fp_"
INPUT_DIR = Path("data/cytotoxicity_ionic_liquids/csv_datasets")
CELL_LINES_CSV = Path("data/cytotoxicity_ionic_liquids/cell_lines.csv")
METHODS_CSV = Path("data/cytotoxicity_ionic_liquids/methods.csv")
OUTPUT_ML_READY = Path("master_data/biological/biological_qsar_ml_ready.csv")

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
QSAR_REG_PATH = MODELS_DIR / "qsar_regressor_rf.pkl"
QSAR_CLF_PATH = MODELS_DIR / "qsar_classifier_rf.pkl"
QSAR_SCALER_PATH = MODELS_DIR / "qsar_scaler.pkl"
QSAR_FEATURES_PATH = MODELS_DIR / "qsar_feature_list.json"

TOXICITY_THRESHOLD_MM = 1.0  # default threshold for toxic_label (adjustable)


# Helpers: SMILES -> fingerprint
# we want to have the function take SMILES as input and an "N_BITS" length of fingerprint
def safe_smiles_to_fingerprint(smiles, n_bits=N_BITS, radius=2):

    """Return numpy array of n_bits for SMILES. Handles salts/malformed."""

    # return an empty array if the input is empty or not a str
    # This prevents RDKit from attempting to parse non-string types or blank values, which would otherwise raise parsing warnings or fail silently
    if not isinstance(smiles, str) or not smiles.strip():
        return np.zeros(n_bits, dtype=int)
    
    # strip and split the SMILES for formatting 
    # Chem.MolFromSmiles(s) parses the SMILES and returns an rdkit.Chem.rdchem.Mol object on success, or None if parsing fails. This parses the SMILES
    # Here we take only the first fragment before semi-colons or dots, which removes salts ("CCO.Cl" -> "CCO") and multi-entry formats ("CCO; note" -> "CCO")
    s = smiles.strip().split(';')[0].split('.')[0]

    # put MolFromSmiles method in a try-except block because it may return None instead of raising anything
    # RDKit commonly returns None for invalid SMILES while printing parser warnings
    try:
        mol = Chem.MolFromSmiles(s)

        # If RDKit cannot parse the SMILES, MolFromSmiles returns None, we convert this case to a zero-vector fingerprint so downstream ML stays consistent
        if mol is None:
            return np.zeros(n_bits, dtype=int)
        
        # The Morgan generator is the modern RDKit API for ECFP-style fingerprints.
        # radius=2 produces ECFP4-like circular neighborhoods.
        # fpSize=n_bits ensures the fingerprint length matches the ML model expectations.
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

        fp = gen.GetFingerprint(mol)  # returns an ExplicitBitVect (bit vector)
        # ExplicitBitVect uses efficient bit storage internally but must be converted, into a numpy array for ML uses, like below

        arr = np.zeros((n_bits,), dtype=int)

        # DataStructs.ConvertToNumpyArray fills an existing numpy array in-place, with the bitvector contents.
        DataStructs.ConvertToNumpyArray(fp, arr)   # explicit and reliable conversion
        return arr

    except Exception as e:
        LOG.warning(f"Failed to convert SMILES '{smiles}': {e}")
        return np.zeros(n_bits, dtype=int)


# Numeric extraction helper
def extract_float(value):

    """Try to parse floats from strings (handles commas, units, ranges)."""
    try:
        if pd.isna(value):
            return np.nan
        s = str(value).strip()
        s = s.replace(",", ".")

        # remove any parentheses or text after space
        s = s.split()[0]

        # handle ranges like "1-2" -> take mean
        if "-" in s:
            parts = [float(p) for p in s.split("-") if p.replace(".","",1).isdigit()]
            return float(np.mean(parts)) if parts else np.nan
        return float(s)
    except Exception:
        return np.nan


# Main data building, here we want to pass as args to the function, the global variables, that are mainly the Input and Output of the function.
# Mainly, we want to pass all of the CSV files of ionic liquids, the cell_lines file and the methods file. The last two csv files are only used for verifying
# the right data is present, and nothing "made up" or incorrect is in the final merged csv file.
def build_qsar_dataset(
    input_dir=INPUT_DIR,
    cell_lines_path=CELL_LINES_CSV,
    methods_path=METHODS_CSV,
    output_path=OUTPUT_ML_READY,
    n_bits=N_BITS,
    threshold_mM=TOXICITY_THRESHOLD_MM
):
    
    """Load CSVs, create fingerprint columns, descriptors, and targets."""

    # set an input directory first for checking via list iteration, if the assigned cell lines and method used is indeed correct. Exclude verification tables
    # handle all the ionic liquid CSVs, log and exti of no files found or error
    input_dir = Path(input_dir)
    files = sorted([p for p in input_dir.glob("*.csv") if p.name not in ["cell_lines.csv", "methods.csv"]])
    if not files:
        LOG.error(f"No CSVs found in {input_dir}")
        return None

    # Then make a list, to which we will append the dataframes of the CSV files we read. Also add error checking
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["Family"] = f.stem
            dfs.append(df)
        except Exception as e:
            LOG.warning(f"Failed to read {f}: {e}")

    # Then concatonate the data from each CSV files, based on a parent row/header and log that as well
    data = pd.concat(dfs, ignore_index=True)
    LOG.info("Loaded %d rows from %d files", len(data), len(dfs))

    # Standardize some columns to expected names (if present), we need to do this because some files may have different cased characters and we want to
    # normalize the data in those columns, into a unified column that can be fed to the model for training.
    if "Mw, g*mol-1" in data.columns:
        data["Mw"] = data["Mw, g*mol-1"].apply(extract_float)

    if "CC50/IC50/EC50, mM" in data.columns:
        data["CC50_mM"] = data["CC50/IC50/EC50, mM"].apply(extract_float)

    if "Incubation time, h" in data.columns:
        data["Incubation_h"] = data["Incubation time, h"].apply(extract_float)

    # Drop entries without SMILES or target CC50, Canonical SMILES is unified ie, its the same string used in all CSV files as the column header
    if "Canonical SMILES" not in data.columns:
        LOG.error("No 'Canonical SMILES' column found in input files")
        return None

    # Fingerprinting CAN NOT exist without SMILES, drop missing CC50 values
    data = data.dropna(subset=["Canonical SMILES", "CC50_mM"]).reset_index(drop=True)
    LOG.info("After dropping missing SMILES/CC50: %d rows", len(data))

    # Merge cell lines and method metadata if available, to fill in the respective columns. Use "left join" as to not delete any experimental data
    try:
        if Path(cell_lines_path).exists():
            cl = pd.read_csv(cell_lines_path)

            # Only add metadata, dont overwrite
            data = data.merge(cl, left_on="Cell line", right_on="Cell name", how="left")

        if Path(methods_path).exists():
            md = pd.read_csv(methods_path)
            data = data.merge(md, left_on="Method", right_on="Method abbreviation", how="left")

    except Exception as e:
        LOG.warning("Could not merge metadata: %s", e)


    # Fingerprints (vectorized)
    LOG.info("Generating %d-bit Morgan fingerprints", n_bits)
    
    # We ideally want to store the canonical SMILE data, as the finger print vectors, by passing in the safe_smiles_to_fingers prints function on the 
    # numpy array we get, which will be the SMILES column, then generate the fingerprints,that were vectorised by pandas.
    # NOTE --- Output will be (n_samples, n_bits) matrix
    fps = np.vstack(data["Canonical SMILES"].apply(lambda s: safe_smiles_to_fingerprint(s, n_bits)))
    fp_columns = [f"{FP_PREFIX}{i}" for i in range(fps.shape[1])]
    fps_df = pd.DataFrame(fps, columns=fp_columns)
    data = pd.concat([data.reset_index(drop=True), fps_df.reset_index(drop=True)], axis=1)  # reset_index will align correctly, check method docstring

    # Calculate simple descriptors using RDkit, mainly:
        # Molecular weight -> MolWt
        # Lipophilicity of the protein
        # Polar surface area
    # We want to set them as empty lists, then populate their values, as seen in the canonical SMILES for each, and append. For failed parsing, append nan.
    molwts, logps, tpsas = [], [], []

    for s in data["Canonical SMILES"]:
        s_small = s.strip().split(';')[0].split('.')[0]
        mol = Chem.MolFromSmiles(s_small)

        if mol:
            molwts.append(Descriptors.MolWt(mol))
            logps.append(Descriptors.MolLogP(mol))
            tpsas.append(Descriptors.TPSA(mol))
        else:
            molwts.append(np.nan)
            logps.append(np.nan)
            tpsas.append(np.nan)

    data["MolWt_calc"] = molwts
    data["LogP_calc"] = logps
    data["TPSA_calc"] = tpsas

    # Create QSAR targets, use log transformation, to improve regression consistency and stability
    # Toxicity label is based on the threshold calculated by the predicted CC50_mM
    data["log_CC50"] = np.log10(data["CC50_mM"].astype(float))

    # NOTE- we are using Binary classification for the boolean output basically, its NOT a weighted spectrum, that is too complicated.
    data["toxic_label"] = data["CC50_mM"].apply(lambda x: 1 if x < threshold_mM else 0)

    # Encode a few useful categorical vars like: Family, Cell type, Organism, Full name of method
    # For each columns, apply pandas "factorise" to produce contiguous integer encodings, BUT leave ORIGINAL LABELS ALONE! 
    for col in ["Family", "Cell type", "Organism", "Full name of method"]:
        if col in data.columns:
            try:
                data[col] = data[col].astype(str).fillna("NA")
                data[f"{col}_enc"] = pd.factorize(data[col])[0]
            except Exception:
                pass

    # Save ML-ready CSV with a new name to avoid overwriting
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    LOG.info("QSAR ML-ready data saved -> %s (%d rows)", output_path, len(data))
    return data


# Training / evaluation
def train_qsar_models(
    ml_ready_df,
    fingerprint_prefix=FP_PREFIX,
    regression_target="log_CC50",
    classification_target="toxic_label",
    test_size=0.2,
    random_state=42,
):
    
    """ Train RF regression & classification models on fingerprint features (plus descriptors). """

    # feature selection: all fp_* and descriptors
    fp_cols = [c for c in ml_ready_df.columns if c.startswith(fingerprint_prefix)]
    descriptor_cols = [c for c in ["MolWt_calc", "LogP_calc", "TPSA_calc"] if c in ml_ready_df.columns]
    categorical_encoded = [c for c in ml_ready_df.columns if c.endswith("_enc")]
    feature_cols = fp_cols + descriptor_cols + categorical_encoded

    X = ml_ready_df[feature_cols].fillna(0).astype(float)
    y_reg = ml_ready_df[regression_target].astype(float)
    y_clf = ml_ready_df[classification_target].astype(int)

    LOG.info("Training data shape: X=%s, y_reg=%s", X.shape, y_reg.shape)

    # Optional: scale descriptors (FPs are binary; scaling more important for descriptors)
    scaler = StandardScaler()

    # scale descriptor + categorical part but keep bits as-is: concatonate scaled descriptors
    if descriptor_cols + categorical_encoded:
        non_fp = X[descriptor_cols + categorical_encoded]
        scaled_non_fp = scaler.fit_transform(non_fp)
        # concat back
        X_full = np.hstack([X[fp_cols].values, scaled_non_fp])
    else:
        X_full = X.values
        scaler = None

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X_full, y_reg.values, y_clf.values, test_size=test_size, random_state=random_state
    )

    # RandomForest regressor
    reg = RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1)
    reg.fit(X_train, y_reg_train)
    y_reg_pred = reg.predict(X_test)
    r2 = r2_score(y_reg_test, y_reg_pred)
    mae = mean_absolute_error(y_reg_test, y_reg_pred)
    LOG.info("Regressor -> R²=%.3f, MAE=%.3f (log units)", r2, mae)

    # RandomForest classifier
    clf = RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)
    clf.fit(X_train, y_clf_train)
    y_clf_pred = clf.predict(X_test)
    acc = accuracy_score(y_clf_test, y_clf_pred)
    # if probabilities present compute ROC AUC
    try:
        y_clf_prob = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_clf_test, y_clf_prob)
    except Exception:
        auc = None
    LOG.info("Classifier -> Accuracy=%.3f, ROC-AUC=%s", acc, f"{auc:.3f}" if auc is not None else "N/A")


    # Save artifacts
    joblib.dump(reg, QSAR_REG_PATH)
    joblib.dump(clf, QSAR_CLF_PATH)
    if scaler is not None:
        joblib.dump(scaler, QSAR_SCALER_PATH)
    with open(QSAR_FEATURES_PATH, "w") as f:
        json.dump(feature_cols, f)

    LOG.info("Saved QSAR models and artifacts: %s, %s", QSAR_REG_PATH, QSAR_CLF_PATH)
    return {"regressor": reg, "classifier": clf, "scaler": scaler, "feature_cols": feature_cols, "metrics": {"r2": r2, "mae": mae, "acc": acc, "auc": auc}}


# Inference helpers
def _load_qsar_artifacts():

    if not QSAR_REG_PATH.exists() or not QSAR_CLF_PATH.exists() or not QSAR_FEATURES_PATH.exists():
        raise FileNotFoundError("QSAR model artifacts not found. Train the models first.")
    reg = joblib.load(QSAR_REG_PATH)
    clf = joblib.load(QSAR_CLF_PATH)
    scaler = joblib.load(QSAR_SCALER_PATH) if QSAR_SCALER_PATH.exists() else None
    with open(QSAR_FEATURES_PATH) as f:
        feature_cols = json.load(f)
    return reg, clf, scaler, feature_cols


def predict_from_smiles(smiles, threshold_mM=TOXICITY_THRESHOLD_MM):

    """Given a SMILES string, return predicted log_CC50 (regression) and toxicity probability."""

    reg, clf, scaler, feature_cols = _load_qsar_artifacts()
    
    # compute fingerprint and descriptors
    fp = safe_smiles_to_fingerprint(smiles, N_BITS).reshape(1, -1)
    s = str(smiles).strip().split(';')[0].split('.')[0]
    mol = Chem.MolFromSmiles(s)
    molwt = Descriptors.MolWt(mol) if mol else 0.0
    logp = Descriptors.MolLogP(mol) if mol else 0.0
    tpsa = Descriptors.TPSA(mol) if mol else 0.0

    # build X vector according to saved feature_cols
    # feature_cols: [fp_0..fp_N, descriptor_cols..., cat_enc...]
    fp_len = fp.shape[1]
    
    # gather descriptor/categorical values in order
    desc_vals = []
    for c in feature_cols[fp_len:]:
        if c == "MolWt_calc":
            desc_vals.append(molwt)
        elif c == "LogP_calc":
            desc_vals.append(logp)
        elif c == "TPSA_calc":
            desc_vals.append(tpsa)
        else:
            # unknown categorical: default 0
            desc_vals.append(0.0)
    if desc_vals:
        if scaler is not None:
            scaled = scaler.transform([desc_vals])  # scaler was trained only on desc+cat
        else:
            scaled = np.array([desc_vals], dtype=float)
        X_full = np.hstack([fp, scaled])
    else:
        X_full = fp

    # Regression output (log CC50)
    log_cc50_pred = float(reg.predict(X_full)[0])
    # Classification probability
    prob_toxic = float(clf.predict_proba(X_full)[0, 1]) if hasattr(clf, "predict_proba") else float(clf.predict(X_full)[0])

    # return both raw regression and classification label (based on threshold in mM)
    return {"log_CC50_pred": log_cc50_pred, "CC50_mM_pred": 10 ** log_cc50_pred, "prob_toxic": prob_toxic, "toxic_label_pred": int(prob_toxic > 0.5)}


# CLI / Entrypoint
def run_all(train_models_flag=True):

    data = build_qsar_dataset()

    if data is None:
        LOG.error("No data built; aborting.")
        return
    
    if train_models_flag:
        res = train_qsar_models(data)
        LOG.info("Training finished. Metrics: %s", res["metrics"])
    return data

if __name__ == "__main__":
    run_all(train_models_flag=True)