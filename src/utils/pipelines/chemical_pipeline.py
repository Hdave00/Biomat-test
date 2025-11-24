#!/usr/bin/env python3
"""
chemical_pipeline.py

Combine ionic-liquid CSVs, compute simple chemical descriptors (RDKit optional),
and produce an ML-ready chemical_features CSV with a conservative biocompatibility label.
Outputs -> master_data/chemical/chemical_features.csv
"""

from pathlib import Path
import pandas as pd
import logging
import re
from typing import Optional, Dict

log = logging.getLogger("chemical_pipeline")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

BASE = Path(__file__).resolve().parents[2]
DATA_DIR = BASE / "data" / "cytotoxicity_ionic_liquids" / "csv_datasets"
MASTER_DIR = BASE / "master_data" / "chemical"
MASTER_DIR.mkdir(parents=True, exist_ok=True)

# Threshold for conservative worst-case aggregation (mM)
BIOCOMPATIBLE_THRESHOLD_MILLIMOLAR = 10.0  # >= => likely OK
TOXIC_THRESHOLD_MILLIMOLAR = 1.0          # <= => toxic

# Try to import RDKit for descriptors; if unavailable, fallback gracefully
RDKit_available = True
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
except Exception:
    RDKit_available = False
    log.warning("RDKit not available — chemical descriptor computation will be limited.")


def safe_str(x):
    return "" if pd.isna(x) else str(x)

def parse_numeric_cc50(value: str) -> Optional[float]:
    """Reuse same parsing approach as biological pipeline to extract numeric mM value (conservative lower bound)."""
    if pd.isna(value):
        return None
    v = safe_str(value).strip()
    if v == "":
        return None
    v = re.sub(r"[,\s]*\(.+?\)", "", v)
    v = v.replace(",", "").replace("–", "-").replace("—", "-")
    # inequality
    if v.startswith(">") or v.startswith("<"):
        try:
            num = float(re.findall(r"[-+]?\d*\.?\d+|\d+", v)[0])
            return num
        except Exception:
            return None
    # range
    if "-" in v:
        parts = [p for p in v.split("-") if p.strip()]
        try:
            num = float(re.findall(r"[-+]?\d*\.?\d+|\d+", parts[0])[0])
            return num
        except Exception:
            return None
    m = re.search(r"[-+]?\d*\.?\d+|\d+", v)
    if not m:
        return None
    try:
        return float(m.group())
    except Exception:
        return None

def compute_rdkit_descriptors(smiles: str) -> Dict[str, float]:
    """Return a small set of RDKit descriptors; if RDKit unavailable returns empty dict."""
    if not RDKit_available or not smiles or smiles == "":
        return {}
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        return {
            "MolWt": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "TPSA": Descriptors.TPSA(mol),
            "HBD": Descriptors.NumHDonors(mol),
            "HBA": Descriptors.NumHAcceptors(mol),
            "AromaticRings": Descriptors.NumAromaticRings(mol)
        }
    except Exception:
        return {}

def collect_chemical_files():
    if not DATA_DIR.exists():
        log.error("Chemical data directory not found: %s", DATA_DIR)
        return []
    files = sorted([p for p in DATA_DIR.glob("*.csv")])
    # exclude methods & cell_lines (they live sibling dir; guard anyway)
    files = [f for f in files if f.name.lower() not in ("cell_lines.csv", "methods.csv")]
    return files

def parse_and_combine():
    files = collect_chemical_files()
    log.info("Found %d chemical files", len(files))
    rows = []
    for f in files:
        log.info("Reading %s", f.name)
        try:
            df = pd.read_csv(f)
        except Exception:
            df = pd.read_csv(f, encoding="latin-1")
        # identify columns robustly
        cols = {c.strip(): c for c in df.columns}
        name_col = cols.get("Name") or cols.get("name") or list(df.columns)[0]
        cc_col = None
        for cand in ["CC50/IC50/EC50, mM", "CC50/IC50/EC50", "CC50", "IC50", "EC50"]:
            if cand in cols:
                cc_col = cols[cand]; break
        cell_col = next((cols[c] for c in ("Cell line","Cell","Cellline") if c in cols), None)
        smiles_col = next((cols[c] for c in ("Canonical SMILES","SMILES","SMILES (Canonical)") if c in cols), None)

        for _, r in df.iterrows():
            name = safe_str(r.get(name_col, "")).strip()
            if name == "":
                continue
            cc_raw = safe_str(r.get(cc_col, "")).strip() if cc_col else ""
            cc_mm = parse_numeric_cc50(cc_raw)
            smiles = safe_str(r.get(smiles_col, "")).strip() if smiles_col else ""
            rw = {
                "source_file": f.name,
                "Name": name,
                "Clean_Name": re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9\-\+\(\)\/ ]+", " ", name.upper())).strip(),
                "CC50_raw": cc_raw,
                "CC50_mM": cc_mm,
                "Canonical_SMILES": smiles
            }
            # RDKit descriptors (if available)
            if smiles:
                descs = compute_rdkit_descriptors(smiles)
                rw.update(descs)
            rows.append(rw)
    return pd.DataFrame(rows)

def aggregate_to_compound(df):
    """Aggregate multiple assays per compound. Conservative (worst-case) logic: if any test is toxic, mark toxic."""
    if df.empty:
        return pd.DataFrame()
    # compute worst (min CC50)
    agg = df.groupby("Canonical_SMILES", dropna=False).agg({
        "CC50_mM": lambda s: s.dropna().min() if s.dropna().size>0 else None,
        "Name": lambda s: s.iloc[0],
        "source_file": "nunique",
        # descriptors: take median if present
        "MolWt": "median",
        "LogP": "median",
        "TPSA": "median",
        "HBD": "median",
        "HBA": "median",
        "AromaticRings": "median"
    }).reset_index().rename(columns={"source_file":"num_sources"})
    # compute binary label via thresholds
    def label_from_cc(cc):
        if pd.isna(cc):
            return -1
        if cc <= TOXIC_THRESHOLD_MILLIMOLAR:
            return 0
        if cc >= BIOCOMPATIBLE_THRESHOLD_MILLIMOLAR:
            return 1
        return -1
    agg["biocompatibility_label"] = agg["CC50_mM"].apply(label_from_cc)
    return agg

def run_chemical_pipeline():

    log.info("Starting chemical pipeline...")
    df_all = parse_and_combine()
    out_raw = MASTER_DIR / "chemical_raw_combined.csv"
    df_all.to_csv(out_raw, index=False)
    log.info("Wrote raw combined chemical measurements -> %s", out_raw)

    agg = aggregate_to_compound(df_all)
    out_agg = MASTER_DIR / "chemical_features.csv"
    agg.to_csv(out_agg, index=False)
    log.info("Wrote aggregated chemical features -> %s (%d rows)", out_agg, len(agg))
    
    log.info("Descriptor coverage: %.1f%%", 100*agg["MolWt"].notna().mean())

    return agg

if __name__ == "__main__":
    run_chemical_pipeline()