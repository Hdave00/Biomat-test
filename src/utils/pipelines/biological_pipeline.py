#!/usr/bin/env python3
"""
utils/pipelines/biological_pipeline.py

Parses ionic-liquid cytotoxicity CSVs, methods.csv, and cell_lines.csv
and produces three outputs in master_data/biological/:

- chemical_toxicity_measurements.csv
- material_identity_map.csv
- biocompatibility_master.csv (binary classification 0/1; -1 = uncertain)

Designed to be conservative and easily tweaked.
"""

from pathlib import Path
import pandas as pd
import re
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("biological_pipeline")

BASE = Path(__file__).resolve().parent.parent.parent  # repo root
DATA_DIR = BASE / "data" / "cytotoxicity_ionic_liquids" / "csv_datasets"
MASTER_DIR = BASE / "master_data" / "biological"
MASTER_DIR.mkdir(parents=True, exist_ok=True)

# Configurable thresholds (mM)
TOXIC_THRESHOLD_MILLIMOLAR = 1.0     # <= => toxic
BIOCOMPATIBLE_THRESHOLD_MILLIMOLAR = 10.0  # >= => biocompatible

# Files expected in DATA_DIR: many ionic-liquid CSVs e.g. ammonium.csv, imidazolium.csv...
# Also expect: cell_lines.csv and methods.csv in same directory or parent directory.
CELL_LINES_FILE = DATA_DIR.parent / "cell_lines.csv"
METHODS_FILE = DATA_DIR.parent / "methods.csv"

# Field names as given in main biological dataset
MEAS_COLUMNS = [
    "Name","Empirical formula","CAS","SMILES","Canonical SMILES",
    "Mw, g*mol-1","CC50/IC50/EC50, mM","Statistics (95% CI, SEM, SD, RMSE, SE etc.)",
    "Incubation time, h","Cell line","Method","Reference (author+year+journal)","doi","Notes"
]

def safe_str(x):
    return "" if pd.isna(x) else str(x)

def make_clean_name(s: str) -> str:
    if s is None:
        return ""
    s = safe_str(s).strip()
    s = re.sub(r'^\s*(ANSI|ISO|DIN|ASTM|BS|SAE|GB|JIS|GOST)[\s,:-]*', '', s, flags=re.I)
    s = s.replace(",", " ")
    s = re.sub(r"[^A-Z0-9\-\+\(\)\/ ]+", " ", s.upper())
    s = re.sub(r"\s+", " ", s).strip()
    return s

def parse_numeric_cc50(value: str) -> Optional[float]:
    """
    Parse a string value of CC50/IC50/EC50 and return numeric value in mM.
    Returns None if unparseable.
    Handles:
     - single numbers "12.5"
     - inequalities ">100", "<0.1"
     - ranges "1-10", "0.5–2.0"
     - values with units (assumes mM if no unit)
    """
    if pd.isna(value):
        return None
    v = str(value).strip()
    if v == "":
        return None

    # Remove commas, extra text in parentheses
    v = re.sub(r"[,\s]*\(.+?\)", "", v)
    v = v.replace(",", "")
    # Common separators in ranges
    v = v.replace("–", "-").replace("—", "-")

    # If contains '>' or '<'
    if v.startswith(">") or v.startswith("<"):
        try:
            num = float(re.findall(r"[-+]?\d*\.?\d+|\d+", v)[0])
            # For inequality, we return the numeric as-is; final aggregator will handle conservatism
            return num
        except Exception:
            return None

    # Range like '1-10' -> take the lower bound (conservative: smallest means more toxic)
    if "-" in v:
        parts = [p for p in v.split("-") if p.strip() != ""]
        try:
            num = float(re.findall(r"[-+]?\d*\.?\d+|\d+", parts[0])[0])
            return num
        except Exception:
            return None

    # Try to extract first numeric
    m = re.search(r"[-+]?\d*\.?\d+|\d+", v)
    if not m:
        return None
    try:
        return float(m.group())
    except Exception:
        return None

def classify_cc50_to_label(cc50_mM: Optional[float]) -> int:
    """
    Conservative mapping:
    - cc50 <= TOXIC_THRESHOLD_MILLIMOLAR -> 0 (toxic)
    - cc50 >= BIOCOMPATIBLE_THRESHOLD_MILLIMOLAR -> 1 (likely biocompatible)
    - otherwise -> -1 (uncertain)
    """
    if cc50_mM is None:
        return -1
    if cc50_mM <= TOXIC_THRESHOLD_MILLIMOLAR:
        return 0
    if cc50_mM >= BIOCOMPATIBLE_THRESHOLD_MILLIMOLAR:
        return 1
    return -1

def load_methods():
    if METHODS_FILE.exists():
        try:
            dfm = pd.read_csv(METHODS_FILE)
            return dfm
        except Exception as e:
            log.warning(f"Failed to load methods file: {e}")
    return pd.DataFrame()

def load_cell_lines():
    if CELL_LINES_FILE.exists():
        try:
            dfl = pd.read_csv(CELL_LINES_FILE)
            return dfl
        except Exception as e:
            log.warning(f"Failed to load cell_lines file: {e}")
    return pd.DataFrame()

def collect_cytotox_files() -> list:
    if not DATA_DIR.exists():
        log.error(f"Cytotox directory not found: {DATA_DIR}")
        return []
    files = sorted([p for p in DATA_DIR.glob("*.csv")])
    # exclude the methods / cell_lines files if they live in same dir
    files = [f for f in files if f.name.lower() not in ("methods.csv", "cell_lines.csv")]
    return files

def parse_all_cytotox():
    files = collect_cytotox_files()
    log.info(f"Found {len(files)} cytotox files")
    records = []
    identity_rows = []

    for f in files:
        log.info(f"Parsing {f.name}")
        try:
            df = pd.read_csv(f)
        except Exception:
            # Try with latin-1 fallback
            df = pd.read_csv(f, encoding="latin-1")

        # normalize column names for robustness
        cols = {c.strip(): c for c in df.columns}
        # try common expected columns
        name_col = cols.get("Name") or cols.get("name") or cols.get("Material") or list(df.columns)[0]
        cc_col = None
        for candidate in ["CC50/IC50/EC50, mM", "CC50/IC50/EC50", "CC50", "IC50", "EC50"]:
            if candidate in cols:
                cc_col = cols[candidate]
                break
        if cc_col is None:
            # heuristics: find column that contains 'CC' or 'IC' in name
            for c in df.columns:
                if re.search(r"CC50|IC50|EC50", c, flags=re.I):
                    cc_col = c
                    break

        cell_col_candidates = ["Cell line","Cell", "Cellline"]
        cell_col = next((cols[c] for c in cell_col_candidates if c in cols), None)
        method_col = next((cols[c] for c in ["Method","method","Assay"] if c in cols), None)
        cas_col = next((cols[c] for c in ["CAS","cas","CAS Number"] if c in cols), None)
        mw_col = next((cols[c] for c in ["Mw, g*mol-1","Mw","Molecular weight"] if c in cols), None)

        for _, r in df.iterrows():
            name = safe_str(r.get(name_col, "")).strip()
            if name == "":
                continue
            cas = safe_str(r.get(cas_col, "")).strip() if cas_col else ""
            raw_cc = safe_str(r.get(cc_col, "")).strip() if cc_col else ""
            cc_numeric = parse_numeric_cc50(raw_cc)
            cell_line = safe_str(r.get(cell_col, "")).strip() if cell_col else ""
            method = safe_str(r.get(method_col, "")).strip() if method_col else ""
            mw = safe_str(r.get(mw_col, "")).strip() if mw_col else ""

            # Clean names
            clean_name = make_clean_name(name)

            records.append({
                "source_file": f.name,
                "Name": name,
                "CAS": cas,
                "Clean_Name": clean_name,
                "CC50_raw": raw_cc,
                "CC50_mM": cc_numeric,
                "Cell_line": cell_line,
                "Method": method,
                "Mw": mw
            })

            identity_rows.append({
                "Name": name,
                "CAS": cas,
                "Clean_Name": clean_name,
                "Source_File": f.name
            })

    # Create DataFrames
    df_records = pd.DataFrame(records)
    df_identity = pd.DataFrame(identity_rows).drop_duplicates(subset=["Name","CAS","Clean_Name"]).reset_index(drop=True)
    return df_records, df_identity

def aggregate_to_material_label(df_records: pd.DataFrame, cell_lines_df: pd.DataFrame):
    """
    Aggregate measurements per material into one conservative label.
    We focus on human cell lines (user chose all human cell lines).
    We'll select the lowest CC50_mM measured on any human cell line (most conservative).
    """
    # Identify human cell line names from the cell_lines file
    human_cells = set()
    if not cell_lines_df.empty:

        # detect rows where Organism suggests human, homo sapien only :)
        for _, r in cell_lines_df.iterrows():
            org = safe_str(r.get("Organism", "")).lower()
            name = safe_str(r.get("Cell name", "")).strip()
            if "human" in org or "homo sapiens" in org.lower():
                human_cells.add(name.upper())

    # Upper-case mapping for matching
    df_records["Cell_line_up"] = df_records["Cell_line"].str.upper().fillna("")

    # Flag which rows are human-derived if cell_lines mapping available; if not available assume all are potentially human
    if len(human_cells) > 0:
        df_records["Is_Human_Cell"] = df_records["Cell_line_up"].apply(lambda x: any(hc in x for hc in human_cells))
    else:
        # no mapping available -> assume all rows may be human (user chose include all human cell lines available)
        df_records["Is_Human_Cell"] = True

    # For aggregation: prefer human cell measurements
    # Group by Clean_Name and pick the minimum CC50_mM among human cell rows if available, otherwise overall min
    agg_rows = []
    for name, g in df_records.groupby("Clean_Name"):
        human_subset = g[g["Is_Human_Cell"] == True]
        if not human_subset.empty:
            min_cc = human_subset["CC50_mM"].dropna().min() if human_subset["CC50_mM"].dropna().size>0 else None
        else:
            min_cc = g["CC50_mM"].dropna().min() if g["CC50_mM"].dropna().size>0 else None

        label = classify_cc50_to_label(min_cc)
        agg_rows.append({
            "Clean_Name": name,
            "min_CC50_mM_on_human": min_cc,
            "biocompatibility_label": label,
            "num_measurements": len(g),
            "num_human_measurements": int(g["Is_Human_Cell"].sum())
        })

    return pd.DataFrame(agg_rows)

def run_biological_pipeline():
    log.info("Starting biological pipeline...")
    df_records, df_identity = parse_all_cytotox()

    # Load reference tables if available
    df_methods = load_methods()
    df_cell_lines = load_cell_lines()

    # Save raw combined measurements for traceability
    out_measurements = MASTER_DIR / "chemical_toxicity_measurements.csv"
    df_records.to_csv(out_measurements, index=False)
    log.info(f"Wrote combined measurements -> {out_measurements} ({len(df_records)} rows)")

    # Save identity map
    out_identity = MASTER_DIR / "material_identity_map.csv"
    df_identity.to_csv(out_identity, index=False)
    log.info(f"Wrote identity map -> {out_identity} ({len(df_identity)} unique identities)")

    # Aggregate to binary label per material
    df_labels = aggregate_to_material_label(df_records, df_cell_lines)

    # Keep only definitive labels (1 or 0). Save uncertain separately.
    df_definitive = df_labels[df_labels["biocompatibility_label"].isin([0,1])].copy()
    df_uncertain = df_labels[df_labels["biocompatibility_label"] == -1].copy()

    out_labels = MASTER_DIR / "biocompatibility_master.csv"
    df_definitive.to_csv(out_labels, index=False)
    log.info(f"Wrote definitive biocompatibility labels -> {out_labels} ({len(df_definitive)} rows)")

    out_uncertain = MASTER_DIR / "biocompatibility_uncertain.csv"
    df_uncertain.to_csv(out_uncertain, index=False)
    log.info(f"Wrote uncertain label set -> {out_uncertain} ({len(df_uncertain)} rows)")

    log.info("Biological pipeline complete.")

if __name__ == "__main__":
    run_biological_pipeline()