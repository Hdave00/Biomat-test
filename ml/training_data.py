"""
prepare_training_data.py
---------------------------------------
Creates a unified machine-learning dataset by merging
biological, chemical, and corrosion domain outputs.

Output:
    master_data/unified_ml_dataset.csv
"""

import pandas as pd
import re
from pathlib import Path
from difflib import get_close_matches

BASE_DIR = Path(__file__).resolve().parents[1]
MASTER_DATA = BASE_DIR / "master_data"
OUTPUT_FILE = MASTER_DATA / "unified_ml_dataset.csv"

def normalize_name(name):
    """Simplify material names for better merging."""
    if pd.isna(name):
        return None
    name = str(name).lower()
    name = re.sub(r"[^a-z0-9]+", "", name)
    return name

def fuzzy_merge(df1, df2, key1, key2, threshold=0.8):
    """Perform fuzzy matching on two DataFrames when direct keys don't overlap."""
    df1 = df1.copy()
    df2 = df2.copy()
    df1["_match"] = None
    df2_keys = df2[key2].dropna().unique()

    for i, val in enumerate(df1[key1]):
        matches = get_close_matches(val, df2_keys, n=1, cutoff=threshold)
        if matches:
            df1.loc[i, "_match"] = matches[0]

    merged = pd.merge(df1, df2, left_on="_match", right_on=key2, how="inner")
    merged.drop(columns=["_match"], inplace=True)
    return merged

def main():
    print("Preparing unified ML dataset...")

    # --- Load all domain CSVs ---
    bio_file = MASTER_DATA / "biological" / "biocompatibility_master.csv"
    chem_file = MASTER_DATA / "chemical" / "chemical_features.csv"
    corr_file = MASTER_DATA / "corrosion" / "corrosion_scores.csv"

    biological = pd.read_csv(bio_file)
    chemical = pd.read_csv(chem_file)
    corrosion = pd.read_csv(corr_file)

    # --- Normalize key columns ---
    biological["Name_norm"] = biological["Clean_Name"].apply(normalize_name)
    chemical["Name_norm"] = chemical["Name"].apply(normalize_name)
    corrosion["Name_norm"] = corrosion["Clean_Name"].apply(normalize_name)

    print(f"Biological rows: {len(biological)}")
    print(f"Chemical rows: {len(chemical)}")
    print(f"Corrosion rows: {len(corrosion)}")

    # --- Step 1: try direct merge ---
    merged = pd.merge(chemical, biological, on="Name_norm", how="inner", suffixes=("_chem", "_bio"))
    print(f"Exact merge rows: {len(merged)}")

    # --- Step 2: fuzzy merge if empty ---
    if len(merged) == 0:
        print("No exact matches found — attempting fuzzy merge...")
        merged = fuzzy_merge(chemical, biological, "Name_norm", "Name_norm", threshold=0.75)
        print(f"Fuzzy merge rows: {len(merged)}")

    # --- Step 3: merge corrosion data if available ---
    if len(merged) > 0:
        merged = pd.merge(merged, corrosion, on="Name_norm", how="left", suffixes=("", "_corr"))
        print(f"After adding corrosion: {len(merged)} rows")

    # --- Step 4: Save the output ---
    if len(merged) > 0:
        merged.to_csv(OUTPUT_FILE, index=False)
        print(f"Unified ML dataset saved -> {OUTPUT_FILE}")
    else:
        print("No data merged successfully. Please check name inconsistencies.")

if __name__ == "__main__":
    main()