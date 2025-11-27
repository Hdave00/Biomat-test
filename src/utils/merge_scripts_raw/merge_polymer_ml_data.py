import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/polymer_tg_density")
OUTPUT_FILE = Path("master_data/unified_polymer_data.csv")

def normalize_name(series):
    return series.astype(str).str.strip().str.upper()


def load_data():
    # Load both datasets
    df_tg = pd.read_csv(DATA_DIR / "tg_density.csv")
    df_main = pd.read_csv(DATA_DIR / "TgSS_enriched_cleaned.csv")

    # Normalize column names
    df_tg.columns = df_tg.columns.str.strip().str.replace(" ", "_")
    df_main.columns = df_main.columns.str.strip().str.replace(" ", "_")

    return df_tg, df_main


def merge_polymer_data(df_tg, df_main):
    """
    Clean unified dataset for Tg-only prediction.
    - Drops density entirely
    - Ensures one 'Tg' column
    - Avoids row multiplication by name normalization
    """

    # Normalize unique identifiers
    df_tg["NAME_NORM"] = normalize_name(df_tg["name"])
    df_main["NAME_NORM"] = normalize_name(df_main["Polymer_Class"])

    # Prepare Tg in each dataset
    df_tg = df_tg.rename(columns={"Tg": "Tg_density"})
    df_main = df_main.rename(columns={"Tg": "Tg_TgSS"})

    # Use Tg column from whichever exists
    df_tg["Tg"] = df_tg["Tg_density"]
    df_main["Tg"] = df_main["Tg_TgSS"]

    # TgSS-only entries
    df_main_only = df_main[~df_main["NAME_NORM"].isin(df_tg["NAME_NORM"])].copy()

    # Combine
    df_final = pd.concat([df_tg, df_main_only], ignore_index=True)

    # Keep only necessary columns for training
    keep_cols = ["SMILES", "Tg", "name", "abr", "Polymer_Class"]
    df_final = df_final[[c for c in keep_cols if c in df_final.columns]]

    return df_final


def clean_for_ml(df):
    # Drop duplicates
    df = df.drop_duplicates(subset=["SMILES"]).copy()

    # Remove rows missing SMILES or missing Tg completely
    df = df.dropna(subset=["SMILES", "Tg"])

    # Convert Tg numeric
    df["Tg"] = pd.to_numeric(df["Tg"], errors="coerce")

    # Drop rows where Tg is NaN/invalid/noisy
    df = df[df["Tg"].notna() & (df["Tg"] > -200) & (df["Tg"] < 1000)]

    # Clean SMILES strings
    df["SMILES"] = df["SMILES"].astype(str).str.strip()

    return df


def save(df):
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nUnified Tg-only dataset saved -> {OUTPUT_FILE}")
    print(f"Total valid polymers: {len(df)}\n")
    print(df.head())


if __name__ == "__main__":
    df_tg, df_main = load_data()
    merged = merge_polymer_data(df_tg, df_main)
    merged = clean_for_ml(merged)
    save(merged)