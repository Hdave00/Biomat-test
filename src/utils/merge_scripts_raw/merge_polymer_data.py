""" 
This merge script is different from the other polymer merge script as this one is intended to classify AND merge the two files for future use in 
    identifying the class of a polymer ie: 'Flexible', 'Rigid' or 'Semi-rigid' which becomes a label for certain data-classification tasks. 
    For handling 'None' values, the label given is 'Unknown'.

    NOTE-- Not used in current architecture.

"""


import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/polymer_tg_density")
OUTPUT_FILE = Path("master_data/unified_polymer_data.csv")

def load_data():

    tgss = pd.read_csv(DATA_DIR / "TgSS_enriched_cleaned.csv")
    density = pd.read_csv(DATA_DIR / "tg_density.csv")

    # Normalize column names globally AFTER data load
    tgss.columns = tgss.columns.str.strip().str.replace(" ", "_")
    density.columns = density.columns.str.strip().str.replace(" ", "_")

    return tgss, density


def normalize_tg_column(df, column_name):

    df[column_name] = pd.to_numeric(df[column_name], errors="coerce")
    # Convert Kelvin to Celsius if too high
    df.loc[df[column_name] > 300, column_name] -= 273.15
    return df


def assign_rigidity_class(tg_c):

    if pd.isna(tg_c): return "Unknown"
    if tg_c < 0: return "Flexible"
    if tg_c <= 100: return "Semi-rigid"
    return "Rigid"


def process_data(tgss, density):

    # Standardize column names
    tgss = tgss.rename(columns={"Tg": "Tg_C"})
    density = density.rename(columns={
        "name": "Polymer",
        "Tg": "Tg_C",
        "Density": "Density_g_cm3"
    })

    # Normalize Tg scale
    tgss = normalize_tg_column(tgss, "Tg_C")
    density = normalize_tg_column(density, "Tg_C")

    # Select columns (Polymer_Class is the correct name)
    tgss = tgss[["Polymer_Class", "Tg_C"]]
    density = density[["Polymer", "SMILES", "Tg_C", "Density_g_cm3"]]

    # Merge & clean
    df = pd.merge(density, tgss, on="Tg_C", how="outer")
    df["Polymer"] = df["Polymer"].fillna("Unknown Polymer")

    # Feature engineering
    df["Rigidity_Class"] = df["Tg_C"].apply(assign_rigidity_class)

    df = df.drop_duplicates(subset=["Polymer"])

    # Final ordering
    df = df.reindex(columns=[
        "Polymer", "Polymer_Class", "SMILES",
        "Tg_C", "Density_g_cm3", "Rigidity_Class"
    ])

    return df


def save_data(df):
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved polymer dataset to {OUTPUT_FILE}")


if __name__ == "__main__":
    tgss, density = load_data()
    df = process_data(tgss, density)
    save_data(df)