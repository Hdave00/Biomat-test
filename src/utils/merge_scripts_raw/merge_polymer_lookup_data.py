import pandas as pd
from pathlib import Path

# Directories (unchanged layout)
DATA_DIR = Path("data/polymer_tg_density")
OUTPUT_FILE = Path("master_data/polymer_tg/polymer_lookup_data.csv")

def load_data():
    df1 = pd.read_csv(DATA_DIR / "TgSS_enriched_cleaned.csv")
    df2 = pd.read_csv(DATA_DIR / "tg_density.csv")

    # Clean column names globally (no transformations)
    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()

    return df1, df2


def merge_raw(df1, df2):
    """
    Combine both datasets by stacking them vertically.
    No modifications, no additional columns, no ML merge.
    """

    # Union (outer=True keeps all columns even if mismatch)
    merged = pd.concat([df1, df2], axis=0, ignore_index=True)

    # Remove full-row duplicates
    merged = merged.drop_duplicates()

    return merged


def save_data(df):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved polymer lookup dataset to: {OUTPUT_FILE}")


if __name__ == "__main__":
    df1, df2 = load_data()
    merged_df = merge_raw(df1, df2)
    save_data(merged_df)