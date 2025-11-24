from pathlib import Path
import re
import pandas as pd
import logging

log = logging.getLogger("material_mapper")
BASE = Path(__file__).resolve().parent.parent.parent
MASTER_DIR = BASE / "master_data"

def safe_str(x):
    return "" if pd.isna(x) else str(x)

def make_clean_name(s: str) -> str:

    """Same clean normalizer used across pipelines."""
    if s is None:
        return ""
    s = safe_str(s).strip()
    s = re.sub(r'^\s*(ANSI|ISO|DIN|ASTM|BS|SAE|GB|JIS|GOST)[\s,:-]*', '', s, flags=re.I)
    s = s.replace(",", " ")
    s = re.sub(r"[^A-Z0-9\-\+\(\)\/ ]+", " ", s.upper())
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_unified_materials(path: Path = None) -> pd.DataFrame:

    """Load existing unified material master. If path not provided, use default master_data/unified_material_data.csv"""
    p = Path(path) if path else MASTER_DIR / "unified_material_data.csv"
    if not p.exists():
        log.warning(f"Unified materials file not found at {p}")
        return pd.DataFrame()
    df = pd.read_csv(p, dtype=str)
    if "Clean_Name" not in df.columns:
        # ensure consistent clean name
        df["Clean_Name"] = df.get("Material_Raw", df.get("Material_Name", "")).apply(make_clean_name)
    return df

def build_material_index(unified_df: pd.DataFrame = None) -> pd.DataFrame:

    """Return DataFrame mapping Clean_Name -> Material_ID (guaranteed unique by first encounter)."""
    if unified_df is None:
        unified_df = load_unified_materials()
    if unified_df.empty:
        return pd.DataFrame(columns=["Material_ID", "Clean_Name", "Material_Raw"])
    idx = unified_df[["Material_ID", "Clean_Name", "Material_Raw"]].drop_duplicates(subset=["Clean_Name"])
    return idx

def match_material_to_master(name: str, unified_df: pd.DataFrame = None) -> str:
    
    """Return Material_ID for a given raw name using exact Clean_Name match. Returns empty string if not found."""
    if unified_df is None:
        unified_df = load_unified_materials()
    if unified_df.empty:
        return ""
    cn = make_clean_name(name)
    match = unified_df[unified_df["Clean_Name"] == cn]
    if not match.empty and "Material_ID" in match.columns:
        return str(match["Material_ID"].iloc[0])
    return ""