#!/usr/bin/env python3
"""
corrosion_pipeline.py

Parse CORR-DATA_Database.csv and compute normalized corrosion scores per material.
Outputs -> master_data/corrosion/corrosion_scores.csv
"""

from pathlib import Path
import pandas as pd
import math
import numpy as np
import re
import logging

log = logging.getLogger("corrosion_pipeline")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

BASE = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE / "data" / "metal_corrosion"
INPUT_FILE = DATA_DIR / "CORR-DATA_Database.csv"
MASTER_DIR = BASE / "master_data" / "corrosion"
MASTER_DIR.mkdir(parents=True, exist_ok=True)

def safe_str(x):
    return "" if pd.isna(x) else str(x)

# map letter ratings -> numeric (higher = better)
LETTER_MAP = {
    "A": 4.0,  # Resistant / excellent
    "B": 3.0,
    "C": 2.0,
    "D": 1.0,
    "E": 0.5,
    "F": 0.0
}

def parse_rating_or_rate(value: str) -> float:
    """
    Accept strings like:
      - 'D (Poor)'
      - '0.05 max'
      - numeric mm/yr or '0.002 in/yr' (we try mm/yr)
    Return a normalized numeric corrosion_rate_mm_per_yr if possible; otherwise np.nan
    """
    if pd.isna(value):
        return np.nan
    s = safe_str(value).strip()
    if s == "":
        return np.nan
    # Letter rating
    m = re.search(r'\b([A-F])\b', s.upper())
    if m:
        # return inverse numeric rate proxy: map letter to small numeric mm/yr surrogate (higher letter -> small rate)
        # we later convert letter map to score
        return np.nan  # indicate non-numeric; handle separately
    
    # Try to extract numeric mm/yr
    # Normalize units: if in mils/yr (thousandth inch), convert to mm/yr (1 mil = 0.0254 mm)
    # handle formats like "0.05 max", "2 max", "0.001"
    num_match = re.findall(r"[-+]?\d*\.?\d+|\d+", s)
    if not num_match:
        return np.nan
    try:
        num = float(num_match[0])
    except:
        return np.nan
    
    # guess units: if value > 10 it's probably mils/yr or other - but dataset mixes mm and mils; we cannot reliably infer
    # If text contains 'mils' or 'in' treat as mils (convert)
    if re.search(r'mil', s, flags=re.I) or re.search(r'inch', s, flags=re.I) or re.search(r'"', s):
        # convert mils to mm: 1 mil = 0.0254 mm
        return num * 0.0254
    # if 'mm' present or no unit, assume the number is mm/yr
    return num

def rating_to_score(letter_or_text: str) -> float:
    """Convert letter rating A-D to 0..1 score; if numeric text provided, returns nan."""
    if pd.isna(letter_or_text):
        return np.nan
    s = safe_str(letter_or_text).upper()
    # detect A/B/C/D tokens
    m = re.search(r'\b([A-F])\b', s)
    if m:
        letter = m.group(1)
        val = LETTER_MAP.get(letter, np.nan)
        # normalize to 0..1 (divide by 4)
        try:
            return float(val) / 4.0
        except:
            return np.nan
    return np.nan

def mmrate_to_score(rate_mm_per_yr: float) -> float:
    """Convert mm/yr numeric corrosion rate into a 0..1 score (lower rate -> closer to 1)."""
    if pd.isna(rate_mm_per_yr):
        return np.nan
    # clamp extremely large values
    r = max(rate_mm_per_yr, 0.0)
    # use a log transform and asymptote
    # score = 1 / (1 + log10(rate + 1)) -> rate=0 -> 1.0 ; large rate -> small score

    try:
        r = max(rate_mm_per_yr, 0.0)
        return 1.0 / (1.0 + math.log10(r + 1.0))
    except Exception:
        return np.nan

def run_corrosion_pipeline():
    log.info("Starting corrosion pipeline...")
    if not INPUT_FILE.exists():
        log.error("Corrosion input file not found: %s", INPUT_FILE)
        return

    # Some CSVs include commas, try default read then fallback with latin-1
    try:
        df = pd.read_csv(INPUT_FILE, dtype=str)
    except Exception:
        df = pd.read_csv(INPUT_FILE, dtype=str, encoding="latin-1")

    # Normalize columns (strip whitespace)
    df.columns = [c.strip() for c in df.columns]

    # Focus on required columns
    expected = ["Environment","Material Group","Material Family","Material",
                "Rate (mm/yr) or Rating","Rate (mils/yr) or Rating",
                "Localized Attack","UNS","Condition/Comment","Concentration (Vol %)",
                "Temperature (deg C)","Temperature (deg F)","Duration","Reference #","Reference"]
    for col in expected:
        if col not in df.columns:
            df[col] = None

    # Parse numeric rates where possible and convert letter ratings to scores
    df["rate_mm_from_mmcol"] = df["Rate (mm/yr) or Rating"].apply(parse_rating_or_rate)
    df["rate_mm_from_milscol"] = df["Rate (mils/yr) or Rating"].apply(parse_rating_or_rate)
    df["rate_mm"] = df["rate_mm_from_mmcol"].fillna(df["rate_mm_from_milscol"])
    
    # also compute rating scores when a letter rating is present
    df["rating_score"] = df["Rate (mm/yr) or Rating"].apply(rating_to_score)
    df["rating_score"] = df["rating_score"].fillna(df["Rate (mils/yr) or Rating"].apply(rating_to_score))

    # For each row compute a corrosion score: prefer numeric rate if present, else letter rating score
    def row_score(row):
        if not pd.isna(row["rate_mm"]):
            return mmrate_to_score(float(row["rate_mm"]))
        if not pd.isna(row["rating_score"]):
            return float(row["rating_score"])
        return np.nan

    df["corr_score_row"] = df.apply(row_score, axis=1)

    # Clean material names
    df["Clean_Name"] = df["Material"].fillna("").apply(lambda s: re.sub(r'\s+', ' ', re.sub(r'[^A-Z0-9\-\+\(\)\/ ]+', ' ', str(s).upper())).strip())

    # Group by material (and UNS if available) to get min/mean/var and count
    group_cols = ["Clean_Name"]
    agg = df.groupby(group_cols).agg(
        corr_score_min=pd.NamedAgg(column="corr_score_row", aggfunc=lambda s: float(s.dropna().min()) if s.dropna().size>0 else float("nan")),
        corr_score_mean=pd.NamedAgg(column="corr_score_row", aggfunc=lambda s: float(s.dropna().mean()) if s.dropna().size>0 else float("nan")),
        corr_score_var=pd.NamedAgg(column="corr_score_row", aggfunc=lambda s: float(s.dropna().var()) if s.dropna().size>1 else 0.0),
        data_points=pd.NamedAgg(column="corr_score_row", aggfunc=lambda s: int(s.dropna().size))
    ).reset_index()

    # Add UNS first-seen (if any)
    uns = df.groupby("Clean_Name")["UNS"].apply(lambda s: next((x for x in s if pd.notna(x) and str(x).strip() != ""), "" )).reset_index(name="UNS")
    agg = agg.merge(uns, on="Clean_Name", how="left")

    out_file = MASTER_DIR / "corrosion_scores.csv"
    agg.to_csv(out_file, index=False)
    log.info("Wrote corrosion summary -> %s (%d rows)", out_file, len(agg))

if __name__ == "__main__":
    run_corrosion_pipeline()