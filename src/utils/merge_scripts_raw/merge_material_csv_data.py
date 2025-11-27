"""
utils/merge_material_csv_data.py

Improved merging / cleaning for materials_properties.csv (Kaggle).
- Prefers 'Std' column for standard when available
- Keeps original Material text and creates a Clean_Name for merging
- Extracts Material_Type with more patterns
- Normalizes E, G, Ro -> GPa, GPa, g/cm3
- Produces missing_report.csv and a summary CSV
"""

import os
import re
import pandas as pd
import hashlib


# adjust as repo data folder location changes
BASE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "materials_data")
INPUT_CSV = os.path.join(BASE_PATH, "materials_properties_data.csv")
OUTPUT_CSV = os.path.join(BASE_PATH, "unified_material_data.csv")
MISSING_REPORT = os.path.join(BASE_PATH, "missing_report.csv")
UNMATCHED_TYPES = os.path.join(BASE_PATH, "unmatched_material_types.csv")

# Helper funcs
def safe_str(x):
    return "" if pd.isna(x) else str(x)

def normalize_whitespace(s):
    return re.sub(r"\s+", " ", safe_str(s)).strip()

def make_clean_name(material_raw):
    """
    Create a normalized, merge-friendly material "clean name"
    - Remove punctuation except hyphen and plus
    - Collapse multiple spaces
    - Uppercase
    - Keep composition tokens (like, Ti-6Al-4V) as it is when possible
    """
    s = normalize_whitespace(material_raw)

    # remove leading standard tokens like "ANSI," if accidentally present in name
    s = re.sub(r'^\s*(ANSI|ISO|DIN|ASTM|BS|SAE|GB|JIS|GOST)[\s,:-]*', '', s, flags=re.I)

    # Replace commas with space
    s = s.replace(",", " ")

    # Keep letters, numbers, dash, plus, parentheses; remove other punct
    s = re.sub(r"[^A-Z0-9\-\+\(\)\/ ]+", " ", s.upper())
    s = re.sub(r"\s+", " ", s).strip()

    return s

def prefer_std_column(row):

    """
    Return the standard: prefer the 'Std' column if present and non-empty,
    otherwise attempt to extract from the Material text (fallback).
    """
    if "Std" in row and pd.notna(row["Std"]) and str(row["Std"]).strip() != "":
        return str(row["Std"]).strip().upper()
    
    # fallback, try to parse first token or known abbreviations in Material string
    mat = safe_str(row.get("Material", ""))

    # look for tokens at start like "ANSI", "ISO", "DIN", "ASTM", "BS", "SAE", "GB" and "NF"
    m = re.match(r'^(ANSI|ISO|DIN|ASTM|BS|SAE|GB|JIS|GOST|NF)\b', mat, flags=re.I)
    if m:
        return m.group(1).upper()
    
    # last fallback: search for any of the known codes anywhere
    for token in ["ANSI","ASTM","ISO","DIN","BS","SAE","GB","JIS","GOST","NF"]:
        if token in mat.upper():
            return token
    return "UNKNOWN"

# expanded material type matcher (ordered, longer-first) save the MATERIAL_TYPE_PATTERNS as a tuple, using regex for exact matches of various materials
# this is done because of the specificity of the type of materials present in the csv
MATERIAL_TYPE_PATTERNS = [
    # Copper alloys first (more specific)
    (r'\bMUNTZ METAL\b', 'COPPER_ALLOY'),
    (r'\bLEADED MUNTZ METAL\b', 'COPPER_ALLOY'),
    (r'\bC\d{3,5}\b', 'COPPER_ALLOY'),  # C28000, C36500 style
    
    # Cast irons
    (r'\bNODULAR CAST IRON\b', 'NODULAR_CAST_IRON'),
    (r'\bMALLEABLE CAST IRON\b', 'MALLEABLE_CAST_IRON'),
    (r'\bGRAY CAST IRON\b', 'GRAY_CAST_IRON'),
    (r'\bCAST IRON\b', 'CAST_IRON'),
    (r'\bGGG\b', 'NODULAR_CAST_IRON'),  # DIN GGG-50, GGG-60, etc.
    (r'\bGG\b', 'GRAY_CAST_IRON'),      # DIN GG-15, GG-20, etc.
    (r'\bGTS\b', 'CAST_IRON'),          # DIN GTS-40, GTS-50
    
    # DIN specific patterns (free-cutting steels and special codes)
    (r'\b(9SMN28|10S2O|35S20)\b', 'FREE_CUTTING_STEEL'),  # Free-cutting steels
    (r'\b(HI|HII)\b', 'STEEL'),                           # DIN quality grades
    (r'\b(UST1303|UST34-2|RST37-2)\b', 'STEEL'),          # Specific DIN codes
    
    # DIN steel composition patterns (after clean_name removes prefixes)
    (r'\b(CK|CF)\d+\b', 'CARBON_STEEL'),                  # CK10, CF35, etc.
    (r'\b(\d+MN\d+|\d+CR\d+|\d+MO\d+)\b', 'ALLOY_STEEL'), # 17MN4, 34CR4, 15MO3
    (r'\b(\d+CRMO\d+|\d+CRV\d+)\b', 'ALLOY_STEEL'),       # 25CRMO4, 50CRV4
    (r'\b(\d+CRNIMO\d+|\d+NI\d+)\b', 'ALLOY_STEEL'),      # 17CRNIMO6, 10NI14
    (r'\b(WTST\d+)\b', 'STEEL'),                          # WTST52-3
    
    # JIS steel patterns (after clean_name)
    (r'\b(SUM22|SMN433|SCR430|SCR435|SCR440|SMNC420)\b', 'ALLOY_STEEL'),
    (r'\b(SNCM815|SNCM447|SNCM630)\b', 'ALLOY_STEEL'),
    (r'\bS(NC|CM|UP|MnC|Cr)\d+', 'ALLOY_STEEL'),          # SCM430, SUP9, etc.
    (r'\bSUS\d+', 'STAINLESS_STEEL'),                     # SUS304, SUS316, etc.
    (r'\bSUH\d+', 'HEAT_RESISTANT_STEEL'),                # SUH1, SUH330
    (r'\bS\d+C\b', 'CARBON_STEEL'),                       # S10C, S45C, etc.
    (r'\bSS\d+\b', 'STEEL'),                              # SS330, SS400, SS490
    (r'\bSTKM\d+', 'STEEL'),                              # STKM13B, STKM16A
    (r'\bSM\d+\b', 'STEEL'),                              # SM570, SM52OC
    
    # BS steel patterns (after clean_name)
    (r'\b(\d+[A-Z]\d+|\d+M\d+)\b', 'STEEL'),              # 230M07, 212M36, 070M26
    (r'\bGRADE\s+\d+\b', 'STEEL'),                        # GRADE 360, GRADE 430
    (r'\b(\d+[A-Z]\d+)\b', 'STEEL'),                      # 530A36, 525A60
    (r'\b(\d+S\d+)\b', 'STAINLESS_STEEL'),                # 304S15, 316S11, 430S17
    (r'\b(CR\d+GP|CR\d+PL|CEW\d+BK|CFS\d+NBK)\b', 'STEEL'), # CR3GP, CEW2BK, etc.
    (r'\b(\d+[A-Z])\b', 'STEEL'),                         # 40B, 40C, 43B, 50D, etc.
    
    # CSN patterns (Czech standards)
    (r'\b1[01]\d{4}\b', 'STEEL'),                         # 10370, 11110, 11301, etc.
    (r'\b4[012]\d{4}\b', 'STEEL'),                        # 422303, 423042, etc.
    
    # NF (French) steel patterns
    (r'\b([XZ]\w+\d+)\b', 'STEEL'),                       # XC10, Z6C13, Z12C13
    (r'\b(\d+[A-Z]+\d+)\b', 'STEEL'),                     # 13MF4, 35MF6
    (r'\b([A-Z]\d+)\b', 'STEEL'),                         # A34-2, E24-2, A48CP
    (r'\b(\d+[CDMNS]+\d+)\b', 'ALLOY_STEEL'),             # 35CD4, 16MC5, 30CND8
    
    # Generic standards detection (catch-all for any remaining standardized materials)
    (r'\b(ST|UST|RST|CK|CF|SMN|SCR|SNCM)\w*\b', 'STEEL'),
    
    # Basic material types
    (r'\bBRASS\b', 'BRASS'),
    (r'\bBRONZE\b', 'BRONZE'),
    (r'\bCOPPER-NICKEL\b', 'COPPER_NICKEL'),
    (r'\bCOPPER\b', 'COPPER'),
    (r'\bMAGNESIUM\b', 'MAGNESIUM'),
    (r'\bALUMINUM|ALUMINIUM\b', 'ALUMINUM'),
    (r'\bTITANIUM\b', 'TITANIUM'),
    (r'\bNICKEL\b', 'NICKEL'),
    (r'\bSTAINLESS\b', 'STAINLESS_STEEL'),
    (r'\bSTEEL\b', 'STEEL'),
    (r'\bALLOY\b', 'ALLOY'),
    (r'\bPEROVSKITE\b', 'PEROVSKITE'),
    (r'\bCERAMIC\b', 'CERAMIC'),
    (r'\bPOLYMER\b', 'POLYMER'),
]

# function to extract material type from a name, because this dataset has some material names as "JIS SM570" which are basically standardised materials
# jsut named after the standard
def extract_material_type_from_name(material_name):

    name_u = safe_str(material_name).upper()

    # First check specific patterns
    for patt, label in MATERIAL_TYPE_PATTERNS:
        if re.search(patt, name_u):
            return label
        
    # Handle code-based heuristics
    if re.search(r'\bC\d{3,5}\b', name_u):  # C28000, C36500 style
        return "COPPER_ALLOY"
    
    # If it contains common steel element symbols (Cr, Ni, Mo, V, Mn, etc.)
    if re.search(r'\b(\d+CR|\d+NI|\d+MO|\d+V|\d+MN|\d+SI|\d+W|\d+CO)\b', name_u):
        return "ALLOY_STEEL"
    
    # If it looks like a standardized code (numbers and letters mixed)
    if re.search(r'^\d+[A-Z]|[A-Z]+\d+', name_u):
        return "STEEL"
    
    # If it starts with a known standard but no other pattern matched
    if re.search(r'^(DIN|BS|CSN|NF|JIS|ISO|ASTM)\b', name_u):
        return "STEEL"
    
    return "UNKNOWN"

# then we want to convert the units given a dataframe
def convert_units(df):

    # rename MPa columns if present
    if "E" in df.columns and df["E"].notna().any():
        df["Youngs_Modulus_MPa"] = df["E"]
    if "G" in df.columns and df["G"].notna().any():
        df["Shear_Modulus_MPa"] = df["G"]

    # density 'Ro' appears to be kg/m3 -> convert to g/cm3
    if "Ro" in df.columns and df["Ro"].notna().any():
        df["Density_kgm3"] = df["Ro"]

    # convert to GPa and g/cm3
    if "Youngs_Modulus_MPa" in df.columns:
        df["Youngs_Modulus_GPa"] = df["Youngs_Modulus_MPa"] / 1000.0
    if "Shear_Modulus_MPa" in df.columns:
        df["Shear_Modulus_GPa"] = df["Shear_Modulus_MPa"] / 1000.0
    if "Density_kgm3" in df.columns:
        df["Density_gcm3"] = df["Density_kgm3"] / 1000.0
    return df


# Load the CSV
print("Loading:", INPUT_CSV)

# read as strings initially to avoid surprises
df = pd.read_csv(INPUT_CSV, dtype=str)

print("Original columns:", df.columns.tolist())

# Try to force numeric columns (some CSVs have missing numeric cells)
num_cols_candidates = ["Su","Sy","E","G","A5","Bhn","HV","mu","Ro"]

for c in num_cols_candidates:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# Create Material_ID if present as "ID" else, fallback to index
if "ID" in df.columns:
    df["Material_ID"] = df["ID"]
else:
    df["Material_ID"] = df.index.astype(str)


if "Material_Raw" not in df.columns:
    df["Material_Raw"] = df.get("Material", "").astype(str)

if "Clean_Name" not in df.columns:
    df["Clean_Name"] = df["Material_Raw"].apply(make_clean_name)

# Ensure Material_Name exists (same as Clean_Name for now)
if "Material_Name" not in df.columns:
    df["Material_Name"] = df["Clean_Name"]

# Standard Selection (ANSI / ISO / DIN / UNKNOWN)
df["Standard"] = df.apply(prefer_std_column, axis=1)

# Rename Desc -> Notes if present
if "Desc" in df.columns:
    df = df.rename(columns={"Desc": "Notes"})

# Material Type extraction
df["Material_Type"] = df["Material_Name"].apply(extract_material_type_from_name)

# Units conversion
df = convert_units(df)



# Load High Entropy Alloys dataset
HEA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "high_entropy_alloys", "high_entropy_alloys_properties.csv")

if os.path.exists(HEA_FILE):
    df_hea = pd.read_csv(HEA_FILE)
    
    # Standardize column names and select relevant properties
    df_hea_clean = pd.DataFrame()
    
    # Map HEA columns to your unified schema
    df_hea_clean["Material_Raw"] = df_hea.get("FORMULA", "")
    df_hea_clean["Material_Name"] = df_hea_clean["Material_Raw"].apply(make_clean_name)
    df_hea_clean["Clean_Name"] = df_hea_clean["Material_Name"]
    df_hea_clean["Standard"] = "HEA"
    df_hea_clean["Material_Type"] = "HIGH_ENTROPY_ALLOY"
    df_hea_clean["Heat treatment"] = df_hea.get("PROPERTY: Processing method", None)
    
    # Mechanical properties mapping
    df_hea_clean["Tensile_Strength_MPa"] = df_hea.get("PROPERTY: UTS (MPa)", None)
    df_hea_clean["Yield_Strength_MPa"] = df_hea.get("PROPERTY: YS (MPa)", None)
    df_hea_clean["Youngs_Modulus_GPa"] = df_hea.get("PROPERTY: Exp. Young modulus (GPa)", 
                                                   df_hea.get("PROPERTY: Calculated Young modulus (GPa)", None))
    df_hea_clean["Hardness_HV"] = df_hea.get("PROPERTY: HV", None)
    df_hea_clean["Elongation_percent"] = df_hea.get("PROPERTY: Elongation (%)", None)
    df_hea_clean["Density_gcm3"] = df_hea.get("PROPERTY: Exp. Density (g/cm$^3$)", 
                                             df_hea.get("PROPERTY: Calculated Density (g/cm$^3$)", None))
    
    # Set missing columns to None
    df_hea_clean["Shear_Modulus_GPa"] = None
    df_hea_clean["Poisson_Ratio"] = None
    df_hea_clean["Hardness_BHN"] = None
    df_hea_clean["Notes"] = "High Entropy Alloy: " + df_hea.get("PROPERTY: Microstructure", "").fillna("")
    
    # Generate IDs
    df_hea_clean["Material_ID"] = df_hea_clean["Material_Name"].apply(
        lambda x: hashlib.md5(x.encode()).hexdigest().upper()
    )
    
    # Drop duplicates and append
    df_hea_clean.drop_duplicates(subset=["Material_Name"], inplace=True)
    df = pd.concat([df, df_hea_clean], ignore_index=True, sort=False)
    print(f"Added High Entropy Alloys dataset ({len(df_hea_clean)} rows)")
else:
    print("High Entropy Alloys dataset not found -> Skipped")


# Load Alloys dataset
ALLOYS_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "alloys", "alloys_dataset.csv")

if os.path.exists(ALLOYS_FILE):
    df_alloys = pd.read_csv(ALLOYS_FILE)
    
    df_alloys_clean = pd.DataFrame()
    
    # Map Alloys columns to your unified schema
    df_alloys_clean["Material_Raw"] = df_alloys.get("Alloy", "")
    df_alloys_clean["Material_Name"] = df_alloys_clean["Material_Raw"].apply(make_clean_name)
    df_alloys_clean["Clean_Name"] = df_alloys_clean["Material_Name"]
    df_alloys_clean["Standard"] = "ASTM"  # Most are ASTM standards
    df_alloys_clean["Material_Type"] = "ALLOY"
    
    # Convert psi to MPa for tensile strength
    if "Tensile Strength: Ultimate (UTS) (psi)" in df_alloys.columns:
        df_alloys_clean["Tensile_Strength_MPa"] = df_alloys["Tensile Strength: Ultimate (UTS) (psi)"] * 0.00689476
    
    # Set other properties to None (this dataset mainly has composition)
    df_alloys_clean["Yield_Strength_MPa"] = None
    df_alloys_clean["Youngs_Modulus_GPa"] = None
    df_alloys_clean["Shear_Modulus_GPa"] = None
    df_alloys_clean["Poisson_Ratio"] = None
    df_alloys_clean["Density_gcm3"] = None
    df_alloys_clean["Elongation_percent"] = None
    df_alloys_clean["Hardness_BHN"] = None
    df_alloys_clean["Hardness_HV"] = None
    df_alloys_clean["Heat treatment"] = None
    df_alloys_clean["Notes"] = "Alloy composition data"
    
    # Generate IDs
    df_alloys_clean["Material_ID"] = df_alloys_clean["Material_Name"].apply(
        lambda x: hashlib.md5(x.encode()).hexdigest().upper()
    )
    
    # Drop duplicates and append
    df_alloys_clean.drop_duplicates(subset=["Material_Name"], inplace=True)
    df = pd.concat([df, df_alloys_clean], ignore_index=True, sort=False)
    print(f"Added Alloys dataset ({len(df_alloys_clean)} rows)")
else:
    print("Alloys dataset not found -> Skipped")


# Load elemental metals Young's Modulus dataset
YOUNGS_MOD_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "youngs_modulus", "youngs_modulus.csv")

if os.path.exists(YOUNGS_MOD_FILE):
    df_young = pd.read_csv(YOUNGS_MOD_FILE, sep=";")

    # Standardize column names
    df_young.rename(columns={
        "Metal": "Material_Raw",
        "Young's Modulus (GPa)": "Youngs_Modulus_GPa"
    }, inplace=True)

    # Assign default metadata fields
    df_young["Material_Name"] = df_young["Material_Raw"].apply(make_clean_name)
    df_young["Clean_Name"] = df_young["Material_Name"]
    df_young["Standard"] = "ELEMENT"
    df_young["Material_Type"] = "METAL"
    df_young["Heat treatment"] = None
    df_young["Tensile_Strength_MPa"] = None
    df_young["Yield_Strength_MPa"] = None
    df_young["Shear_Modulus_GPa"] = None
    df_young["Poisson_Ratio"] = None
    df_young["Density_gcm3"] = None  # optional future enhancement
    df_young["Elongation_percent"] = None
    df_young["Hardness_BHN"] = None
    df_young["Hardness_HV"] = None
    df_young["Notes"] = None

    # Generate unique Material IDs for metals if missing, use a lambda function, then use the hashlib to get a hexadec value for each id in CAPS
    if "Material_ID" not in df_young.columns:
        df_young["Material_ID"] = df_young["Material_Name"].apply(lambda x: hashlib.md5(x.encode()).hexdigest().upper())

    # dropping exact duplicates to avoid double counting if youngs modulus info later overlaps with alloys
    df_young.drop_duplicates(subset=["Material_Name"], inplace=True)

    # Append into master df before filtering/output
    df = pd.concat([df, df_young], ignore_index=True, sort=False)
    print(f"Added elemental metals Young’s modulus dataset ({len(df_young)} rows)")


else:
    print("Young’s Modulus dataset not found -> Skipped")

# Rename common numeric columns for clarity (if present)
rename_map = {
    "Su": "Tensile_Strength_MPa",
    "Sy": "Yield_Strength_MPa",
    "A5": "Elongation_percent",
    "Bhn": "Hardness_BHN",
    "HV": "Hardness_HV",
    "mu": "Poisson_Ratio"
}


for k, v in rename_map.items():
    if k in df.columns:
        df.rename(columns={k: v}, inplace=True)

# Select final columns that matter for this dataset
final_cols = [
    "Material_ID", "Material_Raw", "Material_Name", "Clean_Name",
    "Standard", "Material_Type",
    "Heat treatment",
    "Tensile_Strength_MPa", "Yield_Strength_MPa",
    "Youngs_Modulus_GPa", "Shear_Modulus_GPa", "Poisson_Ratio",
    "Density_gcm3", "Elongation_percent", "Hardness_BHN", "Hardness_HV",
    "Notes",
]


# use list iteration to check if there are any duplicated in the final file
final_existing = [c for c in final_cols if c in df.columns]
unified = df[final_existing].copy()

# Save missing report
missing = unified.isna().sum().sort_values(ascending=False)
missing.to_csv(MISSING_REPORT)
print("Saved missing report ->", MISSING_REPORT)

# Save unmatched material types for inspection
cols_to_show = [c for c in ["Material_ID","Material_Raw","Clean_Name"] if c in unified.columns]

unmatched = unified[unified["Material_Type"] == "UNKNOWN"][cols_to_show]

if len(unmatched) > 0:
    unmatched[cols_to_show].to_csv(UNMATCHED_TYPES, index=False)
    print(f"Unmatched material types found: {len(unmatched)}")
    print("Saved report ->", UNMATCHED_TYPES)
else:
    print("No unmatched material types!")

# Save unified table
unified.to_csv(OUTPUT_CSV, index=False)
print("Saved unified dataset ->", OUTPUT_CSV)

# Print summary
print("Summary:")
print(" Rows:", len(unified))
print(" Distinct standards:", unified["Standard"].nunique(), "examples ->", unified["Standard"].unique()[:10])
print(" Distinct material types:", unified["Material_Type"].nunique(), "examples ->", unified["Material_Type"].unique()[:20])