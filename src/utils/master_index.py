# utils/master_index.py
import pandas as pd
import hashlib

def create_material_index(mechanical, chemical, biological):
    """Create a master index linking all unique materials"""
    materials = set()

    # Collect material names from all datasets
    for df, col in [(mechanical, 'Material'), (chemical, 'Polymer'), (biological, 'Material')]:
        if col in df.columns:
            materials.update(df[col].dropna().str.upper())

    master_index = pd.DataFrame(list(materials), columns=['material_name'])
    master_index['material_id'] = master_index['material_name'].apply(
        lambda x: hashlib.md5(x.encode()).hexdigest().upper()
    )
    return master_index