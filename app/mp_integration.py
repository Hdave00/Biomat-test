# app/mp_integration.py

import os
import pandas as pd
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from mp_api.client import MPRester
from functools import lru_cache
import streamlit as st

# Load key from .env
load_dotenv()
API_KEY = None


# Try Streamlit Secrets first, cause streamlit cloud uses st.secrets
try:
    API_KEY = st.secrets["MP_API_KEY"]
except Exception:
    pass

# if not then fallback to  .env (local)
if not API_KEY:
    API_KEY = os.getenv("MP_API_KEY")

# if no api key at all, then raise error
if not API_KEY:
    raise ValueError("MP_API_KEY not loaded from Streamlit secrets")


def tidy_summary_safe(summary_json: dict) -> dict:
    if not summary_json:
        return {}

    sym = summary_json.get("symmetry", None)
    
    # Always convert to string
    if sym:
        crystal_system = str(getattr(sym, "crystal_system", getattr(sym, "name", "N/A")))
        space_group = str(getattr(sym, "symbol", "N/A"))
    else:
        crystal_system = "N/A"
        space_group = "N/A"

    # Structure: convert to string formula
    struct = summary_json.get("structure", None)
    if struct:
        try:
            struct_str = getattr(struct, "formula", None) or str(struct)
        except Exception:
            struct_str = str(struct)
    else:
        struct_str = None

    return {
        "material_id": summary_json.get("material_id"),
        "formula_pretty": summary_json.get("formula_pretty"),
        "chemsys": summary_json.get("chemsys"),
        "density": summary_json.get("density"),
        "band_gap": summary_json.get("band_gap"),
        "energy_above_hull": summary_json.get("energy_above_hull"),
        "formation_energy_per_atom": summary_json.get("formation_energy_per_atom"),
        "is_stable": summary_json.get("is_stable"),
        "crystal_system": crystal_system,
        "space_group": space_group,
        "volume": summary_json.get("volume"),
        "structure": struct_str
    }


def tidy_thermo_safe(thermo_json: dict) -> dict:
    if not thermo_json:
        return {}
    return {
        "formation_energy_per_atom": thermo_json.get("formation_energy_per_atom"),
        "energy_above_hull": thermo_json.get("energy_above_hull"),
        "is_stable": thermo_json.get("is_stable"),
        "thermo_type": thermo_json.get("thermo_type", "N/A")
    }

def tidy_magnetism_safe(mag_json: dict) -> dict:
    if not mag_json:
        return {}
    return {
        "ordering": mag_json.get("ordering", "N/A"),
        "is_magnetic": bool(mag_json.get("is_magnetic")),
        "total_magnetization": mag_json.get("total_magnetization"),
        "num_magnetic_sites": mag_json.get("num_magnetic_sites"),
        "types_of_magnetic_species": ", ".join(mag_json.get("types_of_magnetic_species", [])),
        "total_magnetization_normalized_vol": mag_json.get("total_magnetization_normalized_vol")
    }


# ----  Caching for mp-api queries  ----
@st.cache_data(show_spinner=False)
def cached_query_mp_advanced_filters(**kwargs):
    """Cached wrapper to avoid re-fetching the same big MP dataset."""
    return query_mp_advanced_filters(api_key=API_KEY, **kwargs)

@st.cache_data(show_spinner=False)
def cached_query_material(material_id):
    """Cache single-material retrieval."""
    return query_materials_project(material_id, API_KEY)


def query_materials_project(term: str, api_key: Optional[str] = None) -> dict:
    """
    Query Materials Project for a given material (by formula, element, or material_id)
    and return a unified data dict including summary, structure, elasticity, thermo,
    magnetism, oxidation states, bonding, and robocrystallographer data.
    """
    from mp_api.client import MPRester
    material_id = None

    # first Resolve material_id if formula given 
    if term.lower().startswith("mp-"):
        material_id = term
    else:
        with MPRester(api_key) as mpr:
            docs = mpr.materials.summary.search(
                formula=term,
                fields=["material_id", "formula_pretty"]
            )
            if docs:
                material_id = getattr(docs[0], "material_id", None)

    if not material_id:
        print(f"No material_id found for '{term}'")
        return {}

    unified = {"material_id": material_id}

    # then Query all relevant endpoints
    with MPRester(api_key) as mpr:

        # get summary from api endpoint
        try:
            summary_docs = mpr.materials.summary.search(
                material_ids=[material_id],
                fields=[
                    "material_id", "formula_pretty", "density", "band_gap",
                    "energy_above_hull", "formation_energy_per_atom", "is_stable",
                    "volume", "structure", "symmetry", "chemsys", "elements",
                    "ordering", "total_magnetization"
                ]
            )
            if summary_docs:
                unified.update(summary_docs[0].model_dump())
        except Exception as e:
            print("Summary query failed:", e)
            unified["summary_error"] = str(e)

        # === ELASTICITY ===
        try:
            elas_docs = mpr.materials.elasticity.search(
                material_ids=[material_id],
                fields=["K_VRH", "G_VRH", "homogeneous_poisson", "elastic_tensor"]
            )
            unified["elasticity"] = elas_docs[0].model_dump() if elas_docs else {}
        except Exception as e:
            unified["elasticity"] = {}
            unified["elasticity_error"] = str(e)

        # === THERMODYNAMICS ===
        try:
            thermo_docs = mpr.materials.thermo.search(
                material_ids=[material_id],
                fields=["formation_energy_per_atom", "energy_above_hull", "is_stable"]
            )
            unified["thermo"] = thermo_docs[0].model_dump() if thermo_docs else {}
        except Exception as e:
            unified["thermo"] = {}
            unified["thermo_error"] = str(e)

        # === MAGNETISM ===
        try:
            mag_docs = mpr.materials.magnetism.search(
                material_ids=[material_id],
                fields=[
                    "ordering", "total_magnetization",
                    "total_magnetization_normalized_vol",
                    "num_magnetic_sites", "types_of_magnetic_species"
                ]
            )
            unified["magnetism"] = mag_docs[0].model_dump() if mag_docs else {}
        except Exception as e:
            unified["magnetism"] = {}
            unified["magnetism_error"] = str(e)

        # === OXIDATION STATES ===
        try:
            oxi_docs = mpr.materials.oxidation_states.search(
                material_ids=[material_id],
                fields=["average_oxidation_state", "oxidation_states", "possible_species"]
            )
            unified["oxidation_states"] = oxi_docs[0].model_dump() if oxi_docs else {}
        except Exception as e:
            unified["oxidation_states"] = {}
            unified["oxidation_states_error"] = str(e)

        # === BONDING ===
        try:
            bond_docs = mpr.materials.bonds.search(
                material_ids=[material_id],
                fields=["average_bond_length", "average_coordination_number", "bonds"]
            )
            unified["bonding"] = bond_docs[0].model_dump() if bond_docs else {}
        except Exception as e:
            unified["bonding"] = {}
            unified["bonding_error"] = str(e)

        # === ROBOCRYSTALLOGRAPHER ===
        try:
            robo_docs = mpr.materials.robocrys.search(
                material_ids=[material_id],
                fields=["description"]
            )
            unified["robocrys"] = robo_docs[0].model_dump() if robo_docs else {}
        except Exception as e:
            unified["robocrys"] = {}
            unified["robocrys_error"] = str(e)

    return unified


def get_mp_property_dataframe(mp_json: Dict[str, Any]) -> pd.DataFrame:
    if not mp_json:
        return pd.DataFrame()
    
    props = {}
    props.update(tidy_summary_safe(mp_json))
    
    if "thermo" in mp_json:
        props.update(tidy_thermo_safe(mp_json["thermo"]))
    
    if "magnetism" in mp_json:
        props.update(tidy_magnetism_safe(mp_json["magnetism"]))
    
    # Elasticity
    elasticity = mp_json.get("elasticity", {})
    K = elasticity.get("K_VRH") or elasticity.get("bulk_modulus")
    G = elasticity.get("G_VRH") or elasticity.get("shear_modulus")
    props["K_VRH"] = K
    props["G_VRH"] = G
    props["bulk_modulus"] = elasticity.get("bulk_modulus")
    props["shear_modulus"] = elasticity.get("shear_modulus")
    
    # Derived Young's modulus, the only way that i got it to work
    if K and G:
        try:
            props["E_est_GPa"] = (9 * K * G) / (3 * K + G)
        except Exception:
            props["E_est_GPa"] = None
    else:
        props["E_est_GPa"] = None
    
    return pd.DataFrame([props])


# ADVANCED FILTER QUERY
def query_mp_advanced_filters(
    api_key: Optional[str] = None,

    # MP official fields
    elements: Optional[list] = None,
    exclude_elements: Optional[list] = None,
    formula: Optional[str] = None,
    chemsys: Optional[list] = None,
    material_ids: Optional[list] = None,

    band_gap: Optional[tuple] = None,
    density: Optional[tuple] = None,
    energy_above_hull: Optional[tuple] = None,
    crystal_system: Optional[str] = None,
    spacegroup_symbol: Optional[str] = None,
    deprecated: Optional[bool] = None,

    # allow extra MP-legal fields without renaming
    **extra_filters
):
    """
    Perform advanced MP summary search using ONLY documented MP parameters.
    All parameters must match MP API exactly.
    """
    KEY = api_key or API_KEY

    try:
        with MPRester(KEY) as mpr:

            # Build query dictionary with only non-None values
            query_kwargs = {}

            # These directly map to MP API fields
            if elements:
                query_kwargs["elements"] = elements

            if exclude_elements:
                query_kwargs["exclude_elements"] = exclude_elements

            if formula:
                query_kwargs["formula"] = formula

            if chemsys:
                query_kwargs["chemsys"] = chemsys

            if material_ids:
                query_kwargs["material_ids"] = material_ids

            if band_gap:
                query_kwargs["band_gap"] = band_gap

            if density:
                query_kwargs["density"] = density

            if energy_above_hull:
                query_kwargs["energy_above_hull"] = energy_above_hull

            if crystal_system:
                query_kwargs["crystal_system"] = crystal_system

            if spacegroup_symbol:
                query_kwargs["spacegroup_symbol"] = spacegroup_symbol

            if deprecated is not None:
                query_kwargs["deprecated"] = deprecated

            # Extra filters: must be valid MP fields
            for k, v in extra_filters.items():
                if v is not None:
                    query_kwargs[k] = v

            # Perform query
            results = mpr.materials.summary.search(**query_kwargs)

            if not results:
                return None

            normalized = []
            for s in results:
                normalized.append({
                    "material_id": s.material_id,
                    "pretty_formula": getattr(s, "formula_pretty", None),
                    "density": getattr(s, "density", None),
                    "band_gap": getattr(s, "band_gap", None),
                    "e_above_hull": getattr(s, "e_above_hull", None),
                    "symmetry": getattr(s, "symmetry", None),
                    "structure": getattr(s, "structure", None),
                })

            return normalized

    except Exception as ex:
        print("Advanced MP filter query failed:", ex)
        return None