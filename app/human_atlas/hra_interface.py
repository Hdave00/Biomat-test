# will contain Human Resource Atlas CDN helper functions.

# hra_interface.py

import requests
import pandas as pd
import json

BASE_URL = "https://cdn.humanatlas.io/digital-objects"
CDN_URL = "https://cdn.humanatlas.io/digital-objects"

def build_object_path(organ_id):
    try:
        return SUPPORTED_ORGANS[organ_id]
    except KeyError:
        raise ValueError(f"Unsupported organ: {organ_id}")

SUPPORTED_ORGANS = {
    "pelvis": "ref-organ/pelvis",
    "left-kidney": "ref-organ/kidney-left",
    "right-kidney": "ref-organ/kidney-right",
    "heart": "ref-organ/heart",
    "liver": "ref-organ/liver",
    "left-knee": "ref-organ/left-knee",
    "right-knee": "ref-organ/right-knee",
    # add more as needed, for now these are working
}

# Mapping from user UI labels -> HRA organ IDs
ORGAN_MAP = {
    "Hip": "pelvis",
    "Knee": "femur-left",
    "Left Femur": "femur-left",
    "Right Femur": "femur-right",
    "Pelvis": "pelvis",
    "Heart": "heart",
    "Kidney (Left)": "kidney-left",
    "Kidney (Right)": "kidney-right",
    "Lung (Left)": "lung-left",
    "Lung (Right)": "lung-right",
}


# first get METADATA
def get_organ_metadata(organ_id, sex):
    obj = build_object_path(organ_id, sex)
    url = f"{CDN_URL}/{obj}/metadata.json"

    r = requests.get(url)

    if r.status_code != 200:
        raise Exception(f"HRA metadata fetch failed [{r.status_code}]: {r.text}")
    
    if "text/html" in r.headers.get("Content-Type", ""):
        raise Exception(f"Expected JSON but got HTML: {r.text[:200]}")

    return r.json()


# then build 3D MODEL URL
def get_3d_model_url(organ_id, sex):
    """
    Given a valid organ ID, build GLB 3D model URL.
    """

    # HRA file naming pattern: NOTE this is pattern to use to tget the correct url from HRA
    #   3d-vh-male-pelvis.glb
    return f"{CDN_URL}/{SUPPORTED_ORGANS[organ_id]}/3d-vh-{sex}-{organ_id}.glb"


# then get ASCT+B table
def get_asctb_table(_: str):
    """
    ASCT+B crosswalk is a global table (same for all organs)
    so organ_id is unused.
    """
    csv_url = f"{CDN_URL}/ref-organ/asct-b-3d-models-crosswalk/asct-b-3d-models-crosswalk.csv"
    df = pd.read_csv(csv_url)
    return df