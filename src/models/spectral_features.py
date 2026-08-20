"""
Unified spectral indices — target-agnostic, multi-surface (station / vc / anomaly).
Curated from Spearman + model-importance analysis; removes nh3n_/tp_/phys_ redundancy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-8

# band group in spec_{group}_B{band}
SURFACES = {
    "sta": "station",
    "vc": "veg_corrected",
    "anom": "anomaly",
}

# Curated index recipes (name -> needs B11/B12 flag)
CURATED_INDICES = (
    "ndre", "ndwi", "b4_b5_ratio", "oci", "grvi", "nir_red",
    "spm_proxy", "flh", "cire", "cdom_proxy", "cdom_b1_b3",
    "turb_b4_b8", "multiband_iss", "b4_b8_prod", "red_edge_slope", "mndwi",
)

ADJ_BANDS = (3, 4, 5, 8)


def _gb(df: pd.DataFrame, group: str, band) -> np.ndarray:
    col = f"spec_{group}_B{band}"
    if col not in df.columns:
        return np.zeros(len(df))
    return df[col].values.astype(float)


def _sdiv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.where(np.abs(b) > EPS, a / (b + EPS), 0.0)


def _bands(df: pd.DataFrame, group: str) -> dict:
    keys = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
    return {k: _gb(df, group, k) for k in keys}


def _compute_index(name: str, B: dict) -> np.ndarray | None:
    B1, B2, B3 = B[1], B[2], B[3]
    B4, B5, B6 = B[4], B[5], B[6]
    B7, B8 = B[7], B[8]
    B11, B12 = B[11], B[12]

    if name == "ndre":
        return _sdiv(B5 - B4, B5 + B4)
    if name == "ndwi":
        return _sdiv(B3 - B8, B3 + B8)
    if name == "b4_b5_ratio":
        return _sdiv(B4, B5)
    if name == "oci":
        return _sdiv(B3, B5)
    if name == "grvi":
        return _sdiv(B3 - B4, B3 + B4)
    if name == "nir_red":
        return _sdiv(B8, B4)
    if name == "spm_proxy":
        return _sdiv(B5, B4)
    if name == "flh":
        return B5 - 0.5 * (B4 + B6)
    if name == "cire":
        return _sdiv(B7, B5) - 1.0
    if name == "cdom_proxy":
        return _sdiv(B1, B4)
    if name == "cdom_b1_b3":
        return _sdiv(B1, B3)
    if name == "turb_b4_b8":
        return _sdiv(B4, B8)
    if name == "multiband_iss":
        return _sdiv(B5 + B6 + B7, 3.0 * B4)
    if name == "b4_b8_prod":
        return B4 * B8 / (B3 ** 2 + EPS)
    if name == "red_edge_slope":
        return _sdiv(B7 - B5, B7 + B5)
    if name == "mndwi":
        return _sdiv(B3 - B11, B3 + B11)
    return None


def add_multisurface_indices(df: pd.DataFrame) -> pd.DataFrame:
    """idx_{sta|vc|anom}_{recipe} for each curated index and surface."""
    for surf_key, group in SURFACES.items():
        B = _bands(df, group)
        for name in CURATED_INDICES:
            val = _compute_index(name, B)
            if val is not None:
                df[f"idx_{surf_key}_{name}"] = val

    # vegetation-pixel NDVI (context only, not duplicated per target)
    vB3, vB8 = _gb(df, "veg", 3), _gb(df, "veg", 8)
    if np.any(vB3 != 0) or np.any(vB8 != 0):
        df["idx_veg_ndvi"] = _sdiv(vB8 - vB3, vB8 + vB3)
    return df


def add_water_veg_adjacency(df: pd.DataFrame) -> pd.DataFrame:
    """Station / nearby-vegetation band ratios (4 bands)."""
    for band in ADJ_BANDS:
        vs = _gb(df, "station", band)
        vv = _gb(df, "veg", band)
        if np.any(vv != 0):
            df[f"adj_sta_veg_B{band}"] = _sdiv(vs, vv)
    return df
