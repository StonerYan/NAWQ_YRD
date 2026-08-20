"""
Feature Engineering for WQI0627
================================
Streamlined input space:
  - 48 raw spectral bands (station / veg / veg-corrected / anomaly)
  - Curated target-agnostic indices on station, veg-corrected, anomaly surfaces
  - Water–veg adjacency ratios, temporal, spatial, ERA5 meteorology
  - Year-matched GLC_FCS30D land-cover fractions
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.spectral_features import add_multisurface_indices, add_water_veg_adjacency

CORE_ALWAYS = [
    "spec_veg_corrected_B9", "spec_station_B9", "spec_station_B3",
    "idx_sta_b4_b5_ratio", "idx_sta_oci", "idx_sta_ndre",
    "idx_sta_spm_proxy", "spec_station_B2",
    "season_sin", "season_cos", "month",
    "meteo_T2m_mean_C", "meteo_T2m_7d", "meteo_T2m_3d",
    "meteo_Precip_7d", "meteo_Precip_3d", "meteo_Precip_1d",
    "meteo_SoilMois_m3m3", "lon", "lat",
]


def add_seasonal_decomp(df: pd.DataFrame) -> pd.DataFrame:
    """Seasonal anomalies for meteorological variables (DOY 10-day bins)."""
    meteo_cols = [c for c in df.columns
                  if c.startswith("meteo_") and "anom" not in c and "seasanom" not in c]
    if not meteo_cols or "doy" not in df.columns:
        return df
    df = df.copy()
    doy_bin = df["doy"] // 10
    for col in meteo_cols:
        clim = df.groupby(doy_bin)[col].transform("median")
        clim_std = df.groupby(doy_bin)[col].transform("std").replace(0, 1.0)
        df[f"{col}_seasanom"] = (df[col] - clim) / clim_std
    return df


def add_glc_fcs30d(df: pd.DataFrame) -> pd.DataFrame:
    """Merge year-matched GLC_FCS30D fractions (clamped to 2000–2022)."""
    from src.config import DATA_DIR
    path = DATA_DIR / "lulc" / "station_year_glc_fcs30d.csv"
    if not path.exists():
        return df
    lucc = pd.read_csv(path)
    feat = [c for c in lucc.columns if c.startswith("lucc_")]
    if "year" not in df.columns:
        return df
    key = df[["station_name"]].copy()
    key["year"] = df["year"].clip(2000, 2022).astype(int)
    merged = key.merge(lucc[["station_name", "year"] + feat], on=["station_name", "year"], how="left")
    for c in feat:
        df[c] = merged[c].to_numpy()
    return df


def add_driver_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Biogeochemical driver products (nitrification, runoff pulse, urban loading)."""
    df = df.copy()
    pairs = [
        ("drv_T_x_sins", "meteo_T2m_mean_C", "season_sin"),
        ("drv_T_x_cos", "meteo_T2m_mean_C", "season_cos"),
        ("drv_p3_x_spm", "meteo_Precip_3d", "idx_sta_spm_proxy"),
        ("drv_p7_x_spm", "meteo_Precip_7d", "idx_sta_spm_proxy"),
        ("drv_built_x_sins", "lucc_built_1km", "season_sin"),
        ("drv_crop_x_p7", "lucc_crop_1km", "meteo_Precip_7d"),
        ("drv_built_x_p3", "lucc_built_1km", "meteo_Precip_3d"),
        ("drv_water_x_ndre", "lucc_water_1km", "idx_sta_ndre"),
    ]
    for name, a, b in pairs:
        if a in df.columns and b in df.columns:
            df[name] = df[a].to_numpy(float) * df[b].to_numpy(float)
    return df


def build_feature_list(df: pd.DataFrame, use_meteo: bool = True) -> list:
    """Ordered feature column list (no target-specific nh3n_/tp_ prefixes)."""
    fc: list[str] = []
    for prefix in (
        "spec_station_", "spec_veg_", "spec_veg_corrected_", "spec_anomaly_",
    ):
        fc += sorted(c for c in df.columns if c.startswith(prefix))
    fc += sorted(c for c in df.columns if c.startswith("idx_"))
    fc += sorted(c for c in df.columns if c.startswith("adj_"))
    for c in ("season_sin", "season_cos", "month", "year"):
        if c in df.columns:
            fc.append(c)
    fc += sorted(c for c in df.columns if c.startswith("trend_"))
    if use_meteo:
        fc += sorted(c for c in df.columns if c.startswith("meteo_"))
    fc += sorted(c for c in df.columns if c.startswith("lucc_"))
    fc += sorted(c for c in df.columns if c.startswith("drv_"))
    for c in ("lon", "lat"):
        if c in df.columns:
            fc.append(c)
    return list(dict.fromkeys(c for c in fc if c in df.columns))


def select_fold_features(tr_df: pd.DataFrame, fc: list, target: str, k: int = 50) -> list:
    """Train-fold-only RF ranking: keep the top-k, no reserved / forced-in block.

    LULC, drivers, spectral and meteo columns compete on the same importance
    ranking. Fit only on the outer training rows of this fold.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer

    cols = [c for c in fc if c in tr_df.columns]
    if not cols:
        return []
    k = max(1, min(int(k), len(cols)))
    y = tr_df[f"target_{target}"].to_numpy(float)
    X = SimpleImputer(strategy="median").fit_transform(tr_df[cols].to_numpy(float))
    m = RandomForestRegressor(
        n_estimators=250, min_samples_leaf=5, max_depth=8,
        n_jobs=-1, random_state=42,
    )
    m.fit(X, y)
    order = np.argsort(-m.feature_importances_)
    return [cols[i] for i in order[:k]]


def engineer_model_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Apply unified feature pipeline; returns (df, feature_list)."""
    df = add_multisurface_indices(df)
    df = add_water_veg_adjacency(df)
    df = add_glc_fcs30d(df)
    df = add_seasonal_decomp(df)
    df = add_driver_interactions(df)
    use_m = any(c.startswith("meteo_") for c in df.columns)
    fc = build_feature_list(df, use_meteo=use_m)
    return df, fc


def _apply_feature_subset(fc: list[str], subset_name: str | None) -> list[str]:
    """Restrict to a named set from feature-selection JSON (order preserved in JSON)."""
    if not subset_name:
        return fc
    from src.config import FEATURE_SELECTION_JSON
    import json
    data = json.loads(FEATURE_SELECTION_JSON.read_text(encoding="utf-8"))
    if subset_name not in data:
        raise KeyError(f"Unknown feature set '{subset_name}' in {FEATURE_SELECTION_JSON}")
    chosen = data[subset_name]["features"]
    missing = [f for f in chosen if f not in fc]
    if missing:
        raise ValueError(f"Selected features not in engineered columns: {missing[:5]}")
    return list(chosen)


def load_engineered_dataset(
    feature_subset: str | None = None,
    spectral_file=None,
    encoding: str | None = None,
) -> tuple[pd.DataFrame, list]:
    """Load spectral + ERA5 data and return (df, feature_list).

    When feature_subset is None, uses SELECTED_FEATURE_SET from config (if set).
    Pass feature_subset='full' or '' to use all engineered features.
    """
    from src.config import SELECTED_FEATURE_SET
    from src.utils.data_loader import (
        load_spectral_data, load_era5_data, merge_era5_features,
    )
    kwargs = {}
    if spectral_file is not None:
        kwargs["file_path"] = spectral_file
    if encoding is not None:
        kwargs["encoding"] = encoding
    df, _ = load_spectral_data(**kwargs)
    era5 = load_era5_data()
    df = merge_era5_features(df, era5, lookback_days=[1, 3, 7])
    df, fc = engineer_model_features(df)
    df = df.reset_index(drop=True)

    if feature_subset is None:
        feature_subset = SELECTED_FEATURE_SET
    if feature_subset and feature_subset != "full":
        fc = _apply_feature_subset(fc, feature_subset)

    import src.config as cfg
    cfg.N_FEATURES = len(fc)
    return df, fc
