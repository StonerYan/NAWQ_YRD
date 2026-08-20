"""
WQI0615 Data Loader
===================
Loads Sentinel-2 spectral data and ERA5 meteorological data.
Data is read from WQI0616/data (see src/config.py).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Data paths (from config.py) ───────────────────────────────────────────────
from src.config import SPECTRAL_FILE, ERA5_FILE as ERA5_ALL_FILE
DATA_DIR = SPECTRAL_FILE.parent

# Column indices (0-based) in the raw CSV
TARGET_COL_MAP = {
    'CODMn': 60,   # Permanganate index (mg/L)
    'NH3N':  61,   # Ammonia nitrogen (mg/L)
    'TP':    62,   # Total phosphorus (mg/L)
}
SPECTRAL_BAND_COLS = {
    'station':       list(range(9, 21)),   # B1–B12 (station water pixel)
    'veg':           list(range(21, 33)),  # B1–B12 (nearby vegetation)
    'anomaly':       list(range(33, 45)),  # B1–B12 (water pixel minus climatology)
    'veg_corrected': list(range(45, 57)),  # B1–B12 (vegetation-corrected)
}
BAND_NUMS = [1, 2, 3, 4, 5, 6, 7, 8, '8A', 9, 11, 12]


from src.utils.eval_verbose import vprint


def load_spectral_data(file_path=None, encoding='gbk'):
    """
    Load Sentinel-2 spectral + water quality matchup dataset.

    Returns
    -------
    df : pd.DataFrame  (rows = matchups, standardised column names)
    col_names : dict   (original column name mapping)
    """
    if file_path is None:
        file_path = SPECTRAL_FILE

    df = pd.read_csv(file_path, encoding=encoding)
    vprint(f"Raw data: {df.shape[0]} rows, {df.shape[1]} cols")

    df['date'] = pd.to_datetime(df['date_beijing'], errors='coerce')
    df = df.dropna(subset=['date'])

    cols = df.columns.tolist()

    # Column name aliases
    col_names = {
        'station': cols[0],
        'lon':     cols[2],
        'lat':     cols[3],
    }
    for key, idx in TARGET_COL_MAP.items():
        if idx < len(cols):
            col_names[key] = cols[idx]

    # Standardised column names
    df['station_name'] = df[cols[0]]
    df['lon']          = df[cols[2]]
    df['lat']          = df[cols[3]]

    # Point ID (row-level unique id that carries station identity)
    df['point_id'] = df['station_name'].astype(str)

    # Time features
    df['year']       = df['date'].dt.year
    df['month']      = df['date'].dt.month
    df['doy']        = df['date'].dt.dayofyear
    df['season_sin'] = np.sin(2 * np.pi * df['doy'] / 365.25)
    df['season_cos'] = np.cos(2 * np.pi * df['doy'] / 365.25)

    # Year-trend features (capture long-term water quality improvement)
    df['trend_year_offset']  = df['year'] - 2020
    df['trend_decimal_year'] = df['trend_year_offset'] + (df['doy'] - 1) / 365.25
    df['trend_year_sq']      = df['trend_year_offset'] ** 2
    df['trend_year_log']     = np.log1p(df['trend_year_offset'].clip(0))
    df['trend_lon_x_year']   = df['lon'] * df['trend_year_offset']
    df['trend_lat_x_year']   = df['lat'] * df['trend_year_offset']

    # Target columns
    actual_cols = df.columns.tolist()
    for key, idx in TARGET_COL_MAP.items():
        if idx < len(actual_cols):
            df[f'target_{key}'] = df[actual_cols[idx]]

    # Spectral band columns
    for group, indices in SPECTRAL_BAND_COLS.items():
        for i, idx in enumerate(indices):
            if idx < len(actual_cols):
                df[f'spec_{group}_B{BAND_NUMS[i]}'] = df[actual_cols[idx]]

    vprint(f"Valid rows: {len(df)}")
    vprint(f"Stations:   {df['station_name'].nunique()}")
    vprint(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    return df, col_names


def load_era5_data(file_path=None):
    """Load ERA5-Land meteorological data (station × day)."""
    if file_path is None:
        file_path = ERA5_ALL_FILE

    era5 = pd.read_csv(file_path)
    era5['date'] = pd.to_datetime(era5['date'])
    vprint(f"ERA5: {era5.shape[0]} rows, {era5.shape[1]} cols")
    meteo_vars = [c for c in era5.columns if c not in ['station_id', 'lon', 'lat', 'date']]
    vprint(f"ERA5 variables: {meteo_vars}")
    return era5


def merge_era5_features(df_spectral, era5_df, lookback_days=(3, 7)):
    """
    Join ERA5 meteorological covariates to the spectral matchup dataset.

    Uses vectorised rolling windows; avoids same-day leakage via shift(1).
    """
    vprint("Merging ERA5 (vectorised)…")
    meteo_vars = ['T2m_mean_C', 'Precip_mm', 'Wind10m_ms', 'SSRD_Jm2', 'SoilMois_m3m3']
    avail = [v for v in meteo_vars if v in era5_df.columns]

    era5_w = era5_df[['station_id', 'date'] + avail].copy()
    era5_w['date'] = pd.to_datetime(era5_w['date']).dt.normalize()
    era5_w = era5_w.sort_values(['station_id', 'date'])

    for n in lookback_days:
        era5_w[f'Precip_{n}d'] = (
            era5_w.groupby('station_id')['Precip_mm']
                  .transform(lambda x: x.shift(1).rolling(n, min_periods=1).sum()))
        era5_w[f'T2m_{n}d'] = (
            era5_w.groupby('station_id')['T2m_mean_C']
                  .transform(lambda x: x.shift(1).rolling(n, min_periods=1).mean()))

    df_w = df_spectral.copy()
    df_w['_date_norm'] = pd.to_datetime(df_w['date']).dt.normalize()

    merge_cols = avail + [f'Precip_{n}d' for n in lookback_days] + [f'T2m_{n}d' for n in lookback_days]
    merged = df_w.merge(
        era5_w[['station_id', 'date'] + merge_cols],
        left_on=['station_name', '_date_norm'],
        right_on=['station_id', 'date'],
        how='left')

    rename_map = {v: f'meteo_{v}' for v in avail}
    for n in lookback_days:
        rename_map[f'Precip_{n}d'] = f'meteo_Precip_{n}d'
        rename_map[f'T2m_{n}d']    = f'meteo_T2m_{n}d'
    merged = merged.rename(columns=rename_map)

    for col in ['_date_norm', 'station_id_y', 'date_y']:
        if col in merged.columns:
            merged = merged.drop(columns=[col])
    if 'station_id' in merged.columns:
        merged = merged.drop(columns=['station_id'], errors='ignore')
    # Rename date_x → date (created when spectral 'date' clashes with ERA5 'date')
    if 'date_x' in merged.columns:
        merged = merged.rename(columns={'date_x': 'date'})
    elif 'date_y' in merged.columns:
        merged = merged.rename(columns={'date_y': 'date'})

    check = f'meteo_{avail[0]}' if avail else None
    if check and check in merged.columns:
        rate = merged[check].notna().mean()
        n_ok = merged[check].notna().sum()
        vprint(f"ERA5 merge rate: {rate:.1%} ({n_ok}/{len(merged)} rows)")

    vprint(f"Total features after merge: {len(merged.columns)}")
    return merged


def get_feature_columns(df, use_meteo=True):
    """Return ordered feature columns (delegates to feature_engineering)."""
    from src.models.feature_engineering import build_feature_list
    return build_feature_list(df, use_meteo=use_meteo)


def create_splits(df, target_col, feature_cols,
                  split_method='temporal', random_state=42, test_ratio=0.2):
    """
    Create train/test splits without leakage.

    Returns
    -------
    X_train, X_test, y_train, y_test, train_idx, test_idx
    """
    spectral = [c for c in feature_cols if c.startswith('spec_')]
    valid    = df[target_col].notna()
    if spectral:
        valid &= df[spectral].notna().any(axis=1)
    vdf = df[valid].copy()
    vprint(f"\nSplit={split_method}, valid rows={len(vdf)}")

    if split_method == 'random':
        from sklearn.model_selection import train_test_split
        idx = vdf.index.tolist()
        tr_idx, te_idx = train_test_split(idx, test_size=test_ratio, random_state=random_state)

    elif split_method == 'temporal':
        from src.config import TEMPORAL_TEST_YEAR
        # Train: all years except held-out test year; Test: complete calendar year
        tr_idx = vdf[vdf['year'] != TEMPORAL_TEST_YEAR].index.tolist()
        te_idx = vdf[vdf['year'] == TEMPORAL_TEST_YEAR].index.tolist()

    elif split_method == 'station':
        from sklearn.model_selection import train_test_split
        stations = vdf['station_name'].unique().tolist()
        tr_sta, te_sta = train_test_split(stations, test_size=test_ratio, random_state=random_state)
        tr_idx = vdf[vdf['station_name'].isin(tr_sta)].index.tolist()
        te_idx = vdf[vdf['station_name'].isin(te_sta)].index.tolist()
        vprint(f"  Train stations={len(tr_sta)}, test stations={len(te_sta)}")
    else:
        raise ValueError(f"Unknown split_method: {split_method}")

    vprint(f"  Train={len(tr_idx)}, Test={len(te_idx)}")

    def _extract(idx_list):
        sub = vdf.loc[idx_list, feature_cols].copy()
        for c in sub.columns:
            if sub[c].isna().any():
                sub[c] = sub[c].fillna(sub[c].median())
        return sub.values

    X_tr = _extract(tr_idx); X_te = _extract(te_idx)
    y_tr = vdf.loc[tr_idx, target_col].values
    y_te = vdf.loc[te_idx, target_col].values
    return X_tr, X_te, y_tr, y_te, tr_idx, te_idx


def compute_metrics(y_true, y_pred):
    """Standard regression metrics."""
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    rng  = float(np.ptp(y_true))
    return dict(R2=r2, MAE=mae, RMSE=rmse,
                NMAE=mae / rng if rng > 0 else np.nan,
                PBIAS=100 * np.sum(y_true - y_pred) / np.sum(y_true))
