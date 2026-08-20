"""
Concentration-stratified grouped k-fold for station-out validation.
===================================================================
Every station is held out exactly once. Official Protocol S ranks stations
by each target's own mean concentration, then fills folds inside rank
blocks so each fold spans that target's range. CODMn / NH3N / TP therefore
have different station-to-fold maps.

shared_station_folds() keeps the old composite-rank map for historical
diagnostics only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import TARGETS, STATION_N_FOLDS, STATION_CV_SEED


def valid_mask(df: pd.DataFrame, fc: list, target: str) -> pd.Series:
    valid = df[f"target_{target}"].notna()
    spec = [c for c in fc if c.startswith("spec_")]
    if spec:
        valid = valid & df[spec].notna().any(axis=1)
    return valid


def station_rank_table(df: pd.DataFrame, fc: list) -> pd.DataFrame:
    rows = []
    for name, g in df.groupby("station_name"):
        row = {"station": name, "n": int(len(g))}
        for t in TARGETS:
            v = g.loc[valid_mask(g, fc, t), f"target_{t}"]
            row[f"n_{t}"] = int(v.notna().sum())
            row[f"mean_{t}"] = float(v.mean()) if v.notna().any() else np.nan
        rows.append(row)
    tab = pd.DataFrame(rows)
    for t in TARGETS:
        tab[f"rank_{t}"] = tab[f"mean_{t}"].rank()
    tab["rank"] = tab[[f"rank_{t}" for t in TARGETS]].mean(axis=1)
    return tab.sort_values("rank").reset_index(drop=True)


def stratified_group_folds(
    stations_ranked: list[str],
    n_splits: int = STATION_N_FOLDS,
    seed: int = STATION_CV_SEED,
) -> list[list[str]]:
    """Assign stations to folds so each fold covers the concentration range."""
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(n_splits)]
    for blk in range(0, len(stations_ranked), n_splits):
        chunk = list(stations_ranked[blk:blk + n_splits])
        order = rng.permutation(len(chunk))
        for slot, j in enumerate(order):
            folds[slot].append(chunk[j])
    return [sorted(f) for f in folds if f]


def _assign_folds(tab: pd.DataFrame, ranked: list[str], col: str,
                  n_splits: int, seed: int) -> list[list[str]]:
    folds = stratified_group_folds(ranked, n_splits=n_splits, seed=seed)
    tab[col] = -1
    for i, stas in enumerate(folds):
        tab.loc[tab["station"].isin(stas), col] = i
    return folds


def target_station_folds(
    df: pd.DataFrame,
    fc: list,
    target: str,
    n_splits: int = STATION_N_FOLDS,
    seed: int = STATION_CV_SEED,
) -> tuple[list[list[str]], pd.DataFrame]:
    """Stratify on one target's station-mean rank only."""
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}; expected one of {TARGETS}")
    tab = station_rank_table(df, fc)
    ranked = tab.sort_values(f"rank_{target}", kind="mergesort")["station"].tolist()
    folds = _assign_folds(tab, ranked, f"fold_{target}", n_splits, seed)
    tab["fold"] = tab[f"fold_{target}"]
    return folds, tab.sort_values(f"rank_{target}").reset_index(drop=True)


def per_target_station_folds(
    df: pd.DataFrame,
    fc: list,
    n_splits: int = STATION_N_FOLDS,
    seed: int = STATION_CV_SEED,
    targets: list[str] | None = None,
) -> tuple[dict[str, list[list[str]]], pd.DataFrame]:
    """Official S: one concentration-stratified 5-fold map per target."""
    targets = list(targets or TARGETS)
    tab = station_rank_table(df, fc)
    folds_by: dict[str, list[list[str]]] = {}
    for t in targets:
        ranked = tab.sort_values(f"rank_{t}", kind="mergesort")["station"].tolist()
        folds_by[t] = _assign_folds(tab, ranked, f"fold_{t}", n_splits, seed)
    return folds_by, tab.sort_values("rank").reset_index(drop=True)


def shared_station_folds(
    df: pd.DataFrame,
    fc: list,
    n_splits: int = STATION_N_FOLDS,
    seed: int = STATION_CV_SEED,
    target: str | None = None,
) -> tuple[list[list[str]], pd.DataFrame]:
    """Legacy composite-rank map, or one-target map when ``target`` is set."""
    if target is not None:
        return target_station_folds(df, fc, target, n_splits=n_splits, seed=seed)
    tab = station_rank_table(df, fc)
    folds = _assign_folds(tab, tab["station"].tolist(), "fold", n_splits, seed)
    return folds, tab


def iter_group_folds(groups, n_splits: int, seed: int = 42):
    """Inner GroupKFold over training stations (no concentration ranking)."""
    groups = np.asarray(groups)
    uniq = np.array(sorted(np.unique(groups), key=str))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_splits = max(2, min(int(n_splits), len(uniq)))
    fold_of = {g: i % n_splits for i, g in enumerate(uniq)}
    labels = np.array([fold_of[g] for g in groups])
    for f in range(n_splits):
        yield np.where(labels != f)[0], np.where(labels == f)[0]


def grouped_holdout_indices(groups, frac: float = 0.15, seed: int = 42, min_val: int = 1):
    """Hold out whole stations (not trailing rows) for early stopping."""
    groups = np.asarray(groups)
    uniq = np.array(sorted(np.unique(groups), key=str))
    rng = np.random.default_rng(seed)
    n_val = max(min_val, int(round(len(uniq) * frac)))
    n_val = min(n_val, max(1, len(uniq) - 1))
    val_g = set(rng.choice(uniq, size=n_val, replace=False).tolist())
    va = np.array([i for i, g in enumerate(groups) if g in val_g], dtype=int)
    tr = np.array([i for i, g in enumerate(groups) if g not in val_g], dtype=int)
    return tr, va


GEO_FEATURE_PREDICATE = (
    "lon", "lat", "month", "year", "season_sin", "season_cos",
)


def is_geo_feature(name: str) -> bool:
    if name in GEO_FEATURE_PREDICATE:
        return True
    return name.startswith(("meteo_", "trend_", "season_"))
