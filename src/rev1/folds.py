"""Load frozen R/T/S fold assignments and map them to row indices."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.rev1.paths import N_FOLDS, SEED, TARGETS, TEMPORAL_TEST_YEARS
from src.utils.grouped_splits import valid_mask


def labeled_index(df: pd.DataFrame, fc: list, target: str) -> np.ndarray:
    return df.index[valid_mask(df, fc, target)].to_numpy(dtype=int)


def load_station_fold_map(path: Path) -> dict[str, dict[int, list[str]]]:
    tab = pd.read_csv(path, encoding="utf-8-sig")
    out: dict[str, dict[int, list[str]]] = {}
    for t in TARGETS:
        col = f"fold_{t}"
        if col not in tab.columns:
            raise KeyError(f"{path} missing {col}")
        out[t] = {
            int(k): sorted(tab.loc[tab[col] == k, "station"].astype(str).tolist())
            for k in sorted(tab[col].unique())
        }
    return out


def station_maps_equal(a: dict, b: dict) -> tuple[bool, str]:
    for t in TARGETS:
        if set(a[t]) != set(b[t]):
            return False, f"{t}: fold ids differ {set(a[t])} vs {set(b[t])}"
        for k in a[t]:
            if list(a[t][k]) != list(b[t][k]):
                return False, f"{t} fold {k}: {a[t][k]} vs {b[t][k]}"
    return True, "ok"


def make_random_folds(df: pd.DataFrame, fc: list, seed: int = SEED) -> pd.DataFrame:
    rows = []
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for t in TARGETS:
        idx = labeled_index(df, fc, t)
        for fold, (_, te) in enumerate(kf.split(idx)):
            for i in idx[te]:
                rows.append({"target": t, "fold": int(fold), "row_index": int(i)})
    return pd.DataFrame(rows)


def make_temporal_folds(df: pd.DataFrame, fc: list) -> pd.DataFrame:
    rows = []
    for t in TARGETS:
        idx = labeled_index(df, fc, t)
        years = df.loc[idx, "year"].astype(int)
        for year in TEMPORAL_TEST_YEARS:
            te = idx[years.to_numpy() == year]
            for i in te:
                rows.append({
                    "target": t, "fold": int(year), "year": int(year),
                    "row_index": int(i),
                })
    return pd.DataFrame(rows)


def sample_inventory(df: pd.DataFrame, fc: list) -> pd.DataFrame:
    rows = []
    for t in TARGETS:
        m = valid_mask(df, fc, t)
        sub = df.loc[m, ["station_name", "year", "month"]]
        g = sub.groupby(["year", "month", "station_name"], dropna=False).size()
        for (year, month, sta), n in g.items():
            rows.append({
                "target": t, "year": int(year), "month": int(month),
                "station": str(sta), "n": int(n),
            })
    return pd.DataFrame(rows)


def split_indices(
    df: pd.DataFrame,
    fc: list,
    protocol: str,
    target: str,
    fold_key,
    folds_dir: Path,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (tr_idx, te_idx, fold_label). fold_label is 0-4 (R/S) or year (T)."""
    labeled = set(labeled_index(df, fc, target).tolist())
    if protocol == "station":
        tab = pd.read_csv(folds_dir / "station_folds.csv", encoding="utf-8-sig")
        fold_id = int(fold_key)
        tes = set(tab.loc[tab[f"fold_{target}"] == fold_id, "station"].astype(str))
        te = np.array([
            i for i in labeled
            if str(df.at[i, "station_name"]) in tes
        ], dtype=int)
        tr = np.array([i for i in labeled if i not in set(te.tolist())], dtype=int)
        return tr, te, fold_id
    if protocol == "temporal":
        year = int(fold_key)
        if year not in TEMPORAL_TEST_YEARS:
            raise ValueError(f"T test year must be 2021-2025, got {year}")
        te = np.array([
            i for i in labeled if int(df.at[i, "year"]) == year
        ], dtype=int)
        tr = np.array([
            i for i in labeled if int(df.at[i, "year"]) != year
        ], dtype=int)
        return tr, te, year
    if protocol == "random":
        rf = pd.read_csv(folds_dir / "random_folds.csv", encoding="utf-8-sig")
        fold_id = int(fold_key)
        te = rf.loc[(rf["target"] == target) & (rf["fold"] == fold_id), "row_index"]
        te = np.array([int(i) for i in te if int(i) in labeled], dtype=int)
        tr = np.array([i for i in labeled if i not in set(te.tolist())], dtype=int)
        return tr, te, fold_id
    raise ValueError(protocol)


def fold_keys(protocol: str) -> list:
    if protocol == "temporal":
        return list(TEMPORAL_TEST_YEARS)
    return list(range(N_FOLDS))


def inner_splits(protocol: str, stations, years, n: int, seed: int = SEED):
    """Index splits over the outer training arrays (protocol-respecting)."""
    from src.utils.grouped_splits import iter_group_folds

    n_rows = len(stations)
    stations = np.asarray(stations)
    years = np.asarray(years, dtype=int)
    if protocol == "station":
        return list(iter_group_folds(stations, n, seed))
    if protocol == "temporal":
        val_years = [y for y in sorted(set(years.tolist())) if y >= 2021]
        folds = []
        for y in val_years[:n]:
            va = np.where(years == y)[0]
            tr = np.where(years != y)[0]
            if len(va) >= 10 and len(tr) >= 30:
                folds.append((tr, va))
        if folds:
            return folds
        return list(iter_group_folds(stations, n, seed))
    kf = KFold(n_splits=max(2, min(n, n_rows)), shuffle=True, random_state=seed)
    return list(kf.split(np.arange(n_rows)))
