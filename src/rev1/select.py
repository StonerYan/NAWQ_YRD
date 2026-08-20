"""Train-fold feature selectors. No reserved / forced-in columns."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

from src.config import FEATURE_SELECTION_JSON
from src.models.feature_engineering import select_fold_features

SCHEMES = ("rf", "union", "union_nolucc", "union50", "vif", "spec")


def drop_lulc(fc: list[str]) -> list[str]:
    """Drop station-level LULC and LULC×driver columns from the candidate pool."""
    return [
        c for c in fc
        if not str(c).startswith("lucc_") and not str(c).startswith("drv_")
    ]


def _rf_rank(tr_df: pd.DataFrame, cols: list[str], target: str, k: int) -> list[str]:
    cols = [c for c in cols if c in tr_df.columns]
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


def _imputed(tr_df: pd.DataFrame, cols: list[str]) -> tuple[list[str], np.ndarray]:
    cols = [c for c in cols if c in tr_df.columns]
    X = SimpleImputer(strategy="median").fit_transform(tr_df[cols].to_numpy(float))
    keep = [i for i in range(len(cols)) if float(np.nanstd(X[:, i])) > 1e-12]
    return [cols[i] for i in keep], X[:, keep]


def _vif_vector(X: np.ndarray) -> np.ndarray:
    """VIF_i = 1 / (1 - R²) from regressing column i on the rest."""
    n, p = X.shape
    Z = (X - X.mean(axis=0)) / np.clip(X.std(axis=0), 1e-12, None)
    vif = np.ones(p)
    for i in range(p):
        y = Z[:, i]
        A = np.delete(Z, i, axis=1)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coef
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        vif[i] = 1.0 / max(1e-8, 1.0 - min(r2, 0.999999))
    return vif


def select_vif(tr_df: pd.DataFrame, fc: list[str], target: str, k: int = 50,
               thresh: float = 10.0) -> list[str]:
    cols, X = _imputed(tr_df, list(fc))
    while len(cols) > 2:
        vif = _vif_vector(X)
        j = int(np.argmax(vif))
        if vif[j] <= thresh:
            break
        cols.pop(j)
        X = np.delete(X, j, axis=1)
    if len(cols) > k:
        return _rf_rank(tr_df, cols, target, k)
    return cols


def select_union50(fc: list[str]) -> list[str]:
    data = json.loads(FEATURE_SELECTION_JSON.read_text(encoding="utf-8"))
    chosen = data["rf_union_top50"]["features"]
    return [c for c in chosen if c in fc]


def select_union_n(tr_df: pd.DataFrame, fc: list[str], k: int = 50) -> list[str]:
    """Paper union-N: score(f) = max RF importance across COD/NH3/TP on this train fold."""
    from src.config import TARGETS

    cols = [c for c in fc if c in tr_df.columns]
    if not cols:
        return []
    k = max(1, min(int(k), len(cols)))
    X = SimpleImputer(strategy="median").fit_transform(tr_df[cols].to_numpy(float))
    scores = {c: 0.0 for c in cols}
    for t in TARGETS:
        y = tr_df[f"target_{t}"].to_numpy(float)
        msk = np.isfinite(y)
        if int(msk.sum()) < 30:
            continue
        m = RandomForestRegressor(
            n_estimators=250, min_samples_leaf=5, max_depth=8,
            n_jobs=-1, random_state=42,
        )
        m.fit(X[msk], y[msk])
        for i, c in enumerate(cols):
            scores[c] = max(scores[c], float(m.feature_importances_[i]))
    return sorted(scores, key=lambda c: -scores[c])[:k]


def select_features(
    tr_df: pd.DataFrame,
    fc: list[str],
    target: str,
    scheme: str,
    k: int = 50,
) -> tuple[list[str], str]:
    scheme = str(scheme).lower()
    if scheme in ("rf", "rf_topk", "rf_train_fold_topk"):
        feats = select_fold_features(tr_df, fc, target, k=k)
        return feats, f"rf_topk_{len(feats)}"
    if scheme == "union50":
        feats = select_union50(fc)
        return feats, "union50_locked"
    if scheme in ("union", "union_n"):
        feats = select_union_n(tr_df, fc, k=k)
        return feats, f"union_n_{len(feats)}"
    if scheme in ("union_nolucc", "union_n_nolucc", "union_spec"):
        pool = drop_lulc(fc)
        feats = select_union_n(tr_df, pool, k=k)
        return feats, f"union_n_nolucc_{len(feats)}"
    if scheme == "vif":
        feats = select_vif(tr_df, fc, target, k=k, thresh=10.0)
        return feats, f"vif10_rf{k}_{len(feats)}"
    if scheme in ("spec", "spec50", "nolucc"):
        pool = [c for c in fc if not str(c).startswith("lucc_")]
        feats = _rf_rank(tr_df, pool, target, k)
        return feats, f"spec_rf_topk_{len(feats)}"
    raise ValueError(f"unknown feature scheme {scheme!r}; expected one of {SCHEMES}")
