"""Fold-wise feature selection, impute, and domain labels (train only)."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from src.config import N_OWT
from src.rev1.paths import SELECT_K, SELECT_SCHEME, preprocess_dir
from src.rev1.select import select_features


@dataclass
class FoldPrep:
    features: list[str]
    imputer: SimpleImputer
    kmeans: KMeans | None
    spec_idx: list[int]
    protocol: str
    target: str
    fold_key: int

    def transform(self, X_raw: np.ndarray) -> np.ndarray:
        return self.imputer.transform(X_raw)

    def domain_labels(self, stations, years, X_imp: np.ndarray) -> tuple[np.ndarray, int]:
        if self.protocol == "station":
            d = LabelEncoder().fit_transform(np.asarray(stations).astype(str))
            return d, max(int(d.max()) + 1, 2)
        if self.protocol == "temporal":
            uy = sorted(set(int(y) for y in years))
            ym = {y: i for i, y in enumerate(uy)}
            d = np.array([ym[int(y)] for y in years], dtype=int)
            return d, max(int(d.max()) + 1, 2)
        if self.kmeans is None or not self.spec_idx:
            d = np.zeros(len(X_imp), dtype=int)
            return d, 2
        d = self.kmeans.predict(X_imp[:, self.spec_idx])
        return np.asarray(d, dtype=int), int(self.kmeans.n_clusters)


def _lock_path(d: Path) -> Path:
    return d.with_suffix(d.suffix + ".lock") if d.suffix else Path(str(d) + ".lock")


def fit_or_load_prep(
    df: pd.DataFrame,
    fc_full: list[str],
    protocol: str,
    target: str,
    fold_key,
    tr_idx: np.ndarray,
    tag: str,
    k: int = SELECT_K,
    scheme: str = SELECT_SCHEME,
) -> FoldPrep:
    d = preprocess_dir(tag, protocol, target, fold_key)
    bundle = d / "preprocess.joblib"
    feat_p = d / "features.json"
    if bundle.exists() and feat_p.exists():
        return joblib.load(bundle)

    d.mkdir(parents=True, exist_ok=True)
    lock = d / ".lock"
    waited = 0.0
    while lock.exists() and waited < 1800:
        time.sleep(2.0)
        waited += 2.0
        if bundle.exists() and feat_p.exists():
            return joblib.load(bundle)
    try:
        lock.write_text("1", encoding="utf-8")
        if bundle.exists() and feat_p.exists():
            return joblib.load(bundle)
        tr = df.loc[list(tr_idx)]
        feats, selector = select_features(tr, fc_full, target, scheme=scheme, k=k)
        X = tr[feats].to_numpy(float)
        imp = SimpleImputer(strategy="median")
        X_imp = imp.fit_transform(X)
        spec_idx = [i for i, c in enumerate(feats) if str(c).startswith("spec_station_")]
        km = None
        if protocol == "random" and len(spec_idx) >= 2:
            km = KMeans(n_clusters=N_OWT, random_state=42, n_init=5)
            km.fit(X_imp[:, spec_idx])
        prep = FoldPrep(
            features=list(feats), imputer=imp, kmeans=km, spec_idx=spec_idx,
            protocol=protocol, target=target, fold_key=int(fold_key),
        )
        joblib.dump(prep, bundle)
        feat_p.write_text(json.dumps({
            "features": feats, "k": len(feats), "selector": selector,
            "scheme": scheme, "protocol": protocol, "target": target,
            "fold": int(fold_key),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return prep
    finally:
        if lock.exists():
            try:
                lock.unlink()
            except OSError:
                pass
