"""Simple ADAE: one between-station mean + DANN + XGB + CatBoost.

Same recipe for COD_Mn / NH3-N / TP. No per-target heads.
Stage-1 is part of ADAE (not a preprocessor given to the baselines).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from src.models.adae import _center_by_group, blend_simplex
from src.rev1.trainers import _finalize, fit_dann_kfold_bundle
from src.utils.grouped_splits import iter_group_folds

try:
    import catboost as cb
    import xgboost as xgb
except ImportError:
    cb = xgb = None


def _grouped_splits(X_tr, groups, seed):
    if groups is not None:
        return list(iter_group_folds(groups, 3, seed))
    return list(KFold(n_splits=3, shuffle=True, random_state=seed).split(X_tr))


def fit_cab_oof(X_tr, y, X_te, best, seed, groups=None):
    p = dict(best)
    oof = np.zeros(len(y), dtype=float)
    for tri, vai in _grouped_splits(X_tr, groups, seed):
        if len(vai) < 5 or len(tri) < 20:
            continue
        m = cb.CatBoostRegressor(colsample_bylevel=0.8, random_seed=seed, verbose=0, **p)
        m.fit(X_tr[tri], y[tri], verbose=False)
        oof[vai] = m.predict(X_tr[vai])
    m = cb.CatBoostRegressor(colsample_bylevel=0.8, random_seed=seed, verbose=0, **p)
    m.fit(X_tr, y, verbose=False)
    return oof, m.predict(X_te)


def fit_xgb_oof(X_tr, y, X_te, best, seed, groups=None):
    p = dict(best)
    oof = np.zeros(len(y), dtype=float)
    for tri, vai in _grouped_splits(X_tr, groups, seed):
        if len(vai) < 5 or len(tri) < 20:
            continue
        m = xgb.XGBRegressor(verbosity=0, random_state=seed, **p)
        m.fit(X_tr[tri], y[tri])
        oof[vai] = m.predict(X_tr[vai])
    m = xgb.XGBRegressor(verbosity=0, random_state=seed, **p)
    m.fit(X_tr, y)
    return oof, m.predict(X_te)


def fit_adae_simple(
    X_tr, y_model, d_tr, X_te, nd, groups,
    dann_best, cab_best, seed, dom_weight, xgb_best=None,
):
    """Train DANN / XGB / CatBoost on the same y_model; blend on training OOF."""
    oof_d, y_d, _ = fit_dann_kfold_bundle(
        X_tr, y_model, d_tr, X_te, nd, n_folds=3, seed=seed,
        grl_scale=float(dann_best["grl_scale"]),
        dom_weight=dom_weight,
        hidden_dim=int(dann_best["hidden_dim"]),
        groups=groups, dann_epochs=50, mae_epochs=20,
    )
    oof_c, y_c = fit_cab_oof(X_tr, y_model, X_te, cab_best, seed, groups=groups)
    oofs = {"dann": oof_d, "cab": oof_c}
    preds = {"dann": y_d, "cab": y_c}
    extra = {
        "w_dann": 0.0, "w_xgb": 0.0, "w_cab": 0.0,
        "oof_r2_dann": float(r2_score(y_model, oof_d)),
        "oof_r2_xgb": float("nan"),
        "oof_r2_cab": float(r2_score(y_model, oof_c)),
    }
    if xgb_best is not None:
        oof_x, y_x = fit_xgb_oof(X_tr, y_model, X_te, xgb_best, seed, groups=groups)
        oofs["xgb"] = oof_x
        preds["xgb"] = y_x
        extra["oof_r2_xgb"] = float(r2_score(y_model, oof_x))
    w = blend_simplex(oofs, y_model, n_trials=80, seed=seed)
    raw = sum(w[k] * preds[k] for k in w)
    oof = sum(w[k] * oofs[k] for k in w)
    extra["w_dann"] = w.get("dann", 0.0)
    extra["w_xgb"] = w.get("xgb", 0.0)
    extra["w_cab"] = w.get("cab", 0.0)
    extra["oof_r2_blend"] = float(r2_score(y_model, oof))
    return raw, y_d, extra


def apply_stage1(raw, y_dann, pack):
    """Always residualize in y_fit space, then map back to concentration."""
    raw = np.asarray(raw, float) + np.asarray(pack["mu_te"], float)
    return _finalize(raw, pack["use_log"])


def official_dann_mu(protocol, target, fold, sta_te, use_log):
    """Station means of the official nested DANN, in y_fit space."""
    from src.rev1.paths import cell_dir

    z = np.load(cell_dir("rev1", protocol, "DANN", target, fold) / "y.npz", allow_pickle=True)
    p = np.asarray(z["y_pred"], float)
    sta = np.asarray(z["station"]).astype(str) if "station" in z.files else np.asarray(sta_te).astype(str)
    if use_log:
        p = np.log1p(np.maximum(p, 0))
    return p - _center_by_group(p, sta)
