"""Nested HPO + final fit for RF / XGB / CaB / DANN / ADAE."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from src.config import DOM_WEIGHT, GRL_LAMBDA
from src.models.adae import (
    STDANN,
    TabularMAE,
    apply_within_ridge_addon,
    bayesian_blend,
    fit_cat5_ensemble,
    fit_dann_kfold,
    fit_tree_oof,
    refit_station_mean_from_df,
    train_dann,
    train_tabmae,
)
from src.models.adae import MOE_MAX, MOE_MIN
from src.models.fair_multiseed_eval_scheme_b import SplitData, predict_adae
from src.rev1.bundle import CachedPredictor, DannEnsemblePredictor, SklearnPredictor
from src.rev1.folds import inner_splits
from src.rev1.paths import N_TUNE_TRIALS, SEED
from src.utils.grouped_splits import iter_group_folds

try:
    import xgboost as xgb
    import catboost as cb
except ImportError:  # pragma: no cover
    xgb = cb = None


def _finalize(y_raw, use_log: bool) -> np.ndarray:
    y = np.expm1(np.maximum(y_raw, 0)) if use_log else np.asarray(y_raw, dtype=float).copy()
    return np.maximum(y, 0)


def _pick(rng, space: dict) -> dict:
    out = {}
    for k, vals in space.items():
        vals = list(vals)
        out[k] = vals[int(rng.integers(0, len(vals)))]
    return out


def _score_inner(fit_fn, X, y, splits) -> float:
    scores = []
    for tr, va in splits:
        if len(va) < 5 or len(tr) < 20:
            continue
        pred = fit_fn(X[tr], y[tr], X[va])
        scores.append(float(r2_score(y[va], pred)))
    return float(np.mean(scores)) if scores else -1e9


def tune_and_fit_rf(X_tr, y_fit, X_te, splits, seed: int, n_jobs: int, n_trials: int):
    rng = np.random.default_rng(seed)
    space = {
        "n_estimators": [200, 400, 600, 800],
        "min_samples_leaf": [2, 5, 8, 12],
        "max_depth": [8, 12, 16, None],
        "max_features": ["sqrt", 0.3, 0.5, 0.8],
    }
    rows, best_r2, best = [], -1e9, {"n_estimators": 500, "min_samples_leaf": 5}
    for i in range(n_trials):
        p = _pick(rng, space)

        def _fit(a, b, c, p=p):
            m = RandomForestRegressor(n_jobs=n_jobs, random_state=seed, **p)
            m.fit(a, b)
            return m.predict(c)

        r = _score_inner(_fit, X_tr, y_fit, splits)
        rows.append({"trial": i, "inner_r2": r, **{k: str(v) for k, v in p.items()}})
        if r > best_r2:
            best_r2, best = r, p
    m = RandomForestRegressor(n_jobs=n_jobs, random_state=seed, **best)
    m.fit(X_tr, y_fit)
    return m.predict(X_te), m, best, best_r2, pd.DataFrame(rows)


def tune_and_fit_xgb(X_tr, y_fit, X_te, splits, seed: int, n_trials: int):
    rng = np.random.default_rng(seed)
    space = {
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.03, 0.05, 0.08, 0.10],
        "n_estimators": [400, 600, 800, 1000],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5, 8],
        "reg_lambda": [1.0, 3.0, 5.0],
    }
    rows, best_r2, best, best_n = [], -1e9, None, 800
    hold = splits[0]
    for i in range(n_trials):
        p = _pick(rng, space)
        scores, ns = [], []
        for tr, va in splits:
            m = xgb.XGBRegressor(early_stopping_rounds=30, verbosity=0, random_state=seed, **p)
            m.fit(X_tr[tr], y_fit[tr], eval_set=[(X_tr[va], y_fit[va])], verbose=False)
            scores.append(float(r2_score(y_fit[va], m.predict(X_tr[va]))))
            ns.append(int(getattr(m, "best_iteration", p["n_estimators"]) or p["n_estimators"]) + 1)
        r = float(np.mean(scores))
        rows.append({"trial": i, "inner_r2": r, **{k: str(v) for k, v in p.items()}})
        if r > best_r2:
            best_r2, best, best_n = r, p, int(np.median(ns))
    p = dict(best)
    p["n_estimators"] = max(100, best_n)
    # early-stop on first inner-style holdout carved from full train, then refit length
    tr_i, va_i = hold
    m_es = xgb.XGBRegressor(early_stopping_rounds=30, verbosity=0, random_state=seed, **best)
    m_es.fit(X_tr[tr_i], y_fit[tr_i], eval_set=[(X_tr[va_i], y_fit[va_i])], verbose=False)
    p["n_estimators"] = max(100, int(getattr(m_es, "best_iteration", p["n_estimators"]) or p["n_estimators"]) + 1)
    m = xgb.XGBRegressor(verbosity=0, random_state=seed, **p)
    m.fit(X_tr, y_fit)
    return m.predict(X_te), m, p, best_r2, pd.DataFrame(rows)


def tune_and_fit_cab(X_tr, y_fit, X_te, splits, seed: int, n_trials: int):
    rng = np.random.default_rng(seed)
    space = {
        "depth": [4, 5, 6, 8],
        "learning_rate": [0.03, 0.05, 0.08],
        "iterations": [400, 600, 800],
        "l2_leaf_reg": [1.0, 3.0, 5.0, 8.0],
        "subsample": [0.7, 0.8, 1.0],
    }
    rows, best_r2, best, best_n = [], -1e9, None, 600
    hold = splits[0]
    for i in range(n_trials):
        p = _pick(rng, space)
        scores, ns = [], []
        for tr, va in splits:
            m = cb.CatBoostRegressor(colsample_bylevel=0.8, early_stopping_rounds=30,
                                     random_seed=seed, verbose=0, **p)
            m.fit(X_tr[tr], y_fit[tr], eval_set=(X_tr[va], y_fit[va]), verbose=False)
            scores.append(float(r2_score(y_fit[va], m.predict(X_tr[va]))))
            ns.append(int(m.get_best_iteration() or p["iterations"]) + 1)
        r = float(np.mean(scores))
        rows.append({"trial": i, "inner_r2": r, **{k: str(v) for k, v in p.items()}})
        if r > best_r2:
            best_r2, best, best_n = r, p, int(np.median(ns))
    p = dict(best)
    p["iterations"] = max(100, best_n)
    tr_i, va_i = hold
    m_es = cb.CatBoostRegressor(colsample_bylevel=0.8, early_stopping_rounds=30,
                                random_seed=seed, verbose=0, **best)
    m_es.fit(X_tr[tr_i], y_fit[tr_i], eval_set=(X_tr[va_i], y_fit[va_i]), verbose=False)
    p["iterations"] = max(100, int(m_es.get_best_iteration() or p["iterations"]) + 1)
    m = cb.CatBoostRegressor(colsample_bylevel=0.8, random_seed=seed, verbose=0, **p)
    m.fit(X_tr, y_fit, verbose=False)
    return m.predict(X_te), m, p, best_r2, pd.DataFrame(rows)


def _fit_one_dann(Xtr, ytr, dtr, Xva, yva, Xte, hidden, grl, domw, nd, epochs, mae_epochs, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    sc, ysc = StandardScaler(), StandardScaler()
    Xtr_s, Xva_s, Xte_s = sc.fit_transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    ytr_s = ysc.fit_transform(ytr.reshape(-1, 1)).ravel()
    yva_s = ysc.transform(yva.reshape(-1, 1)).ravel()
    mae = TabularMAE(Xtr.shape[1], hidden_dim=hidden)
    train_tabmae(mae, np.vstack([Xtr_s, Xva_s]), epochs=mae_epochs)
    model = STDANN(Xtr.shape[1], mae.encoder, nd, hidden_dim=hidden)
    bw = train_dann(model, Xtr_s, ytr_s, dtr, Xva_s, yva_s,
                    epochs=epochs, grl_scale=grl, dom_weight=domw)
    model.load_state_dict(bw)
    model.eval()
    with torch.no_grad():
        te = ysc.inverse_transform(
            model(torch.tensor(Xte_s, dtype=torch.float32))[0].numpy().reshape(-1, 1)
        ).ravel()
        va = ysc.inverse_transform(
            model(torch.tensor(Xva_s, dtype=torch.float32))[0].numpy().reshape(-1, 1)
        ).ravel()
    return va, te, {"sc": sc, "ysc": ysc, "state": deepcopy(model.state_dict())}


def fit_dann_kfold_bundle(X_train, y_train, d_train, X_test, num_domains,
                          n_folds=3, seed=42, grl_scale=1.0, dom_weight=0.3,
                          hidden_dim=96, groups=None, dann_epochs=50, mae_epochs=20):
    torch.manual_seed(seed)
    np.random.seed(seed)
    oof = np.zeros(len(X_train))
    te = np.zeros(len(X_test))
    members = []
    fold_iter = (
        iter_group_folds(groups, n_folds, seed) if groups is not None
        else __import__("sklearn.model_selection", fromlist=["KFold"]).KFold(
            n_splits=n_folds, shuffle=True, random_state=seed
        ).split(X_train)
    )
    n_done = 0
    for tri, vai in fold_iter:
        va, tp, mem = _fit_one_dann(
            X_train[tri], y_train[tri], d_train[tri],
            X_train[vai], y_train[vai], X_test,
            hidden_dim, grl_scale, dom_weight, num_domains,
            dann_epochs, mae_epochs, seed + n_done,
        )
        oof[vai] = va
        te += tp
        members.append(mem)
        n_done += 1
    te /= max(n_done, 1)
    return oof, te, members


def tune_and_fit_dann(X_tr, y_fit, d_tr, X_te, nd, protocol, splits, seed: int, n_trials: int):
    default = {"grl_scale": GRL_LAMBDA[protocol], "hidden_dim": 96}
    grid = [default]
    for g in (
        {"grl_scale": 0.20, "hidden_dim": 96},
        {"grl_scale": 0.40, "hidden_dim": 96},
        {"grl_scale": 0.60, "hidden_dim": 96},
        {"grl_scale": 0.40, "hidden_dim": 128},
        {"grl_scale": 0.05, "hidden_dim": 96},
        {"grl_scale": 0.20, "hidden_dim": 128},
        {"grl_scale": 0.80, "hidden_dim": 96},
        {"grl_scale": 0.40, "hidden_dim": 64},
        {"grl_scale": 0.10, "hidden_dim": 96},
        {"grl_scale": 0.30, "hidden_dim": 96},
        {"grl_scale": 0.50, "hidden_dim": 128},
        {"grl_scale": 0.60, "hidden_dim": 128},
        {"grl_scale": 0.15, "hidden_dim": 64},
        {"grl_scale": 0.25, "hidden_dim": 96},
    ):
        if (g["grl_scale"], g["hidden_dim"]) not in {(c["grl_scale"], c["hidden_dim"]) for c in grid}:
            grid.append(g)
    grid = grid[:n_trials]
    # score on the first protocol-correct inner split (15 full inner-CV DANNs is too heavy)
    tr_i, va_i = splits[0]
    d_inner = d_tr[tr_i]
    nd_inner = max(int(d_inner.max()) + 1, 2)
    rows, best_r2, best = [], -1e9, default
    for i, g in enumerate(grid):
        _, yva_hat = fit_dann_kfold(
            X_tr[tr_i], y_fit[tr_i], d_inner, X_tr[va_i], nd_inner,
            n_folds=2, seed=seed, grl_scale=g["grl_scale"],
            dom_weight=DOM_WEIGHT[protocol], hidden_dim=g["hidden_dim"],
            groups=None, dann_epochs=30, mae_epochs=15,
        )
        r = float(r2_score(y_fit[va_i], yva_hat))
        rows.append({"trial": i, "inner_r2": r, **g})
        if r > best_r2:
            best_r2, best = r, g
    _, y_raw, members = fit_dann_kfold_bundle(
        X_tr, y_fit, d_tr, X_te, nd, n_folds=3, seed=seed,
        grl_scale=best["grl_scale"], dom_weight=DOM_WEIGHT[protocol],
        hidden_dim=best["hidden_dim"], groups=None,
        dann_epochs=50, mae_epochs=20,
    )
    return y_raw, members, best, best_r2, pd.DataFrame(rows)


def fit_adae_cell(sd: SplitData, seed: int, lite: bool, df, test_stations, feat_names):
    extra = {}
    if sd.protocol == "station":
        yp, w = predict_adae(sd, seed, lite=lite)
        weights = w if isinstance(w, dict) else {"dann": w[0], "moe": w[1], "stack": w[2]}
        if sd.target == "NH3N":
            yp, extra = refit_station_mean_from_df(yp, sd.sta_te, df, test_stations, sd.target)
            weights = {**weights, **extra}
        if sd.target == "TP":
            yp, extra = apply_within_ridge_addon(
                yp, sd.sta_te, df, test_stations, sd.target, list(feat_names),
                blend=0.3, te_idx=sd.te_idx,
            )
            weights = {**weights, **extra}
        return np.asarray(yp, dtype=float), weights, extra
    # R/T: Optuna blend on train OOF (same as predict_adae_optuna)
    torch.manual_seed(seed)
    np.random.seed(seed)
    grl, dom = GRL_LAMBDA[sd.protocol], DOM_WEIGHT[sd.protocol]
    y_stack, _, meta_tr, _ = fit_tree_oof(
        sd.X_tr, sd.y_fit, sd.X_te, seed=seed, split=sd.protocol, target=sd.target,
    )
    from sklearn.linear_model import ElasticNet, Ridge
    scm = StandardScaler()
    Xm = scm.fit_transform(meta_tr)
    oof_stack = 0.5 * Ridge(alpha=1.0).fit(Xm, sd.y_fit).predict(Xm)
    oof_stack = oof_stack + 0.5 * ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=2000).fit(Xm, sd.y_fit).predict(Xm)
    oof_moe, y_moe = fit_cat5_ensemble(sd.X_tr, sd.y_fit, sd.X_te, seed=seed, split=sd.protocol)
    oof_dann, y_dann = fit_dann_kfold(
        sd.X_tr, sd.y_fit, sd.d_tr, sd.X_te, sd.nd,
        n_folds=3, seed=seed, grl_scale=grl, dom_weight=dom,
    )
    w1, w2, w3 = bayesian_blend(
        oof_dann, oof_moe, oof_stack, sd.y_fit,
        moe_min=MOE_MIN[sd.protocol], moe_max=MOE_MAX[sd.protocol],
    )
    y_pred = w1 * y_dann + w2 * y_moe + w3 * y_stack
    yp = _finalize(y_pred, sd.use_log)
    weights = {"dann": float(w1), "moe": float(w2), "stack": float(w3), "source": "optuna_oof"}
    return yp, weights, extra


def run_method(method, sd: SplitData, protocol, seed, n_jobs, n_trials, lite, df, test_stations):
    splits = inner_splits(protocol, sd.groups_tr, getattr(sd, "years_tr", np.zeros(len(sd.X_tr))), 3, seed)
    use_log = sd.use_log
    if method == "RF":
        raw, est, best, inner, trials = tune_and_fit_rf(
            sd.X_tr, sd.y_fit, sd.X_te, splits, seed, n_jobs, n_trials,
        )
        yp = _finalize(raw, use_log)
        return yp, SklearnPredictor(est, use_log), best, inner, trials, {"kind": "sklearn"}
    if method == "XGB":
        raw, est, best, inner, trials = tune_and_fit_xgb(
            sd.X_tr, sd.y_fit, sd.X_te, splits, seed, n_trials,
        )
        yp = _finalize(raw, use_log)
        return yp, SklearnPredictor(est, use_log), best, inner, trials, {"kind": "sklearn"}
    if method == "CaB":
        raw, est, best, inner, trials = tune_and_fit_cab(
            sd.X_tr, sd.y_fit, sd.X_te, splits, seed, n_trials,
        )
        yp = _finalize(raw, use_log)
        return yp, SklearnPredictor(est, use_log), best, inner, trials, {"kind": "sklearn"}
    if method == "DANN":
        raw, members, best, inner, trials = tune_and_fit_dann(
            sd.X_tr, sd.y_fit, sd.d_tr, sd.X_te, sd.nd, protocol, splits, seed, n_trials,
        )
        yp = _finalize(raw, use_log)
        pred = DannEnsemblePredictor(members, best["hidden_dim"], sd.nd, use_log)
        return yp, pred, best, inner, trials, {"kind": "dann_kfold", "dom_weight": DOM_WEIGHT[protocol]}
    if method == "ADAE":
        yp, weights, extra = fit_adae_cell(sd, seed, lite, df, test_stations, sd.feat_names)
        pred = CachedPredictor(sd.X_te, yp, "ADAE")
        trials = pd.DataFrame([{"trial": 0, "inner_r2": weights.get("oof_r2_dann"), **weights}])
        inner = weights.get("oof_r2_dann")
        return yp, pred, weights, inner, trials, {"kind": "adae_cached", **extra, **weights}
    raise ValueError(method)
