"""
Permutation feature importance for the full ADAE model (no component split).
Fits ADAE once, then measures test-set R² drop when each feature is shuffled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from src.models.adae import (
    TabularMAE,
    STDANN,
    train_tabmae,
    train_dann,
    bayesian_blend,
    MOE_MIN,
    MOE_MAX,
)
from src.models.fair_multiseed_eval_scheme_b import SplitData, DANN_HIDDEN
from src.config import GRL_LAMBDA, DOM_WEIGHT


def _finalize(y_raw: np.ndarray, use_log: bool) -> np.ndarray:
    y = np.expm1(np.maximum(y_raw, 0)) if use_log else y_raw.copy()
    return np.maximum(y, 0)


def _tree_params(protocol: str, target: str) -> tuple[dict, dict, dict, dict]:
    if protocol == "station" and target == "CODMn":
        xp = dict(n_estimators=700, learning_rate=0.04, max_depth=5,
                  subsample=0.75, colsample_bytree=0.75, min_child_weight=5,
                  reg_alpha=0.3, reg_lambda=2.0, early_stopping_rounds=40)
        lp = dict(n_estimators=700, learning_rate=0.04, max_depth=6,
                  num_leaves=48, subsample=0.75, colsample_bytree=0.75,
                  min_child_samples=20, reg_alpha=0.3, reg_lambda=2.0,
                  early_stopping_rounds=40)
        cp = dict(iterations=600, learning_rate=0.04, depth=6,
                  l2_leaf_reg=5.0, subsample=0.75, early_stopping_rounds=40)
        et_kw = dict(n_estimators=500, max_depth=10, min_samples_leaf=10)
    elif protocol == "station":
        xp = dict(n_estimators=600, learning_rate=0.05, max_depth=4,
                  subsample=0.7, colsample_bytree=0.7, min_child_weight=10,
                  reg_alpha=0.5, reg_lambda=3.0, early_stopping_rounds=40)
        lp = dict(n_estimators=600, learning_rate=0.05, max_depth=5,
                  num_leaves=31, subsample=0.7, colsample_bytree=0.7,
                  min_child_samples=30, reg_alpha=0.5, reg_lambda=3.0,
                  early_stopping_rounds=40)
        cp = dict(iterations=500, learning_rate=0.05, depth=5,
                  l2_leaf_reg=5.0, subsample=0.7, early_stopping_rounds=40)
        et_kw = dict(n_estimators=400, max_depth=8, min_samples_leaf=15)
    else:
        xp = dict(n_estimators=600, learning_rate=0.04, max_depth=5,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                  reg_alpha=0.1, reg_lambda=1.5, early_stopping_rounds=40)
        lp = dict(n_estimators=600, learning_rate=0.04, max_depth=6,
                  num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                  min_child_samples=20, reg_alpha=0.1, reg_lambda=1.5,
                  early_stopping_rounds=40)
        cp = dict(iterations=600, learning_rate=0.04, depth=6,
                  l2_leaf_reg=3.0, subsample=0.8, early_stopping_rounds=40)
        et_kw = dict(n_estimators=500, max_depth=None, min_samples_leaf=8)
    return xp, lp, cp, et_kw


@dataclass
class _DannFold:
    model: STDANN
    x_scaler: StandardScaler
    y_scaler: StandardScaler


@dataclass
class ADAEPredictor:
    weights: tuple[float, float, float]
    use_log: bool
    dann_folds: list[_DannFold]
    moe_models: list[cb.CatBoostRegressor]
    stack_tree_models: list[tuple]
    stack_scm: StandardScaler
    stack_ridge: Ridge
    stack_enet: ElasticNet
    n_stack_folds: int

    def _predict_dann(self, X: np.ndarray) -> np.ndarray:
        te = np.zeros(len(X))
        for fold in self.dann_folds:
            Xs = fold.x_scaler.transform(X)
            with torch.no_grad():
                raw = fold.model(torch.tensor(Xs, dtype=torch.float32))[0].numpy()
            te += fold.y_scaler.inverse_transform(raw.reshape(-1, 1)).flatten() / len(self.dann_folds)
        return te

    def _predict_moe(self, X: np.ndarray) -> np.ndarray:
        te = np.zeros(len(X))
        for m in self.moe_models:
            te += m.predict(X) / len(self.moe_models)
        return te

    def _predict_stack(self, X: np.ndarray) -> np.ndarray:
        meta = np.zeros((len(X), 4))
        for mx, ml, mc, me in self.stack_tree_models:
            meta[:, 0] += mx.predict(X) / self.n_stack_folds
            meta[:, 1] += ml.predict(X) / self.n_stack_folds
            meta[:, 2] += mc.predict(X) / self.n_stack_folds
            meta[:, 3] += me.predict(X) / self.n_stack_folds
        Xm = self.stack_scm.transform(meta)
        return 0.5 * self.stack_ridge.predict(Xm) + 0.5 * self.stack_enet.predict(Xm)

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        w1, w2, w3 = self.weights
        return w1 * self._predict_dann(X) + w2 * self._predict_moe(X) + w3 * self._predict_stack(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return _finalize(self.predict_raw(X), self.use_log)


def fit_adae_predictor(sd: SplitData, seed: int = 42) -> ADAEPredictor:
    """Fit ADAE once and return an inference-only predictor."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    grl = GRL_LAMBDA[sd.protocol]
    dom = DOM_WEIGHT[sd.protocol]
    n_folds = 3
    n_f = sd.X_tr.shape[1]

    # ── DANN k-fold (OOF + stored models) ───────────────────────────────────
    oof_dann = np.zeros(len(sd.X_tr))
    dann_folds: list[_DannFold] = []
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (tri, vai) in enumerate(kf.split(sd.X_tr)):
        Xtr, Xva = sd.X_tr[tri], sd.X_tr[vai]
        ytr, yva = sd.y_fit[tri], sd.y_fit[vai]
        dtr = sd.d_tr[tri]
        xsc = StandardScaler()
        Xtr_s = xsc.fit_transform(Xtr)
        Xva_s = xsc.transform(Xva)
        ysc = StandardScaler()
        ytr_s = ysc.fit_transform(ytr.reshape(-1, 1)).flatten()
        yva_s = ysc.transform(yva.reshape(-1, 1)).flatten()
        mae_m = TabularMAE(n_f, hidden_dim=DANN_HIDDEN)
        train_tabmae(mae_m, np.vstack([Xtr_s, Xva_s]), epochs=20)
        model = STDANN(n_f, mae_m.encoder, sd.nd, hidden_dim=DANN_HIDDEN)
        bw = train_dann(model, Xtr_s, ytr_s, dtr, Xva_s, yva_s,
                        grl_scale=grl, dom_weight=dom)
        model.load_state_dict(bw)
        model.eval()
        with torch.no_grad():
            vp = ysc.inverse_transform(
                model(torch.tensor(Xva_s, dtype=torch.float32))[0]
                .numpy().reshape(-1, 1)).flatten()
        oof_dann[vai] = vp
        dann_folds.append(_DannFold(model, xsc, ysc))

    # ── MoE Cat-5 (OOF via inner k-fold + 5 test models) ────────────────────
    cat_params = dict(iterations=500, learning_rate=0.05, depth=6,
                      l2_leaf_reg=3.0, subsample=0.8, colsample_bylevel=0.8,
                      early_stopping_rounds=30, verbose=0)
    oof_moe = np.zeros(len(sd.X_tr))
    nv = max(30, int(len(sd.X_tr) * 0.12))
    kf_moe = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (tri, vai) in enumerate(kf_moe.split(sd.X_tr)):
        Xtr, ytr = sd.X_tr[tri], sd.y_fit[tri]
        fold_va = np.zeros(len(vai))
        for k in range(5):
            cat = cb.CatBoostRegressor(**cat_params, random_seed=seed * 10 + k + fold * 100)
            cat.fit(Xtr[:-nv], ytr[:-nv], eval_set=(Xtr[-nv:], ytr[-nv:]), verbose=False)
            fold_va += cat.predict(sd.X_tr[vai]) / 5
        oof_moe[vai] = fold_va
    moe_models: list[cb.CatBoostRegressor] = []
    for k in range(5):
        cat = cb.CatBoostRegressor(**cat_params, random_seed=seed * 10 + k)
        cat.fit(sd.X_tr[:-nv], sd.y_fit[:-nv],
                eval_set=(sd.X_tr[-nv:], sd.y_fit[-nv:]), verbose=False)
        moe_models.append(cat)

    # ── Stack (OOF meta + stored fold models) ───────────────────────────────
    xp, lp, cp, et_kw = _tree_params(sd.protocol, sd.target)
    stack_tree_models: list[tuple] = []
    oof_trees = {k: np.zeros(len(sd.X_tr)) for k in ["xgb", "lgbm", "cat", "et"]}
    kf2 = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (tri, vai) in enumerate(kf2.split(sd.X_tr)):
        Xtr, Xva = sd.X_tr[tri], sd.X_tr[vai]
        ytr, yva = sd.y_fit[tri], sd.y_fit[vai]
        mx = xgb.XGBRegressor(**xp, verbosity=0, random_state=seed + fold)
        mx.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        oof_trees["xgb"][vai] = mx.predict(Xva)
        ml = lgb.LGBMRegressor(**lp, random_state=seed + fold, verbose=-1)
        ml.fit(Xtr, ytr, eval_set=[(Xva, yva)],
               callbacks=[lgb.early_stopping(lp["early_stopping_rounds"], verbose=False),
                          lgb.log_evaluation(-1)])
        oof_trees["lgbm"][vai] = ml.predict(Xva)
        mc = cb.CatBoostRegressor(**cp, random_seed=seed + fold, verbose=0)
        mc.fit(Xtr, ytr, eval_set=(Xva, yva), verbose=False)
        oof_trees["cat"][vai] = mc.predict(Xva)
        me = ExtraTreesRegressor(**et_kw, random_state=seed + fold, n_jobs=-1)
        me.fit(Xtr, ytr)
        oof_trees["et"][vai] = me.predict(Xva)
        stack_tree_models.append((mx, ml, mc, me))

    meta_tr = np.column_stack([oof_trees[k] for k in ["xgb", "lgbm", "cat", "et"]])
    stack_scm = StandardScaler()
    Xm = stack_scm.fit_transform(meta_tr)
    stack_ridge = Ridge(alpha=1.0).fit(Xm, sd.y_fit)
    stack_enet = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=2000).fit(Xm, sd.y_fit)
    oof_stack = 0.5 * stack_ridge.predict(Xm) + 0.5 * stack_enet.predict(Xm)

    w1, w2, w3 = bayesian_blend(
        oof_dann, oof_moe, oof_stack, sd.y_fit,
        moe_min=MOE_MIN[sd.protocol], moe_max=MOE_MAX[sd.protocol],
    )

    return ADAEPredictor(
        weights=(w1, w2, w3),
        use_log=sd.use_log,
        dann_folds=dann_folds,
        moe_models=moe_models,
        stack_tree_models=stack_tree_models,
        stack_scm=stack_scm,
        stack_ridge=stack_ridge,
        stack_enet=stack_enet,
        n_stack_folds=n_folds,
    )


def permutation_importance(
    predictor: ADAEPredictor,
    X_te: np.ndarray,
    y_te: np.ndarray,
    n_repeats: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return per-feature mean R² drop (importance), std, baseline R²."""
    rng = np.random.default_rng(seed)
    y_base = predictor.predict(X_te)
    r2_base = float(r2_score(y_te, y_base))
    n_feat = X_te.shape[1]
    imp = np.zeros(n_feat)
    std = np.zeros(n_feat)

    for j in range(n_feat):
        drops = []
        for _ in range(n_repeats):
            Xp = X_te.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops.append(r2_base - r2_score(y_te, predictor.predict(Xp)))
        imp[j] = float(np.mean(drops))
        std[j] = float(np.std(drops, ddof=1)) if n_repeats > 1 else 0.0

    return np.maximum(imp, 0.0), std, r2_base


def pathway_permutation_importance(
    predictor: ADAEPredictor,
    X_te: np.ndarray,
    y_te: np.ndarray,
    pathway_cols: dict[str, list[int]],
    n_repeats: int = 5,
    seed: int = 42,
) -> dict[str, dict]:
    """
    Grouped permutation: shuffle all features in a pathway with the same index.
    Returns {pathway: {importance, importance_std}} as ΔR².
    """
    rng = np.random.default_rng(seed)
    y_base = predictor.predict(X_te)
    r2_base = float(r2_score(y_te, y_base))
    out: dict[str, dict] = {}
    for name, cols in pathway_cols.items():
        if not cols:
            continue
        drops = []
        for _ in range(n_repeats):
            Xp = X_te.copy()
            perm = rng.permutation(len(X_te))
            for j in cols:
                Xp[:, j] = X_te[perm, j]
            drops.append(r2_base - r2_score(y_te, predictor.predict(Xp)))
        out[name] = {
            "importance": float(max(np.mean(drops), 0.0)),
            "importance_std": float(np.std(drops, ddof=1)) if n_repeats > 1 else 0.0,
            "n_features": len(cols),
        }
    return out
