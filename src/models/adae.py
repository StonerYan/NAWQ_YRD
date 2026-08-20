"""
ADAE Model — WQI0616 v12
=========================
Adaptive Domain-Adversarial Ensemble (ADAE) for water quality retrieval.

Architecture (v12):
  TabularMAE (128-D)  →  spectral representation pre-training
  ST-DANN (GRL)       →  domain-invariant feature alignment
  Tree Stack          →  XGB + LGBM + CaB + ET → Ridge/ElasticNet meta
  Cat-5 ensemble      →  5-seed CatBoost OWT-MoE component
  Bayesian blend      →  optimised DANN / MoE / Stack weights

Split-specific behaviour:
  random    :  standard ensemble + Bayesian blend
  temporal  :  same as random (year-based domains)
  station   :  hierarchical ADAE (spectral station-mean + residual DANN/trees)
"""

import sys, copy, warnings
from src.utils.eval_verbose import vprint
import numpy as np
import pandas as pd
import torch
import optuna
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge, ElasticNet, LinearRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from src.utils.grouped_splits import iter_group_folds, is_geo_feature, grouped_holdout_indices

import lightgbm as lgb
import xgboost as xgb
import catboost as cb

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
torch.set_num_threads(4)

# ── Hyper-parameters ──────────────────────────────────────────────────────────
GRL_LAMBDA   = {'station': 0.40, 'temporal': 0.20, 'random': 0.05}
DOM_WEIGHT   = {'station': 0.35, 'temporal': 0.25, 'random': 0.15}
GRL_TP_STA   = 0.40   # Match standalone DANN baseline config (was 0.55)
N_OWT        = 5
OWT_M        = 2.0
TP_DANN_SEEDS = [42, 123, 456, 789, 2024, 1337, 9999]  # 7 seeds (was 5)

MOE_MIN = {'random': 0.25, 'temporal': 0.30, 'station': 0.00}
MOE_MAX = {'random': 0.60, 'temporal': 0.65, 'station': 0.20}

STATION_BLEND = {
    'CODMn': {'dann': 0.00, 'moe': 0.40, 'stack': 0.60},
    'NH3N':  {'dann': 1.00, 'moe': 0.00, 'stack': 0.00},
    'TP':    {'dann': 0.80, 'moe': 0.20, 'stack': 0.00},
}


# ══════════════════════════════════════════════════════════════════════════════
# Neural network components
# ══════════════════════════════════════════════════════════════════════════════

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g.neg() * ctx.alpha, None


class TabularMAE(torch.nn.Module):
    """Tabular Masked Autoencoder for unsupervised spectral pre-training."""
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256), torch.nn.LayerNorm(256),
            torch.nn.GELU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(256, hidden_dim), torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU())
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 256), torch.nn.GELU(),
            torch.nn.Dropout(0.1), torch.nn.Linear(256, input_dim))

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


class STDANN(torch.nn.Module):
    """Split-Adaptive Spatiotemporal DANN."""
    def __init__(self, input_dim, encoder, num_domains, hidden_dim=128):
        super().__init__()
        self.encoder = encoder
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 96), torch.nn.LayerNorm(96),
            torch.nn.GELU(), torch.nn.Dropout(0.15),
            torch.nn.Linear(96, 48), torch.nn.GELU(),
            torch.nn.Linear(48, 1))
        self.domain_classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 64), torch.nn.GELU(),
            torch.nn.Dropout(0.2), torch.nn.Linear(64, num_domains))

    def forward(self, x, alpha=1.0):
        z = self.encoder(x)
        return (self.regressor(z).squeeze(-1),
                self.domain_classifier(GradientReversal.apply(z, alpha)))


# ── Training helpers ──────────────────────────────────────────────────────────

def train_tabmae(model, X_all, epochs=30, bs=256, lr=1e-3, mask_ratio=0.30):
    """Pre-train TabularMAE via masked feature reconstruction."""
    X_t = torch.tensor(X_all, dtype=torch.float32)
    ld  = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t), batch_size=bs, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    crit = torch.nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for (xb,) in ld:
            mask = (torch.rand_like(xb) < mask_ratio).float()
            opt.zero_grad()
            crit(model(xb * (1 - mask))[0], xb).backward()
            opt.step()
        sch.step()


def train_dann(model, X_tr, y_tr, d_tr, X_val, y_val,
               epochs=50, bs=256, lr=1e-3, gamma=10.0,
               dom_weight=0.3, grl_scale=1.0, y_loss="smoothl1",
               X_unlab=None, unlab_domain=None):
    Xt = torch.tensor(X_tr,  dtype=torch.float32)
    yt = torch.tensor(y_tr,  dtype=torch.float32)
    dt = torch.tensor(d_tr,  dtype=torch.long)
    Xv = torch.tensor(X_val, dtype=torch.float32)
    yv = torch.tensor(y_val, dtype=torch.float32)
    Xu = None
    if X_unlab is not None and len(X_unlab) and unlab_domain is not None:
        Xu = torch.tensor(X_unlab, dtype=torch.float32)
        du = torch.full((len(Xu),), int(unlab_domain), dtype=torch.long)
    ld = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xt, yt, dt), batch_size=bs, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    ly  = torch.nn.MSELoss() if y_loss == "mse" else torch.nn.SmoothL1Loss()
    ldl = torch.nn.CrossEntropyLoss()
    best_val = float('inf')
    best_w   = copy.deepcopy(model.state_dict())
    n_steps  = len(ld) * epochs
    step     = 0
    for _ in range(epochs):
        model.train()
        for xb, yb, db in ld:
            p     = step / n_steps
            alpha = grl_scale * (2.0 / (1.0 + np.exp(-gamma * p)) - 1.0)
            opt.zero_grad()
            yp, dp = model(xb, alpha=alpha)
            loss = ly(yp, yb) + dom_weight * ldl(dp, db)
            if Xu is not None:
                n_u = min(int(xb.size(0)), int(Xu.size(0)))
                ui = torch.randint(0, Xu.size(0), (n_u,))
                _, dp_u = model(Xu[ui], alpha=alpha)
                loss = loss + dom_weight * ldl(dp_u, du[ui])
            loss.backward()
            opt.step()
            step += 1
        sch.step()
        model.eval()
        with torch.no_grad():
            vl = float(ly(model(Xv)[0], yv).item())
        if vl < best_val:
            best_val = vl
            best_w   = copy.deepcopy(model.state_dict())
    return best_w


# ── K-fold DANN ───────────────────────────────────────────────────────────────

def fit_dann_kfold(X_train, y_train, d_train, X_test, num_domains,
                   n_folds=3, seed=42, grl_scale=1.0, dom_weight=0.3,
                   hidden_dim=128, groups=None, dann_epochs=50, mae_epochs=None,
                   test_oversample=1, y_loss="smoothl1", align_test=False):
    """K-fold DANN training. hidden_dim=96 matches baseline DANN config."""
    torch.manual_seed(seed); np.random.seed(seed)
    n_f  = X_train.shape[1]
    oof  = np.zeros(len(X_train))
    te   = np.zeros(len(X_test))
    tabmae_epochs = mae_epochs if mae_epochs is not None else (20 if hidden_dim == 96 else 30)
    fold_iter = (
        iter_group_folds(groups, n_folds, seed) if groups is not None
        else KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X_train)
    )
    n_src = int(num_domains)
    n_dom = n_src + 1 if align_test else n_src
    n_done = 0
    for fold, (tri, vai) in enumerate(fold_iter):
        Xtr, Xva  = X_train[tri], X_train[vai]
        ytr, yva  = y_train[tri], y_train[vai]
        dtr       = d_train[tri]
        sc        = StandardScaler()
        Xtr_s     = sc.fit_transform(Xtr)
        Xva_s     = sc.transform(Xva)
        Xte_s     = sc.transform(X_test)
        ysc       = StandardScaler()
        ytr_s     = ysc.fit_transform(ytr.reshape(-1, 1)).flatten()
        yva_s     = ysc.transform(yva.reshape(-1, 1)).flatten()
        mae_m     = TabularMAE(n_f, hidden_dim=hidden_dim)
        X_mae = np.vstack([Xtr_s, Xva_s, Xte_s])
        if test_oversample and int(test_oversample) > 1:
            X_mae = np.vstack([X_mae, np.repeat(Xte_s, int(test_oversample) - 1, axis=0)])
        train_tabmae(mae_m, X_mae, epochs=tabmae_epochs)
        model     = STDANN(n_f, mae_m.encoder, n_dom, hidden_dim=hidden_dim)
        bw        = train_dann(
            model, Xtr_s, ytr_s, dtr, Xva_s, yva_s,
            epochs=dann_epochs,
            grl_scale=grl_scale, dom_weight=dom_weight,
            y_loss=y_loss,
            X_unlab=Xte_s if align_test else None,
            unlab_domain=n_src if align_test else None,
        )
        model.load_state_dict(bw); model.eval()
        with torch.no_grad():
            vp = ysc.inverse_transform(
                model(torch.tensor(Xva_s, dtype=torch.float32))[0]
                     .numpy().reshape(-1, 1)).flatten()
            tp = ysc.inverse_transform(
                model(torch.tensor(Xte_s, dtype=torch.float32))[0]
                     .numpy().reshape(-1, 1)).flatten()
        oof[vai] = vp
        te      += tp
        n_done  += 1
    te /= max(n_done, 1)
    return oof, te


def fit_dann_multiseed(X_train, y_train, d_train, X_test, num_domains,
                       seeds=TP_DANN_SEEDS, n_folds=3,
                       grl_scale=1.0, dom_weight=0.3, hidden_dim=128,
                       groups=None, dann_epochs=50, mae_epochs=None,
                       test_oversample=1, y_loss="smoothl1", align_test=False):
    """Multi-seed DANN ensemble for TP station-out (reduces variance)."""
    vprint(f"      Multi-seed DANN: {len(seeds)} seeds × {n_folds} folds (hidden={hidden_dim})…")
    te_all, oof_all = [], []
    for si, seed in enumerate(seeds):
        oof, te = fit_dann_kfold(X_train, y_train, d_train, X_test, num_domains,
                                 n_folds=n_folds, seed=seed,
                                 grl_scale=grl_scale, dom_weight=dom_weight,
                                 hidden_dim=hidden_dim, groups=groups,
                                 dann_epochs=dann_epochs, mae_epochs=mae_epochs,
                                 test_oversample=test_oversample, y_loss=y_loss,
                                 align_test=align_test)
        te_all.append(te)
        oof_all.append(oof)
        vprint(f"        seed {seed}: done")
    return np.mean(oof_all, axis=0), np.mean(te_all, axis=0)


# ── Multi-seed XGB ensemble ───────────────────────────────────────────────────

def fit_xgb_multiseed(X_train, y_train, X_test,
                      seeds=[42, 123, 456, 789, 2024], n_folds=3):
    """
    Multi-seed XGB ensemble for CODMn station-out.
    Uses the same holdout approach as the standalone XGB baseline
    (train on first 85%, validate on last 15% for early stopping),
    then averages across 5 seeds to reduce variance.
    This ensures we use the full training data (like baseline) while
    gaining stability from seed averaging.
    """
    xp = dict(n_estimators=800, learning_rate=0.05, max_depth=5,
              subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
              early_stopping_rounds=30, verbosity=0)
    nv = max(30, int(len(X_train) * 0.15))  # 15% holdout for early stopping
    te_all = []
    vprint(f"      XGB multiseed: {len(seeds)} seeds × holdout…")
    for seed in seeds:
        mx = xgb.XGBRegressor(**xp, random_state=seed)
        mx.fit(X_train[:-nv], y_train[:-nv],
               eval_set=[(X_train[-nv:], y_train[-nv:])],
               verbose=False)
        te_all.append(mx.predict(X_test))
        vprint(f"        seed {seed}: done")
    return np.mean(te_all, axis=0)


# ── Tree stacking ─────────────────────────────────────────────────────────────

def fit_tree_oof(X_train, y_train, X_test,
                 seed=42, n_folds=3, split='random', target='CODMn',
                 groups=None, lite=False):
    """
    XGB + LGBM + CatBoost + ExtraTrees k-fold OOF stacking.
    Meta-learner: 0.5 × Ridge + 0.5 × ElasticNet.
    When groups is given, inner folds hold out whole stations.
    """
    n    = len(X_train)
    keys = ['xgb', 'cat'] if lite else ['xgb', 'lgbm', 'cat', 'et']
    oof  = {k: np.zeros(n) for k in keys}
    te   = {k: np.zeros(len(X_test)) for k in keys}
    fold_iter = (
        iter_group_folds(groups, n_folds, seed) if groups is not None
        else KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X_train)
    )

    if split == 'station' and target == 'CODMn':
        # Moderate regularisation for CODMn station-out
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
    elif split == 'station':
        # High regularisation for NH3N/TP station (NH3N uses DANN; TP uses blend)
        xp = dict(n_estimators=600, learning_rate=0.05, max_depth=4,
                  subsample=0.7, colsample_bytree=0.7, min_child_weight=10,
                  reg_alpha=0.5, reg_lambda=3.0, early_stopping_rounds=40)
        lp = dict(n_estimators=600, learning_rate=0.05, max_depth=5,
                  num_leaves=31, subsample=0.7, colsample_bytree=0.7,
                  min_child_samples=30, reg_alpha=0.5, reg_lambda=3.0,
                  early_stopping_rounds=40)
        cp = dict(iterations=500, learning_rate=0.05, depth=5,
                  l2_leaf_reg=10.0, subsample=0.7, early_stopping_rounds=40)
        et_kw = dict(n_estimators=400, max_depth=8, min_samples_leaf=15)
    else:
        # random / temporal splits
        xp = dict(n_estimators=800, learning_rate=0.04, max_depth=5,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                  reg_alpha=0.1, reg_lambda=1.5, early_stopping_rounds=40)
        lp = dict(n_estimators=800, learning_rate=0.04, max_depth=6,
                  num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                  min_child_samples=20, reg_alpha=0.1, reg_lambda=1.5,
                  early_stopping_rounds=40)
        cp = dict(iterations=600, learning_rate=0.04, depth=6,
                  l2_leaf_reg=3.0, subsample=0.8, early_stopping_rounds=40)
        et_kw = dict(n_estimators=500, max_depth=None, min_samples_leaf=8)

    n_done = 0
    for fold, (tri, vai) in enumerate(fold_iter):
        Xtr, Xva = X_train[tri], X_train[vai]
        ytr, yva = y_train[tri], y_train[vai]

        mx = xgb.XGBRegressor(**xp, verbosity=0, random_state=seed + fold)
        mx.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        oof['xgb'][vai] = mx.predict(Xva)
        te['xgb']      += mx.predict(X_test)

        if 'lgbm' in keys:
            ml = lgb.LGBMRegressor(**lp, random_state=seed + fold, verbose=-1)
            ml.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                   callbacks=[lgb.early_stopping(lp['early_stopping_rounds'], verbose=False),
                              lgb.log_evaluation(-1)])
            oof['lgbm'][vai] = ml.predict(Xva)
            te['lgbm']      += ml.predict(X_test)

        mc = cb.CatBoostRegressor(**cp, random_seed=seed + fold, verbose=0)
        mc.fit(Xtr, ytr, eval_set=(Xva, yva), verbose=False)
        oof['cat'][vai] = mc.predict(Xva)
        te['cat']      += mc.predict(X_test)

        if 'et' in keys:
            me = ExtraTreesRegressor(**et_kw, random_state=seed + fold, n_jobs=-1)
            me.fit(Xtr, ytr)
            oof['et'][vai] = me.predict(Xva)
            te['et']      += me.predict(X_test)
        n_done += 1

    for k in keys:
        te[k] /= max(n_done, 1)

    meta_tr = np.column_stack([oof[k] for k in keys])
    meta_te = np.column_stack([te[k]  for k in keys])
    scm     = StandardScaler()
    Xm      = scm.fit_transform(meta_tr)
    Xt      = scm.transform(meta_te)
    ri      = Ridge(alpha=1.0);            ri.fit(Xm, y_train)
    en      = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=2000); en.fit(Xm, y_train)
    y_te    = 0.5 * ri.predict(Xt) + 0.5 * en.predict(Xt)
    return y_te, oof, meta_tr, meta_te


# ── 5-CatBoost ensemble ───────────────────────────────────────────────────────

def fit_cat5_ensemble(X_train, y_train, X_test,
                      n_models=5, seed=42, n_folds=3, split='random'):
    cat_params = dict(iterations=500, learning_rate=0.05, depth=6,
                      l2_leaf_reg=3.0, subsample=0.8, colsample_bylevel=0.8,
                      early_stopping_rounds=30, verbose=0)
    n_v_frac = 0.12
    oof = np.zeros(len(X_train))
    kf  = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (tri, vai) in enumerate(kf.split(X_train)):
        Xtr, ytr = X_train[tri], y_train[tri]
        fold_va  = np.zeros(len(vai))
        for k in range(n_models):
            cat = cb.CatBoostRegressor(**cat_params, random_seed=seed * 10 + k + fold * 100)
            nv  = max(30, int(len(Xtr) * n_v_frac))
            cat.fit(Xtr[:-nv], ytr[:-nv], eval_set=(Xtr[-nv:], ytr[-nv:]), verbose=False)
            fold_va += cat.predict(X_train[vai]) / n_models
        oof[vai] = fold_va
    te = np.zeros(len(X_test))
    for k in range(n_models):
        cat = cb.CatBoostRegressor(**cat_params, random_seed=seed * 10 + k)
        nv  = max(30, int(len(X_train) * n_v_frac))
        cat.fit(X_train[:-nv], y_train[:-nv],
                eval_set=(X_train[-nv:], y_train[-nv:]), verbose=False)
        te += cat.predict(X_test) / n_models
    return oof, te


# ── Bayesian blend ────────────────────────────────────────────────────────────

def bayesian_blend(oof_dann, oof_moe, oof_stack, y_train,
                   n_trials=80, moe_min=0.25, moe_max=0.60,
                   w1_max=0.40, min_w3=0.05):
    def obj(t):
        w1 = t.suggest_float('w1', 0.00, w1_max)
        w2 = t.suggest_float('w2', moe_min, moe_max)
        w3 = 1.0 - w1 - w2
        if w3 < min_w3 or w3 > 0.80:
            return -1.0
        return float(r2_score(y_train, w1 * oof_dann + w2 * oof_moe + w3 * oof_stack))
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    bp = study.best_params
    w1, w2 = bp['w1'], bp['w2']
    return w1, w2, 1.0 - w1 - w2


def blend_simplex(oofs: dict, y, n_trials=100, seed=42):
    """Non-negative weights summing to 1 over named OOF predictors."""
    keys = list(oofs.keys())
    mats = {k: np.asarray(oofs[k], dtype=float) for k in keys}

    def obj(t):
        raw = np.array([t.suggest_float(k, 0.0, 1.0) for k in keys])
        s = raw.sum()
        if s < 1e-8:
            return -1.0
        w = raw / s
        pred = sum(w[i] * mats[keys[i]] for i in range(len(keys)))
        return float(r2_score(y, pred))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    raw = np.array([study.best_params[k] for k in keys])
    w = raw / max(raw.sum(), 1e-8)
    return {k: float(w[i]) for i, k in enumerate(keys)}


def fit_geo_ridge(X_train, y_train, X_test, feat_names, groups,
                  n_folds=3, seed=42, alpha=10.0):
    """Ridge on transferable geo/meteo/trend features with grouped OOF."""
    idx = [i for i, c in enumerate(feat_names) if is_geo_feature(c)]
    if len(idx) < 2:
        return np.zeros(len(X_train)), np.zeros(len(X_test))
    Xg_tr = X_train[:, idx]
    Xg_te = X_test[:, idx]
    sc = StandardScaler()
    Xg_tr = sc.fit_transform(Xg_tr)
    Xg_te = sc.transform(Xg_te)
    oof = np.zeros(len(X_train))
    te = np.zeros(len(X_test))
    n_done = 0
    for tri, vai in iter_group_folds(groups, n_folds, seed):
        m = Ridge(alpha=alpha)
        m.fit(Xg_tr[tri], y_train[tri])
        oof[vai] = m.predict(Xg_tr[vai])
        te += m.predict(Xg_te)
        n_done += 1
    te /= max(n_done, 1)
    return oof, te


def _train_val_from_groups(X, y, d, groups, seed, frac=0.15):
    tr, va = grouped_holdout_indices(groups, frac=frac, seed=seed)
    return X[tr], y[tr], d[tr], X[va], y[va]


def fit_dann_full(X_train, y_train, d_train, X_test, num_domains,
                  seed=42, grl_scale=1.0, dom_weight=0.3, hidden_dim=128,
                  groups=None, dann_epochs=50, mae_epochs=None,
                  test_oversample=1):
    """Single DANN refit on all training stations (grouped holdout for early stop)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    tabmae_epochs = mae_epochs if mae_epochs is not None else (20 if hidden_dim == 96 else 30)
    if groups is not None:
        Xtr, ytr, dtr, Xva, yva = _train_val_from_groups(
            X_train, y_train, d_train, groups, seed,
        )
    else:
        nv = max(30, int(len(X_train) * 0.15))
        Xtr, ytr, dtr = X_train[:-nv], y_train[:-nv], d_train[:-nv]
        Xva, yva = X_train[-nv:], y_train[-nv:]
    sc = StandardScaler()
    Xtr_s, Xva_s, Xte_s = sc.fit_transform(Xtr), sc.transform(Xva), sc.transform(X_test)
    ysc = StandardScaler()
    ytr_s = ysc.fit_transform(ytr.reshape(-1, 1)).flatten()
    yva_s = ysc.transform(yva.reshape(-1, 1)).flatten()
    mae_m = TabularMAE(X_train.shape[1], hidden_dim=hidden_dim)
    X_mae = np.vstack([Xtr_s, Xva_s, Xte_s])
    if test_oversample and int(test_oversample) > 1:
        X_mae = np.vstack([X_mae, np.repeat(Xte_s, int(test_oversample) - 1, axis=0)])
    train_tabmae(mae_m, X_mae, epochs=tabmae_epochs)
    model = STDANN(X_train.shape[1], mae_m.encoder, num_domains, hidden_dim=hidden_dim)
    bw = train_dann(model, Xtr_s, ytr_s, dtr, Xva_s, yva_s,
                    epochs=dann_epochs, grl_scale=grl_scale, dom_weight=dom_weight)
    model.load_state_dict(bw)
    model.eval()
    with torch.no_grad():
        tp = ysc.inverse_transform(
            model(torch.tensor(Xte_s, dtype=torch.float32))[0]
                 .numpy().reshape(-1, 1)
        ).flatten()
    return tp


def fit_tree_full(X_train, y_train, X_test, seed=42, split="station",
                  target="CODMn", groups=None, lite=False):
    """Refit tree stack on all training stations; grouped holdout for early stop."""
    if groups is not None:
        tr, va = grouped_holdout_indices(groups, frac=0.15, seed=seed)
        Xtr, ytr, Xva, yva = X_train[tr], y_train[tr], X_train[va], y_train[va]
    else:
        nv = max(30, int(len(X_train) * 0.15))
        Xtr, ytr, Xva, yva = X_train[:-nv], y_train[:-nv], X_train[-nv:], y_train[-nv:]

    if split == "station" and target == "CODMn":
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
    elif split == "station":
        xp = dict(n_estimators=600, learning_rate=0.05, max_depth=4,
                  subsample=0.7, colsample_bytree=0.7, min_child_weight=10,
                  reg_alpha=0.5, reg_lambda=3.0, early_stopping_rounds=40)
        lp = dict(n_estimators=600, learning_rate=0.05, max_depth=5,
                  num_leaves=31, subsample=0.7, colsample_bytree=0.7,
                  min_child_samples=30, reg_alpha=0.5, reg_lambda=3.0,
                  early_stopping_rounds=40)
        cp = dict(iterations=500, learning_rate=0.05, depth=5,
                  l2_leaf_reg=10.0, subsample=0.7, early_stopping_rounds=40)
        et_kw = dict(n_estimators=400, max_depth=8, min_samples_leaf=15)
    else:
        xp = dict(n_estimators=800, learning_rate=0.04, max_depth=5,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                  reg_alpha=0.1, reg_lambda=1.5, early_stopping_rounds=40)
        lp = dict(n_estimators=800, learning_rate=0.04, max_depth=6,
                  num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                  min_child_samples=20, reg_alpha=0.1, reg_lambda=1.5,
                  early_stopping_rounds=40)
        cp = dict(iterations=600, learning_rate=0.04, depth=6,
                  l2_leaf_reg=3.0, subsample=0.8, early_stopping_rounds=40)
        et_kw = dict(n_estimators=500, max_depth=None, min_samples_leaf=8)

    preds = []
    mx = xgb.XGBRegressor(**xp, verbosity=0, random_state=seed)
    mx.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    preds.append(mx.predict(X_test))
    if not lite:
        ml = lgb.LGBMRegressor(**lp, random_state=seed, verbose=-1)
        ml.fit(Xtr, ytr, eval_set=[(Xva, yva)],
               callbacks=[lgb.early_stopping(lp["early_stopping_rounds"], verbose=False),
                          lgb.log_evaluation(-1)])
        preds.append(ml.predict(X_test))
    mc = cb.CatBoostRegressor(**cp, random_seed=seed, verbose=0)
    mc.fit(Xtr, ytr, eval_set=(Xva, yva), verbose=False)
    preds.append(mc.predict(X_test))
    if not lite:
        me = ExtraTreesRegressor(**et_kw, random_state=seed, n_jobs=-1)
        me.fit(X_train, y_train)
        preds.append(me.predict(X_test))
    return np.mean(preds, axis=0)


STATION_BLEND_PRIOR = {
    "CODMn": {"dann": 0.50, "stack": 0.50},
    "NH3N": {"dann": 0.55, "stack": 0.45},
    "TP": {"dann": 0.90, "stack": 0.10},
}
STATION_STACK_CAP = {"CODMn": 0.55, "NH3N": 0.50, "TP": 0.30}
STATION_BLEND_SHRINK = 0.40

SPEC8_MEAN = [
    "spec_veg_corrected_B9", "spec_station_B9", "spec_station_B3",
    "idx_sta_b4_b5_ratio", "idx_sta_oci", "idx_sta_ndre",
    "idx_sta_spm_proxy", "spec_station_B2",
]
LUCC_MEAN = [
    "lucc_crop_1km", "lucc_built_1km", "lucc_water_1km",
    "lucc_crop_5km", "lucc_built_5km",
    "lucc_water_150m", "lucc_built_gain_2km",
]
# nitrification seasonality: station-median T×sin + season lifts NH3N LOSO mean R2 0.444→0.640
NH3N_MEAN_EXTRA = ["drv_T_x_sins", "season_sin", "season_cos"]
STATION_MEAN_CFG = {
    "CODMn": {
        "features": "spec8", "kind": "gp", "gp_kernel": "rbf",
        "embed": "median", "blend": 1.0, "residualize": False,
    },
    # compact LUCC + spec8 + T-season: station-mean LOSO R2 0.640
    "NH3N": {
        "features": "lucc_spec8_t", "kind": "ridge", "alpha": 4.0,
        "embed": "median", "blend": 1.0, "residualize": False,
    },
    "TP": {
        "features": "spec8", "kind": "knn", "k": 3,
        "embed": "median", "blend": 0.0, "residualize": False,
    },
}


def _mean_feat_idx(feat_names, mode):
    feat_names = list(feat_names)
    if mode == "spec8":
        wanted = set(SPEC8_MEAN)
        return [i for i, c in enumerate(feat_names) if c in wanted]
    if mode == "spec_all":
        return [i for i, c in enumerate(feat_names)
                if str(c).startswith("spec_") or str(c).startswith("idx_")]
    if mode == "lucc":
        wanted = set(LUCC_MEAN)
        return [i for i, c in enumerate(feat_names) if c in wanted]
    if mode == "lucc_spec8":
        wanted = set(LUCC_MEAN) | set(SPEC8_MEAN)
        return [i for i, c in enumerate(feat_names) if c in wanted]
    if mode == "lucc_spec8_t":
        wanted = set(LUCC_MEAN) | set(SPEC8_MEAN) | set(NH3N_MEAN_EXTRA)
        return [i for i, c in enumerate(feat_names) if c in wanted]
    return list(range(len(feat_names)))


def _sorted_groups(groups):
    return np.array(sorted(np.unique(np.asarray(groups)), key=str))


def _group_embed(X, groups, col_idx, mode="median"):
    """Station-level spectral prototype: median or distributional moments."""
    groups = np.asarray(groups)
    uniq = _sorted_groups(groups)
    Xs = np.asarray(X)[:, col_idx]
    if mode == "moments":
        rows = []
        for g in uniq:
            x = Xs[groups == g]
            parts = [np.nanmean(x, 0), np.nanmedian(x, 0), np.nanstd(x, 0)]
            for q in (10, 25, 50, 75, 90):
                parts.append(np.nanpercentile(x, q, axis=0))
            rows.append(np.concatenate(parts))
        return uniq, np.vstack(rows)
    med = np.empty((len(uniq), len(col_idx)), dtype=float)
    for i, g in enumerate(uniq):
        med[i] = np.nanmedian(Xs[groups == g], axis=0)
    return uniq, med


def _group_means(y, groups):
    groups = np.asarray(groups)
    uniq = _sorted_groups(groups)
    mu = np.array([float(np.mean(y[groups == g])) for g in uniq], dtype=float)
    return uniq, mu


def _map_group_values(uniq, values, groups):
    lookup = {g: float(v) for g, v in zip(uniq, values)}
    return np.array([lookup[g] for g in groups], dtype=float)


def _center_by_group(y, groups):
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    out = np.empty_like(y)
    for g in np.unique(groups):
        m = groups == g
        out[m] = y[m] - np.mean(y[m])
    return out


def _append_group_deltas(X, groups, col_idx):
    """Append within-station deviations (test station uses its own unlabeled median)."""
    if not col_idx:
        return X
    groups = np.asarray(groups)
    Xs = np.asarray(X, dtype=float)[:, col_idx]
    dlt = np.zeros_like(Xs)
    for g in np.unique(groups):
        m = groups == g
        med = np.nanmedian(Xs[m], axis=0)
        dlt[m] = Xs[m] - med
    return np.hstack([X, np.nan_to_num(dlt)])


# NH3N: unlabeled within-station deltas (test station uses its own median).
# TP: post-hoc deltas hurt; lift comes from MSE DANN, not extra features.
WITHIN_DELTA_FEATS = {
    "NH3N": [
        "meteo_T2m_mean_C", "meteo_T2m_7d", "meteo_T2m_3d",
        "meteo_T2m_mean_C_seasanom", "season_sin",
        "spec_station_B3", "spec_station_B9", "idx_sta_ndre", "idx_sta_oci",
    ],
}


def _fit_mean_estimator(Xtr, ytr, Xte, kind, k=3, alpha=10.0, gp_kernel="rbf"):
    imp, sc = SimpleImputer(), StandardScaler()
    Xtr_s = sc.fit_transform(imp.fit_transform(Xtr))
    Xte_s = sc.transform(imp.transform(Xte))
    if kind == "knn":
        m = KNeighborsRegressor(n_neighbors=min(int(k), len(Xtr_s)), weights="distance")
        m.fit(Xtr_s, ytr)
        return m.predict(Xte_s)
    if kind == "gp":
        if gp_kernel == "matern":
            ker = (ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=1.0, nu=1.5)
                   + WhiteKernel(0.2, (1e-5, 1.0)))
        else:
            ker = (ConstantKernel(1.0, (1e-2, 1e2)) * RBF(1.0)
                   + WhiteKernel(0.2, (1e-5, 1.0)))
        m = GaussianProcessRegressor(
            kernel=ker, normalize_y=True, random_state=0, n_restarts_optimizer=2,
        )
        m.fit(Xtr_s, ytr)
        return m.predict(Xte_s)
    m = Ridge(alpha=alpha)
    m.fit(Xtr_s, ytr)
    return m.predict(Xte_s)


def _loso_mean_estimator(X, y, kind, k=3, alpha=10.0, gp_kernel="rbf"):
    oof = np.zeros(len(y), dtype=float)
    for i in range(len(y)):
        oof[i] = _fit_mean_estimator(
            np.delete(X, i, 0), np.delete(y, i), X[i:i + 1],
            kind, k=k, alpha=alpha, gp_kernel=gp_kernel,
        )[0]
    return oof


def fit_station_mean(X_tr, y, groups, X_te, sta_te, feat_names, target):
    """Leave-one-station-out spectral mean model → sample-level μ_tr / μ_te."""
    cfg = dict(STATION_MEAN_CFG[target])
    if target == "NH3N" and not any(str(c).startswith("lucc_") for c in feat_names):
        cfg["features"] = "spec_all"
        cfg["alpha"] = 10.0
    if target == "NH3N" and cfg.get("features") == "lucc_spec8_t":
        if not any(c in feat_names for c in NH3N_MEAN_EXTRA):
            cfg["features"] = "lucc_spec8"
    if cfg.get("blend", 0.0) <= 0.0 and not cfg.get("residualize", False):
        return None
    col_idx = _mean_feat_idx(feat_names, cfg["features"])
    if len(col_idx) < 2 or groups is None or sta_te is None:
        return None
    embed = cfg.get("embed", "median")
    uniq_tr, med_tr = _group_embed(X_tr, groups, col_idx, mode=embed)
    uniq_mu, mu_tr = _group_means(y, groups)
    if not np.array_equal(uniq_tr, uniq_mu):
        mu_lookup = dict(zip(uniq_mu, mu_tr))
        mu_tr = np.array([mu_lookup[g] for g in uniq_tr], dtype=float)
    uniq_te, med_te = _group_embed(X_te, sta_te, col_idx, mode=embed)
    kw = dict(
        kind=cfg["kind"], k=cfg.get("k", 3), alpha=cfg.get("alpha", 10.0),
        gp_kernel=cfg.get("gp_kernel", "rbf"),
    )
    mu_te_sta = _fit_mean_estimator(med_tr, mu_tr, med_te, **kw)
    if cfg.get("residualize"):
        mu_tr_oof = _loso_mean_estimator(med_tr, mu_tr, **kw)
        oof_r2 = float(r2_score(mu_tr, mu_tr_oof)) if len(mu_tr) > 1 else 0.0
    else:
        mu_tr_oof = np.zeros_like(mu_tr)
        oof_r2 = float("nan")
    return {
        "mu_tr": _map_group_values(uniq_tr, mu_tr_oof, groups),
        "mu_te": _map_group_values(uniq_te, mu_te_sta, sta_te),
        "cfg": cfg,
        "oof_r2": None if not np.isfinite(oof_r2) else float(oof_r2),
        "n_feat": int(len(col_idx)),
    }


def _fit_station_heads(X_tr, y_model, d_tr, X_te, nd, groups, target, seed,
                       n_inner, lite, dann_seeds, hidden, grl, domw,
                       dann_folds, dann_epochs, mae_epochs, test_oversample,
                       skip_stack=False, y_loss="smoothl1", align_test=False,
                       stack_mu_tr=None, stack_mu_te=None, sta_te=None):
    """DANN + grouped tree stack on a (possibly residual) training target."""
    if skip_stack:
        oof_dann, y_dann = (
            fit_dann_multiseed(
                X_tr, y_model, d_tr, X_te, nd, seeds=dann_seeds,
                n_folds=dann_folds, grl_scale=grl, dom_weight=domw,
                hidden_dim=hidden, groups=None, dann_epochs=dann_epochs,
                mae_epochs=mae_epochs, test_oversample=test_oversample,
                y_loss=y_loss, align_test=align_test,
            ) if len(dann_seeds) > 1 else
            fit_dann_kfold(
                X_tr, y_model, d_tr, X_te, nd, seed=dann_seeds[0],
                n_folds=dann_folds, grl_scale=grl, dom_weight=domw,
                hidden_dim=hidden, groups=None, dann_epochs=dann_epochs,
                mae_epochs=mae_epochs, test_oversample=test_oversample,
                y_loss=y_loss, align_test=align_test,
            )
        )
        return y_dann, {
            "dann": 1.0, "stack": 0.0,
            "w_oof_dann": 1.0, "w_oof_stack": 0.0,
            "oof_r2_dann": float(r2_score(y_model, oof_dann)),
            "oof_r2_stack": 0.0,
        }
    y_tree = y_model
    mu_tr_add = 0.0
    mu_te_add = 0.0
    if stack_mu_tr is not None:
        y_tree = y_model - np.asarray(stack_mu_tr, dtype=float)
        mu_tr_add = np.asarray(stack_mu_tr, dtype=float)
        if stack_mu_te is not None:
            mu_te_add = np.asarray(stack_mu_te, dtype=float)
        vprint("      residual tree stack (within-station y)")
    _, _, meta_tr, _ = fit_tree_oof(
        X_tr, y_tree, X_te, seed=seed, split="station", target=target,
        groups=groups, n_folds=n_inner, lite=lite,
    )
    scm_oof = StandardScaler()
    Xm_oof = scm_oof.fit_transform(meta_tr)
    ri_oof = Ridge(alpha=1.0)
    ri_oof.fit(Xm_oof, y_tree)
    en_oof = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=2000)
    en_oof.fit(Xm_oof, y_tree)
    oof_stack = 0.5 * ri_oof.predict(Xm_oof) + 0.5 * en_oof.predict(Xm_oof) + mu_tr_add

    dann_kw = dict(
        n_folds=dann_folds, grl_scale=grl, dom_weight=domw,
        hidden_dim=hidden, groups=None,
        dann_epochs=dann_epochs, mae_epochs=mae_epochs,
        test_oversample=test_oversample, y_loss=y_loss, align_test=align_test,
    )
    if len(dann_seeds) > 1:
        oof_dann, y_dann = fit_dann_multiseed(
            X_tr, y_model, d_tr, X_te, nd, seeds=dann_seeds, **dann_kw,
        )
    else:
        oof_dann, y_dann = fit_dann_kfold(
            X_tr, y_model, d_tr, X_te, nd, seed=dann_seeds[0], **dann_kw,
        )

    if (stack_mu_tr is not None and stack_mu_te is None and sta_te is not None):
        uniq_te, mu_te_imp = _group_means(y_dann, sta_te)
        mu_te_add = _map_group_values(uniq_te, mu_te_imp, sta_te)
        vprint("      residual trees + DANN-implied test means")

    y_stack = fit_tree_full(
        X_tr, y_tree, X_te, seed=seed, split="station", target=target,
        groups=groups, lite=lite,
    ) + mu_te_add
    if stack_mu_tr is not None:
        w_oof = blend_simplex(
            {"dann": oof_dann - mu_tr_add, "stack": oof_stack - mu_tr_add},
            y_tree, n_trials=80, seed=seed,
        )
        oof_r2_dann = float(r2_score(y_tree, oof_dann - mu_tr_add))
        oof_r2_stack = float(r2_score(y_tree, oof_stack - mu_tr_add))
    else:
        w_oof = blend_simplex(
            {"dann": oof_dann, "stack": oof_stack}, y_model, n_trials=80, seed=seed,
        )
        oof_r2_dann = float(r2_score(y_model, oof_dann))
        oof_r2_stack = float(r2_score(y_model, oof_stack))
    prior = STATION_BLEND_PRIOR[target]
    w_stack = STATION_BLEND_SHRINK * w_oof["stack"] + (1.0 - STATION_BLEND_SHRINK) * prior["stack"]
    w_stack = min(w_stack, STATION_STACK_CAP[target])
    w_dann = 1.0 - w_stack
    y_pred = w_dann * y_dann + w_stack * y_stack
    return y_pred, {
        "dann": float(w_dann), "stack": float(w_stack),
        "w_oof_dann": float(w_oof["dann"]), "w_oof_stack": float(w_oof["stack"]),
        "oof_r2_dann": oof_r2_dann,
        "oof_r2_stack": oof_r2_stack,
    }


def _apply_mean_swap(y_pred, sta_te, mu_te, blend):
    if blend <= 0.0 or sta_te is None:
        return y_pred
    centered = _center_by_group(y_pred, sta_te)
    implied = y_pred - centered
    return centered + blend * mu_te + (1.0 - blend) * implied


def refit_station_mean_from_df(y_pred, sta_te, df, test_stations, target, feat_mode=None, alpha=4.0):
    """Replace predicted station means using train-station ridge on df columns.

    Used after dual-view averaging so complementary residuals keep one shared mean.
    """
    if sta_te is None or target != "NH3N":
        return np.asarray(y_pred, dtype=float), {}
    feat_mode = feat_mode or STATION_MEAN_CFG[target]["features"]
    dummy_names = list(df.columns)
    col_idx = _mean_feat_idx(dummy_names, feat_mode)
    cols = [dummy_names[i] for i in col_idx if dummy_names[i] in df.columns]
    if len(cols) < 2:
        return np.asarray(y_pred, dtype=float), {}
    ycol = f"target_{target}"
    te = set(map(str, test_stations))
    tr_mask = ~df["station_name"].astype(str).isin(te) & df[ycol].notna()
    te_mask = df["station_name"].astype(str).isin(te) & df[ycol].notna()

    def _embed(mask):
        sub = df.loc[mask, ["station_name", ycol] + cols]
        g = sub.groupby("station_name")
        med = g[cols].median()
        mu = g[ycol].mean()
        return med, mu

    med_tr, mu_tr = _embed(tr_mask)
    med_te, _ = _embed(te_mask)
    common = [c for c in cols if c in med_tr.columns and c in med_te.columns]
    if len(common) < 2 or len(med_tr) < 3 or len(med_te) == 0:
        return np.asarray(y_pred, dtype=float), {}
    mu_te_sta = _fit_mean_estimator(
        med_tr[common].to_numpy(float), mu_tr.to_numpy(float),
        med_te[common].to_numpy(float),
        kind="ridge", alpha=alpha,
    )
    mu_map = dict(zip(med_te.index.astype(str), mu_te_sta))
    sta = np.asarray(sta_te).astype(str)
    mu_te = np.array([mu_map.get(s, np.nan) for s in sta], dtype=float)
    if not np.isfinite(mu_te).all():
        return np.asarray(y_pred, dtype=float), {}
    y_new = _apply_mean_swap(np.asarray(y_pred, dtype=float), sta, mu_te, 1.0)
    return y_new, {"mean_refit": feat_mode, "mean_refit_n": int(len(common))}


def apply_within_ridge_addon(y_pred, sta_te, df, test_stations, target, cols,
                             blend=0.3, alpha=10.0, te_idx=None):
    """Blend ADAE within residuals with a train-only station-demeaned Ridge.

    Does not touch station means. Pre-specified blend=0.3 (not test-tuned).
    """
    y_pred = np.asarray(y_pred, dtype=float)
    if sta_te is None or not cols:
        return y_pred, {}
    ycol = f"target_{target}"
    use = [c for c in cols if c in df.columns]
    if len(use) < 2 or ycol not in df.columns:
        return y_pred, {}
    te = set(map(str, test_stations))
    tr = df.loc[~df["station_name"].astype(str).isin(te) & df[ycol].notna(), ["station_name", ycol] + use]
    if te_idx is not None:
        te_df = df.loc[list(te_idx), ["station_name"] + use]
    else:
        te_df = df.loc[df["station_name"].astype(str).isin(te) & df[ycol].notna(), ["station_name"] + use]
    if len(tr) < 30 or len(te_df) != len(y_pred):
        return y_pred, {}
    gtr = tr["station_name"].astype(str).to_numpy()
    ytr = tr[ycol].to_numpy(float)
    ytr_c = ytr - pd.Series(ytr).groupby(gtr).transform("mean").to_numpy()
    imp, sc = SimpleImputer(), StandardScaler()
    Ztr = sc.fit_transform(imp.fit_transform(tr[use].to_numpy(float)))
    Zte = sc.transform(imp.transform(te_df[use].to_numpy(float)))
    m = Ridge(alpha=alpha)
    m.fit(Ztr, ytr_c)
    add = m.predict(Zte)
    sta = np.asarray(sta_te).astype(str)
    add = add - pd.Series(add).groupby(sta).transform("mean").to_numpy()
    mu = pd.Series(y_pred).groupby(sta).transform("mean").to_numpy()
    pc = y_pred - mu
    y_new = mu + (1.0 - blend) * pc + blend * add
    return y_new, {"within_ridge": float(blend), "within_ridge_n": int(len(use))}


def fit_adae_station(X_tr, y_fit, d_tr, X_te, nd, groups, feat_names,
                     target, seed=42, lite=False, sta_te=None,
                     disable_mean_head=False, disable_residual_stack=False):
    """
    Hierarchical station-out ADAE.

    Between-station: spec8 Gaussian-process (CODMn) or spectral Ridge (NH3N)
    on station prototypes. Within-station: DANN + grouped tree stack.
    TP keeps DANN-implied station means.

    disable_mean_head / disable_residual_stack turn off the per-target
    station-mean and residual-tree patches (ablation only; official path
    keeps both False).
    """
    n_inner = 2 if lite else 3
    dann_folds = 2 if lite else 3
    if target == "TP":
        dann_epochs = 40 if lite else 50
        mae_epochs = 20 if lite else 30
        test_oversample = 1
        hidden = 96
        grl = GRL_LAMBDA["station"]
        dann_seeds = [seed] if lite else list(TP_DANN_SEEDS)
    elif target == "NH3N":
        dann_epochs = 40 if lite else 60
        mae_epochs = 20 if lite else 35
        test_oversample = 3
        hidden = 128
        grl = GRL_LAMBDA["station"]
        dann_seeds = [seed] if lite else [seed, 123, 456]
    else:
        dann_epochs = 40 if lite else 60
        mae_epochs = 20 if lite else 35
        test_oversample = 3
        hidden = 128
        grl = GRL_LAMBDA["station"]
        dann_seeds = [seed] if lite else [seed, 123, 456]
    mean_pack = None
    if sta_te is not None and not disable_mean_head:
        mean_pack = fit_station_mean(
            X_tr, y_fit, groups, X_te, sta_te, feat_names, target,
        )

    y_loss = "smoothl1"
    align_test = False

    res_mode = mean_pack["cfg"].get("residualize", False) if mean_pack else False
    mean_blend = float(mean_pack["cfg"].get("blend", 0.0)) if mean_pack else 0.0

    vprint(
        f"      station ADAE  target={target}  lite={lite}  "
        f"tree_inner={n_inner} dann_seeds={len(dann_seeds)} hidden={hidden}  "
        f"residualize={res_mode}  mean={None if mean_pack is None else mean_pack['cfg']['kind']}/{None if mean_pack is None else mean_pack['cfg']['features']}"
    )

    skip_trees = STATION_STACK_CAP[target] <= 0.0
    stack_mu_tr = stack_mu_te = None
    if target in ("NH3N", "TP") and groups is not None and not disable_residual_stack:
        uniq, mu = _group_means(y_fit, groups)
        stack_mu_tr = _map_group_values(uniq, mu, groups)
        if target == "NH3N" and mean_pack is not None:
            stack_mu_te = mean_pack["mu_te"]
            vprint("      NH3N trees on station-demeaned y + ridge mu_te")
        elif target == "TP":
            vprint("      TP trees on station-demeaned y + DANN-implied mu_te")
    head_kw = dict(
        X_tr=X_tr, d_tr=d_tr, X_te=X_te, nd=nd, groups=groups,
        target=target, seed=seed, n_inner=n_inner, lite=lite,
        dann_seeds=dann_seeds, hidden=hidden,
        grl=grl, domw=DOM_WEIGHT["station"],
        dann_folds=dann_folds, dann_epochs=dann_epochs,
        mae_epochs=mae_epochs, test_oversample=test_oversample,
        skip_stack=skip_trees, y_loss=y_loss, align_test=align_test,
        stack_mu_tr=stack_mu_tr, stack_mu_te=stack_mu_te, sta_te=sta_te,
    )

    dual = res_mode == "both" and mean_pack is not None
    residualize = res_mode is True
    if dual:
        y_full, w_full = _fit_station_heads(y_model=y_fit, **head_kw)
        y_res, w_res = _fit_station_heads(
            y_model=y_fit - mean_pack["mu_tr"], **head_kw,
        )
        y_swap = _apply_mean_swap(y_full, sta_te, mean_pack["mu_te"], mean_blend)
        y_hier = y_res + mean_pack["mu_te"]
        y_pred = 0.5 * y_swap + 0.5 * y_hier
        w_use = dict(w_full)
        w_use["w_res_dann"] = w_res["dann"]
        w_use["oof_r2_dann_res"] = w_res["oof_r2_dann"]
    else:
        y_model = (y_fit - mean_pack["mu_tr"]) if residualize else y_fit
        y_pred, w_use = _fit_station_heads(y_model=y_model, **head_kw)
        if residualize:
            y_pred = y_pred + mean_pack["mu_te"]
        elif mean_blend > 0.0:
            y_pred = _apply_mean_swap(y_pred, sta_te, mean_pack["mu_te"], mean_blend)

    lo, hi = np.percentile(y_fit, [1, 99])
    y_pred = np.clip(y_pred, lo, hi)
    if dual or residualize or mean_blend >= 0.5:
        alpha = 0.0
    else:
        alpha = {"CODMn": 0.12, "NH3N": 0.06, "TP": 0.08}.get(target, 0.10)
        if alpha > 0:
            y_pred = (1.0 - alpha) * y_pred + alpha * float(np.mean(y_fit))

    mean_oof_r2 = None if mean_pack is None else mean_pack["oof_r2"]
    weights = {
        **w_use, "geo": 0.0,
        "shrink_alpha": float(alpha),
        "mean_blend": float(mean_blend),
        "mean_oof_r2": None if mean_oof_r2 is None else float(mean_oof_r2),
        "residualize": res_mode if isinstance(res_mode, str) else bool(residualize),
    }
    vprint(
        f"      w dann={w_use['dann']:.2f} stack={w_use['stack']:.2f}  "
        f"OOF d={w_use['oof_r2_dann']:.3f} s={w_use['oof_r2_stack']:.3f}  "
        f"mean_oof={mean_oof_r2 if mean_oof_r2 is None else f'{mean_oof_r2:.3f}'}"
    )
    return y_pred, weights


# ══════════════════════════════════════════════════════════════════════════════
# Per-target, per-split runner
# ══════════════════════════════════════════════════════════════════════════════


def run_target_split(target, split_method, df, fc,
                     tr_idx, te_idx, seed=42):
    """
    Train ADAE for one (target, split) combination.

    Returns
    -------
    dict with keys: R2, MAE, y_true, y_pred, weights (optional),
                    dates, station_ids, station_names
    """
    vprint(f"\n  [{target}|{split_method}]", end='', flush=True)
    use_log = target in ('TP', 'NH3N')

    # Pull arrays from df using pre-computed indices
    imp    = SimpleImputer(strategy='median')
    valid  = df[f'target_{target}'].notna()
    spec   = [c for c in fc if c.startswith('spec_')]
    if spec:
        valid &= df[spec].notna().any(axis=1)
    vdf    = df[valid].copy()

    # Filter tr/te indices to valid rows
    tr_set = set(tr_idx)
    te_set = set(te_idx)
    tr_valid = [i for i in vdf.index if i in tr_set]
    te_valid = [i for i in vdf.index if i in te_set]

    X_tr = imp.fit_transform(vdf.loc[tr_valid, fc].values)
    X_te = imp.transform(vdf.loc[te_valid, fc].values)
    y_tr = vdf.loc[tr_valid, f'target_{target}'].values
    y_te = vdf.loc[te_valid, f'target_{target}'].values

    # Metadata for output CSV
    dates_te   = vdf.loc[te_valid, 'date'].dt.strftime('%Y-%m-%d').tolist()
    sta_ids_te = vdf.loc[te_valid, 'station_name'].tolist()

    y_fit  = np.log1p(np.maximum(y_tr, 0)) if use_log else y_tr.copy()
    y_true = y_te.copy()

    # Domain labels
    if split_method == 'station':
        le = LabelEncoder()
        d_tr = le.fit_transform(vdf.loc[tr_valid, 'station_name'].values)
        nd   = max(int(d_tr.max()) + 1, 2)
    elif split_method == 'temporal':
        years = vdf.loc[tr_valid, 'year'].values if 'year' in vdf.columns else np.zeros(len(tr_valid))
        uy    = sorted(np.unique(years))
        ym    = {y: i for i, y in enumerate(uy)}
        d_tr  = np.array([ym.get(y, 0) for y in years])
        nd    = max(int(d_tr.max()) + 1, 2)
    else:
        spec_cols = [c for c in fc if c.startswith('spec_station_')]
        if spec_cols:
            si   = [fc.index(c) for c in spec_cols]
            km   = KMeans(n_clusters=N_OWT, random_state=42, n_init=5)
            d_tr = km.fit_predict(X_tr[:, si])
        else:
            d_tr = np.zeros(len(X_tr), dtype=int)
        nd = N_OWT

    grl  = GRL_LAMBDA[split_method]
    domw = DOM_WEIGHT[split_method]

    # ── Standard path: random / temporal ──────────────────────────────────
    vprint(" [standard]", end='', flush=True)
    y_stack, oof_stack_dict, meta_tr, meta_te = fit_tree_oof(
        X_tr, y_fit, X_te, seed=seed, split=split_method, target=target)

    scm_oof = StandardScaler()
    Xm_oof  = scm_oof.fit_transform(meta_tr)
    ri_oof  = Ridge(alpha=1.0);  ri_oof.fit(Xm_oof, y_fit)
    en_oof  = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=2000); en_oof.fit(Xm_oof, y_fit)
    oof_stack = 0.5 * ri_oof.predict(Xm_oof) + 0.5 * en_oof.predict(Xm_oof)

    oof_moe, y_moe   = fit_cat5_ensemble(X_tr, y_fit, X_te, seed=seed, split=split_method)
    oof_dann, y_dann = fit_dann_kfold(X_tr, y_fit, d_tr, X_te, nd,
                                      n_folds=3, seed=seed,
                                      grl_scale=grl, dom_weight=domw)
    w1, w2, w3 = bayesian_blend(oof_dann, oof_moe, oof_stack, y_fit,
                                 moe_min=MOE_MIN[split_method],
                                 moe_max=MOE_MAX[split_method])
    y_pred = w1 * y_dann + w2 * y_moe + w3 * y_stack

    if use_log:
        y_pred  = np.expm1(np.maximum(y_pred,  0))
        y_dann  = np.expm1(np.maximum(y_dann,  0))
        y_moe   = np.expm1(np.maximum(y_moe,   0))
        y_stack = np.expm1(np.maximum(y_stack, 0))
    y_pred = np.maximum(y_pred, 0)

    r2 = r2_score(y_true, y_pred)
    vprint(f" w=({w1:.2f}/{w2:.2f}/{w3:.2f}) R²={r2:.4f}", end='', flush=True)
    return _build_result(r2, y_true, y_pred, dates_te, sta_ids_te,
                         weights=[w1, w2, w3])


def _build_result(r2, y_true, y_pred, dates, station_ids, weights=None):
    out = dict(
        R2=float(r2),
        MAE=float(mean_absolute_error(y_true, y_pred)),
        y_true=y_true.tolist(),
        y_pred=y_pred.tolist(),
        dates=dates,
        station_ids=station_ids,
    )
    if weights is not None:
        out['weights'] = [float(w) for w in weights]
    return out
