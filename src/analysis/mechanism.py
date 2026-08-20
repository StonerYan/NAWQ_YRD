"""
Mechanistic analysis for WQI0627
================================
(D) ICC / Moran's I — validation inflation structure.
Top-3 per target — data relationships for the three highest model-importance
categories (from category permutation CSVs), not fixed CDOM/SPM anchors.

Run after: python src/analysis/run_attribution.py
          python src/analysis/mechanism.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.models.feature_engineering import load_engineered_dataset
from src.analysis.mechanism_top3 import top3_mechanism_analysis, multilevel_mechanism_summary
from src.utils.eval_verbose import set_verbose

OUT = ROOT / "results" / "analysis"

# Sentinel-2 MSI band centre wavelengths (nm)
S2_WL = {1: 443, 2: 492, 3: 560, 4: 665, 5: 704, 6: 740,
         7: 783, 8: 833, 9: 945, 11: 1614, 12: 2202}


def _pearson(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return float("nan"), float("nan"), int(m.sum())
    r, p = stats.pearsonr(x[m], y[m])
    return float(r), float(p), int(m.sum())


def _spearman(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return float("nan"), float("nan"), int(m.sum())
    r, p = stats.spearmanr(x[m], y[m])
    return float(r), float(p), int(m.sum())


# ── (A) CDOM optical slope vs COD_Mn ─────────────────────────────────────────
def cdom_analysis(df: pd.DataFrame) -> dict:
    bands = [1, 2, 3, 4]
    wl = np.array([S2_WL[b] for b in bands], dtype=float)
    cols = [f"spec_station_B{b}" for b in bands]
    R = df[cols].values.astype(float)
    # Spectral slope S of reflectance: ln R = a - S*(wl-wl0); fit per sample.
    wl0 = wl[0]
    dx = wl - wl0
    slopes = np.full(len(df), np.nan)
    for i in range(len(df)):
        r = R[i]
        if np.all(np.isfinite(r)) and np.all(r > 0):
            lr = np.log(r)
            b = np.polyfit(dx, lr, 1)[0]
            slopes[i] = -b  # positive S = steeper blue-enriched decay
    df = df.assign(cdom_slope=slopes)

    g = df.groupby("station_name").agg(
        codmn=("target_CODMn", "mean"),
        cdom_proxy=("phys_CDOM_proxy", "mean"),
        cdom_chl=("phys_CDOM_chl", "mean"),
        slope=("cdom_slope", "mean"),
    ).dropna()

    r_proxy, p_proxy, n = _pearson(g["cdom_proxy"].values, g["codmn"].values)
    r_slope, p_slope, _ = _pearson(g["slope"].values, g["codmn"].values)
    # sample-level (not just station means)
    s_prox = df["phys_CDOM_proxy"].values.astype(float)
    s_cod = df["target_CODMn"].values.astype(float)
    mask = np.isfinite(s_prox) & np.isfinite(s_cod)
    rs_proxy, ps_proxy, ns = _spearman(s_prox, s_cod)

    return {
        "description": "CDOM exponential reflectance slope S (443-665 nm) and "
                       "blue/red CDOM proxy vs COD_Mn",
        "n_stations": n,
        "station_r_proxy_codmn": r_proxy, "station_p_proxy_codmn": p_proxy,
        "station_r_slope_codmn": r_slope, "station_p_slope_codmn": p_slope,
        "sample_spearman_proxy_codmn": rs_proxy,
        "sample_spearman_p": ps_proxy, "sample_n": ns,
        "_arrays": {
            "cdom_proxy_station": g["cdom_proxy"].values,
            "cdom_slope_station": g["slope"].values,
            "codmn_station": g["codmn"].values,
            "cdom_proxy_sample": s_prox[mask],
            "codmn_sample": s_cod[mask],
        },
    }


# ── (B) Arrhenius / Q10 nitrification temperature sensitivity ────────────────
def nitrification_analysis(df: pd.DataFrame) -> dict:
    tcol = "meteo_T2m_7d" if "meteo_T2m_7d" in df.columns else "meteo_T2m_mean_C"
    sub = df[[tcol, "target_NH3N", "target_TP"]].copy()
    sub = sub[np.isfinite(sub[tcol]) & np.isfinite(sub["target_NH3N"])]
    sub = sub[sub["target_NH3N"] > 0]
    T_C = sub[tcol].values.astype(float)
    T_K = T_C + 273.15
    nh3 = sub["target_NH3N"].values.astype(float)

    # Arrhenius form: ln(NH3) = c + (Ea/R) * (1/T)  (standing stock rises when
    # removal slows at low T). slope_invT = Ea/R.
    inv_T = 1.0 / T_K
    ln_nh3 = np.log(nh3)
    sl, ic, r_arr, p_arr, _ = stats.linregress(inv_T, ln_nh3)
    R_gas = 8.314  # J/mol/K
    Ea = sl * R_gas / 1000.0  # kJ/mol (sign carries direction)

    # Effective Q10 from ln(NH3) vs T (degC): slope_T -> Q10 = exp(10*slope)
    sl_t, ic_t, r_t, p_t, _ = stats.linregress(T_C, ln_nh3)
    Q10 = float(np.exp(10.0 * sl_t))

    # temperature-binned means for the figure
    bins = np.linspace(np.nanpercentile(T_C, 1), np.nanpercentile(T_C, 99), 11)
    idx = np.digitize(T_C, bins)
    bx, by, bs = [], [], []
    for k in range(1, len(bins)):
        m = idx == k
        if m.sum() >= 10:
            bx.append(float(np.mean(T_C[m])))
            by.append(float(np.mean(nh3[m])))
            bs.append(float(np.std(nh3[m]) / np.sqrt(m.sum())))

    return {
        "description": "Apparent temperature sensitivity of standing NH3-N "
                       "(Arrhenius / Q10), consistent with nitrification kinetics",
        "temp_column": tcol,
        "arrhenius_slope_invT": float(sl),
        "apparent_Ea_kJ_per_mol": float(Ea),
        "arrhenius_r": float(r_arr), "arrhenius_p": float(p_arr),
        "lnNH3_vs_T_slope_perC": float(sl_t),
        "effective_Q10": Q10,
        "Q10_r": float(r_t), "Q10_p": float(p_t),
        "n": int(len(nh3)),
        "_arrays": {
            "bin_T_C": np.array(bx), "bin_NH3": np.array(by), "bin_se": np.array(bs),
            "T_C_all": T_C, "NH3_all": nh3,
        },
    }


# ── (C) TP-ISS coupling and Redfield N:P ─────────────────────────────────────
def phosphorus_analysis(df: pd.DataFrame) -> dict:
    turb_cols = [c for c in ["tp_log_turb", "phys_Turb_proxy", "tp_SPM_proxy"]
                 if c in df.columns]
    out = {"description": "Particulate-P / ISS coupling and N:P vs Redfield (16:1)"}
    corr = {}
    for c in turb_cols:
        r, p, n = _spearman(df[c].values.astype(float),
                            df["target_TP"].values.astype(float))
        corr[c] = {"spearman_r": r, "p": p, "n": n}
    out["tp_turbidity_coupling"] = corr

    # molar N:P using NH3-N as a dissolved-inorganic-N proxy (lower bound on TN)
    sub = df[["target_NH3N", "target_TP"]].copy()
    sub = sub[(sub["target_NH3N"] > 0) & (sub["target_TP"] > 0)]
    np_molar = (sub["target_NH3N"].values / 14.0) / (sub["target_TP"].values / 31.0)
    out["NP_molar_median"] = float(np.median(np_molar))
    out["NP_molar_q25"] = float(np.percentile(np_molar, 25))
    out["NP_molar_q75"] = float(np.percentile(np_molar, 75))
    out["redfield_NP"] = 16.0
    out["frac_above_redfield"] = float(np.mean(np_molar > 16.0))
    out["n_NP"] = int(len(np_molar))
    out["note_NP"] = ("NH3-N is only the dissolved-inorganic-ammonia pool, a small "
                      "fraction of TN; this ratio is descriptive and NOT used to "
                      "infer nutrient limitation.")
    # sample-level arrays for the figure: SPM proxy (B5/B4) shows positive coupling
    spm = df["tp_SPM_proxy"].values.astype(float)
    tp = df["target_TP"].values.astype(float)
    m2 = np.isfinite(spm) & np.isfinite(tp) & (tp > 0)
    out["_arrays"] = {
        "np_molar": np_molar,
        "spm_sample": spm[m2],
        "tp_sample": tp[m2],
    }
    return out


# ── (D) Spatial autocorrelation (Moran's I) ──────────────────────────────────
def _haversine_km(lat, lon):
    lat = np.radians(lat)
    lon = np.radians(lon)
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _morans_I(z, W):
    n = len(z)
    zc = z - z.mean()
    num = np.sum(W * np.outer(zc, zc))
    den = np.sum(zc ** 2)
    S0 = W.sum()
    I = (n / S0) * (num / den) if den > 0 and S0 > 0 else float("nan")
    # permutation p-value
    rng = np.random.default_rng(0)
    perm = np.empty(999)
    for i in range(999):
        zp = rng.permutation(zc)
        perm[i] = (n / S0) * (np.sum(W * np.outer(zp, zp)) / np.sum(zp ** 2))
    p = (np.sum(perm >= I) + 1) / (len(perm) + 1)
    return (float(I), float(p), float(perm.mean()), float(perm.std()),
            float(np.percentile(perm, 2.5)), float(np.percentile(perm, 97.5)))


def _icc1(values: np.ndarray, groups: np.ndarray) -> tuple[float, float, float]:
    """One-way random-effects ICC(1): fraction of variance between stations.
    Returns (ICC, between_var_frac_simple, n0)."""
    df = pd.DataFrame({"y": values, "g": groups}).dropna()
    grand = df["y"].mean()
    k = df["g"].nunique()
    N = len(df)
    gm = df.groupby("g")["y"]
    ni = gm.size().values
    means = gm.mean().values
    ssb = np.sum(ni * (means - grand) ** 2)
    ssw = np.sum((df["y"].values - df["g"].map(gm.mean()).values) ** 2)
    msb = ssb / (k - 1)
    msw = ssw / (N - k)
    n0 = (N - np.sum(ni ** 2) / N) / (k - 1)
    icc = (msb - msw) / (msb + (n0 - 1) * msw) if (msb + (n0 - 1) * msw) > 0 else float("nan")
    simple = ssb / (ssb + ssw)  # raw between-group SS fraction
    return float(icc), float(simple), float(n0)


def group_structure_analysis(df: pd.DataFrame) -> dict:
    """Why random-split inflates skill: variance decomposition (station ICC) plus
    a (deliberately secondary) Moran's I check showing the between-station field
    is NOT spatially smooth -> the inflation is grouped pseudo-replication, not
    spatial interpolation."""
    g = df.groupby("station_name").agg(
        lat=("lat", "mean"), lon=("lon", "mean"),
        codmn=("target_CODMn", "mean"),
        nh3n=("target_NH3N", "mean"),
        tp=("target_TP", "mean"),
    ).dropna()
    lat, lon = g["lat"].values, g["lon"].values
    D = _haversine_km(lat, lon)
    with np.errstate(divide="ignore"):
        W = 1.0 / D
    np.fill_diagonal(W, 0.0)
    W = W / W.sum(axis=1, keepdims=True)

    out = {"description": "Variance decomposition (station ICC) and Moran's I: the "
                          "random-vs-station-out gap is driven by between-station "
                          "pseudo-replication, not spatial smoothness",
           "n_stations": int(len(g)),
           "mean_pairwise_dist_km": float(D[np.triu_indices_from(D, 1)].mean())}

    icc, morans = {}, {}
    for col, lab in [("CODMn", "CODMn"), ("NH3N", "NH3N"), ("TP", "TP")]:
        v = df[f"target_{col}"].values.astype(float)
        gg = df["station_name"].values
        ic, simple, n0 = _icc1(v, gg)
        icc[lab] = {"ICC1": ic, "between_station_var_frac": simple, "avg_group_size": n0}
        key = {"CODMn": "codmn", "NH3N": "nh3n", "TP": "tp"}[lab]
        I, p, em, es, p025, p975 = _morans_I(g[key].values, W)
        morans[lab] = {"morans_I": I, "p_perm": p,
                       "expected_I": float(-1 / (len(g) - 1)),
                       "perm_p025": p025, "perm_p975": p975}
    out["icc"] = icc
    out["morans_I"] = morans
    out["_arrays"] = {
        "station_lat": lat, "station_lon": lon,
        "station_codmn": g["codmn"].values, "station_nh3n": g["nh3n"].values,
        "station_tp": g["tp"].values,
    }
    return out


def main():
    set_verbose(False)
    OUT.mkdir(parents=True, exist_ok=True)
    df, fc = load_engineered_dataset()
    print(f"Loaded {len(df)} rows, {df['station_name'].nunique()} stations")

    results = {
        "group_structure": group_structure_analysis(df),
        "multilevel_synthesis": multilevel_mechanism_summary(),
    }
    for target in ("CODMn", "NH3N", "TP"):
        results[f"top3_{target}"] = top3_mechanism_analysis(df, fc, target)

    # split arrays out to npz; keep scalars in json
    arrays = {}
    clean = {}
    for sec, d in results.items():
        cd = {}
        for k, v in d.items():
            if k == "_arrays":
                for ak, av in v.items():
                    key = ak if sec.startswith("top3_") else f"{sec}__{ak}"
                    arrays[key] = np.asarray(av)
            else:
                cd[k] = v
        clean[sec] = cd

    np.savez(OUT / "mechanism_arrays.npz", **arrays)
    (OUT / "mechanism.json").write_text(
        json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Top-3 mechanism (model-driven) ===")
    for target in ("CODMn", "NH3N", "TP"):
        panels = clean[f"top3_{target}"]["panels"]
        print(f"  {target}:")
        for p in panels:
            print(f"    #{p['rank']} {p['category_label']}: "
                  f"model ΔR²={p['model_delta_r2']:.3f}, "
                  f"data |r|={p['data_max_abs_spearman']:.2f}, "
                  f"feat={p['representative_feature']}")
    gs = clean["group_structure"]
    for t in ["CODMn", "NH3N", "TP"]:
        print(f"  {t} ICC={gs['icc'][t]['ICC1']:.3f}")
    print(f"\n  -> {OUT}/mechanism.json")


if __name__ == "__main__":
    main()
