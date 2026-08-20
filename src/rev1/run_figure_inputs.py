"""Build Stage-6 figure inputs from official rev1 products + the analysis table.

Writes:
  results/{tag}/downstream/structure.json          ICC1 + Moran I
  results/{tag}/downstream/conformal/conformal.json
  results/{tag}/downstream/conformal/conformal_arrays.npz
  results/{tag}/downstream/q10/q10.json
  results/{tag}/downstream/q10/q10_arrays.npz
  results/{tag}/downstream/mech/hexbin_arrays.npz
"""
from __future__ import annotations

import argparse
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

from src.analysis.mechanism import group_structure_analysis
from src.config import CONFORMAL_ALPHA, QA_THRESHOLDS
from src.models.feature_engineering import load_engineered_dataset
from src.rev1.paths import DEFAULT_DATA, TARGETS, rev1_dir
from src.utils.data_loader import load_spectral_data


def _oof(tag: str, protocol: str, method: str, target: str) -> dict:
    p = rev1_dir(tag) / "summaries" / "oof" / f"{protocol}_{method}_{target}.npz"
    if not p.exists():
        p = rev1_dir(tag) / "cells" / protocol / method / target / "oof.npz"
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in z.files}


def _conformal_one(y, yhat, alpha, seed=42) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)
    n_cal = n // 2
    cal, ev = idx[:n_cal], idx[n_cal:]
    scores = np.abs(y[cal] - yhat[cal])
    k = min(float(np.ceil((1 - alpha) * (len(cal) + 1))) / len(cal), 1.0)
    qhat = float(np.quantile(scores, k, method="higher"))
    lo = yhat[ev] - qhat
    hi = yhat[ev] + qhat
    cover = float(np.mean((y[ev] >= lo) & (y[ev] <= hi)))
    order = np.argsort(yhat[ev])
    return {
        "qhat": qhat,
        "empirical_coverage": cover,
        "n_cal": int(n_cal),
        "n_eval": int(len(ev)),
        "y": y[ev][order],
        "yhat": yhat[ev][order],
    }


def write_structure(tag: str, df_spec: pd.DataFrame) -> dict:
    gs = group_structure_analysis(df_spec)
    clean = {k: v for k, v in gs.items() if k != "_arrays"}
    dest = rev1_dir(tag) / "downstream" / "structure.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  structure -> {dest}")
    for t in TARGETS:
        print(f"    {t} ICC1={clean['icc'][t]['ICC1']:.3f}  I={clean['morans_I'][t]['morans_I']:.3f}")
    return clean


def write_conformal(tag: str) -> None:
    alpha = CONFORMAL_ALPHA
    out = rev1_dir(tag) / "downstream" / "conformal"
    out.mkdir(parents=True, exist_ok=True)
    results = {"primary_protocol": "station", "alpha": alpha, "protocols": {}}
    arrays = {}
    for protocol in ("station", "temporal"):
        results["protocols"][protocol] = {}
        for target in TARGETS:
            d = _oof(tag, protocol, "ADAE", target)
            y = np.asarray(d["y_true"], dtype=float)
            yhat = np.asarray(d["y_pred"], dtype=float)
            cell = _conformal_one(y, yhat, alpha, seed=42)
            thr = QA_THRESHOLDS[target]
            results["protocols"][protocol][target] = {
                "alpha": alpha,
                "qhat": cell["qhat"],
                "pi_width": 2 * cell["qhat"],
                "empirical_coverage": cell["empirical_coverage"],
                "n_cal": cell["n_cal"],
                "n_eval": cell["n_eval"],
                "halfwidth_frac_of_threshold": cell["qhat"] / thr,
                "threshold": thr,
                "n_oof": int(len(y)),
            }
            prefix = f"{protocol}__{target}"
            arrays[f"{prefix}__y"] = cell["y"]
            arrays[f"{prefix}__yhat"] = cell["yhat"]
            arrays[f"{prefix}__qhat"] = np.array([cell["qhat"]])
            print(
                f"  conformal {protocol}/{target}: "
                f"cover={cell['empirical_coverage']:.3f} qhat={cell['qhat']:.4f}"
            )
    np.savez(out / "conformal_arrays.npz", **arrays)
    (out / "conformal.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8",
    )


def _q10_bins(T, y, n_bins=10):
    m = np.isfinite(T) & np.isfinite(y) & (y > 0)
    T, y = T[m], y[m]
    if len(y) < 30:
        return {}, {}
    sl_t, _, r_t, p_t, _ = stats.linregress(T, np.log(y))
    q10 = float(np.exp(10.0 * sl_t))
    bins = np.linspace(np.nanpercentile(T, 1), np.nanpercentile(T, 99), n_bins + 1)
    idx = np.digitize(T, bins)
    bx, by, bs = [], [], []
    for k in range(1, len(bins)):
        sel = idx == k
        if sel.sum() >= 10:
            bx.append(float(np.mean(T[sel])))
            by.append(float(np.median(y[sel])))
            bs.append(float(np.std(y[sel]) / np.sqrt(sel.sum())))
    meta = {
        "effective_Q10": q10,
        "ln_vs_T_r": float(r_t),
        "ln_vs_T_p": float(p_t),
        "n": int(len(y)),
        "spearman_r": float(stats.spearmanr(T, y).statistic),
    }
    arr = {
        "bin_T": np.asarray(bx),
        "bin_y": np.asarray(by),
        "bin_se": np.asarray(bs),
        "T": T,
        "y": y,
    }
    return meta, arr


def write_q10_and_hex(tag: str, df: pd.DataFrame) -> None:
    tcol = "meteo_T2m_7d" if "meteo_T2m_7d" in df.columns else "meteo_T2m_mean_C"
    oci = "idx_sta_oci" if "idx_sta_oci" in df.columns else None
    ndre = "idx_sta_ndre" if "idx_sta_ndre" in df.columns else None

    q10_out = rev1_dir(tag) / "downstream" / "q10"
    mech_out = rev1_dir(tag) / "downstream" / "mech"
    q10_out.mkdir(parents=True, exist_ok=True)
    mech_out.mkdir(parents=True, exist_ok=True)

    payload = {"temp_column": tcol}
    arrays = {}
    hex_arrays = {}

    # in-situ NH3-N Q10
    meta, arr = _q10_bins(df[tcol].to_numpy(float), df["target_NH3N"].to_numpy(float))
    payload["insitu_NH3N"] = meta
    for k, v in arr.items():
        arrays[f"insitu_NH3N__{k}"] = v

    # satellite Q10 from S ADAE OOF
    oof = _oof(tag, "station", "ADAE", "NH3N")
    idx = np.asarray(oof["row_index"], dtype=int)
    T_sat = df.loc[idx, tcol].to_numpy(float)
    y_sat = np.asarray(oof["y_pred"], dtype=float)
    meta_s, arr_s = _q10_bins(T_sat, y_sat)
    payload["satellite_S_NH3N"] = meta_s
    for k, v in arr_s.items():
        arrays[f"sat_S_NH3N__{k}"] = v

    # month / year vs in-situ NH3-N (temporal–trend is the lead process group)
    m = df["target_NH3N"].notna()
    hex_arrays["nh3n_month_x"] = df.loc[m, "month"].to_numpy(float)
    hex_arrays["nh3n_month_y"] = df.loc[m, "target_NH3N"].to_numpy(float)
    year_nh3 = "year" if "year" in df.columns else "trend_decimal_year"
    hex_arrays["nh3n_year_x"] = df.loc[m, year_nh3].to_numpy(float)
    hex_arrays["nh3n_year_y"] = df.loc[m, "target_NH3N"].to_numpy(float)
    payload["nh3n_year_feature"] = year_nh3

    # TP: OCI vs in-situ and vs S OOF
    if oci:
        m = df["target_TP"].notna() & df[oci].notna()
        hex_arrays["tp_oci_x"] = df.loc[m, oci].to_numpy(float)
        hex_arrays["tp_oci_y"] = df.loc[m, "target_TP"].to_numpy(float)
        payload["tp_oci_spearman"] = float(stats.spearmanr(
            hex_arrays["tp_oci_x"], hex_arrays["tp_oci_y"],
        ).statistic)
        oof_tp = _oof(tag, "station", "ADAE", "TP")
        idx_tp = np.asarray(oof_tp["row_index"], dtype=int)
        hex_arrays["tp_oci_sat_x"] = df.loc[idx_tp, oci].to_numpy(float)
        hex_arrays["tp_oci_sat_y"] = np.asarray(oof_tp["y_pred"], dtype=float)

    # TP temporal
    m = df["target_TP"].notna()
    year_col = "trend_lat_x_year" if "trend_lat_x_year" in df.columns else "year"
    hex_arrays["tp_year_x"] = df.loc[m, year_col].to_numpy(float)
    hex_arrays["tp_year_y"] = df.loc[m, "target_TP"].to_numpy(float)
    payload["tp_year_feature"] = year_col

    # COD_Mn: lat, year, ndre
    m = df["target_CODMn"].notna()
    hex_arrays["cod_lat_x"] = df.loc[m, "lat"].to_numpy(float)
    hex_arrays["cod_lat_y"] = df.loc[m, "target_CODMn"].to_numpy(float)
    hex_arrays["cod_year_x"] = df.loc[m, "year"].to_numpy(float)
    hex_arrays["cod_year_y"] = df.loc[m, "target_CODMn"].to_numpy(float)
    if ndre:
        m2 = m & df[ndre].notna()
        hex_arrays["cod_ndre_x"] = df.loc[m2, ndre].to_numpy(float)
        hex_arrays["cod_ndre_y"] = df.loc[m2, "target_CODMn"].to_numpy(float)
        payload["cod_ndre_spearman"] = float(stats.spearmanr(
            hex_arrays["cod_ndre_x"], hex_arrays["cod_ndre_y"],
        ).statistic)
    oof_c = _oof(tag, "station", "ADAE", "CODMn")
    idx_c = np.asarray(oof_c["row_index"], dtype=int)
    hex_arrays["cod_lat_sat_x"] = df.loc[idx_c, "lat"].to_numpy(float)
    hex_arrays["cod_lat_sat_y"] = np.asarray(oof_c["y_pred"], dtype=float)

    payload["n_rows"] = int(len(df))
    np.savez(q10_out / "q10_arrays.npz", **arrays)
    (q10_out / "q10.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=float), encoding="utf-8",
    )
    np.savez(mech_out / "hexbin_arrays.npz", **hex_arrays)
    print(f"  Q10 in-situ={payload.get('insitu_NH3N', {}).get('effective_Q10')}  "
          f"sat={payload.get('satellite_S_NH3N', {}).get('effective_Q10')}")
    print(f"  q10 -> {q10_out}")
    print(f"  hex -> {mech_out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="rev1")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    args = ap.parse_args()
    tag = args.tag

    print("  load spectral table for ICC/Moran …", flush=True)
    df_spec, _ = load_spectral_data(file_path=args.data)
    write_structure(tag, df_spec)

    print("  conformal from OOF …", flush=True)
    write_conformal(tag)

    print("  load engineered table for Q10 / hexbins …", flush=True)
    df, _ = load_engineered_dataset(
        feature_subset="full",
        spectral_file=args.data,
    )
    write_q10_and_hex(tag, df)
    print("  figure inputs done")


if __name__ == "__main__":
    main()
