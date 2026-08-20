"""Stage 4: grouped permutation importance on official station-out ADAE folds.

Writes results/rev1/downstream/attribution/{domain,band,function}/category_{T}_station.csv
and per-fold tables. Does not write into results/analysis/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[2]
# Running as a file puts src/rev1 on sys.path and shadows stdlib `select`.
_here = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p).resolve() != _here]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.analysis.category_data_stats import category_data_correlation
from src.analysis.grouping_schemes import ATTR_SCHEMES, get_scheme
from src.config import ATTR_PERM_REPEATS, ATTR_SEED
from src.models.fair_multiseed_eval_scheme_b import SplitData
from src.models.feature_engineering import load_engineered_dataset
from src.rev1.folds import fold_keys, split_indices
from src.rev1.paths import DEFAULT_DATA, SEED, SELECT_K, SELECT_SCHEME, TARGETS, cell_dir, rev1_dir
from src.rev1.preprocess import fit_or_load_prep
from src.rev1.station_predict import StationAdaePredict, fit_station_predictor


def _attach(pred: StationAdaePredict, df, test_stations, te_idx, sta_te):
    pred._df = df
    pred._test_stations = list(test_stations)
    pred._te_idx = list(te_idx)
    pred._sta_te = np.asarray(sta_te).astype(str)
    return pred


def _perm_groups(predictor, X, y, cat_idx, n_repeats, seed):
    rng = np.random.default_rng(seed)
    y0 = predictor.predict(X)
    r2_base = float(r2_score(y, y0))
    out = {}
    for name, cols in cat_idx.items():
        if not cols:
            continue
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            for j in cols:
                Xp[:, j] = X[perm, j]
            drops.append(r2_base - float(r2_score(y, predictor.predict(Xp))))
        out[name] = {
            "importance": float(np.mean(drops)),
            "importance_std": float(np.std(drops, ddof=1)) if n_repeats > 1 else 0.0,
            "n_features": len(cols),
        }
    return out, r2_base


def _build_split(df, fc, protocol, target, fold_key, tag, scheme, k):
    folds_dir = rev1_dir(tag) / "folds"
    tr_idx, te_idx, fold_label = split_indices(df, fc, protocol, target, fold_key, folds_dir)
    prep = fit_or_load_prep(df, fc, protocol, target, fold_key, tr_idx, tag, k=k, scheme=scheme)
    feats = prep.features
    X_tr = prep.transform(df.loc[tr_idx, feats].to_numpy(float))
    X_te = prep.transform(df.loc[te_idx, feats].to_numpy(float))
    y_tr = df.loc[tr_idx, f"target_{target}"].to_numpy(float)
    y_te = df.loc[te_idx, f"target_{target}"].to_numpy(float)
    use_log = target in ("TP", "NH3N")
    y_fit = np.log1p(np.maximum(y_tr, 0)) if use_log else y_tr.copy()
    sta_tr = df.loc[tr_idx, "station_name"].astype(str).to_numpy()
    sta_te = df.loc[te_idx, "station_name"].astype(str).to_numpy()
    year_tr = df.loc[tr_idx, "year"].astype(int).to_numpy()
    d_tr, nd = prep.domain_labels(sta_tr, year_tr, X_tr)
    sd = SplitData(
        X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te, y_fit=y_fit,
        d_tr=d_tr, nd=nd, tr_idx=list(tr_idx), te_idx=list(te_idx),
        use_log=use_log, target=target, protocol=protocol,
        groups_tr=sta_tr, feat_names=list(feats), sta_te=sta_te,
    )
    return sd, prep, int(fold_label), sorted(set(sta_te.tolist()))


def _load_or_fit(pred_path: Path, sd, seed, lite, df, test_stations, force: bool):
    if pred_path.exists() and not force:
        pred = joblib.load(pred_path)
        pred = _attach(pred, df, test_stations, sd.te_idx, sd.sta_te)
        yp = pred.predict(sd.X_te)
        r2 = float(r2_score(sd.y_te, yp))
        print(f"    loaded predictor  R2={r2:+.4f}", flush=True)
        return pred, r2
    t0 = time.time()
    pred = fit_station_predictor(sd, seed, lite, df, test_stations)
    r2 = float(r2_score(sd.y_te, pred.predict(sd.X_te)))
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    # drop live df before dump
    pred._df = None
    joblib.dump(pred, pred_path)
    pred = _attach(pred, df, test_stations, sd.te_idx, sd.sta_te)
    print(f"    fitted predictor  R2={r2:+.4f}  {time.time() - t0:.0f}s", flush=True)
    return pred, r2


def _aggregate(fold_rows: list[dict], data_stats: dict, scheme) -> pd.DataFrame:
    df = pd.DataFrame(fold_rows)
    rows = []
    for cat, g in df.groupby("category"):
        rows.append({
            "category": cat,
            "label": scheme.labels.get(cat, cat),
            "perm_delta_r2": float(g["perm_delta_r2"].mean()),
            "perm_std": float(g["perm_delta_r2"].std(ddof=1)) if len(g) > 1 else 0.0,
            "n_features": int(g["n_features"].median()),
            "n_folds": int(len(g)),
            "data_mean_abs_spearman": data_stats.get(cat, {}).get("mean_abs_spearman", 0.0),
            "data_max_abs_spearman": data_stats.get(cat, {}).get("max_abs_spearman", 0.0),
            "data_top_feature": data_stats.get(cat, {}).get("top_feature"),
        })
    return pd.DataFrame(rows).sort_values("perm_delta_r2", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="rev1")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--protocol", default="station")
    ap.add_argument("--schemes", nargs="+", default=list(ATTR_SCHEMES), choices=list(ATTR_SCHEMES))
    ap.add_argument("--targets", nargs="+", default=list(TARGETS), choices=list(TARGETS))
    ap.add_argument("--folds", nargs="+", default=None)
    ap.add_argument("--n-repeats", type=int, default=ATTR_PERM_REPEATS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--lite", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.protocol != "station":
        raise SystemExit("Stage 4 official attribution is station-out only")

    dest = rev1_dir(args.tag) / "downstream" / "attribution"
    dest.mkdir(parents=True, exist_ok=True)
    pred_root = dest / "predictors"
    pred_root.mkdir(exist_ok=True)

    print("=== load engineered ===", flush=True)
    df, fc = load_engineered_dataset(feature_subset="full", spectral_file=args.data)
    keys = [int(x) for x in args.folds] if args.folds else list(fold_keys("station"))

    cat_data = {
        t: {s: category_data_correlation(df, fc, t, scheme=s) for s in args.schemes}
        for t in args.targets
    }

    journal = dest / "_fold_rows.jsonl"
    done_pairs = set()
    if journal.exists() and not args.force:
        for line in journal.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            done_pairs.add((rec["target"], int(rec["fold"]), rec["scheme"]))

    fold_store: dict[str, list] = {s: [] for s in args.schemes}
    if journal.exists() and not args.force:
        for line in journal.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["scheme"] in fold_store:
                fold_store[rec["scheme"]].append(rec)
    sanity = []
    if (dest / "refit_sanity.csv").exists() and not args.force:
        sanity = pd.read_csv(dest / "refit_sanity.csv").to_dict("records")

    jf = journal.open("a", encoding="utf-8") if not args.force else journal.open("w", encoding="utf-8")
    try:
        for target in args.targets:
            for fold_key in keys:
                need = [(target, int(fold_key), s) for s in args.schemes]
                if all(p in done_pairs for p in need) and not args.force:
                    print(f"\n== {target} f{fold_key} skip (journal) ==", flush=True)
                    continue
                print(f"\n== {target} f{fold_key} ==", flush=True)
                sd, prep, fold_label, test_stations = _build_split(
                    df, fc, "station", target, fold_key, args.tag, SELECT_SCHEME, SELECT_K,
                )
                cell = cell_dir(args.tag, "station", "ADAE", target, fold_key)
                official = None
                if (cell / "metrics.json").exists():
                    official = json.loads((cell / "metrics.json").read_text(encoding="utf-8")).get("R2")
                pred_path = pred_root / f"{target}_f{fold_key}.joblib"
                pred, r2 = _load_or_fit(pred_path, sd, args.seed, args.lite, df, test_stations, args.force)
                sanity.append({
                    "target": target, "fold": fold_label, "refit_R2": r2,
                    "official_R2": official, "n_test": int(len(sd.y_te)),
                    "n_features": len(sd.feat_names),
                })
                print(f"    official R2={official}  refit={r2:+.4f}", flush=True)

                for scheme_name in args.schemes:
                    if (target, int(fold_label), scheme_name) in done_pairs and not args.force:
                        print(f"    {scheme_name}: skip (journal)", flush=True)
                        continue
                    scheme = get_scheme(scheme_name)
                    cat_idx = scheme.category_indices(list(sd.feat_names))
                    imp, r2_base = _perm_groups(
                        pred, sd.X_te, sd.y_te, cat_idx, args.n_repeats, ATTR_SEED + fold_label,
                    )
                    for cat, rec in imp.items():
                        row = {
                            "scheme": scheme_name,
                            "target": target, "fold": fold_label, "category": cat,
                            "perm_delta_r2": rec["importance"],
                            "perm_std_repeats": rec["importance_std"],
                            "n_features": rec["n_features"],
                            "r2_base": r2_base,
                        }
                        fold_store[scheme_name].append(row)
                        jf.write(json.dumps(row, ensure_ascii=False) + "\n")
                        jf.flush()
                        done_pairs.add((target, int(fold_label), scheme_name))
                    top = sorted(imp.items(), key=lambda kv: -kv[1]["importance"])[:3]
                    print(
                        f"    {scheme_name}: "
                        + ", ".join(f"{scheme.labels.get(k, k)}={v['importance']:+.3f}" for k, v in top),
                        flush=True,
                    )

    finally:
        jf.close()
    pd.DataFrame(sanity).to_csv(dest / "refit_sanity.csv", index=False, encoding="utf-8-sig")
    summary_rows = []
    synthesis = {}
    for scheme_name in args.schemes:
        scheme = get_scheme(scheme_name)
        (dest / scheme_name).mkdir(exist_ok=True)
        raw = pd.DataFrame(fold_store[scheme_name])
        raw.to_csv(dest / scheme_name / "per_fold.csv", index=False, encoding="utf-8-sig")
        for target in args.targets:
            sub = raw[raw.target == target]
            tab = _aggregate(sub.to_dict("records"), cat_data[target][scheme_name], scheme)
            tab.to_csv(
                dest / scheme_name / f"category_{target}_station.csv",
                index=False, encoding="utf-8-sig",
            )
            if scheme_name == "function":
                tab.to_csv(dest / f"category_{target}_station.csv", index=False, encoding="utf-8-sig")
            print(f"\n[{scheme_name}/{target}]", flush=True)
            print(tab.head(5).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
            for rec in tab.head(3).to_dict("records"):
                summary_rows.append({
                    "scheme": scheme_name, "protocol": "station", "target": target, **rec,
                })

    for target in args.targets:
        hits = {}
        for row in summary_rows:
            if row["target"] != target:
                continue
            hits[row["label"]] = hits.get(row["label"], 0) + 1
        synthesis[target] = sorted(
            [{"label": k, "n_schemes_top3": v} for k, v in hits.items() if v >= 2],
            key=lambda x: -x["n_schemes_top3"],
        )

    payload = {
        "method": "rev1_station_grouped_permutation",
        "tag": args.tag,
        "n_repeats": args.n_repeats,
        "lite": args.lite,
        "schemes": args.schemes,
        "refit_sanity": sanity,
        "multilevel_top3": summary_rows,
        "cross_level_synthesis": synthesis,
    }
    (dest / "multilevel_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=float), encoding="utf-8",
    )
    print("\nStage 4 done →", dest)


if __name__ == "__main__":
    main()
