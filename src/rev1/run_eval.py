"""Stage 1: nested-HPO R/T/S evaluation. Resume via DONE files."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.models.fair_multiseed_eval_scheme_b import SplitData
from src.models.feature_engineering import load_engineered_dataset
from src.rev1.bundle import cell_is_done, save_bundle
from src.rev1.folds import fold_keys, split_indices
from src.rev1.paths import (
    DEFAULT_DATA,
    METHODS,
    N_TUNE_TRIALS,
    PROTOCOLS,
    SEED,
    SELECT_K,
    SELECT_SCHEME,
    TARGETS,
    cell_dir,
    cell_id,
    combo_dir,
    rev1_dir,
)
from src.rev1.preprocess import fit_or_load_prep
from src.rev1.trainers import run_method


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


def _write_oof(tag, protocol, method, target, expected_n: int | None):
    from src.rev1.summarise import write_oof_if_ready
    write_oof_if_ready(tag, protocol, method, target, expected_n=expected_n)


def run_one_cell(args, df, fc, protocol, method, target, fold_key):
    tag = args.tag
    cid = cell_id(protocol, method, target, fold_key)
    cell = cell_dir(tag, protocol, method, target, fold_key)
    log_path = rev1_dir(tag) / "logs" / "cells" / f"{cid}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if cell_is_done(cell) and not args.force:
        print(f"  skip {cid} (DONE)", flush=True)
        return json.loads((cell / "metrics.json").read_text(encoding="utf-8"))

    if cell.exists():
        shutil.rmtree(cell)
    cell.mkdir(parents=True, exist_ok=True)

    lf = log_path.open("w", encoding="utf-8")
    old = sys.stdout
    sys.stdout = _Tee(old, lf)
    t0 = time.time()
    try:
        print(f"== {cid}  {time.strftime('%Y-%m-%d %H:%M:%S')} ==")
        folds_dir = rev1_dir(tag) / "folds"
        tr_idx, te_idx, fold_label = split_indices(
            df, fc, protocol, target, fold_key, folds_dir,
        )
        if len(te_idx) < 5 or len(tr_idx) < 30:
            raise RuntimeError(f"tiny split n_tr={len(tr_idx)} n_te={len(te_idx)}")
        print(f"  n_train={len(tr_idx)}  n_test={len(te_idx)}  fold_label={fold_label}")

        prep = fit_or_load_prep(
            df, fc, protocol, target, fold_key, tr_idx, tag,
            k=args.select_k, scheme=getattr(args, "scheme", SELECT_SCHEME),
        )
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
        year_te = df.loc[te_idx, "year"].astype(int).to_numpy()
        month_te = df.loc[te_idx, "month"].astype(int).to_numpy()
        d_tr, nd = prep.domain_labels(sta_tr, year_tr, X_tr)
        test_stations = sorted(set(sta_te.tolist())) if protocol == "station" else []

        sd = SplitData(
            X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te, y_fit=y_fit,
            d_tr=d_tr, nd=nd, tr_idx=list(tr_idx), te_idx=list(te_idx),
            use_log=use_log, target=target, protocol=protocol,
            groups_tr=sta_tr, feat_names=list(feats), sta_te=sta_te,
        )
        sd.years_tr = year_tr

        yp, predictor, best, inner_r2, trials, extra = run_method(
            method, sd, protocol, args.seed, args.n_jobs, args.n_trials,
            args.lite, df, test_stations,
        )
        r2 = float(r2_score(y_te, yp))
        mae = float(mean_absolute_error(y_te, yp))
        elapsed = round(time.time() - t0, 1)
        print(f"  R2={r2:+.4f}  MAE={mae:.4f}  inner={inner_r2}  {elapsed}s")
        print(f"  best={best}")

        y = {
            "y_true": np.asarray(y_te, dtype=np.float64),
            "y_pred": np.asarray(yp, dtype=np.float64),
            "fold": np.full(len(y_te), int(fold_label), dtype=np.int32),
            "station": np.asarray(sta_te, dtype=object),
            "year": np.asarray(year_te, dtype=np.int16),
            "month": np.asarray(month_te, dtype=np.int16),
            "row_index": np.asarray(te_idx, dtype=np.int32),
        }
        metrics = {
            "protocol": protocol, "method": method, "target": target,
            "fold": int(fold_label), "fold_key": fold_key,
            "R2": r2, "MAE": mae, "n": int(len(y_te)),
            "n_train": int(len(tr_idx)), "n_stations": int(len(set(sta_te))),
            "seed": args.seed, "elapsed_s": elapsed, "inner_r2": inner_r2,
            "cell_id": cid,
        }
        hparams = {
            "best": best, "n_trials": args.n_trials, "inner": "protocol_cv",
            "inner_best_r2": inner_r2, "seed": args.seed, "lite": args.lite,
            "grl_lambda": None if method in ("RF", "XGB", "CaB") else extra,
        }
        features = {
            "features": feats, "k": len(feats),
            "selector": getattr(args, "scheme", "rf"),
            "scheme": getattr(args, "scheme", "rf"),
            "protocol": protocol, "target": target, "fold": int(fold_label),
        }
        save_bundle(
            cell, prep=prep, predictor=predictor, hparams=hparams,
            features=features, metrics=metrics, y=y, inner_cv=trials, extra=extra,
        )
        from src.utils.grouped_splits import valid_mask
        expected = int(valid_mask(df, fc, target).sum()) if protocol != "temporal" else None
        _write_oof(tag, protocol, method, target, expected)
        return metrics
    except Exception:
        print(traceback.format_exc())
        raise
    finally:
        sys.stdout = old
        lf.close()


def _parse_only(s: str | None):
    if not s:
        return None
    out = {}
    for part in s.split(","):
        k, _, v = part.partition("=")
        out[k.strip()] = v.strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="rev1")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--encoding", default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--protocols", nargs="+", default=list(PROTOCOLS), choices=PROTOCOLS)
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    ap.add_argument("--targets", nargs="+", default=list(TARGETS), choices=TARGETS)
    ap.add_argument("--folds", nargs="+", default=None, help="fold keys, e.g. 0 1 or 2024")
    ap.add_argument("--n-trials", type=int, default=N_TUNE_TRIALS)
    ap.add_argument("--select-k", type=int, default=SELECT_K)
    ap.add_argument(
        "--scheme", default=SELECT_SCHEME,
        help="feature scheme: rf | union | union_nolucc | union50 | vif | spec",
    )
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--lite", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default=None, help="protocol=station,method=RF,target=TP,fold=0")
    ap.add_argument("--parallel-methods", action="store_true",
                    help="spawn one process per method (do not nest)")
    args = ap.parse_args()

    only = _parse_only(args.only)
    if only:
        if "protocol" in only:
            args.protocols = [only["protocol"]]
        if "method" in only:
            args.methods = [only["method"]]
        if "target" in only:
            args.targets = [only["target"]]
        if "fold" in only:
            args.folds = [only["fold"]]

    out = rev1_dir(args.tag)
    folds = out / "folds" / "station_folds.csv"
    if not folds.exists():
        src = rev1_dir("rev1") / "folds"
        if src.exists() and args.tag != "rev1":
            import shutil
            shutil.copytree(src, out / "folds", dirs_exist_ok=True)
        else:
            raise SystemExit("Stage 0 missing: run python src/rev1/freeze_folds.py first")

    if args.parallel_methods and len(args.methods) > 1:
        import subprocess
        procs = []
        for m in args.methods:
            cmd = [
                sys.executable, str(Path(__file__).resolve()),
                "--tag", args.tag, "--data", args.data, "--seed", str(args.seed),
                "--protocols", *args.protocols, "--methods", m,
                "--targets", *args.targets, "--n-trials", str(args.n_trials),
                "--select-k", str(args.select_k), "--n-jobs", str(args.n_jobs),
                "--scheme", args.scheme,
            ]
            if args.encoding:
                cmd += ["--encoding", args.encoding]
            if args.folds:
                cmd += ["--folds", *map(str, args.folds)]
            if args.lite:
                cmd.append("--lite")
            if args.force:
                cmd.append("--force")
            log = out / "logs" / f"stage1_{m}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            fh = log.open("w", encoding="utf-8")
            print(f"  spawn {m} -> {log}")
            procs.append((m, subprocess.Popen(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT), fh))
        rc = 0
        for m, p, fh in procs:
            p.wait()
            fh.close()
            print(f"  {m} exit={p.returncode}")
            rc = rc or p.returncode
        raise SystemExit(rc)

    import torch
    torch.set_num_threads(max(1, min(4, args.n_jobs)))

    df, fc = load_engineered_dataset(
        feature_subset="full", spectral_file=args.data, encoding=args.encoding,
    )
    print(
        f"rev1 eval  protocols={args.protocols}  methods={args.methods}  "
        f"targets={args.targets}  rows={len(df)}",
        flush=True,
    )

    jobs = []
    for protocol in args.protocols:
        keys = args.folds if args.folds is not None else fold_keys(protocol)
        keys = [int(k) for k in keys]
        for method in args.methods:
            for target in args.targets:
                for fk in keys:
                    jobs.append((protocol, method, target, fk))

    t0 = time.time()
    done = 0
    for protocol, method, target, fk in jobs:
        done += 1
        print(f"\n[{done}/{len(jobs)}] {protocol} {method} {target} f{fk}", flush=True)
        run_one_cell(args, df, fc, protocol, method, target, fk)
    print(f"\n  Stage 1 chunk done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
