"""Read DONE cells → per_fold / fold_mean / pooled / R−S / oof.npz."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.rev1.folds import fold_keys as _fold_keys
from src.rev1.paths import METHODS, PROTOCOLS, TARGETS, cell_dir, combo_dir, rev1_dir


def _load_y(cell: Path):
    z = np.load(cell / "y.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


def write_oof_if_ready(tag: str, protocol: str, method: str, target: str,
                       expected_n: int | None = None) -> Path | None:
    keys = _fold_keys(protocol)
    cells = [cell_dir(tag, protocol, method, target, k) for k in keys]
    if not all((c / "DONE").exists() and (c / "y.npz").exists() for c in cells):
        return None
    parts = [_load_y(c) for c in cells]
    keys_arr = ("y_true", "y_pred", "fold", "station", "year", "month", "row_index")
    cat = {k: np.concatenate([p[k] for p in parts]) for k in keys_arr}
    _, uniq = np.unique(cat["row_index"], return_index=True)
    if len(uniq) != len(cat["row_index"]):
        for k in keys_arr:
            cat[k] = cat[k][np.sort(uniq)]
    if expected_n is not None and protocol != "temporal" and len(cat["row_index"]) != expected_n:
        print(
            f"  oof skip {protocol}/{method}/{target}: n={len(cat['row_index'])} expected={expected_n}",
            flush=True,
        )
        return None
    dests = [
        combo_dir(tag, protocol, method, target) / "oof.npz",
        rev1_dir(tag) / "summaries" / "oof" / f"{protocol}_{method}_{target}.npz",
    ]
    for dest in dests:
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            dest, **cat,
            protocol=np.asarray(protocol), method=np.asarray(method), target=np.asarray(target),
        )
    return dests[1]


def collect_cells(tag: str) -> list[dict]:
    rows = []
    root = rev1_dir(tag) / "cells"
    if not root.exists():
        return rows
    for protocol in PROTOCOLS:
        for method in METHODS:
            for target in TARGETS:
                for k in _fold_keys(protocol):
                    cell = cell_dir(tag, protocol, method, target, k)
                    if not (cell / "DONE").exists() or not (cell / "metrics.json").exists():
                        continue
                    m = json.loads((cell / "metrics.json").read_text(encoding="utf-8"))
                    rows.append(m)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="rev1")
    args = ap.parse_args()
    tag = args.tag
    rows = collect_cells(tag)
    if not rows:
        print("  no DONE cells")
        return
    per = pd.DataFrame(rows)
    out = rev1_dir(tag) / "summaries"
    out.mkdir(parents=True, exist_ok=True)
    per.to_csv(out / "per_fold.csv", index=False, encoding="utf-8-sig")

    mean_rows, pool_rows = [], []
    for (protocol, method, target), g in per.groupby(["protocol", "method", "target"]):
        mean_rows.append({
            "protocol": protocol, "method": method, "target": target,
            "n_folds": int(g["fold"].nunique()),
            "R2_fold_mean": float(g["R2"].mean()),
            "R2_fold_std": float(g["R2"].std(ddof=1)) if len(g) > 1 else 0.0,
            "MAE_fold_mean": float(g["MAE"].mean()),
            "MAE_fold_std": float(g["MAE"].std(ddof=1)) if len(g) > 1 else 0.0,
        })
        ys, ps = [], []
        for _, r in g.iterrows():
            z = _load_y(cell_dir(tag, protocol, method, target, r["fold_key"] if "fold_key" in r else r["fold"]))
            ys.append(z["y_true"])
            ps.append(z["y_pred"])
        y, p = np.concatenate(ys), np.concatenate(ps)
        pool_rows.append({
            "protocol": protocol, "method": method, "target": target,
            "R2_pooled": float(r2_score(y, p)),
            "MAE_pooled": float(mean_absolute_error(y, p)),
            "n": int(len(y)),
        })
        write_oof_if_ready(tag, protocol, method, target)

    fm = pd.DataFrame(mean_rows)
    po = pd.DataFrame(pool_rows)
    fm.to_csv(out / "fold_mean.csv", index=False, encoding="utf-8-sig")
    po.to_csv(out / "pooled.csv", index=False, encoding="utf-8-sig")

    gap_rows = []
    for method in METHODS:
        for target in TARGETS:
            r = fm[(fm.method == method) & (fm.target == target) & (fm.protocol == "random")]
            s = fm[(fm.method == method) & (fm.target == target) & (fm.protocol == "station")]
            if len(r) and len(s) and int(r.iloc[0]["n_folds"]) == 5 and int(s.iloc[0]["n_folds"]) == 5:
                gap_rows.append({
                    "method": method, "target": target,
                    "R2_R": float(r.iloc[0]["R2_fold_mean"]),
                    "R2_S": float(s.iloc[0]["R2_fold_mean"]),
                    "R_minus_S": float(r.iloc[0]["R2_fold_mean"] - s.iloc[0]["R2_fold_mean"]),
                })
    pd.DataFrame(gap_rows).to_csv(out / "r_minus_s.csv", index=False, encoding="utf-8-sig")

    print(fm.to_string(index=False))
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
