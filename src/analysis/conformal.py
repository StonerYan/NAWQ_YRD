"""
Split-conformal prediction intervals for WQI0627
================================================
ADAE 7-seed ensemble predictions on the held-out test set.
Primary: station-out (operational extrapolation).
Secondary: temporal (2024 holdout).

Outputs -> results/analysis/conformal.json (+ conformal_arrays.npz)

Run:
  cd g:/论文/WQI/WQI0627
  python src/analysis/conformal.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import TARGETS, CONFORMAL_ALPHA, CONFORMAL_PROTOCOLS, QA_THRESHOLDS

PRED = ROOT / "results" / "evaluation" / "predictions"
OUT = ROOT / "results" / "analysis"


def _conformal_one(y: np.ndarray, yhat: np.ndarray, alpha: float, seed: int) -> dict:
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
        "alpha": alpha,
        "qhat": qhat,
        "pi_width": 2 * qhat,
        "empirical_coverage": cover,
        "n_cal": int(n_cal),
        "n_eval": int(n - n_cal),
        "arrays": {
            "y": y[ev][order],
            "yhat": yhat[ev][order],
            "qhat": np.array([qhat]),
        },
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    alpha = CONFORMAL_ALPHA
    results = {"primary_protocol": "station", "protocols": {}}
    all_arrays = {}

    for protocol in CONFORMAL_PROTOCOLS:
        results["protocols"][protocol] = {}
        print(f"\n=== Conformal ({protocol}) ===")
        for target in TARGETS:
            f = PRED / f"ADAE_{target}_{protocol}.npz"
            if not f.exists():
                print(f"  [skip] {f.name} not found (run pipeline first)")
                continue
            d = np.load(f)
            y = d["y_true"]
            yhat = d["y_pred_ens"]
            cell = _conformal_one(y, yhat, alpha, seed=42)
            thr = QA_THRESHOLDS[target]
            results["protocols"][protocol][target] = {
                "alpha": alpha,
                "qhat": cell["qhat"],
                "pi_width": cell["pi_width"],
                "empirical_coverage": cell["empirical_coverage"],
                "n_cal": cell["n_cal"],
                "n_eval": cell["n_eval"],
                "halfwidth_frac_of_threshold": cell["qhat"] / thr,
                "threshold": thr,
            }
            prefix = f"{protocol}__{target}"
            all_arrays[f"{prefix}__y"] = cell["arrays"]["y"]
            all_arrays[f"{prefix}__yhat"] = cell["arrays"]["yhat"]
            all_arrays[f"{prefix}__qhat"] = cell["arrays"]["qhat"]
            print(
                f"  {target}: coverage={cell['empirical_coverage']:.3f}, "
                f"qhat={cell['qhat']:.4f} ({cell['qhat']/thr:.1%} of Class III)"
            )

    if all_arrays:
        np.savez(OUT / "conformal_arrays.npz", **all_arrays)
    (OUT / "conformal.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"\n  -> {OUT}/conformal.json")


if __name__ == "__main__":
    main()
