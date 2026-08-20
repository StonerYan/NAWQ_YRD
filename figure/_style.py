"""
Shared publication style for WQI0627_k figures.

Official campaign: results/rev1  (union_nolucc N=50, nested HPO, 5-fold R/T/S).
Do not read results/evaluation/summary_table.csv or old results/analysis/ as current.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REV1 = ROOT / "results" / "rev1"
SUMM = REV1 / "summaries"
OOF_DIR = SUMM / "oof"
FOLDS = REV1 / "folds"
DOWN = REV1 / "downstream"
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(parents=True, exist_ok=True)

# Elsevier double-column
MM = 1.0 / 25.4
COL1 = 90 * MM
COL15 = 140 * MM
COL2 = 190 * MM

METHOD_COLORS = {
    "RF":   "#9DB4C0",
    "XGB":  "#E9A663",
    "CaB":  "#A7C4A0",
    "DANN": "#B9A2D6",
    "ADAE": "#1F4E79",
}
METHOD_ERRBAR_COLORS = {
    "RF":   "#5A7385",
    "XGB":  "#B8752F",
    "CaB":  "#5F8A58",
    "DANN": "#7E5FA8",
    "ADAE": "#0A2238",
}
PROTOCOL_COLORS = {
    "random":   "#4292C6",
    "temporal": "#FD8D3C",
    "station":  "#CB181D",
}
TARGET_COLORS = {
    "CODMn": "#2171B5",
    "NH3N":  "#238B45",
    "TP":    "#CB4B16",
}
FOLD_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"]
YEAR_COLORS = {
    2020: "#BDBDBD",
    2021: "#4C78A8",
    2022: "#54A24B",
    2023: "#F58518",
    2024: "#B279A2",
    2025: "#E45756",
}
DENSITY_CMAP = "YlGnBu"

PARAM_LABEL = {"CODMn": r"COD$_\mathrm{Mn}$", "NH3N": r"NH$_3$-N", "TP": "TP"}
PARAM_UNIT = {"CODMn": r"mg L$^{-1}$", "NH3N": r"mg L$^{-1}$", "TP": r"mg L$^{-1}$"}
PARAM_THRESH = {"CODMn": 6.0, "NH3N": 1.0, "TP": 0.2}
PROTOCOL_LABEL = {
    "random":   "Random (R)",
    "temporal": "Temporal (T)",
    "station":  "Station-out (S)",
}
METHOD_ORDER = ["RF", "XGB", "CaB", "DANN", "ADAE"]
TARGET_ORDER = ["CODMn", "NH3N", "TP"]
PROTOCOL_ORDER = ["random", "temporal", "station"]


def set_style() -> None:
    mpl.rcParams.update({
        "font.family":          "sans-serif",
        "font.sans-serif":      ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype":         "none",
        "pdf.fonttype":         42,
        "ps.fonttype":          42,
        "font.size":            8,
        "axes.titlesize":       9,
        "axes.labelsize":       8,
        "xtick.labelsize":      7,
        "ytick.labelsize":      7,
        "legend.fontsize":      7,
        "axes.spines.right":    False,
        "axes.spines.top":      False,
        "axes.linewidth":       0.8,
        "xtick.major.width":    0.8,
        "ytick.major.width":    0.8,
        "xtick.direction":      "out",
        "ytick.direction":      "out",
        "legend.frameon":       False,
        "figure.dpi":           150,
        "savefig.dpi":          600,
        "axes.axisbelow":       True,
        "lines.linewidth":      1.2,
        "mathtext.default":     "regular",
        "axes.grid":            True,
        "grid.linewidth":       0.4,
        "grid.alpha":           0.5,
        "grid.color":           "#CCCCCC",
    })


def panel_label(ax, letter: str, dx: float = -0.12, dy: float = 1.03,
                fontsize: float = 10) -> None:
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes, fontsize=fontsize,
            fontweight="bold", va="bottom", ha="left")


def save(fig, name: str, svg: bool = True, aliases: tuple[str, ...] = (),
         pad_inches: float = 0.04) -> None:
    targets = (name,) + tuple(aliases)
    kw = dict(bbox_inches="tight", pad_inches=pad_inches)
    for n in targets:
        fig.savefig(OUT / f"{n}.png", dpi=600, **kw)
        fig.savefig(OUT / f"{n}.pdf", **kw)
        if svg:
            fig.savefig(OUT / f"{n}.svg", **kw)
    extra = f" (+ {', '.join(aliases)})" if aliases else ""
    print(f"  Saved figure/output/{name}.(png|pdf{'|svg' if svg else ''}){extra}")


def load_fold_mean(tag: str = "rev1") -> pd.DataFrame:
    path = ROOT / "results" / tag / "summaries" / "fold_mean.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run summarise.py --tag {tag}")
    return pd.read_csv(path)


def load_r_minus_s(tag: str = "rev1") -> pd.DataFrame:
    path = ROOT / "results" / tag / "summaries" / "r_minus_s.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    return pd.read_csv(path)


def load_per_fold(tag: str = "rev1") -> pd.DataFrame:
    return pd.read_csv(ROOT / "results" / tag / "summaries" / "per_fold.csv")


def oof_path(protocol: str, method: str, target: str, tag: str = "rev1") -> Path:
    return ROOT / "results" / tag / "summaries" / "oof" / f"{protocol}_{method}_{target}.npz"


def load_oof(protocol: str, method: str, target: str, tag: str = "rev1") -> dict:
    path = oof_path(protocol, method, target, tag)
    if not path.exists():
        alt = ROOT / "results" / tag / "cells" / protocol / method / target / "oof.npz"
        path = alt
    if not path.exists():
        raise FileNotFoundError(f"missing OOF for {protocol}/{method}/{target}")
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def fold_mean_r2(df: pd.DataFrame, protocol: str, method: str, target: str) -> tuple[float, float]:
    row = df[(df.protocol == protocol) & (df.method == method) & (df.target == target)]
    if row.empty:
        raise KeyError(f"no fold-mean row {protocol}/{method}/{target}")
    return float(row.iloc[0]["R2_fold_mean"]), float(row.iloc[0]["R2_fold_std"])


def fold_mean_mae(df: pd.DataFrame, protocol: str, method: str, target: str) -> tuple[float, float]:
    row = df[(df.protocol == protocol) & (df.method == method) & (df.target == target)]
    return float(row.iloc[0]["MAE_fold_mean"]), float(row.iloc[0]["MAE_fold_std"])


def density_z(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    from scipy.stats import gaussian_kde
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    z = np.full(x.shape, 0.5, dtype=float)
    if m.sum() < 8:
        return z
    try:
        xy = np.vstack([x[m], y[m]])
        kde = gaussian_kde(xy)
        zz = kde(xy)
        zz = (zz - zz.min()) / (zz.max() - zz.min() + 1e-12)
        z[m] = zz
    except Exception:
        pass
    return z
