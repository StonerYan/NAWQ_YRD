"""
Fig. S4 — OOF diagnostics that do not belong on the main Fig. 4.

(a) S by fold (3 × 5)
(b) T coloured by year (1 × 3)
(c) S station-mean scatter, 26 points (1 × 3)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from _style import (
    COL2,
    DENSITY_CMAP,
    PARAM_LABEL,
    PARAM_UNIT,
    TARGET_COLORS,
    TARGET_ORDER,
    YEAR_COLORS,
    density_z,
    load_oof,
    load_per_fold,
    panel_label,
    save,
    set_style,
)

set_style()
pf = load_per_fold()


def _limits(y, p):
    mn = min(float(np.nanmin(y)), float(np.nanmin(p)))
    mx = max(float(np.nanmax(y)), float(np.nanmax(p)))
    pad = (mx - mn) * 0.06
    return mn - pad, mx + pad


def _one_to_one(ax, lo, hi):
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.7)


def main():
    fig = plt.figure(figsize=(COL2, COL2 * 1.18))
    outer = fig.add_gridspec(
        3, 1, height_ratios=[2.35, 1.20, 1.20],
        left=0.07, right=0.985, top=0.965, bottom=0.06,
        hspace=0.16,
    )
    gs_a = outer[0].subgridspec(3, 5, hspace=0.16, wspace=0.18)
    gs_b = outer[1].subgridspec(1, 3, wspace=0.22)
    gs_c = outer[2].subgridspec(1, 3, wspace=0.22)

    axes_a = np.empty((3, 5), dtype=object)
    for ri in range(3):
        for fi in range(5):
            axes_a[ri, fi] = fig.add_subplot(gs_a[ri, fi])

    for ri, target in enumerate(TARGET_ORDER):
        oof = load_oof("station", "ADAE", target)
        y = np.asarray(oof["y_true"], dtype=float)
        p = np.asarray(oof["y_pred"], dtype=float)
        fold = np.asarray(oof["fold"], dtype=int)
        lo, hi = _limits(y, p)
        for fi in range(5):
            ax = axes_a[ri, fi]
            m = fold == fi
            ax.scatter(p[m], y[m], c=density_z(p[m], y[m]), cmap=DENSITY_CMAP,
                       s=4, alpha=0.75, linewidths=0, rasterized=True, vmin=0, vmax=1)
            _one_to_one(ax, lo, hi)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            row = pf[(pf.protocol == "station") & (pf.method == "ADAE")
                     & (pf.target == target) & (pf.fold == fi)]
            r2 = float(row.iloc[0]["R2"]) if len(row) else np.nan
            ax.text(0.05, 0.95, f"$R^2$={r2:.2f}", transform=ax.transAxes,
                    fontsize=5.8, va="top")
            if ri == 0:
                ax.set_title(f"Fold {fi}", fontsize=7, fontweight="bold", pad=2)
            if fi == 0:
                ax.set_ylabel(f"Obs. {PARAM_LABEL[target]}", fontsize=6.5)
            if ri == 2:
                ax.set_xlabel("Pred.", fontsize=6.5)
            ax.tick_params(labelsize=5.5)
    panel_label(axes_a[0, 0], "a", dx=-0.22, dy=1.08, fontsize=10)
    axes_a[0, 2].text(0.5, 1.28, "S out-of-fold by station fold",
                      transform=axes_a[0, 2].transAxes, ha="center", va="bottom",
                      fontsize=8, fontweight="bold", color="#444444")

    axes_b = [fig.add_subplot(gs_b[0, ci]) for ci in range(3)]
    for ci, target in enumerate(TARGET_ORDER):
        ax = axes_b[ci]
        oof = load_oof("temporal", "ADAE", target)
        y = np.asarray(oof["y_true"], dtype=float)
        p = np.asarray(oof["y_pred"], dtype=float)
        year = np.asarray(oof["year"], dtype=int)
        lo, hi = _limits(y, p)
        for yr in range(2021, 2026):
            m = year == yr
            ax.scatter(p[m], y[m], s=5, alpha=0.55, linewidths=0,
                       color=YEAR_COLORS[yr], rasterized=True)
        _one_to_one(ax, lo, hi)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.text(0.50, 0.97, PARAM_LABEL[target], transform=ax.transAxes,
                ha="center", va="top", fontsize=8, fontweight="bold",
                color=TARGET_COLORS[target], zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
        ax.set_xlabel(f"Predicted ({PARAM_UNIT[target]})", fontsize=6.5)
        if ci == 0:
            ax.set_ylabel("Observed", fontsize=7)
            panel_label(ax, "b", dx=-0.16, dy=0.98, fontsize=10)
        ax.tick_params(labelsize=6)

    handles = [Line2D([], [], marker="o", linestyle="None", markersize=4,
                      color=YEAR_COLORS[y], label=str(y)) for y in range(2021, 2026)]
    axes_b[2].legend(handles=handles, fontsize=5.5, loc="lower right",
                     title="Year", title_fontsize=5.5, frameon=True,
                     fancybox=False, edgecolor="0.7")

    axes_c = [fig.add_subplot(gs_c[0, ci]) for ci in range(3)]
    for ci, target in enumerate(TARGET_ORDER):
        ax = axes_c[ci]
        oof = load_oof("station", "ADAE", target)
        g = pd.DataFrame({
            "station": np.asarray(oof["station"]).astype(str),
            "y": np.asarray(oof["y_true"], dtype=float),
            "p": np.asarray(oof["y_pred"], dtype=float),
        }).groupby("station").mean(numeric_only=True)
        r, _ = spearmanr(g["y"], g["p"])
        lo, hi = _limits(g["y"].to_numpy(), g["p"].to_numpy())
        ax.scatter(g["p"], g["y"], s=28, color=TARGET_COLORS[target],
                   edgecolors="white", linewidths=0.5, zorder=3)
        _one_to_one(ax, lo, hi)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.text(0.50, 0.97, PARAM_LABEL[target], transform=ax.transAxes,
                ha="center", va="top", fontsize=8, fontweight="bold",
                color=TARGET_COLORS[target], zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
        ax.text(0.04, 0.86, rf"$n$=26  $|r|_S$={abs(r):.2f}",
                transform=ax.transAxes, fontsize=6.5, va="top")
        ax.set_xlabel("Station-mean predicted", fontsize=6.5)
        if ci == 0:
            ax.set_ylabel("Station-mean observed", fontsize=7)
            panel_label(ax, "c", dx=-0.16, dy=0.98, fontsize=10)
        ax.tick_params(labelsize=6)

    save(fig, "figS4_oof", pad_inches=0.02)
    plt.close(fig)
    print("figS4_oof done")


if __name__ == "__main__":
    main()
