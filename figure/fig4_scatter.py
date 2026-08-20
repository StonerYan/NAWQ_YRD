"""
Fig. 4 — ADAE OOF observed vs predicted (3 indicators × 3 designs).

Points: five-fold OOF (each observation once). Colour = local density + colorbar.
Corner statistic: fold-mean R² ± SD, not the pooled cloud R².

Input : results/rev1/summaries/oof + fold_mean.csv
Output: figure/output/fig4_scatter.(png|pdf|svg)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from _style import (
    COL2,
    PARAM_LABEL,
    PARAM_UNIT,
    PROTOCOL_LABEL,
    PROTOCOL_ORDER,
    TARGET_ORDER,
    density_z,
    fold_mean_mae,
    fold_mean_r2,
    load_fold_mean,
    load_oof,
    panel_label,
    save,
    set_style,
)

# Match WQI0627 Fig. 4: plasma_r density, axes start at 0, one square window per row.
CMAP = "plasma_r"
AXIS_LIM = {"CODMn": (0.0, 8.0), "NH3N": (0.0, 2.4), "TP": (0.0, 0.50)}

set_style()
fm = load_fold_mean()
LETTERS = "abcdefghi"

fig, axes = plt.subplots(3, 3, figsize=(COL2, COL2 * 0.98))
fig.subplots_adjust(wspace=0.32, hspace=0.38, right=0.88)

for ri, target in enumerate(TARGET_ORDER):
    lo, hi = AXIS_LIM[target]
    for ci, protocol in enumerate(PROTOCOL_ORDER):
        ax = axes[ri, ci]
        letter = LETTERS[ri * 3 + ci]
        oof = load_oof(protocol, "ADAE", target)
        y_true = np.asarray(oof["y_true"], dtype=float)
        y_pred = np.asarray(oof["y_pred"], dtype=float)
        z = density_z(y_pred, y_true)
        r2, r2sd = fold_mean_r2(fm, protocol, "ADAE", target)
        mae, _ = fold_mean_mae(fm, protocol, "ADAE", target)

        ax.scatter(y_pred, y_true, c=z, cmap=CMAP,
                   s=3, alpha=0.65, linewidths=0, rasterized=True, vmin=0, vmax=1)

        ax.plot([lo, hi], [lo, hi], "k--", lw=0.9, zorder=3)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")

        ax.text(
            0.05, 0.95,
            rf"$R^2$ = {r2:.3f} ± {r2sd:.3f}" + "\n" + f"MAE = {mae:.3f}",
            transform=ax.transAxes, fontsize=6.2, va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.78),
        )
        if ri == 0:
            ax.set_title(PROTOCOL_LABEL[protocol], fontsize=7.5, fontweight="bold")
        if ci == 0:
            ax.set_ylabel(
                f"Observed\n{PARAM_LABEL[target]} ({PARAM_UNIT[target]})", fontsize=7,
            )
        if ri == 2:
            ax.set_xlabel(f"Predicted ({PARAM_UNIT[target]})", fontsize=7)
        panel_label(ax, letter)

cax = fig.add_axes([0.905, 0.15, 0.016, 0.70])
cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=CMAP), cax=cax)
cb.set_label("Point density (scaled 0–1)", fontsize=7)
cb.ax.tick_params(labelsize=6.5)

save(fig, "fig4_scatter")
plt.close(fig)
print("fig4_scatter done")
