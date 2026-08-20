"""
Fig. 8 — COD_Mn observational relationships (promoted from former Fig. S2).

(a) latitude vs in-situ  (b) year vs in-situ
(c) NDRE vs in-situ      (d) latitude vs S OOF
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from _style import (
    COL2,
    DOWN,
    PARAM_UNIT,
    TARGET_COLORS,
    panel_label,
    save,
    set_style,
)

set_style()
FS_TITLE, FS_LABEL, FS_TICK, FS_STAT = 10, 10, 9, 9
COL = TARGET_COLORS["CODMn"]
hx = np.load(DOWN / "mech" / "hexbin_arrays.npz", allow_pickle=True)

fig, axes = plt.subplots(2, 2, figsize=(COL2, COL2 * 0.90))
fig.subplots_adjust(left=0.10, right=0.92, top=0.93, bottom=0.10, wspace=0.42, hspace=0.38)


def hex_panel(ax, x, y, xlabel, ylabel, title, letter):
    hb = ax.hexbin(x, y, gridsize=32, cmap="Blues", mincnt=1, linewidths=0.12)
    cb = fig.colorbar(hb, ax=ax, label="Count", fraction=0.040, pad=0.02, shrink=0.82)
    cb.ax.tick_params(labelsize=FS_TICK)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() > 5:
        coef = np.polyfit(x[m], y[m], 1)
        xl = np.linspace(float(x[m].min()), float(x[m].max()), 100)
        ax.plot(xl, np.polyval(coef, xl), color="black", lw=1.2, linestyle="--")
        r = abs(float(spearmanr(x[m], y[m]).statistic))
    else:
        r = np.nan
    ax.set_xlabel(xlabel, fontsize=FS_LABEL)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_TITLE, pad=5)
    ax.tick_params(labelsize=FS_TICK)
    ax.text(0.58, 0.97, rf"$|r|_S$ = {r:.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=FS_STAT,
            bbox=dict(boxstyle="round,pad=0.30", fc="white", ec=COL, lw=0.9))
    panel_label(ax, letter, fontsize=11)


ylab = f"COD$_\\mathrm{{Mn}}$ ({PARAM_UNIT['CODMn']})"
yp_lab = f"Predicted COD$_\\mathrm{{Mn}}$ ({PARAM_UNIT['CODMn']})"

hex_panel(axes[0, 0], hx["cod_lat_x"], hx["cod_lat_y"],
          "Latitude (°N)", ylab, "Latitude vs in-situ", "a")
hex_panel(axes[0, 1], hx["cod_year_x"], hx["cod_year_y"],
          "Year", ylab, "Year vs in-situ", "b")
if "cod_ndre_x" in hx.files:
    hex_panel(axes[1, 0], hx["cod_ndre_x"], hx["cod_ndre_y"],
              "NDRE", ylab, "NDRE vs in-situ", "c")
else:
    axes[1, 0].text(0.5, 0.5, "NDRE not in table", ha="center", va="center")
    panel_label(axes[1, 0], "c", fontsize=11)
hex_panel(axes[1, 1], hx["cod_lat_sat_x"], hx["cod_lat_sat_y"],
          "Latitude (°N)", yp_lab, r"Latitude vs satellite COD$_{\mathrm{Mn}}$ (S)", "d")

save(fig, "fig8_codmn_mech")
plt.close(fig)
print("fig8_codmn_mech done")
