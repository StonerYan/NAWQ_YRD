"""
Fig. 10 — TP optical pathway on the official 26-station table.

(a) OCI vs in-situ  (b) temporal feature vs in-situ
(c) OCI vs S OOF    (d) phytoplankton-P pathway (conceptual)
"""
import json
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
FS_TITLE, FS_LABEL, FS_TICK, FS_STAT, FS_FLOW = 10, 10, 9, 9, 8.5
COL = TARGET_COLORS["TP"]

q10 = json.loads((DOWN / "q10" / "q10.json").read_text(encoding="utf-8"))
hx = np.load(DOWN / "mech" / "hexbin_arrays.npz", allow_pickle=True)
if "tp_oci_x" not in hx.files:
    raise SystemExit("missing TP hexbin arrays; run src/rev1/run_figure_inputs.py")

fig, axes = plt.subplots(2, 2, figsize=(COL2, COL2 * 0.90))
fig.subplots_adjust(left=0.10, right=0.92, top=0.93, bottom=0.10, wspace=0.42, hspace=0.38)
ax_a, ax_b = axes[0]
ax_c, ax_d = axes[1]


def hex_panel(ax, x, y, xlabel, ylabel, title, letter):
    hb = ax.hexbin(x, y, gridsize=34, cmap="YlOrRd", mincnt=1, linewidths=0.15)
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


hex_panel(ax_a, hx["tp_oci_x"], hx["tp_oci_y"],
          r"S:OCI ($B_3$/$B_5$)", f"TP ({PARAM_UNIT['TP']})",
          "OCI vs in-situ TP", "a")
feat = q10.get("tp_year_feature", "year")
hex_panel(ax_b, hx["tp_year_x"], hx["tp_year_y"],
          feat.replace("_", " "), f"TP ({PARAM_UNIT['TP']})",
          "Temporal feature vs in-situ TP", "b")
hex_panel(ax_c, hx["tp_oci_sat_x"], hx["tp_oci_sat_y"],
          r"S:OCI ($B_3$/$B_5$)", f"Predicted TP ({PARAM_UNIT['TP']})",
          "OCI vs satellite TP (S OOF)", "c")

ax_d.set_xlim(0, 10)
ax_d.set_ylim(-0.3, 10)
ax_d.axis("off")
box_data = [
    (5, 8.8, "High TP\n(dissolved inorganic P)", "#FDBB84"),
    (5, 6.6, "Phytoplankton growth\n(P-limitation lifted)", "#FC8D59"),
    (5, 4.4, "Chlorophyll-a ↑\n(Redfield stoichiometry)", "#E34A33"),
    (5, 2.2, "Red-edge reflectance ↑\n(Sentinel-2 B3, B4, B5)", "#B30000"),
    (5, 0.0, "OCI / GRVI / NDRE\n→ satellite signal", COL),
]
for (xi, yi, txt, fc) in box_data:
    ax_d.text(xi, yi, txt, ha="center", va="center", fontsize=FS_FLOW,
              fontweight="bold", color="black" if fc == "#FDBB84" else "white",
              bbox=dict(boxstyle="round,pad=0.55", fc=fc, ec="white",
                        alpha=0.92, linewidth=0.8))
for i in range(len(box_data) - 1):
    ax_d.annotate("", xy=(5, box_data[i + 1][1] + 0.62),
                  xytext=(5, box_data[i][1] - 0.62),
                  arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.3))
ax_d.set_title("Phosphorus optical pathway\n(conceptual)", fontsize=FS_TITLE, pad=5)
panel_label(ax_d, "d", fontsize=11)

save(fig, "fig10_tp_mech")
plt.close(fig)
print("fig10_tp_mech done")
