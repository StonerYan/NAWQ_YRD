"""
Fig. 5 — Test-design structure: ICC1 vs fold-mean R−S, plus Moran's I.

Input : results/rev1/summaries/r_minus_s.csv
        results/rev1/downstream/structure.json
Output: figure/output/fig5_structure.(png|pdf|svg)
        (alias fig5_inflation for the existing manuscript link)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from _style import (
    COL2,
    DOWN,
    METHOD_ORDER,
    PARAM_LABEL,
    TARGET_COLORS,
    TARGET_ORDER,
    load_r_minus_s,
    panel_label,
    save,
    set_style,
)

LEGEND_KW = dict(frameon=True, fancybox=False, framealpha=1.0,
                 edgecolor="0.35", facecolor="white")

set_style()
gap = load_r_minus_s()
struct = json.loads((DOWN / "structure.json").read_text(encoding="utf-8"))
icc_vals = {t: struct["icc"][t]["ICC1"] for t in TARGET_ORDER}
moran = struct["morans_I"]

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(COL2, COL2 * 0.42))
fig.subplots_adjust(wspace=0.38)

for method in METHOD_ORDER:
    for target in TARGET_ORDER:
        row = gap[(gap.method == method) & (gap.target == target)]
        if row.empty:
            continue
        g = float(row.iloc[0]["R_minus_S"])
        hero = method == "ADAE"
        ax_a.scatter(
            icc_vals[target], g,
            color=TARGET_COLORS[target],
            marker="D" if hero else "o",
            s=70 if hero else 22,
            alpha=0.95 if hero else 0.55,
            edgecolors="black" if hero else "none",
            linewidths=0.8, zorder=5 if hero else 3,
        )

for target in TARGET_ORDER:
    ys = gap.loc[gap.target == target, "R_minus_S"].astype(float)
    ax_a.text(icc_vals[target] + 0.012, float(ys.mean()),
              PARAM_LABEL[target], fontsize=7, color=TARGET_COLORS[target],
              fontweight="bold", va="center")

x_ref = np.linspace(0.0, 0.75, 80)
ax_a.plot(x_ref, x_ref, "k:", lw=0.9)
ax_a.set_xlabel(r"ICC$_1$ (between-station variance fraction)")
ax_a.set_ylabel(r"Random − station-out $\Delta R^2$ (fold mean)")
ax_a.set_xlim(0.05, 0.80)
ymax = float(gap["R_minus_S"].max()) + 0.06
ax_a.set_ylim(-0.05, max(0.62, ymax))
ax_a.legend(
    handles=[
        plt.scatter([], [], marker="D", s=50, color="grey",
                    edgecolors="black", linewidths=0.7, label="ADAE"),
        plt.scatter([], [], marker="o", s=18, color="grey",
                    edgecolors="none", label="Other models"),
        Line2D([], [], color="k", linestyle=":", lw=0.9, label="1:1"),
    ],
    fontsize=6.0, loc="lower right", **LEGEND_KW,
)
panel_label(ax_a, "a")

y_pos = [0, 1, 2]
moranI = [moran[t]["morans_I"] for t in TARGET_ORDER]
E_I = moran[TARGET_ORDER[0]]["expected_I"]
ci_lo = min(moran[t]["perm_p025"] for t in TARGET_ORDER)
ci_hi = max(moran[t]["perm_p975"] for t in TARGET_ORDER)
ax_b.barh(y_pos, moranI, height=0.55,
          color=[TARGET_COLORS[t] for t in TARGET_ORDER], alpha=0.78, zorder=3)
ax_b.axvline(0, color="black", lw=1.0, zorder=4)
ax_b.axvline(E_I, color="grey", lw=0.8, linestyle="--", zorder=3)
ax_b.axvspan(ci_lo, ci_hi, color="grey", alpha=0.12, zorder=1)
ax_b.set_yticks(y_pos)
ax_b.set_yticklabels([PARAM_LABEL[t] for t in TARGET_ORDER])
ax_b.set_xlabel("Moran's I (station-mean field)")
span = max(abs(ci_lo), abs(ci_hi), max(abs(v) for v in moranI)) + 0.08
ax_b.set_xlim(-0.4, span)
ax_b.legend(
    handles=[
        Line2D([0], [0], color="black", lw=1.0, label=r"$I = 0$"),
        Line2D([0], [0], color="grey", lw=0.8, linestyle="--",
               label=rf"E[$I$] = {E_I:.2f}"),
        Patch(facecolor="grey", alpha=0.12, edgecolor="none",
              label="95% permutation null"),
    ],
    fontsize=6.0, loc="upper right", **LEGEND_KW,
)
for i, target in enumerate(TARGET_ORDER):
    p_val = moran[target]["p_perm"]
    xv = moranI[i]
    ax_b.text(xv - 0.012 if xv < 0 else xv + 0.012, i,
              f"I = {xv:.3f}\np = {p_val:.2f}",
              va="center", ha="right" if xv < 0 else "left", fontsize=6.0)
panel_label(ax_b, "b")

save(fig, "fig5_structure", aliases=("fig5_inflation",))
plt.close(fig)
print("fig5_structure done")
