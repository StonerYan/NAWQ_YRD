"""
Fig. 3 — Fold-mean test R² ± SD for 5 models × 3 designs × 3 indicators.

Input : results/rev1/summaries/fold_mean.csv
Output: figure/output/fig3_performance.(png|pdf|svg)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from _style import (
    COL2,
    METHOD_COLORS,
    METHOD_ERRBAR_COLORS,
    METHOD_ORDER,
    PARAM_LABEL,
    PROTOCOL_LABEL,
    PROTOCOL_ORDER,
    TARGET_ORDER,
    load_fold_mean,
    panel_label,
    save,
    set_style,
)

set_style()
df = load_fold_mean()

fig, axes = plt.subplots(1, 3, figsize=(COL2, COL2 * 0.42), sharey=False)
fig.subplots_adjust(wspace=0.30)

N_METH = len(METHOD_ORDER)
Y_SPACE = 1.0
Y_METH = 0.20

for ax, target in zip(axes, TARGET_ORDER):
    sub = df[df["target"] == target]
    ytick_pos, ytick_lab = [], []

    for pi, protocol in enumerate(PROTOCOL_ORDER):
        base_y = pi * (N_METH * Y_METH + Y_SPACE)
        psub = sub[sub["protocol"] == protocol]
        for mi, method in enumerate(METHOD_ORDER):
            row = psub[psub["method"] == method]
            if row.empty:
                continue
            r2 = float(row["R2_fold_mean"].iloc[0])
            r2sd = float(row["R2_fold_std"].iloc[0])
            y = base_y + mi * Y_METH
            col = METHOD_COLORS[method]
            err = METHOD_ERRBAR_COLORS[method]
            hero = method == "ADAE"
            ax.plot([0, r2], [y, y], color=col,
                    lw=1.8 if hero else 0.9,
                    alpha=1.0 if hero else 0.65, zorder=3)
            ax.plot(r2, y, "o", color=col,
                    markersize=8 if hero else 5,
                    markeredgecolor="white" if hero else col,
                    markeredgewidth=1.0 if hero else 0, zorder=4)
            if r2sd > 0:
                ax.errorbar(r2, y, xerr=r2sd, fmt="none", color=err,
                            elinewidth=1.0, capsize=2.5, capthick=1.0, zorder=6)

        ytick_pos.append(base_y + (N_METH - 1) * Y_METH / 2)
        ytick_lab.append(PROTOCOL_LABEL[protocol])

    ax.axvline(0, color="#888888", lw=0.6, linestyle="--", zorder=1)
    for pi in range(len(PROTOCOL_ORDER)):
        base_y = pi * (N_METH * Y_METH + Y_SPACE)
        if pi % 2 == 0:
            ax.axhspan(base_y - 0.15, base_y + N_METH * Y_METH - 0.05,
                       color="#F5F5F5", zorder=0)

    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(ytick_lab)
    ax.set_xlabel(r"Fold-mean test $R^2$ ± SD")
    ax.set_title(PARAM_LABEL[target], fontweight="bold")
    ax.set_xlim(-0.2, 0.8)
    ax.invert_yaxis()
    ax.grid(axis="x", lw=0.4, alpha=0.5)
    ax.set_axisbelow(True)

panel_label(axes[0], "a")
panel_label(axes[1], "b")
panel_label(axes[2], "c")

handles = [
    mlines.Line2D([], [], color=METHOD_COLORS[m], marker="o", linestyle="None",
                  markersize=8 if m == "ADAE" else 5,
                  markeredgecolor="white" if m == "ADAE" else METHOD_COLORS[m],
                  markeredgewidth=1.0 if m == "ADAE" else 0, label=m)
    for m in METHOD_ORDER
]
fig.legend(handles=handles, loc="lower center", ncol=5,
           bbox_to_anchor=(0.5, -0.10), frameon=False, fontsize=7)

save(fig, "fig3_performance")
plt.close(fig)
print("fig3_performance done")
