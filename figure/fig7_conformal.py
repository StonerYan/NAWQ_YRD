"""
Fig. 7 — Split-conformal 90% intervals on official rev1 ADAE OOF.

Input : results/rev1/downstream/conformal/
Output: figure/output/fig7_conformal.(png|pdf|svg)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from _style import (
    COL2,
    DOWN,
    PARAM_LABEL,
    PARAM_THRESH,
    PARAM_UNIT,
    TARGET_ORDER,
    panel_label,
    save,
    set_style,
)

set_style()

PANEL = {
    "CODMn": {"line": "#3C5A8B", "band": "#8FAFD4", "dot": "#5C7CAA", "tint": "#F4F7FB"},
    "NH3N":  {"line": "#2F7D68", "band": "#7FB8A8", "dot": "#4F9A86", "tint": "#F3F9F7"},
    "TP":    {"line": "#B85C38", "band": "#D9A08A", "dot": "#C97855", "tint": "#FBF5F2"},
}
THRESH_COLOR = "#8B4049"
THRESH_FMT = {"CODMn": "6.0", "NH3N": "1.0", "TP": "0.2"}
QHAT_FMT = {"CODMn": ".3f", "NH3N": ".3f", "TP": ".3f"}
PROTO_ROWS = ["station", "temporal"]
PROTO_LABEL = {"station": "Station-out (S)", "temporal": "Temporal (T)"}
ROW_TINT = {"station": "#FAFAFA", "temporal": "#FFFFFF"}

conf_dir = DOWN / "conformal"
if not (conf_dir / "conformal.json").exists():
    raise SystemExit("missing rev1 conformal; run src/rev1/run_figure_inputs.py")

with open(conf_dir / "conformal.json", encoding="utf-8") as fh:
    conf = json.load(fh)
arrays = np.load(conf_dir / "conformal_arrays.npz", allow_pickle=True)

fig, axes = plt.subplots(2, 3, figsize=(COL2, COL2 * 0.68))
fig.subplots_adjust(left=0.09, right=0.98, top=0.90, bottom=0.16, hspace=0.38, wspace=0.28)

for ci, target in enumerate(TARGET_ORDER):
    axes[0, ci].set_title(PARAM_LABEL[target], fontweight="bold", fontsize=9,
                          color=PANEL[target]["line"], pad=12)

letter_idx = 0
for ri, protocol in enumerate(PROTO_ROWS):
    for ci, target in enumerate(TARGET_ORDER):
        ax = axes[ri, ci]
        pal = PANEL[target]
        panel_label(ax, chr(ord("a") + letter_idx), dx=-0.13, dy=1.01, fontsize=9)
        letter_idx += 1
        ax.set_facecolor(ROW_TINT[protocol])
        for spine in ax.spines.values():
            spine.set_color("#D8D8D8")
            spine.set_linewidth(0.6)

        stats = conf["protocols"][protocol][target]
        qhat = float(stats["qhat"])
        coverage = float(stats["empirical_coverage"])
        frac_thr = float(stats["halfwidth_frac_of_threshold"])
        thresh = PARAM_THRESH[target]
        y_true = arrays[f"{protocol}__{target}__y"]
        y_pred = arrays[f"{protocol}__{target}__yhat"]
        x_idx = np.arange(len(y_pred))
        lo = np.clip(y_pred - qhat, 0, None)
        hi = y_pred + qhat

        ax.fill_between(x_idx, lo, hi, color=pal["band"], alpha=0.35, linewidth=0, zorder=1)
        ax.plot(x_idx, y_pred, color=pal["line"], lw=1.4, zorder=3)
        ax.scatter(x_idx, y_true, color=pal["dot"], s=5, alpha=0.28,
                   linewidths=0, rasterized=True, zorder=2)
        ax.axhline(thresh, color=THRESH_COLOR, lw=0.9, linestyle=(0, (4, 3)), zorder=4)

        ymax = max(float(np.max(hi)), float(np.max(y_true)), thresh) * 1.10
        ax.set_xlim(0, len(y_pred) - 1)
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.4, alpha=0.9)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7, colors="#444444")
        ax.text(len(y_pred) * 0.96, thresh + 0.015 * ymax,
                f"Class-III = {THRESH_FMT[target]}",
                ha="right", va="bottom", fontsize=7, color=THRESH_COLOR, zorder=6)
        ax.text(
            0.03, 0.97,
            f"{PROTO_LABEL[protocol]}\n"
            f"Coverage  {coverage * 100:.1f}%\n"
            rf"$\hat{{q}}$ = {qhat:{QHAT_FMT[target]}}  ({frac_thr * 100:.0f}% of Class-III)",
            transform=ax.transAxes, ha="left", va="top", fontsize=7, color="#333333",
            linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=pal["tint"],
                      edgecolor="none", alpha=0.95),
        )
        if ci == 0:
            ax.set_ylabel(f"Concentration ({PARAM_UNIT[target]})", fontsize=7.5)
        if ri == 1:
            ax.set_xlabel(r"Test sample (sorted by $\hat{y}$)", fontsize=7.5)

fig.legend(
    handles=[
        mlines.Line2D([], [], color="#555555", lw=1.4, label=r"Predicted $\hat{y}$"),
        mlines.Line2D([], [], marker="o", linestyle="None", markersize=4.5,
                      markerfacecolor="#888888", markeredgecolor="none", alpha=0.55,
                      label="Observed"),
        mpatches.Patch(facecolor="#A8BFD4", edgecolor="none", alpha=0.55,
                       label="90% conformal interval"),
        mlines.Line2D([], [], color=THRESH_COLOR, lw=0.9, linestyle=(0, (4, 3)),
                      label="GB3838-2002 Class-III threshold"),
    ],
    loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.01), fontsize=7.5,
    frameon=True, fancybox=False, edgecolor="#CCCCCC", facecolor="white",
)

save(fig, "fig7_conformal")
plt.close(fig)
print("fig7_conformal done")
