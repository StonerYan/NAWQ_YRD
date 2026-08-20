"""
Fig. 9 — NH3-N temperature and temporal pathway.

(a) In-situ and station-out Q10 overlaid
(b) year vs in-situ  (lead process group)
(c) month hexbin
(d) Arrhenius vs in-situ Q10
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
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
COL = TARGET_COLORS["NH3N"]
COL_SAT = "#1F4E79"
T_REF = 10.0

q10_dir = DOWN / "q10"
mech_dir = DOWN / "mech"
if not (q10_dir / "q10.json").exists():
    raise SystemExit("missing Q10 arrays; run src/rev1/run_figure_inputs.py")

meta = json.loads((q10_dir / "q10.json").read_text(encoding="utf-8"))
arr = np.load(q10_dir / "q10_arrays.npz", allow_pickle=True)
hx = np.load(mech_dir / "hexbin_arrays.npz", allow_pickle=True)
partial = meta.get("partial", {})


def q10_model(T, C0, Q10):
    return C0 * Q10 ** ((T - T_REF) / 10)


def _fit(bin_x, bin_y, fallback):
    try:
        popt, _ = curve_fit(q10_model, bin_x, bin_y, p0=[float(np.median(bin_y)), fallback],
                            maxfev=5000)
        T_fit = np.linspace(float(bin_x.min()) - 1, float(bin_x.max()) + 1, 200)
        return T_fit, q10_model(T_fit, *popt), float(popt[1])
    except Exception:
        T_fit = np.linspace(0, 32, 200)
        return T_fit, np.mean(bin_y) * (fallback ** ((T_fit - T_REF) / 10)), fallback


fig, axes = plt.subplots(2, 2, figsize=(COL2, COL2 * 0.86))
fig.subplots_adjust(left=0.09, right=0.90, top=0.93, bottom=0.10, wspace=0.42, hspace=0.36)
ax_a, ax_b = axes[0]
ax_c, ax_d = axes[1]

# (a) in-situ + station-out Q10 on one axes
bx, by, se = arr["insitu_NH3N__bin_T"], arr["insitu_NH3N__bin_y"], arr["insitu_NH3N__bin_se"]
q_in = float(meta["insitu_NH3N"]["effective_Q10"])
r_in = abs(float(meta["insitu_NH3N"]["spearman_r"]))
q_in_p = float(partial.get("insitu", {}).get("partial_Q10", np.nan))
T_fit, y_fit, _ = _fit(bx, by, q_in)
ax_a.scatter(bx, by, color=COL, s=52, zorder=4, edgecolors="white", linewidths=0.6,
             label="In-situ")
for x, y, s in zip(bx, by, se):
    ax_a.vlines(x, y - s, y + s, color=COL, lw=1.1, alpha=0.65)
ax_a.plot(T_fit, y_fit, color=COL, lw=2.2, zorder=5)

bx2, by2, se2 = arr["sat_S_NH3N__bin_T"], arr["sat_S_NH3N__bin_y"], arr["sat_S_NH3N__bin_se"]
q_sat = float(meta["satellite_S_NH3N"]["effective_Q10"])
r_sat = abs(float(meta["satellite_S_NH3N"]["spearman_r"]))
q_sat_p = float(partial.get("satellite_S", {}).get("partial_Q10", np.nan))
T2, y2, _ = _fit(bx2, by2, q_sat)
ax_a.scatter(bx2, by2, color=COL_SAT, s=44, marker="D", zorder=4,
             edgecolors="white", linewidths=0.6, label="Station-out")
for x, y, s in zip(bx2, by2, se2):
    ax_a.vlines(x, y - s, y + s, color=COL_SAT, lw=1.1, alpha=0.65)
ax_a.plot(T2, y2, color=COL_SAT, lw=2.2, linestyle="--", zorder=5)

ax_a.text(0.50, 0.97,
          r"In-situ $Q_{10}$" + f" = {q_in:.2f} ({q_in_p:.2f})\n"
          + r"Station-out $Q_{10}$" + f" = {q_sat:.2f} ({q_sat_p:.2f})\n"
          + f"n = {meta['insitu_NH3N']['n']}",
          transform=ax_a.transAxes, ha="left", va="top", fontsize=FS_STAT,
          bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="0.70", lw=0.8))
ax_a.legend(loc="lower left", fontsize=8, frameon=True, edgecolor="0.75",
            fancybox=False, borderpad=0.35)
ax_a.set_title(r"Temperature–NH$_3$-N", fontsize=FS_TITLE, pad=5)
ax_a.set_xlabel(r"7-day mean $T_{2\mathrm{m}}$ (°C)", fontsize=FS_LABEL)
ax_a.set_ylabel(r"NH$_3$-N" + f" ({PARAM_UNIT['NH3N']})", fontsize=FS_LABEL)
ax_a.tick_params(labelsize=FS_TICK)
panel_label(ax_a, "a", fontsize=11)

# (b) year — lead temporal / trend group
if "nh3n_year_x" in hx.files:
    yx, yy = hx["nh3n_year_x"], hx["nh3n_year_y"]
else:
    raise SystemExit("missing nh3n_year arrays; re-run src/rev1/run_figure_inputs.py")
hb = ax_b.hexbin(yx, yy, gridsize=28, cmap="Greens", mincnt=1, linewidths=0.12)
cb = fig.colorbar(hb, ax=ax_b, label="Count", fraction=0.040, pad=0.02)
cb.ax.tick_params(labelsize=FS_TICK)
m = np.isfinite(yx) & np.isfinite(yy)
if m.sum() > 5:
    coef = np.polyfit(yx[m], yy[m], 1)
    xl = np.linspace(float(yx[m].min()), float(yx[m].max()), 100)
    ax_b.plot(xl, np.polyval(coef, xl), color="black", lw=1.2, linestyle="--")
    r_y = abs(float(spearmanr(yx[m], yy[m]).statistic))
else:
    r_y = np.nan
ax_b.text(0.58, 0.97, rf"$|r|_S$ = {r_y:.2f}",
          transform=ax_b.transAxes, ha="left", va="top", fontsize=FS_STAT,
          bbox=dict(boxstyle="round,pad=0.30", fc="white", ec=COL, lw=0.9))
ax_b.set_title("Year (in-situ)", fontsize=FS_TITLE, pad=5)
ax_b.set_xlabel("Year", fontsize=FS_LABEL)
ax_b.set_ylabel(r"NH$_3$-N" + f" ({PARAM_UNIT['NH3N']})", fontsize=FS_LABEL)
ax_b.tick_params(labelsize=FS_TICK)
panel_label(ax_b, "b", fontsize=11)

# (c) month hexbin — no colorbar; Count scale is already on (b)
ax_c.hexbin(hx["nh3n_month_x"], hx["nh3n_month_y"], gridsize=28,
            cmap="Greens", mincnt=1, linewidths=0.12)
ax_c.set_title("Calendar month (in-situ)", fontsize=FS_TITLE, pad=5)
ax_c.set_xlabel("Month", fontsize=FS_LABEL)
ax_c.set_ylabel(r"NH$_3$-N" + f" ({PARAM_UNIT['NH3N']})", fontsize=FS_LABEL)
ax_c.set_xticks(range(1, 13))
ax_c.tick_params(labelsize=FS_TICK)
panel_label(ax_c, "c", fontsize=11)

# (d) Arrhenius
T_c = np.linspace(2, 34, 200)
k_nit = np.exp(-70e3 / (8.314 * (T_c + 273.15)))
k_nit = k_nit / k_nit.max()
c_nh = float(np.mean(by)) * (q_in ** ((T_c - T_REF) / 10))
ax2 = ax_d.twinx()
l1, = ax_d.plot(T_c, k_nit, color="#E34A33", lw=2.2, label=r"$k_{\rm nit}$ (Arrhenius)")
l2, = ax2.plot(T_c, np.clip(c_nh, 0, None), color=COL, lw=2.2, linestyle="--",
               label=r"In-situ $Q_{10}$" + f" = {q_in:.2f}")
ax_d.set_title(r"Arrhenius kinetics vs. observed $Q_{10}$", fontsize=FS_TITLE, pad=5)
ax_d.set_xlabel("Temperature (°C)", fontsize=FS_LABEL)
ax_d.set_ylabel(r"$k_{\rm nit}$ (relative)", color="#E34A33", fontsize=FS_LABEL)
ax_d.yaxis.labelpad = 2
ax2.set_ylabel(r"NH$_3$-N" + f" ({PARAM_UNIT['NH3N']})", color=COL, fontsize=FS_LABEL)
ax_d.tick_params(axis="y", colors="#E34A33", labelsize=FS_TICK)
ax2.tick_params(axis="y", colors=COL, labelsize=FS_TICK)
ax2.spines["right"].set_visible(True)
ax2.spines["right"].set_color(COL)
ax_d.legend(handles=[l1, l2], fontsize=8, loc="upper center",
            bbox_to_anchor=(0.5, 0.96), frameon=True, edgecolor="0.75")
panel_label(ax_d, "d", fontsize=11)

save(fig, "fig9_nh3n_mech")
plt.close(fig)
print("fig9_nh3n_mech done")
