"""
Fig. 6 — Grouped permutation importance under official S (ADAE).

Reads results/rev1/downstream/attribution/ only.
Skips (does not fall back to old results/analysis/) if Stage 4 has not been run.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from _style import (
    COL2,
    DOWN,
    PARAM_LABEL,
    TARGET_COLORS,
    TARGET_ORDER,
    panel_label,
    save,
    set_style,
)

ATTR = DOWN / "attribution"
SCHEMES = ("domain", "band", "function")
SCHEME_ROW = {"domain": "Acquisition surface", "band": "Sentinel-2 band", "function": "Process"}
N_SHOW = 8


def load_category(target, scheme):
    path = ATTR / scheme / f"category_{target}_station.csv"
    if not path.exists() and scheme == "function":
        path = ATTR / f"category_{target}_station.csv"
    if not path.exists():
        return pd.DataFrame(), None
    df = pd.read_csv(path)
    col = "perm_delta_r2" if "perm_delta_r2" in df.columns else "delta_r2_mean"
    return df.sort_values(col, ascending=False).reset_index(drop=True), col


def main():
    if not ATTR.exists() or not any(ATTR.rglob("category_*station*.csv")):
        print("SKIP fig6_importance: Stage 4 attribution not in results/rev1/downstream/attribution/")
        print("  Will not draw from the old results/analysis/ folder.")
        return

    set_style()
    fig, axes = plt.subplots(3, 3, figsize=(COL2, COL2 * 1.05))
    fig.subplots_adjust(wspace=0.38, hspace=0.38, left=0.06, right=0.98, top=0.94, bottom=0.08)
    letters = "abcdefghi"
    k = 0
    for ri, scheme in enumerate(SCHEMES):
        for ci, target in enumerate(TARGET_ORDER):
            ax = axes[ri, ci]
            df, col = load_category(target, scheme)
            color = TARGET_COLORS[target]
            if df.empty:
                ax.text(0.5, 0.5, "No attribution CSV", ha="center", va="center",
                        transform=ax.transAxes, color="#888888")
            else:
                df = df.head(N_SHOW)
                xmax = 0.02
                for yi in range(len(df)):
                    row = df.iloc[yi]
                    dr2 = float(row[col])
                    std = float(row.get("perm_std", 0) or 0)
                    lab = str(row.get("label", row.get("category", f"Group {yi}")))
                    top = yi < 3
                    ax.barh(yi, dr2, height=0.72,
                            color=color if top else "#CCCCCC", alpha=0.88,
                            edgecolor="black" if top else "none",
                            linewidth=1.5 if top else 0,
                            xerr=std if std > 0 else None,
                            error_kw=dict(elinewidth=0.6, capsize=1.6, ecolor="#444444"))
                    ax.text(max(dr2, 0) + (std if std > 0 else 0) + 0.008, yi, lab[:24],
                            va="center", ha="left", fontsize=5.6,
                            fontweight="bold" if top else "normal")
                    xmax = max(xmax, dr2 + (std if std > 0 else 0))
                ax.set_xlim(-0.02, xmax * 1.72 + 0.02)
                ax.invert_yaxis()
            ax.set_yticks([])
            if ri == 2:
                ax.set_xlabel(r"Fold-mean $\Delta$R²", fontsize=7)
            if ri == 0:
                ax.set_title(PARAM_LABEL[target], fontsize=9, fontweight="bold", color=color)
            if ci == 0:
                ax.set_ylabel(SCHEME_ROW[scheme], fontsize=7.5, fontweight="bold")
            panel_label(ax, letters[k])
            k += 1

    fig.legend(
        handles=[
            mpatches.Patch(facecolor="white", edgecolor="black", linewidth=1.5,
                           label="Top-3 category"),
            mpatches.Patch(facecolor="#CCCCCC", edgecolor="none", label="Other categories"),
        ],
        loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=2, fontsize=7,
    )
    save(fig, "fig6_importance")
    plt.close(fig)
    print("fig6_importance done")


if __name__ == "__main__":
    main()
