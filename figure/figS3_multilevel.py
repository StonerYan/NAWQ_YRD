"""
Fig. S3 — Cross-level importance heatmap.

Reads results/rev1/downstream/attribution/{domain,band,function}/ only.
Skips if Stage 4 has not been run.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
SCHEMES = ["domain", "band", "function"]
SCHEME_LABEL = {"domain": "Domain", "band": "Band", "function": "Function"}


def load_top_n(target, scheme, n=8):
    path = ATTR / scheme / f"category_{target}_station.csv"
    if not path.exists() and scheme == "function":
        path = ATTR / f"category_{target}_station.csv"
    if not path.exists():
        return pd.DataFrame(columns=["label", "perm_delta_r2"])
    df = pd.read_csv(path)
    col = "perm_delta_r2" if "perm_delta_r2" in df.columns else "delta_r2_mean"
    return df.sort_values(col, ascending=False).head(n).reset_index(drop=True)


def main():
    if not ATTR.exists() or not any(ATTR.rglob("category_*station*.csv")):
        print("SKIP figS3_multilevel: Stage 4 attribution not in results/rev1/downstream/attribution/")
        return

    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(COL2, COL2 * 0.88))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.11, wspace=0.22)
    for ax, target in zip(axes, TARGET_ORDER):
        col = TARGET_COLORS[target]
        fn_df = load_top_n(target, "function", n=10)
        fn_labs = fn_df["label"].tolist() if not fn_df.empty else []
        mat = pd.DataFrame(index=fn_labs, columns=SCHEMES, dtype=float)
        top3 = {}
        for scheme in SCHEMES:
            sdf = load_top_n(target, scheme, n=20)
            top3[scheme] = set(sdf.head(3)["label"].tolist())
            if sdf.empty:
                continue
            vcol = "perm_delta_r2" if "perm_delta_r2" in sdf.columns else "delta_r2_mean"
            sdf_idx = sdf.set_index("label")[vcol]
            for lab in fn_labs:
                if lab in sdf_idx:
                    mat.loc[lab, scheme] = sdf_idx[lab]
        vals = mat.values.astype(float)
        vmax = np.nanmax(vals) if not np.all(np.isnan(vals)) else 0.35
        cmap = mcolors.LinearSegmentedColormap.from_list("custom", ["white", col], N=256)
        im = ax.imshow(vals, aspect="auto", cmap=cmap, vmin=0, vmax=vmax,
                       interpolation="nearest")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="4.5%", pad=0.20)
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_label(r"$\Delta$R²", fontsize=9)
        ax.set_xticks(range(3))
        ax.set_xticklabels([SCHEME_LABEL[s] for s in SCHEMES], fontsize=9)
        ax.set_yticks(range(len(fn_labs)))
        trans = blended_transform_factory(ax.transAxes, ax.transData)
        ax.set_yticklabels([])
        stable = {lab for lab in fn_labs
                  if sum(1 for sch in SCHEMES if lab in top3.get(sch, set())) >= 2}
        for yi, lab in enumerate(fn_labs):
            ax.text(0.04, yi, lab[:24], transform=trans, ha="left", va="center",
                    fontsize=8, fontstyle="italic" if lab in stable else "normal",
                    bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.72))
        for ri, lab in enumerate(fn_labs):
            for ci, scheme in enumerate(SCHEMES):
                if lab in top3.get(scheme, set()):
                    ax.add_patch(plt.Rectangle((ci - 0.5, ri - 0.5), 1, 1,
                                               fill=False, edgecolor="black", lw=1.5))
        ax.set_title(PARAM_LABEL[target], fontsize=10, fontweight="bold", color=col)
        panel_label(ax, "abc"[TARGET_ORDER.index(target)], fontsize=11)
    fig.text(0.5, 0.018,
             "Italic labels = top-3 in ≥2 schemes. Black outlines = top-3 in that scheme.",
             ha="center", va="bottom", fontsize=8, color="grey")
    save(fig, "figS2_multilevel", aliases=("figS3_multilevel",))
    plt.close(fig)
    print("figS3_multilevel done")


if __name__ == "__main__":
    main()
