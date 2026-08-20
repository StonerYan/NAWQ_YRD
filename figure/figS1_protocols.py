"""
Fig. S1 — Official 5-fold R / T / S assignment (station × year).

Illustrated on COD_Mn labelled matchups. S uses the COD_Mn station map;
NH3-N and TP maps are independent (see station_folds.csv).

Input : results/rev1/folds/ + ADAE CODMn OOF
Output: figure/output/figS1_protocols.(png|pdf|svg)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from _style import (
    COL2,
    FOLDS,
    FOLD_COLORS,
    load_oof,
    panel_label,
    save,
    set_style,
)

set_style()

YEARS = list(range(2020, 2026))
C_NODATA = "#F0F0F0"
C_TRAINONLY = "#BDBDBD"
FS_TITLE, FS_AXIS, FS_NOTE, FS_TICK = 10, 9, 8, 7.5


def _stations() -> list[str]:
    tab = pd.read_csv(FOLDS / "station_folds.csv", encoding="utf-8-sig")
    return tab["station"].astype(str).tolist()


def _empty(n_st: int) -> np.ndarray:
    return np.full((n_st, len(YEARS)), np.nan)


def _paint(grid, stations, sta, year, value):
    if sta not in stations or int(year) not in YEARS:
        return
    grid[stations.index(sta), YEARS.index(int(year))] = value


def grid_random(stations):
    oof = load_oof("random", "ADAE", "CODMn")
    df = pd.DataFrame({
        "station": np.asarray(oof["station"]).astype(str),
        "year": np.asarray(oof["year"]).astype(int),
        "fold": np.asarray(oof["fold"]).astype(int),
    })
    grid = _empty(len(stations))
    mixed = np.zeros_like(grid, dtype=bool)
    for (sta, yr), g in df.groupby(["station", "year"]):
        folds = g["fold"].to_numpy()
        if len(folds) == 0:
            continue
        mode = int(g["fold"].mode().iloc[0])
        _paint(grid, stations, sta, yr, mode)
        if g["fold"].nunique() > 1 and sta in stations and int(yr) in YEARS:
            mixed[stations.index(sta), YEARS.index(int(yr))] = True
    return grid, mixed


def grid_temporal(stations):
    oof = load_oof("temporal", "ADAE", "CODMn")
    df = pd.DataFrame({
        "station": np.asarray(oof["station"]).astype(str),
        "year": np.asarray(oof["year"]).astype(int),
    })
    # 2020 is train-only: take labelled station-years from station OOF
    soof = load_oof("station", "ADAE", "CODMn")
    all_sy = pd.DataFrame({
        "station": np.asarray(soof["station"]).astype(str),
        "year": np.asarray(soof["year"]).astype(int),
    }).drop_duplicates()
    grid = _empty(len(stations))
    for sta, yr in all_sy.itertuples(index=False):
        if int(yr) == 2020:
            _paint(grid, stations, sta, yr, -1)  # train-only
        elif 2021 <= int(yr) <= 2025:
            _paint(grid, stations, sta, yr, int(yr) - 2021)  # fold 0-4
    return grid


def grid_station(stations):
    tab = pd.read_csv(FOLDS / "station_folds.csv", encoding="utf-8-sig")
    fmap = dict(zip(tab["station"].astype(str), tab["fold_CODMn"].astype(int)))
    soof = load_oof("station", "ADAE", "CODMn")
    df = pd.DataFrame({
        "station": np.asarray(soof["station"]).astype(str),
        "year": np.asarray(soof["year"]).astype(int),
    }).drop_duplicates()
    grid = _empty(len(stations))
    for sta, yr in df.itertuples(index=False):
        if sta in fmap:
            _paint(grid, stations, sta, yr, int(fmap[sta]))
    return grid


def _draw(ax, grid, title, subtitle, letter, mixed=None):
    ny, nx = grid.shape
    rgba = np.ones((ny, nx, 4))
    rgba[:] = mcolors.to_rgba(C_NODATA)
    for i in range(ny):
        for j in range(nx):
            v = grid[i, j]
            if not np.isfinite(v):
                continue
            if v < 0:
                rgba[i, j] = mcolors.to_rgba(C_TRAINONLY)
            else:
                rgba[i, j] = mcolors.to_rgba(FOLD_COLORS[int(v) % 5])
    ax.imshow(rgba, aspect="auto", origin="upper", interpolation="nearest")
    if mixed is not None:
        ii, jj = np.where(mixed)
        ax.scatter(jj, ii, s=8, marker="x", c="#222222", linewidths=0.6, zorder=3)
    ax.set_xticks(range(nx))
    ax.set_xticklabels([str(y) for y in YEARS], fontsize=FS_TICK)
    ax.set_xlabel("Year", fontsize=FS_AXIS, labelpad=4)
    ax.set_yticks(np.arange(0, ny, max(1, ny // 13)))
    ax.set_yticklabels([str(int(i) + 1) for i in ax.get_yticks()], fontsize=FS_TICK)
    ax.set_ylabel("Station", fontsize=FS_AXIS, labelpad=4)
    ax.tick_params(length=2, pad=1)
    ax.set_title(title, fontsize=FS_TITLE, fontweight="bold", pad=6, loc="left")
    ax.text(0.0, -0.16, subtitle, transform=ax.transAxes,
            ha="left", va="top", fontsize=FS_NOTE, color="#444444")
    panel_label(ax, letter, dx=-0.10, dy=1.02, fontsize=11)


def main():
    stations = _stations()
    g_r, mixed = grid_random(stations)
    g_t = grid_temporal(stations)
    g_s = grid_station(stations)

    fig, axes = plt.subplots(3, 1, figsize=(COL2 * 0.72, COL2 * 1.02))
    fig.subplots_adjust(left=0.12, right=0.96, top=0.93, bottom=0.10, hspace=0.55)

    _draw(axes[0], g_r, "Random (R)",
          "Observation-level 5-fold (seed 42). Colour = modal test fold in that "
          "station–year; × = more than one fold in the cell.",
          "a", mixed=mixed)
    _draw(axes[1], g_t, "Temporal (T)",
          "Year-blocked 5-fold: 2021–2025 each held out once. Grey = 2020, train-only, never a test year.",
          "b")
    _draw(axes[2], g_s, "Station-out (S)",
          "Concentration-stratified GroupKFold on COD$_\\mathrm{Mn}$ (6/5/5/5/5). "
          "NH$_3$-N and TP use independent maps.",
          "c")

    fold_handles = [
        Patch(facecolor=FOLD_COLORS[i], edgecolor="0.5", linewidth=0.4,
              label=f"Fold {i}")
        for i in range(5)
    ] + [
        Patch(facecolor=C_TRAINONLY, edgecolor="0.5", linewidth=0.4, label="Train only (2020)"),
        Patch(facecolor=C_NODATA, edgecolor="0.5", linewidth=0.4, label="No labelled matchup"),
    ]
    fig.legend(handles=fold_handles, loc="upper center", ncol=7, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, 0.995))

    save(fig, "figS1_protocols")
    plt.close(fig)
    print("figS1_protocols done")


if __name__ == "__main__":
    main()
