"""Campaign paths for rev1. Never write into results/evaluation/gkfold/cells."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "spectral_water_quality_analysis.csv"
OFFICIAL_STATION_FOLDS = ROOT / "results" / "evaluation" / "gkfold" / "station_folds.csv"
FRAMEWORK_MD = ROOT / "paper" / "run_framework.md"

PROTOCOLS = ("station", "temporal", "random")
METHODS = ("RF", "XGB", "CaB", "DANN", "ADAE")
TARGETS = ("CODMn", "NH3N", "TP")
TEMPORAL_TEST_YEARS = (2021, 2022, 2023, 2024, 2025)
N_FOLDS = 5
SELECT_K = 50
SELECT_SCHEME = "union_nolucc"
N_TUNE_TRIALS = 15
SEED = 42


def rev1_dir(tag: str = "rev1") -> Path:
    return ROOT / "results" / tag


def cell_dir(tag: str, protocol: str, method: str, target: str, fold_key) -> Path:
    return rev1_dir(tag) / "cells" / protocol / method / target / f"f{fold_key}"


def combo_dir(tag: str, protocol: str, method: str, target: str) -> Path:
    return rev1_dir(tag) / "cells" / protocol / method / target


def preprocess_dir(tag: str, protocol: str, target: str, fold_key) -> Path:
    return rev1_dir(tag) / "preprocess" / protocol / target / f"f{fold_key}"


def cell_id(protocol: str, method: str, target: str, fold_key) -> str:
    return f"{protocol}_{method}_{target}_f{fold_key}"
