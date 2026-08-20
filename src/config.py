"""
WQI0627 Central Configuration
================================
ALL hyperparameters, paths and constants live here.
No magic numbers anywhere else in the codebase.
"""
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).parent.parent
DATA_DIR      = ROOT_DIR / "data"
# Canonical analysis dataset: 26-station network (see src/make_dataset.py)
SPECTRAL_FILE = DATA_DIR / "spectral_water_quality_analysis.csv"
ERA5_FILE     = DATA_DIR / "era5land_output" / "era5land_all_stations.csv"
N_FEATURES    = None  # dynamic; see load_engineered_dataset() feature list length
# Final 50-feature subset (RF union on station-train; see feature_selection/)
SELECTED_FEATURE_SET = "rf_union_top50"
FEATURE_SELECTION_JSON = ROOT_DIR / "results" / "analysis" / "feature_selection" / "selected_sets.json"
N_STATIONS    = 26    # analysis network size
RESULTS_DIR   = ROOT_DIR / "results"
FIGURES_DIR   = ROOT_DIR / "figures" / "output"

# ── Targets & Protocols ───────────────────────────────────────────────────────
TARGETS   = ["CODMn", "NH3N", "TP"]
PROTOCOLS = ["random", "temporal", "station"]

# ── Data column mapping (0-based index in raw CSV) ───────────────────────────
TARGET_COL_IDX = {"CODMn": 60, "NH3N": 61, "TP": 62}
SPEC_BAND_IDX  = {
    "station":       list(range(9,  21)),
    "veg":           list(range(21, 33)),
    "anomaly":       list(range(33, 45)),
    "veg_corrected": list(range(45, 57)),
}
BAND_NUMS      = [1, 2, 3, 4, 5, 6, 7, 8, "8A", 9, 11, 12]

# ── Validation split parameters ───────────────────────────────────────────────
TEMPORAL_TEST_YEAR = 2024                   # Protocol T: held-out calendar year
STATION_TEST_FRAC  = 0.20                   # legacy Protocol S: fraction of stations held out
RANDOM_SEEDS_R     = [42, 123, 456]         # legacy Protocol R (superseded by EVAL_SEEDS)
EVAL_SEEDS         = [42, 123, 456, 789, 2024, 1337, 9999]  # unified 7-seed evaluation (all protocols)
STATION_SPLIT_SEED = 42                     # legacy fixed holdout (superseded by grouped k-fold)
STATION_N_FOLDS    = 5                      # Protocol S: concentration-stratified GroupKFold
STATION_CV         = "grouped_kfold"        # "grouped_kfold" | "fixed_holdout"
STATION_CV_SEED    = 42                     # shuffle seed inside concentration blocks
# Fixed-split ADAE station-out means (seed-42 hold-out, 7 model seeds) — recovery target
# Official station-out lock: ADAE s26t fold-mean R2 (headline). Pooled: 0.550 / 0.404 / 0.372
# Historical: v23 composite pooled 0.516/0.415/0.345; s26r 0.499/0.371/0.363
STATION_REF_R2     = {"CODMn": 0.534, "NH3N": 0.410, "TP": 0.345}

# ── ADAE hyperparameters ──────────────────────────────────────────────────────
HIDDEN_DIM    = 128
TABMAE_EPOCHS = 35
DANN_EPOCHS   = 50
MASK_RATIO    = 0.30
N_FOLDS       = 3
GRL_LAMBDA    = {"random": 0.05, "temporal": 0.20, "station": 0.40}
DOM_WEIGHT    = {"random": 0.15, "temporal": 0.25, "station": 0.35}
N_OWT         = 5                           # optical water type clusters

# Bayesian blending Optuna bounds by protocol
OPTUNA_TRIALS = 80
MOE_BOUNDS    = {
    "random":   (0.25, 0.60),
    "temporal": (0.30, 0.65),
    "station":  (0.00, 0.50),
}

# ── Feature attribution (ADAE permutation importance) ───────────────────────
ATTR_PROTOCOLS        = ["station", "temporal"]   # station-out primary narrative
ATTR_PRIMARY_PROTOCOL = "station"
ATTR_SCHEMES          = ("domain", "band", "function")  # multi-level grouping
ATTR_PRIMARY_SCHEME   = "function"                      # mechanism narrative
ATTR_SEED             = 42
ATTR_PERM_REPEATS     = 5                         # shuffle repeats per feature
SHAP_TOP_N            = 20                        # features to save in CSV

# ── Conformal prediction ──────────────────────────────────────────────────────
CONFORMAL_ALPHA     = 0.10    # 90% nominal coverage
CONFORMAL_PROTOCOLS = ["station", "temporal"]  # station-out primary

# ── Figure output ─────────────────────────────────────────────────────────────
FIG_DPI       = 300
FIG_FORMAT    = ["png", "pdf"]

# ── WQ quality thresholds (China GB3838-2002 Class III) ──────────────────────
QA_THRESHOLDS = {"CODMn": 6.0, "NH3N": 1.0, "TP": 0.2}   # mg/L

# ── Units and display names ───────────────────────────────────────────────────
PARAM_UNITS   = {"CODMn": "mg L⁻¹", "NH3N": "mg L⁻¹", "TP": "mg L⁻¹"}
PARAM_LABEL   = {"CODMn": "COD$_{Mn}$", "NH3N": "NH$_3$-N", "TP": "TP"}
METHOD_COLORS = {
    "RF":    "#6BA3BE",
    "XGB":   "#F4A261",
    "CaB":   "#A8C5A0",
    "DANN":  "#C9ADE7",
    "ADAE":  "#2C4E80",
}
PROTOCOL_LABELS = {"random": "Random (R)", "temporal": "Temporal (T)", "station": "Station-out (S)"}
