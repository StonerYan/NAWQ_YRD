# Satellite retrieval of NAWQ indicators at ungauged river stations

Code, analysis table and summary products for the manuscript *Satellite retrieval of non-optically active water quality indicators at ungauged river stations: domain-adversarial ensemble learning, validation and environmental drivers*.

## What this deposit contains

- `data/spectral_water_quality_analysis.csv` — 26-station Sentinel-2–in-situ matchup table (4,757 rows).
- `src/rev1/` — five-fold evaluation: concentration-stratified station GroupKFold, leave-one-year-out on 2021–2025, observation-level random KFold. Variable selection and tuning are inside each outer training fold.
- `src/rev1/adae_simple.py` — ADAE used in the paper: station-mean stage (station-out only) + residual DANN + XGBoost + CatBoost.
- `results/rev1/summaries/fold_mean.csv` — Table 1 (fold-mean R² ± SD).
- `results/rev1/summaries/oof/` — out-of-fold predictions for Fig. 4 and Fig. 7.
- `figure/` — scripts for Figs. 3–10 and Supplementary Figs. S1, S2 and S4.

Training-fold model dumps are not included (large). The OOF files and `fold_mean.csv` are enough to redraw the tables and scatter plots.

## Headline numbers (station-out ADAE)

| Indicator | Fold-mean R² ± SD | Labelled n |
|---|---|---|
| COD_Mn | 0.523 ± 0.045 | 3,496 |
| NH₃-N | 0.381 ± 0.140 | 3,387 |
| TP | 0.332 ± 0.137 | 3,490 |

## Reproduce Table 1 from the summaries

```python
import pandas as pd
df = pd.read_csv("results/rev1/summaries/fold_mean.csv")
print(df[df.method == "ADAE"][["protocol", "target", "R2_fold_mean", "R2_fold_std"]])
```

## Environment

Python 3.10+ with numpy, pandas, scikit-learn, scipy, xgboost, catboost, torch, matplotlib, optuna.

## Licence

Research deposit for the accompanying paper. In-situ observations follow the terms of the China National Surface Water Quality Automatic Monitoring Dataset.
