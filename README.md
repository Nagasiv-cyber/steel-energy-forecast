# Steel Plant Energy Forecasting (Track 2)

Predicts the plant's electricity use for the **next 15-minute interval** (`next_interval_Usage_kWh`), using only readings up to now.

**Result:** RMSE **8.57**, MAE **4.06** on unseen future weeks, against 12.79 for the "next = now" baseline (33% better).

## Files

| File | What it is |
|---|---|
| `predict.py` | **Inference function** `predict(df)`: loads the saved model and returns one forecast per row |
| `my_model.joblib` | **Saved model**: average of 4 models (LightGBM ×2, XGBoost, CatBoost) on 43 features, plus the last 7 days of training data used as history |
| `steel_energy_forecast.ipynb` | Full pipeline with explanations: data exploration, baseline, features, 5-period validation, training |
| `Steel_Energy_Methodology.pdf` | Methodology summary (2 pages) |
| `predictions.csv` | Example output on the last 500 rows of train.csv |
| `requirements.txt` | Libraries needed |

## Run inference

```bash
pip install -r requirements.txt
```

```python
import pandas as pd
from predict import predict

test = pd.read_csv("test.csv")   # the 11 feature columns, chronological, 15-minute steps
preds = predict(test)            # numpy array, one prediction per row, same order
```

Or from the command line:

```bash
python predict.py test.csv predictions.csv
```

- The target column is not needed (ignored if present).
- No retraining: prediction takes about 2 seconds.
- If `test.csv` starts 15 minutes after `train.csv` ends, the stored training history is added in front automatically, so the first test rows get full lag and same-time-yesterday features.

## How it works

1. **Features (43 in 5 groups):** raw readings and calendar, time of the next interval, recent usage (lags, rolling averages), same moment on earlier days, and electrical signals. All use data up to the current interval only.
2. **Models:** LightGBM (squared error), LightGBM (Huber), XGBoost and CatBoost, averaged.
3. **Validation:** expanding-window TimeSeriesSplit, 5 folds of 4 weeks each. Every model is trained only on the past and tested on the future.

| Model | RMSE (5 folds) |
|---|---|
| Baseline: next = now | 12.79 |
| Best single model (CatBoost) | 8.64 |
| **Team of 4 (final)** | **8.57 ± 0.40** |

## Retrain

Put `train.csv` next to the notebook and run all cells (about 5 minutes). It rewrites `my_model.joblib`.
