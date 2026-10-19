"""
predict.py: inference for the Steel Plant Energy Forecasting model (Track 2).

The saved model (my_model.joblib) is a team of 4 models (LightGBM x2, XGBoost,
CatBoost) trained in steel_energy_forecast.ipynb on 43 clues (features).
Their predictions are averaged. It also stores the last 7 days of train.csv
as "history".

Use from Python (this is the function the organisers call):
    import pandas as pd
    from predict import predict
    preds = predict(pd.read_csv("test.csv"))   # numpy array, one value per row

Use from the command line:
    python predict.py test.csv predictions.csv

Input: the 11 feature columns of train.csv, in chronological order at 15-minute
steps. The target column is not needed (it is ignored if present).
Output: predicted kWh for the NEXT 15-minute interval of every row, same order.

Why history: clues such as "same time last week" need earlier readings. If
test.csv starts exactly 15 minutes after train.csv ends, the stored history is
placed in front automatically, so even the first test rows get full clues.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).with_name("my_model.joblib")
TARGET = "next_interval_Usage_kWh"
FEATURE_COLUMNS = [
    "date", "Usage_kWh", "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh", "CO2(tCO2)",
    "Lagging_Current_Power_Factor", "Leading_Current_Power_Factor",
    "NSM", "WeekStatus", "Day_of_week", "Load_Type",
]
DAY = 96  # readings per day (24 hours x 4)


def to_time(dates):
    """Parse 'dd/mm/yyyy HH:MM'. The data stamps each midnight reading with the
    previous day's date, so 00:00 readings are moved forward one day."""
    t = pd.to_datetime(dates, format="%d/%m/%Y %H:%M", errors="coerce")
    is_midnight = (t.dt.hour == 0) & (t.dt.minute == 0)
    return t + pd.to_timedelta(is_midnight.astype(int), unit="D")


def make_clues(df):
    """Add the 43 clues. Every clue uses only readings up to now (no future data)."""
    df = df.copy()
    u = df["Usage_kWh"]
    react = df["Lagging_Current_Reactive.Power_kVarh"]
    nxt = to_time(df["date"]) + pd.Timedelta(minutes=15)          # time of the reading we predict
    nxt_min = nxt.dt.hour * 60 + nxt.dt.minute

    # Group 1: calendar and plant state
    df["dow"] = df["Day_of_week"].map({"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
                                       "Friday": 4, "Saturday": 5, "Sunday": 6})
    df["is_weekend"] = (df["WeekStatus"] == "Weekend").astype(int)
    df["load_type"] = df["Load_Type"].map({"Light_Load": 0, "Medium_Load": 1, "Maximum_Load": 2})
    df["month"] = to_time(df["date"]).dt.month

    # Group 2: time of the NEXT reading (the 08:00 start, lunch break, etc.)
    df["next_slot"] = nxt_min // 15                                # 0..95
    df["next_hour"] = nxt.dt.hour
    df["next_dow"] = nxt.dt.dayofweek
    df["next_hour_sin"] = np.sin(2 * np.pi * nxt_min / 1440)       # clock as a circle, so
    df["next_hour_cos"] = np.cos(2 * np.pi * nxt_min / 1440)       # 23:45 sits next to 00:00

    # Group 3: recent usage (lags, changes, rolling windows)
    for k in [1, 2, 3, 4, 8, 12]:
        df[f"usage_lag{k}"] = u.shift(k)                           # reading k steps ago
    df["usage_diff1"] = u - u.shift(1)                             # change in last 15 min
    df["usage_diff4"] = u - u.shift(4)                             # change in last hour
    for w in [4, 8, 16, 96]:                                       # 1h, 2h, 4h, 24h windows
        r = u.rolling(w, min_periods=1)
        df[f"usage_rmean{w}"] = r.mean()
        df[f"usage_rstd{w}"] = r.std()                             # how jumpy it has been
    df["usage_rmax16"] = u.rolling(16, min_periods=1).max()        # highest in last 4h
    df["usage_ewm4"] = u.ewm(span=4, adjust=False).mean()          # average favouring recent readings

    # Group 4: same moment on earlier days
    df["usage_sameslot_1d"] = u.shift(DAY - 1)                     # next slot, yesterday
    df["usage_sameslot_7d"] = u.shift(7 * DAY - 1)                 # next slot, last week
    df["sameslot_4w_max"] = pd.concat([u.shift(7 * DAY * k - 1) for k in range(1, 5)],
                                      axis=1).max(axis=1)          # busiest this slot got in 4 weeks
    df["yday_jump"] = u.shift(DAY - 1) - u.shift(DAY)              # jump at this moment yesterday

    # Group 5: electrical signals
    df["lagkvarh_lag1"] = react.shift(1)
    df["lagkvarh_diff1"] = react - react.shift(1)
    df["apparent_kva"] = np.sqrt(u ** 2 + react ** 2)              # total electrical load
    df["pf_lag_diff1"] = df["Lagging_Current_Power_Factor"].diff()
    df["load_type_lag1"] = df["load_type"].shift(1)
    return df


CLUES = [
    # raw readings (7)
    "Usage_kWh", "Lagging_Current_Reactive.Power_kVarh", "Leading_Current_Reactive_Power_kVarh",
    "CO2(tCO2)", "Lagging_Current_Power_Factor", "Leading_Current_Power_Factor", "NSM",
    # calendar and plant state (4)
    "dow", "is_weekend", "load_type", "month",
    # time of next reading (5)
    "next_slot", "next_hour", "next_dow", "next_hour_sin", "next_hour_cos",
    # recent usage (18)
    "usage_lag1", "usage_lag2", "usage_lag3", "usage_lag4", "usage_lag8", "usage_lag12",
    "usage_diff1", "usage_diff4",
    "usage_rmean4", "usage_rstd4", "usage_rmean8", "usage_rstd8",
    "usage_rmean16", "usage_rstd16", "usage_rmean96", "usage_rstd96",
    "usage_rmax16", "usage_ewm4",
    # same moment on earlier days (4)
    "usage_sameslot_1d", "usage_sameslot_7d", "sameslot_4w_max", "yday_jump",
    # electrical (5)
    "lagkvarh_lag1", "lagkvarh_diff1", "apparent_kva", "pf_lag_diff1", "load_type_lag1",
]


def make_team():
    """The 4 models. Each learns differently; their average makes fewer big mistakes."""
    import catboost as cb
    import lightgbm as lgb
    import xgboost as xgb
    shared = dict(learning_rate=0.03, num_leaves=63, min_child_samples=30, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, n_estimators=400, verbose=-1)
    return [
        lgb.LGBMRegressor(**shared, objective="regression", random_state=1),
        lgb.LGBMRegressor(**shared, objective="huber", alpha=10, random_state=2),
        xgb.XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=7, subsample=0.8,
                         colsample_bytree=0.8, min_child_weight=5, random_state=3),
        cb.CatBoostRegressor(iterations=1500, learning_rate=0.05, depth=8, verbose=0,
                             random_seed=4, allow_writing_files=False),
    ]


def follows_on(history, new_df):
    """True if new_df starts exactly 15 minutes after history ends."""
    try:
        gap = to_time(new_df["date"].iloc[:1]).iloc[0] - to_time(history["date"].iloc[-1:]).iloc[0]
        return gap == pd.Timedelta(minutes=15)
    except Exception:
        return False


def predict(new_df, history=None, model_path=MODEL_PATH):
    """Predict next-interval kWh for every row of new_df (a DataFrame).
    history: earlier readings to put in front; by default the stored end of train.csv,
    used only when new_df follows on from it in time."""
    saved = joblib.load(model_path)
    new_df = new_df.drop(columns=[TARGET], errors="ignore").reset_index(drop=True)

    if history is None:
        history = saved.get("history")
    n_hist = 0
    if history is not None and follows_on(history, new_df):
        history = history.drop(columns=[TARGET], errors="ignore").tail(7 * DAY)
        n_hist = len(history)
        new_df = pd.concat([history, new_df], ignore_index=True)

    data = make_clues(new_df)
    guesses = [m.predict(data[saved["clues"]]) for m in saved["models"]]
    pred = np.clip(np.mean(guesses, axis=0), 0, None)              # energy can't be negative
    return pred[n_hist:]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python predict.py <test.csv> [predictions.csv]")
    test = pd.read_csv(sys.argv[1])
    out_path = sys.argv[2] if len(sys.argv) > 2 else "predictions.csv"
    out = pd.DataFrame({"date": test["date"], "predicted_kWh": predict(test)})
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} predictions to {out_path}")
