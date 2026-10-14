"""Short-horizon telemetry forecasting (spec §20).

Predicts a metric's value ``h`` minutes ahead. Compares a naive last-value
baseline and a seasonal-adjusted baseline against a gradient-boosted regressor
on lag/rolling features. Reported with MAE / RMSE / MAPE, so we only claim ML
value where it measurably beats the baselines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FORECAST_TARGETS = [
    "api_gateway__request_rate",     # strongly seasonal -> ML should win
    "database__db_latency_ms",       # trends during DB incidents
    "order_service__cpu_pct",        # load-driven
    "api_gateway__latency_p95_ms",   # noisy -> naive is a strong baseline
]


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    denom = np.where(np.abs(y_true) < 1e-6, np.nan, np.abs(y_true))
    mape = float(np.nanmean(np.abs(err) / denom) * 100)
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "mape": round(mape, 2)}


def forecast_eval(df: pd.DataFrame, horizon: int = 10, seed: int = 42) -> dict:
    """Evaluate forecasting for each target using the feature dataset.

    ``{target}__last`` is the value at t; target label is that value ``horizon``
    minutes ahead (shift(-h) within each run). Naive = last value; seasonal =
    last value adjusted by the mean15/mean30 slope; ML = gradient boosting on
    lag features.
    """
    from xgboost import XGBRegressor

    results: dict[str, dict] = {}
    for target in FORECAST_TARGETS:
        val_col = f"{target}__last"
        if val_col not in df.columns:
            continue
        feats = [c for c in (f"{target}__last", f"{target}__mean15",
                             f"{target}__mean30", f"{target}__trend15",
                             f"{target}__delta5", "ctx__tod_sin", "ctx__tod_cos")
                 if c in df.columns]

        parts = []
        for _, g in df.sort_values(["run_id", "minute"]).groupby("run_id", sort=False):
            g = g.copy()
            g["_y"] = g[val_col].shift(-horizon)  # future value (label only)
            parts.append(g)
        d = pd.concat(parts).dropna(subset=["_y"])

        tr, te = d[d.split == "train"], d[d.split == "test"]
        if len(tr) < 50 or len(te) < 20:
            continue

        # baselines: naive (persistence) and moving-average (mean of last 15)
        naive = te[val_col].to_numpy(float)
        ma_col = f"{target}__mean15"
        moving_avg = te[ma_col].to_numpy(float) if ma_col in te else naive
        y_te = te["_y"].to_numpy(float)

        model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                             subsample=0.8, random_state=seed, n_jobs=-1)
        model.fit(tr[feats].to_numpy(float), tr["_y"].to_numpy(float))
        ml = model.predict(te[feats].to_numpy(float))

        results[target] = {
            "horizon_min": horizon,
            "naive": _metrics(y_te, naive),
            "moving_avg": _metrics(y_te, moving_avg),
            "ml_xgboost": _metrics(y_te, ml),
            "test_rows": len(te),
        }
    return results
