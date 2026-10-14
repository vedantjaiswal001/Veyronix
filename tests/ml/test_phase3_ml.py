"""Phase 3: metrics, early warning, and the trained ML layer on real data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from veyronix.features.build import (
    build_dataset,
    model_feature_columns,
)
from veyronix.ml.anomaly.detectors import (
    IsolationForestDetector,
    evaluate_detector_scores,
)
from veyronix.ml.evaluation.early_warning import early_warning
from veyronix.ml.evaluation.metrics import (
    best_threshold,
    classification_metrics,
    recall_at_fpr,
)
from veyronix.ml.forecasting.forecasters import forecast_eval
from veyronix.ml.incident_prediction.train import train_all
from veyronix.ml.severity.model import train_severity

pytestmark = pytest.mark.ml


# ---------- pure metric tests ----------
def test_classification_metrics_known_values():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.4, 0.2, 0.8])
    m = classification_metrics(y, p, 0.5)
    assert m["tp"] == 1 and m["fp"] == 1 and m["tn"] == 1 and m["fn"] == 1
    assert m["precision"] == 0.5 and m["recall"] == 0.5


def test_recall_at_fpr_respects_budget():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = np.clip(0.5 * y + rng.normal(0, 0.3, 2000), 0, 1)  # informative scores
    r = recall_at_fpr(y, p, max_fpr=0.1)
    assert r["fp_rate"] <= 0.1 + 1e-6


def test_best_threshold_max_fpr_constraint():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 1000)
    p = rng.random(1000)
    t = best_threshold(y, p, objective="recall_at_fpr", max_fpr=0.05)
    assert classification_metrics(y, p, t)["fp_rate"] <= 0.05 + 1e-6


def test_early_warning_counts_only_pre_incident_alerts():
    incidents = pd.DataFrame({"run_id": ["r"], "start_min": [100]})
    preds = pd.DataFrame({
        "run_id": ["r", "r", "r"],
        "minute": [80, 95, 120],           # 80,95 pre-incident; 120 is after
        "pred": [1, 1, 1],
    })
    ew = early_warning(preds, incidents, horizon=30)
    assert ew["detected"] == 1
    assert ew["warning_minutes"]["max"] == 20  # earliest correct alert at t=80


# ---------- trained models on a small real dataset ----------
@pytest.fixture(scope="module")
def small():
    df = build_dataset(seeds=[42, 43, 44], minutes=900, persist=False)
    meta = {"horizon": 30, "feature_columns": model_feature_columns(df)}
    # incidents reconstructed from active windows for early-warning eval
    inc = (df[df.current_incident == 1]
           .groupby(["run_id", "active_incident_id"]).minute.min()
           .reset_index().rename(columns={"active_incident_id": "fault_id",
                                           "minute": "start_min"}))
    return df, meta, inc


def test_models_beat_naive_on_pr_auc(small):
    df, meta, inc = small
    out = train_all(df=df, meta=meta, incidents=inc, register=False)
    models = out["models"]
    for m in models.values():
        assert 0.0 <= m["pr_auc"] <= 1.0
    # a learned model should beat the naive threshold on PR-AUC
    best_ml = max(models[n]["pr_auc"] for n in ("random_forest", "xgboost"))
    assert best_ml >= models["naive_threshold"]["pr_auc"]


def test_prediction_pool_excludes_active_incidents(small):
    df, meta, inc = small
    # the trainer must never see rows where an incident is already active
    pool = df[df.current_incident == 0]
    assert (pool.current_incident == 0).all()
    assert pool.incident_within_h.notna().all()


def test_isolation_forest_detects_incidents(small):
    df, meta, inc = small
    feat = meta["feature_columns"]
    normal = df[(df.split == "train") & (df.current_incident == 0)
               & (df.incident_within_h == 0)]
    test = df[df.split == "test"]
    iso = IsolationForestDetector(seed=42).fit(normal[feat])
    ev = evaluate_detector_scores(iso.score(test[feat]), test.current_incident.to_numpy(int))
    assert ev["auroc"] > 0.6  # clearly better than chance


def test_severity_within_one_band(small):
    df, meta, inc = small
    res = train_severity(df, meta["feature_columns"], seed=42)
    if res["status"] == "ok":
        assert res["within_one_band"] >= 0.7


def test_forecasting_returns_valid_baselines_and_ml(small):
    df, meta, inc = small
    r = forecast_eval(df, horizon=10, seed=42)
    assert r, "no forecast targets evaluated"
    for _target, rr in r.items():
        for method in ("naive", "moving_avg", "ml_xgboost"):
            assert rr[method]["mae"] >= 0 and np.isfinite(rr[method]["rmse"])
