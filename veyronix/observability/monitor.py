"""Model-health monitoring (spec §36).

Combines feature/prediction/calibration drift into a single health verdict and
pushes gauges to Prometheus. A returning operator sees at a glance whether the
prediction model can still be trusted.
"""

from __future__ import annotations

import pandas as pd

from veyronix.core.logging import get_logger
from veyronix.observability import metrics as m
from veyronix.observability.drift import feature_drift, prediction_drift

log = get_logger("observability.monitor")


def assess_model_health(
    reference: pd.DataFrame, current: pd.DataFrame, feature_cols: list[str],
    ref_probs=None, cur_probs=None, push_metrics: bool = True,
) -> dict:
    """Return a model-health report and (optionally) publish gauges."""
    fdrift = feature_drift(reference, current, feature_cols)
    pdrift = (prediction_drift(ref_probs, cur_probs)
              if ref_probs is not None and cur_probs is not None else None)

    if fdrift["n_flagged"] >= 3 or (pdrift and pdrift["drift_detected"]):
        status, score = "degraded", 0.0
    elif fdrift["n_flagged"] >= 1:
        status, score = "warning", 0.5
    else:
        status, score = "healthy", 1.0

    if push_metrics:
        m.MODEL_HEALTH.set(score)
        for feat, val in fdrift["top_drifted"].items():
            m.MODEL_DRIFT.labels(feature=feat).set(val)

    report = {"status": status, "health_score": score, "feature_drift": fdrift,
              "prediction_drift": pdrift,
              "recommendation": {
                  "healthy": "model within training distribution",
                  "warning": "some feature drift; monitor and consider retraining",
                  "degraded": "significant drift; retrain before trusting predictions",
              }[status]}
    log.info("model.health", status=status, n_flagged=fdrift["n_flagged"],
             max_psi=fdrift["max_psi"])
    return report


def demo_model_drift(shift_scale: float = 3.0) -> dict:
    """MODEL DRIFT demo (§60): shift the input distribution and detect it.

    Uses the built dataset's test split as 'reference' and a copy with a few
    features shifted as 'current', proving the monitor raises a warning.
    """
    from veyronix.features.build import load_dataset, load_meta

    df = load_dataset()
    meta = load_meta()
    feat_cols = meta["feature_columns"]
    ref = df[df.split == "test"].copy()
    cur = ref.copy()
    # simulate an input-distribution shift on a handful of load-related features
    shifted = [c for c in feat_cols if c.endswith("request_rate__last")
               or c.endswith("cpu_pct__mean15")][:6]
    for c in shifted:
        cur[c] = cur[c] * shift_scale + cur[c].std() * 2
    report = assess_model_health(ref, cur, feat_cols, push_metrics=False)
    report["shifted_features"] = shifted
    return report
