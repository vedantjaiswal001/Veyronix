"""Model / feature / calibration drift monitoring (spec §36/§60).

Detects when the world has moved away from what a model was trained on:
  * feature drift   — Population Stability Index (PSI) per feature
  * prediction drift — PSI on the predicted-probability distribution
  * calibration drift — change in expected calibration error

This is the basis of the MODEL DRIFT demo (§60): a shifted input distribution is
caught here and raised as a model-health warning before silent degradation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PSI_MODERATE = 0.1
PSI_SIGNIFICANT = 0.25


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two samples of one variable."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if len(ref) < 20 or len(cur) < 20:
        return 0.0
    quantiles = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(quantiles) < 3:
        return 0.0
    quantiles[0], quantiles[-1] = -np.inf, np.inf
    ref_counts = np.histogram(ref, bins=quantiles)[0].astype(float)
    cur_counts = np.histogram(cur, bins=quantiles)[0].astype(float)
    ref_p = np.clip(ref_counts / ref_counts.sum(), 1e-4, None)
    cur_p = np.clip(cur_counts / cur_counts.sum(), 1e-4, None)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def feature_drift(reference: pd.DataFrame, current: pd.DataFrame,
                  feature_cols: list[str], top_n: int = 15) -> dict:
    """Per-feature PSI and a summary of the most-drifted features."""
    scores = {}
    for c in feature_cols:
        if c in reference.columns and c in current.columns:
            scores[c] = round(psi(reference[c].to_numpy(), current[c].to_numpy()), 4)
    flagged = {k: v for k, v in scores.items() if v >= PSI_SIGNIFICANT}
    top = dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n])
    return {
        "n_features": len(scores),
        "n_flagged": len(flagged),
        "max_psi": round(max(scores.values(), default=0.0), 4),
        "mean_psi": round(float(np.mean(list(scores.values()))) if scores else 0.0, 4),
        "flagged": flagged,
        "top_drifted": top,
        "drift_detected": len(flagged) > 0,
    }


def prediction_drift(ref_probs: np.ndarray, cur_probs: np.ndarray) -> dict:
    val = round(psi(ref_probs, cur_probs), 4)
    return {"psi": val, "drift_detected": val >= PSI_SIGNIFICANT}


def calibration_drift(ref_ece: float, cur_ece: float) -> dict:
    delta = round(cur_ece - ref_ece, 4)
    return {"ref_ece": round(ref_ece, 4), "cur_ece": round(cur_ece, 4),
            "delta": delta, "drift_detected": delta > 0.1}
