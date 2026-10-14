"""Unsupervised anomaly detection (spec §18/§19).

Two complementary detectors:
  * IsolationForest — learns the shape of "normal" multivariate telemetry and
    scores how isolated a point is. Detects novel/unseen patterns.
  * StatisticalDetector — rolling robust z-score (median/MAD) + EWMA on key
    system signals; transparent and fast, no training.

Both are unsupervised (fit on data without using incident labels), so they can
flag *previously unseen* failure modes — the basis of the unknown-failure
experiment (spec §19). Anomaly ground truth for evaluation is "an incident is
currently active"; scores are compared to that with AUROC/AUPRC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from veyronix.core.constants import Severity


class IsolationForestDetector:
    name = "isolation_forest"

    def __init__(self, seed: int = 42, contamination: float = 0.08):
        self.seed = seed
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=250, contamination=contamination,
            random_state=seed, n_jobs=-1)
        self.columns: list[str] = []
        self.threshold_: float = 0.0

    def fit(self, X: pd.DataFrame) -> IsolationForestDetector:
        self.columns = list(X.columns)
        Xt = self.scaler.fit_transform(X.to_numpy(dtype=float))
        self.model.fit(Xt)
        scores = self.score(X)
        self.threshold_ = float(np.quantile(scores, 1 - self.contamination))
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        Xt = self.scaler.transform(X[self.columns].to_numpy(dtype=float))
        # higher = more anomalous
        return -self.model.decision_function(Xt)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.score(X) >= self.threshold_).astype(int)


class StatisticalDetector:
    """Rolling robust z-score on system-health signals (no training)."""

    name = "statistical"
    SIGNALS = ["sys__max_error_rate", "sys__max_latency_p95", "sys__max_saturation",
               "sys__db_latency", "sys__mean_error_rate"]

    def __init__(self, window: int = 30, z: float = 3.5):
        self.window = window
        self.z = z

    def score_frame(self, df: pd.DataFrame) -> np.ndarray:
        cols = [c for c in self.SIGNALS if c in df.columns]
        out = np.zeros(len(df))
        for _run_id, g in df.groupby("run_id", sort=False):
            idx = g.index
            zmax = np.zeros(len(g))
            for c in cols:
                s = g[c]
                med = s.rolling(self.window, min_periods=5).median()
                mad = (s - med).abs().rolling(self.window, min_periods=5).median()
                z = (s - med).abs() / (1.4826 * mad + 1e-6)
                zmax = np.maximum(zmax, z.fillna(0.0).to_numpy())
            out[df.index.get_indexer(idx)] = zmax
        return out

    def predict_frame(self, df: pd.DataFrame) -> np.ndarray:
        return (self.score_frame(df) >= self.z).astype(int)


def severity_from_score(score: float, threshold: float) -> str:
    """Map a normalized anomaly ratio to a severity band (spec §18)."""
    r = score / max(threshold, 1e-6)
    if r < 1.0:
        return "none"
    if r < 1.5:
        return Severity.LOW.value
    if r < 2.5:
        return Severity.MEDIUM.value
    if r < 4.0:
        return Severity.HIGH.value
    return Severity.CRITICAL.value


def evaluate_detector_scores(scores: np.ndarray, is_anomaly: np.ndarray) -> dict:
    y = np.asarray(is_anomaly).astype(int)
    s = np.asarray(scores, dtype=float)
    if y.sum() == 0 or (y == 0).sum() == 0:
        return {"auroc": float("nan"), "auprc": float("nan"), "n": int(len(y)),
                "anomalies": int(y.sum())}
    return {"auroc": round(float(roc_auc_score(y, s)), 4),
            "auprc": round(float(average_precision_score(y, s)), 4),
            "n": int(len(y)), "anomalies": int(y.sum())}
