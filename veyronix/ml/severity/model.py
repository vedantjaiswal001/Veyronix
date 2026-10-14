"""Incident severity classification (spec §21).

Predicts LOW/MEDIUM/HIGH/CRITICAL for an *active* incident from the current
system state (blast radius via topology-derived features, error/latency,
business exposure). Trained on active-incident rows, evaluated chronologically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from veyronix.core.constants import Severity

SEV_ORDER = [s.value for s in Severity]


class SeverityModel:
    name = "severity_rf"

    def __init__(self, seed: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=4,
            class_weight="balanced", random_state=seed, n_jobs=-1)
        self.columns: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SeverityModel:
        self.columns = list(X.columns)
        self.model.fit(X.to_numpy(dtype=float), y.astype(str))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self.columns].to_numpy(dtype=float))


def train_severity(df: pd.DataFrame, feat_cols: list[str], seed: int = 42) -> dict:
    """Train + evaluate the severity classifier on active-incident rows."""
    active = df[df.current_incident == 1].copy()
    tr = active[active.split == "train"]
    te = active[active.split == "test"]
    if len(tr) < 20 or len(te) < 10 or tr.active_severity.nunique() < 2:
        return {"status": "insufficient_data", "train": len(tr), "test": len(te)}

    model = SeverityModel(seed).fit(tr[feat_cols], tr.active_severity)
    pred = model.predict(te[feat_cols])
    true = te.active_severity.to_numpy(str)
    acc = float((pred == true).mean())
    macro_f1 = float(f1_score(true, pred, labels=SEV_ORDER, average="macro", zero_division=0))
    # off-by-one tolerant accuracy (adjacent severity counts as near-hit)
    rank = {s: i for i, s in enumerate(SEV_ORDER)}
    near = float(np.mean([abs(rank.get(p, 0) - rank.get(t, 0)) <= 1
                          for p, t in zip(pred, true, strict=True)]))
    dist = {s: int((true == s).sum()) for s in SEV_ORDER}
    return {"status": "ok", "accuracy": round(acc, 4), "macro_f1": round(macro_f1, 4),
            "within_one_band": round(near, 4), "train_rows": len(tr),
            "test_rows": len(te), "test_class_distribution": dist,
            "model": model}
