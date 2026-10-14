"""Early-warning metric (spec §17).

For each incident, warning time = (incident start) - (first correct pre-incident
alert). A "correct" alert is a positive prediction at minute t within the
lookahead window before the start, i.e. the incident starts within (t, t+H].
We report detection rate and the warning-time distribution over detected
incidents.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def early_warning(
    preds: pd.DataFrame, incidents: pd.DataFrame, horizon: int,
) -> dict:
    """Compute warning-time statistics.

    ``preds`` columns: run_id, minute, pred (0/1). ``incidents`` columns:
    run_id, start_min. Only pre-incident alerts count (t < start_min and the
    start is within the horizon), so this rewards *predicting* not *detecting*.
    """
    warnings: list[float] = []
    detected = 0
    total = 0
    for run_id, inc in incidents.groupby("run_id"):
        p = preds[preds.run_id == run_id]
        pos_minutes = set(p[p.pred == 1].minute.tolist())
        for start in inc.start_min.tolist():
            total += 1
            # correct pre-alerts: minute t in [start-horizon, start) predicted positive
            hits = [t for t in range(start - horizon, start) if t in pos_minutes]
            if hits:
                detected += 1
                warnings.append(float(start - min(hits)))  # earliest correct alert

    if not warnings:
        return {"incidents": total, "detected": detected, "detection_rate": 0.0,
                "warning_minutes": {"mean": None, "median": None, "p25": None,
                                    "p75": None, "min": None, "max": None}}
    w = np.array(warnings)
    return {
        "incidents": total, "detected": detected,
        "detection_rate": round(detected / total, 4),
        "warning_minutes": {
            "mean": round(float(w.mean()), 2), "median": round(float(np.median(w)), 2),
            "p25": round(float(np.percentile(w, 25)), 2),
            "p75": round(float(np.percentile(w, 75)), 2),
            "min": round(float(w.min()), 2), "max": round(float(w.max()), 2),
        },
    }
