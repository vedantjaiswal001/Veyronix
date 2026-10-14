"""Label construction for the incident-prediction task (spec §13/§16/§17).

Primary target: ``incident_within_h`` = 1 if an incident STARTS in (t, t+H].
We deliberately mark rows where an incident is already active so the trainer can
exclude them — otherwise the model would learn to *detect* an ongoing incident
rather than *predict* a future one. Precursor rows (before the labelled start)
carry the signal and are correctly labelled positive.

All labels look strictly FORWARD from t; features look strictly backward. That
separation is what keeps the task leakage-free.
"""

from __future__ import annotations

import pandas as pd

from veyronix.core.constants import Severity


def build_labels(feats: pd.DataFrame, incidents: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Attach labels to a per-(run_id, minute) feature frame.

    ``incidents`` columns: run_id?, fault_id, fault_type, root_cause_service,
    start_min, end_min, severity.
    """
    out = feats[["run_id", "minute", "ts"]].copy()
    out["incident_within_h"] = 0
    out["current_incident"] = 0
    out["eta_minutes"] = pd.NA
    out["upcoming_fault_type"] = pd.NA
    out["upcoming_severity"] = pd.NA
    out["upcoming_incident_id"] = pd.NA
    out["active_fault_type"] = pd.NA
    out["active_severity"] = pd.NA
    out["active_incident_id"] = pd.NA

    # group incidents by run (single-run frames have a constant run_id)
    has_run = "run_id" in incidents.columns
    for run_id, grp in out.groupby("run_id", sort=False):
        inc = incidents[incidents["run_id"] == run_id] if has_run else incidents
        idx = grp.index
        minutes = grp["minute"].to_numpy()
        for i, t in zip(idx, minutes, strict=True):
            # currently active?
            active = inc[(inc["start_min"] <= t) & (t < inc["end_min"])]
            if not active.empty:
                a = active.sort_values("severity", key=_sev_key).iloc[-1]
                out.at[i, "current_incident"] = 1
                out.at[i, "active_fault_type"] = a["fault_type"]
                out.at[i, "active_severity"] = a["severity"]
                out.at[i, "active_incident_id"] = int(a["fault_id"])
            # incident starting within (t, t+H]?
            upcoming = inc[(inc["start_min"] > t) & (inc["start_min"] <= t + horizon)]
            if not upcoming.empty:
                nearest = upcoming.sort_values("start_min").iloc[0]
                worst = upcoming.sort_values("severity", key=_sev_key).iloc[-1]
                out.at[i, "incident_within_h"] = 1
                out.at[i, "eta_minutes"] = int(nearest["start_min"] - t)
                out.at[i, "upcoming_fault_type"] = worst["fault_type"]
                out.at[i, "upcoming_severity"] = worst["severity"]
                out.at[i, "upcoming_incident_id"] = int(nearest["fault_id"])
    return out


def _sev_key(col: pd.Series) -> pd.Series:
    order = {s.value: s.rank for s in Severity}
    return col.map(lambda v: order.get(v, -1))
