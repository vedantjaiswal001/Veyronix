"""Data-quality layer (spec §11).

Detects — and *reports*, never silently fixes — missing data, duplicates,
invalid timestamps, impossible values, schema drift, sudden distribution shifts
and delayed events. The philosophy is observability, not silent cleaning: the
pipeline surfaces exactly what happened so downstream ML can trust its inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from veyronix.core.logging import get_logger
from veyronix.simulator.generator import METRIC_CHANNELS
from veyronix.simulator.profiles import METRIC_BOUNDS

log = get_logger("ingestion.quality")

REQUIRED_METRIC_COLUMNS = {"run_id", "ts", "minute", "service", *METRIC_CHANNELS}


@dataclass
class QualityIssue:
    check: str
    severity: str            # info | warn | error
    count: int
    detail: str
    sample: list = field(default_factory=list)


@dataclass
class QualityReport:
    rows: int
    issues: list[QualityIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    def add(self, check, severity, count, detail, sample=None) -> None:
        if count:
            self.issues.append(QualityIssue(check, severity, int(count), detail, sample or []))

    def to_dict(self) -> dict:
        return {
            "rows": self.rows, "ok": self.ok,
            "issues": [i.__dict__ for i in self.issues],
        }


def check_metrics_frame(df: pd.DataFrame, expected_step_s: int = 60) -> QualityReport:
    """Run all data-quality checks on a metrics frame."""
    rep = QualityReport(rows=len(df))

    # 1. schema drift — missing / unexpected columns
    missing = REQUIRED_METRIC_COLUMNS - set(df.columns)
    extra = set(df.columns) - REQUIRED_METRIC_COLUMNS
    rep.add("schema_missing_columns", "error", len(missing), f"missing: {sorted(missing)}")
    rep.add("schema_unexpected_columns", "info", len(extra), f"extra: {sorted(extra)}")
    if missing:
        return rep  # can't check further without the schema

    # 2. missing values
    nulls = int(df[list(METRIC_CHANNELS)].isna().sum().sum())
    rep.add("missing_values", "warn", nulls, "null metric values present")

    # 3. duplicate (run_id, service, ts)
    dup = int(df.duplicated(subset=["run_id", "service", "ts"]).sum())
    rep.add("duplicate_rows", "warn", dup, "duplicate (run_id, service, ts) keys")

    # 4. invalid timestamps (non-monotonic within a service)
    bad_ts = 0
    for _, g in df.sort_values("ts").groupby(["run_id", "service"], observed=True):
        deltas = g["ts"].diff().dropna().dt.total_seconds()
        bad_ts += int((deltas <= 0).sum())
    rep.add("invalid_timestamps", "warn", bad_ts, "non-increasing timestamps within a service")

    # 5. impossible values (outside physical bounds)
    impossible = 0
    for ch, (lo, hi) in METRIC_BOUNDS.items():
        if ch in df.columns:
            impossible += int(((df[ch] < lo) | (df[ch] > hi)).sum())
    rep.add("impossible_values", "error", impossible, "metric values outside physical bounds")

    # 6. delayed events — gaps larger than the expected step
    delayed = 0
    for _, g in df.sort_values("ts").groupby(["run_id", "service"], observed=True):
        deltas = g["ts"].diff().dropna().dt.total_seconds()
        delayed += int((deltas > expected_step_s * 1.5).sum())
    rep.add("delayed_events", "info", delayed,
            f"inter-sample gap > {expected_step_s * 1.5:.0f}s")

    log.info("quality.metrics", **{i.check: i.count for i in rep.issues} or {"clean": True})
    return rep


def distribution_shift(reference: pd.Series, current: pd.Series, z_thresh: float = 4.0) -> dict:
    """Population Stability-style shift check via standardized mean difference.

    Returns the standardized shift and whether it exceeds the threshold. Used
    both for data-quality (unexpected shift) and later for feature-drift
    monitoring (spec §36/§60).
    """
    ref = reference.dropna().to_numpy(dtype=float)
    cur = current.dropna().to_numpy(dtype=float)
    if len(ref) < 10 or len(cur) < 10:
        return {"shift": 0.0, "flagged": False, "reason": "insufficient_data"}
    mu, sd = float(ref.mean()), float(ref.std() + 1e-9)
    z = abs(float(cur.mean()) - mu) / sd
    return {"shift": round(z, 3), "flagged": bool(z > z_thresh),
            "ref_mean": round(mu, 4), "cur_mean": round(float(cur.mean()), 4)}
