"""Phase 2: data-quality layer detects (and reports) issues, never hides them."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from veyronix.ingestion.quality import check_metrics_frame, distribution_shift
from veyronix.simulator.run import run_simulation

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def clean_metrics():
    return run_simulation(minutes=180, seed=1, run_id="q").metrics


def test_clean_frame_has_no_errors(clean_metrics):
    rep = check_metrics_frame(clean_metrics)
    assert rep.ok
    assert not any(i.severity == "error" for i in rep.issues)


def test_detects_impossible_values(clean_metrics):
    df = clean_metrics.copy()
    df.loc[df.index[:5], "cpu_pct"] = 999.0  # impossible
    rep = check_metrics_frame(df)
    assert not rep.ok
    assert any(i.check == "impossible_values" and i.count >= 5 for i in rep.issues)


def test_detects_duplicates(clean_metrics):
    df = pd.concat([clean_metrics, clean_metrics.iloc[:10]], ignore_index=True)
    rep = check_metrics_frame(df)
    assert any(i.check == "duplicate_rows" and i.count >= 10 for i in rep.issues)


def test_detects_missing_columns(clean_metrics):
    df = clean_metrics.drop(columns=["error_rate"])
    rep = check_metrics_frame(df)
    assert not rep.ok
    assert any(i.check == "schema_missing_columns" for i in rep.issues)


def test_detects_missing_values(clean_metrics):
    df = clean_metrics.copy()
    df.loc[df.index[:7], "latency_p95_ms"] = np.nan
    rep = check_metrics_frame(df)
    assert any(i.check == "missing_values" and i.count >= 7 for i in rep.issues)


def test_distribution_shift_flags_large_shift():
    ref = pd.Series(np.random.default_rng(0).normal(50, 5, 500))
    same = pd.Series(np.random.default_rng(1).normal(50, 5, 500))
    shifted = pd.Series(np.random.default_rng(2).normal(90, 5, 500))
    assert not distribution_shift(ref, same)["flagged"]
    assert distribution_shift(ref, shifted)["flagged"]
