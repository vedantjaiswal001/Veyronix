"""Phase 8: metrics, tracing (no-op), and drift/model-health monitoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from veyronix.observability import metrics as m
from veyronix.observability.drift import calibration_drift, feature_drift, psi
from veyronix.observability.monitor import assess_model_health
from veyronix.observability.tracing import traced

pytestmark = pytest.mark.unit
RNG = np.random.default_rng(0)


def _frame(mean, n=400, cols=("f1", "f2", "f3")):
    return pd.DataFrame({c: RNG.normal(mean, 5, n) for c in cols})


def test_psi_zero_for_same_distribution():
    a = RNG.normal(50, 5, 1000)
    b = RNG.normal(50, 5, 1000)
    assert psi(a, b) < 0.1


def test_psi_high_for_shifted_distribution():
    a = RNG.normal(50, 5, 1000)
    b = RNG.normal(80, 5, 1000)
    assert psi(a, b) > 0.25


def test_feature_drift_detects_shifted_columns():
    ref = _frame(50)
    cur = _frame(50)
    cur["f2"] = cur["f2"] + 40  # shift one feature
    rep = feature_drift(ref, cur, ["f1", "f2", "f3"])
    assert rep["drift_detected"]
    assert "f2" in rep["flagged"]
    assert "f1" not in rep["flagged"]


def test_model_health_healthy_when_stable():
    ref = _frame(50)
    cur = _frame(50)
    rep = assess_model_health(ref, cur, ["f1", "f2", "f3"], push_metrics=False)
    assert rep["status"] == "healthy" and rep["health_score"] == 1.0


def test_model_health_degraded_under_broad_drift():
    ref = _frame(50)
    cur = _frame(50)
    for c in ["f1", "f2", "f3"]:
        cur[c] = cur[c] + 40
    rep = assess_model_health(ref, cur, ["f1", "f2", "f3"], push_metrics=False)
    assert rep["status"] == "degraded" and rep["health_score"] == 0.0


def test_calibration_drift_flag():
    assert calibration_drift(0.05, 0.20)["drift_detected"] is True
    assert calibration_drift(0.05, 0.06)["drift_detected"] is False


def test_metrics_record_and_export():
    m.record_tool_call("query_metrics", True)
    m.record_tokens("offline", 0)  # zero should not error
    m.record_error("test")
    m.set_incident_probability(0.87)
    payload, ctype = m.export_metrics()
    text = payload.decode()
    assert "veyronix_tool_calls_total" in text
    assert "veyronix_incident_probability" in text
    assert "text/plain" in ctype


def test_observe_timer_records():
    with m.observe(m.MODEL_LATENCY, model="unit_test"):
        _ = sum(range(1000))
    text = m.export_metrics()[0].decode()
    assert "veyronix_model_inference_seconds" in text


def test_tracing_is_noop_without_otel():
    with traced("unit-span", attr="x"):
        value = 1 + 1
    assert value == 2  # context manager works even with OTel absent
