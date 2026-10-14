"""Phase 10: evaluation suites produce sane, structured, honest results."""

from __future__ import annotations

import pytest

from veyronix.eval.baselines import run_baselines
from veyronix.eval.unknown_failure import run_unknown_failure
from veyronix.features.build import build_dataset

pytestmark = [pytest.mark.ml, pytest.mark.slow]


@pytest.fixture(scope="module", autouse=True)
def ensure_dataset():
    build_dataset(
        seeds=[42, 43, 44, 45, 46, 47, 48, 49],
        minutes=1440,
        persist=True,
    )


def test_baselines_ml_beats_static_on_pr_auc():
    b = run_baselines()["configs"]
    for name in ("static_threshold", "ml_only", "ml_plus_anomaly", "veyronix_full"):
        c = b[name]
        assert 0 <= c["fp_rate"] <= 0.12  # respects the false-alarm budget
        assert 0 <= c["early_warning_rate"] <= 1
        assert 0 <= c["incident_detection_rate"] <= 1
    assert b["ml_only"]["pr_auc"] >= b["static_threshold"]["pr_auc"]
    # ML detects incidents better than a static threshold
    assert b["ml_only"]["incident_detection_rate"] >= b["static_threshold"]["incident_detection_rate"]


def test_unknown_failure_generalizes_to_unseen_fault():
    u = run_unknown_failure()
    assert u["unknown_incidents"] > 0 and u["known_incidents"] > 0
    # the unsupervised detector flags the held-out cascading failure meaningfully
    assert u["unknown_detection_rate"] >= 0.4
    assert u["observed_fpr"] <= 0.2
    assert u["auroc"] > 0.7
