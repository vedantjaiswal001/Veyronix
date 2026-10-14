"""Phase 2: feature engineering, labels, splits, and leakage audits."""

from __future__ import annotations

import pytest

from veyronix.core.config import get_settings
from veyronix.features.build import build_dataset, incidents_to_df
from veyronix.features.labels import build_labels
from veyronix.features.leakage import split_audit, truncation_invariance_report
from veyronix.features.windows import build_feature_matrix, feature_columns
from veyronix.simulator.run import run_simulation

pytestmark = pytest.mark.ml


@pytest.fixture(scope="module")
def run():
    return run_simulation(minutes=420, seed=42, run_id="p2")


def test_feature_matrix_has_no_nans_and_expected_shape(run):
    feats = build_feature_matrix(run.metrics, run.business)
    cols = feature_columns(feats)
    assert len(feats) == 420  # one row per minute
    assert len(cols) > 200
    assert not feats[cols].isna().any().any()


def test_features_are_leakage_free_truncation_invariant(run):
    rep = truncation_invariance_report(
        run.metrics, run.business, probe_minutes=[100, 200, 300, 400])
    assert rep["leak_free"], rep["leaky_features"]


def test_labels_predict_future_not_present(run):
    horizon = get_settings().prediction_horizon_minutes
    feats = build_feature_matrix(run.metrics, run.business)
    labels = build_labels(feats, incidents_to_df(run), horizon)
    inc = incidents_to_df(run)
    # a positive row must have an incident starting within the horizon ahead
    for _, row in labels[labels.incident_within_h == 1].sample(
            min(20, int(labels.incident_within_h.sum())), random_state=0).iterrows():
        t = row.minute
        upcoming = inc[(inc.start_min > t) & (inc.start_min <= t + horizon)]
        assert not upcoming.empty
    # a row flagged current_incident must actually be inside an incident window
    for _, row in labels[labels.current_incident == 1].sample(
            min(20, int(labels.current_incident.sum())), random_state=0).iterrows():
        t = row.minute
        active = inc[(inc.start_min <= t) & (t < inc.end_min)]
        assert not active.empty


def test_label_columns_are_not_features(run):
    feats = build_feature_matrix(run.metrics, run.business)
    cols = set(feature_columns(feats))
    leaky = {"incident_within_h", "current_incident", "eta_minutes",
             "upcoming_severity", "active_severity"}
    assert cols.isdisjoint(leaky)


def test_dataset_split_is_chronological_with_embargo():
    df = build_dataset(seeds=[42, 43], minutes=420, persist=False)
    s = get_settings()
    embargo = max(s.prediction_horizon_minutes, s.lookback_minutes)
    audit = split_audit(df, embargo=embargo)
    assert audit["ok"], audit["problems"]
    # test split minutes are strictly later than train split minutes (per run)
    for _, g in df.groupby("run_id"):
        if (g.split == "train").any() and (g.split == "test").any():
            assert g[g.split == "test"].minute.min() > g[g.split == "train"].minute.max()
