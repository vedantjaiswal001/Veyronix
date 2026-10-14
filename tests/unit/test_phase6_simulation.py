"""Phase 6: counterfactual simulation, business impact, action recommender."""

from __future__ import annotations

import pytest

from veyronix.core.constants import ActionType, FaultType, RiskTier, Severity
from veyronix.simulation.actions import candidate_actions, efficacy
from veyronix.simulation.business_impact import estimate_impact
from veyronix.simulation.counterfactual import IncidentState, simulate_actions
from veyronix.simulation.recommender import recommend

pytestmark = pytest.mark.unit


def _state(fault=FaultType.BAD_DEPLOYMENT, sev=Severity.CRITICAL):
    return IncidentState(fault, sev, baseline_revenue_per_min=190000,
                         degraded_revenue_per_min=62000, active_users=5000,
                         remaining_minutes=45, baseline_orders_per_min=340,
                         degraded_orders_per_min=110, affected_fraction=0.7,
                         root_cause_confidence=0.9)


def test_efficacy_correct_high_noop_zero():
    assert efficacy(ActionType.ROLLBACK_DEPLOYMENT, FaultType.BAD_DEPLOYMENT) >= 0.85
    assert efficacy(ActionType.NO_OP, FaultType.BAD_DEPLOYMENT) == 0.0
    assert efficacy(ActionType.FLUSH_CACHE, FaultType.DATABASE_OVERLOAD) <= 0.2


def test_blocked_action_never_a_candidate():
    for ft in FaultType:
        assert ActionType.DELETE_DATABASE not in candidate_actions(ft)


def test_do_nothing_has_the_most_loss():
    est = {e.action: e for e in simulate_actions(_state())}
    noop_loss = est[ActionType.NO_OP.value].business_impact["revenue_loss"]
    for e in est.values():
        assert e.business_impact["revenue_loss"] <= noop_loss + 1e-6


def test_correct_action_reduces_downtime_vs_nothing():
    est = {e.action: e for e in simulate_actions(_state())}
    assert (est[ActionType.ROLLBACK_DEPLOYMENT.value].expected_downtime_min
            < est[ActionType.NO_OP.value].expected_downtime_min)


def test_estimates_carry_uncertainty_and_estimate_label():
    for e in simulate_actions(_state()):
        assert len(e.recovery_probability_range) == 2
        assert len(e.downtime_range) == 2
        assert e.kind == "counterfactual_estimate"
        assert "estimate" in e.business_impact["disclaimer"]


def test_recommender_ranks_correct_action_first():
    recs = recommend(_state())
    assert recs[0].action == ActionType.ROLLBACK_DEPLOYMENT.value
    assert recs[0].expected_benefit > 0
    assert any(r.action == ActionType.NO_OP.value for r in recs)  # do-nothing considered


def test_recommendation_reason_flags_approval_for_medium_risk():
    top = recommend(_state())[0]
    assert top.risk == RiskTier.MEDIUM.value
    assert "approval" in top.reason.lower()


def test_business_impact_is_labelled_and_nonnegative():
    imp = estimate_impact(100000, 40000, 30, 5000, 300, 120, 0.6).to_dict()
    assert imp["revenue_loss"] >= 0 and imp["lost_orders"] >= 0
    assert "SIMULATED" in imp["disclaimer"]


def test_database_overload_prefers_scale_db():
    recs = recommend(_state(FaultType.DATABASE_OVERLOAD, Severity.HIGH))
    assert recs[0].action == ActionType.SCALE_UP_DATABASE.value
