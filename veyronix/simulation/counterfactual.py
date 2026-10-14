"""Counterfactual action simulation (spec §30/§31).

For a detected incident we estimate, for each candidate action (including "do
nothing"), the recovery probability, expected downtime, business impact, cost and
risk — each with uncertainty. These are COUNTERFACTUAL ESTIMATES ("what would
happen if we did X"), explicitly distinguished from the deterministic PREDICTION
of the incident itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from veyronix.core.constants import ActionType, FaultType, Severity
from veyronix.simulation.actions import (
    ACTION_MODELS,
    candidate_actions,
    efficacy,
)
from veyronix.simulation.business_impact import estimate_impact

# Natural remaining duration (minutes) if nothing is done, by severity.
_REMAINING_BY_SEVERITY = {
    Severity.LOW: 12.0, Severity.MEDIUM: 22.0, Severity.HIGH: 34.0, Severity.CRITICAL: 45.0,
}


@dataclass
class IncidentState:
    fault_type: FaultType
    severity: Severity
    baseline_revenue_per_min: float
    degraded_revenue_per_min: float
    active_users: float
    remaining_minutes: float
    baseline_orders_per_min: float = 0.0
    degraded_orders_per_min: float = 0.0
    affected_fraction: float = 1.0
    root_cause_confidence: float = 0.7


@dataclass
class ActionEstimate:
    action: str
    recovery_probability: float
    recovery_probability_range: tuple[float, float]
    expected_downtime_min: float
    downtime_range: tuple[float, float]
    business_impact: dict
    cost: float
    risk: str
    confidence: float
    kind: str = "counterfactual_estimate"
    note: str = field(default="what-if estimate; not a guaranteed outcome")


def _clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def simulate_action(state: IncidentState, action: ActionType) -> ActionEstimate:
    model = ACTION_MODELS[action]
    p = efficacy(action, state.fault_type)
    success_dt = model.exec_minutes + 2.0
    fail_dt = state.remaining_minutes

    def expected_dt(prob):
        return prob * success_dt + (1 - prob) * fail_dt

    dt = expected_dt(p)
    p_lo, p_hi = _clip(p - 0.1), _clip(p + 0.1)
    # higher recovery prob => lower downtime, so the range flips
    dt_range = (round(expected_dt(p_hi), 1), round(expected_dt(p_lo), 1))

    impact = estimate_impact(
        state.baseline_revenue_per_min, state.degraded_revenue_per_min, dt,
        state.active_users, state.baseline_orders_per_min,
        state.degraded_orders_per_min, state.affected_fraction)

    conf = _clip(0.4 + 0.45 * state.root_cause_confidence
                 + (0.1 if p >= 0.85 else 0.0), 0.0, 0.95)
    return ActionEstimate(
        action=action.value, recovery_probability=round(p, 3),
        recovery_probability_range=(round(p_lo, 3), round(p_hi, 3)),
        expected_downtime_min=round(dt, 1), downtime_range=dt_range,
        business_impact=impact.to_dict(), cost=model.cost_units,
        risk=model.risk.value, confidence=round(conf, 3))


def simulate_actions(state: IncidentState) -> list[ActionEstimate]:
    """Simulate the focused candidate action set for the incident."""
    return [simulate_action(state, a) for a in candidate_actions(state.fault_type)]


def build_incident_state(
    fault_type: FaultType, severity: Severity, business: dict,
    root_cause_confidence: float = 0.7, affected_fraction: float = 0.6,
) -> IncidentState:
    """Construct an IncidentState from a business-metrics tool result (spec §22)."""
    kpis = business.get("kpis", {})
    base_rev = kpis.get("revenue_per_min", {}).get("baseline", 100000.0)
    deg_rev = kpis.get("revenue_per_min", {}).get("window", base_rev)
    base_ord = kpis.get("orders_per_min", {}).get("baseline", 300.0)
    deg_ord = kpis.get("orders_per_min", {}).get("window", base_ord)
    # active users: approximate from baseline revenue if not provided
    active_users = 5000.0
    return IncidentState(
        fault_type=fault_type, severity=severity,
        baseline_revenue_per_min=base_rev, degraded_revenue_per_min=deg_rev,
        active_users=active_users, baseline_orders_per_min=base_ord,
        degraded_orders_per_min=deg_ord, affected_fraction=affected_fraction,
        remaining_minutes=_REMAINING_BY_SEVERITY.get(severity, 22.0),
        root_cause_confidence=root_cause_confidence)
