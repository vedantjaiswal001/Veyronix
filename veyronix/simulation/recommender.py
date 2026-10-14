"""Action recommender (spec §32).

Ranks candidate interventions by expected net benefit — revenue saved versus
"do nothing", discounted by operational cost and risk. Each recommendation
carries expected benefit, risk, cost, confidence and a human-readable reason.
"""

from __future__ import annotations

from dataclasses import dataclass

from veyronix.core.constants import ActionType, RiskTier
from veyronix.simulation.counterfactual import ActionEstimate, IncidentState, simulate_actions

COST_WEIGHT = 500.0  # converts an operational-cost unit into currency terms
_RISK_PENALTY = {RiskTier.LOW.value: 0.0, RiskTier.MEDIUM.value: 4000.0,
                 RiskTier.HIGH.value: 20000.0, RiskTier.BLOCKED.value: 1e12}


@dataclass
class Recommendation:
    action: str
    expected_benefit: float     # revenue saved vs. doing nothing
    net_score: float
    recovery_probability: float
    expected_downtime_min: float
    residual_revenue_loss: float
    cost: float
    risk: str
    confidence: float
    reason: str


def recommend(state: IncidentState) -> list[Recommendation]:
    estimates = simulate_actions(state)
    by_action = {e.action: e for e in estimates}
    noop = by_action.get(ActionType.NO_OP.value)
    noop_loss = noop.business_impact["revenue_loss"] if noop else 0.0

    recs: list[Recommendation] = []
    for e in estimates:
        loss = e.business_impact["revenue_loss"]
        benefit = max(0.0, noop_loss - loss)
        net = benefit - e.cost * COST_WEIGHT - _RISK_PENALTY.get(e.risk, 0.0)
        recs.append(Recommendation(
            action=e.action, expected_benefit=round(benefit, 2), net_score=round(net, 2),
            recovery_probability=e.recovery_probability,
            expected_downtime_min=e.expected_downtime_min,
            residual_revenue_loss=loss, cost=e.cost, risk=e.risk,
            confidence=e.confidence, reason=_reason(e, benefit)))
    recs.sort(key=lambda r: r.net_score, reverse=True)
    return recs


def _reason(e: ActionEstimate, benefit: float) -> str:
    if e.action == ActionType.NO_OP.value:
        return (f"Doing nothing lets the incident persist ~{e.expected_downtime_min} min "
                f"with ~₹{e.business_impact['revenue_loss']:.0f} revenue loss.")
    gate = {"LOW": "auto-eligible", "MEDIUM": "requires human approval",
            "HIGH": "requires elevated approval", "BLOCKED": "blocked"}.get(e.risk, "")
    return (f"{e.action.replace('_', ' ').title()}: ~{e.recovery_probability:.0%} likely to "
            f"resolve, cutting downtime to ~{e.expected_downtime_min} min and saving "
            f"~₹{benefit:.0f} vs. doing nothing. {e.risk} risk ({gate}).")


def best_recommendation(state: IncidentState) -> Recommendation:
    return recommend(state)[0]
