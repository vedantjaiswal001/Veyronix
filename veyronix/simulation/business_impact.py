"""Business-impact model (spec §22).

Maps technical degradation to business outcomes: affected users, lost orders,
revenue loss and downtime cost. Every figure is explicitly a MODEL ESTIMATE from
a SIMULATED environment, not financial truth.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BusinessImpact:
    downtime_minutes: float
    revenue_loss: float
    lost_orders: float
    affected_users: float
    downtime_cost: float
    currency: str = "INR"
    disclaimer: str = "model estimate from SIMULATED environment; not financial truth"

    def to_dict(self) -> dict:
        return {
            "downtime_minutes": round(self.downtime_minutes, 1),
            "revenue_loss": round(self.revenue_loss, 2),
            "lost_orders": round(self.lost_orders, 1),
            "affected_users": round(self.affected_users, 0),
            "downtime_cost": round(self.downtime_cost, 2),
            "currency": self.currency, "disclaimer": self.disclaimer,
        }


# Assumed cost of an engineer-hour handling an incident (relative units).
INCIDENT_HANDLING_COST_PER_MIN = 50.0


def estimate_impact(
    baseline_revenue_per_min: float, degraded_revenue_per_min: float,
    downtime_minutes: float, active_users: float,
    baseline_orders_per_min: float = 0.0, degraded_orders_per_min: float = 0.0,
    affected_fraction: float = 1.0,
) -> BusinessImpact:
    """Estimate business impact over a downtime window."""
    rev_loss_rate = max(0.0, baseline_revenue_per_min - degraded_revenue_per_min)
    revenue_loss = rev_loss_rate * downtime_minutes
    order_loss_rate = max(0.0, baseline_orders_per_min - degraded_orders_per_min)
    lost_orders = order_loss_rate * downtime_minutes
    affected_users = active_users * affected_fraction
    downtime_cost = downtime_minutes * INCIDENT_HANDLING_COST_PER_MIN
    return BusinessImpact(downtime_minutes, revenue_loss, lost_orders,
                          affected_users, downtime_cost)
