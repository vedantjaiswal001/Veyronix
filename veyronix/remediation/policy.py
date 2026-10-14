"""Remediation safety policy (spec §33/§53).

The AI can PROPOSE; it can never arbitrarily EXECUTE. This module enforces that
boundary with: an allowlist (actions are an enum — there is no code path that
runs an arbitrary string or shell command), per-action parameter validation,
risk-tiered gating (auto / approval / elevated approval / blocked), and
authorization by actor role. ``DELETE_DATABASE`` is permanently BLOCKED.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, field_validator

from veyronix.core.constants import ActionType, RiskTier, ServiceName
from veyronix.simulation.actions import ACTION_MODELS

VALID_SERVICES = {s.value for s in ServiceName}


@dataclass(frozen=True)
class Actor:
    name: str
    role: str  # operator | approver | admin | system


class ActionRequest(BaseModel):
    """A validated, allowlisted remediation request. Construction fails if the
    action is unknown or its parameters are invalid."""

    action_type: ActionType          # enum — cannot be an arbitrary string
    params: dict = {}
    run_id: str | None = None
    incident_id: int | None = None
    proposed_by: str = "recommender"
    reason: str = ""

    @field_validator("params")
    @classmethod
    def _validate_params(cls, v, info):
        action = info.data.get("action_type")
        if action is not None:
            validate_params(action, v)
        return v


# Per-action required parameters and their validators.
def validate_params(action: ActionType, params: dict) -> None:
    """Raise ValueError if params are missing / invalid for the action."""
    def need_service():
        svc = params.get("service")
        if svc not in VALID_SERVICES:
            raise ValueError(f"invalid or missing 'service' (got {svc!r})")

    def need_int(key, lo, hi):
        val = params.get(key)
        if not isinstance(val, int) or isinstance(val, bool) or not (lo <= val <= hi):
            raise ValueError(f"'{key}' must be an int in [{lo}, {hi}] (got {val!r})")

    if action in (ActionType.RESTART_SERVICE, ActionType.SCALE_UP_SERVICE,
                  ActionType.FLUSH_CACHE, ActionType.FAILOVER_DEPENDENCY,
                  ActionType.CLEAR_QUEUE):
        need_service()
    if action == ActionType.SCALE_UP_SERVICE:
        need_int("replicas", 1, 20)
    if action == ActionType.SCALE_UP_DATABASE:
        need_int("size_gb", 1, 4096)
    if action == ActionType.ROLLBACK_DEPLOYMENT and not (
            params.get("deployment_id") or params.get("service")):
        raise ValueError("rollback requires 'deployment_id' or 'service'")
    # NO_OP and DELETE_DATABASE take no params; extras are ignored but the latter
    # is blocked at gating regardless.


def assess_risk(action: ActionType) -> RiskTier:
    return ACTION_MODELS[action].risk


class GateDecision(BaseModel):
    action: str
    risk: str
    decision: str        # auto | requires_approval | requires_elevated_approval | blocked
    rationale: str


def gate(action: ActionType, allow_auto: bool) -> GateDecision:
    """Decide how an action may proceed given its risk tier + the kill-switch."""
    tier = assess_risk(action)
    if tier == RiskTier.BLOCKED:
        decision, why = "blocked", "action is permanently blocked"
    elif tier == RiskTier.HIGH:
        decision, why = "requires_elevated_approval", "high-risk action needs admin approval"
    elif tier == RiskTier.MEDIUM:
        decision, why = "requires_approval", "medium-risk action needs human approval"
    else:  # LOW
        if allow_auto:
            decision, why = "auto", "low-risk action; auto-remediation enabled"
        else:
            decision, why = "requires_approval", "auto-remediation disabled by kill-switch"
    return GateDecision(action=action.value, risk=tier.value, decision=decision, rationale=why)


def can_approve(actor: Actor, action: ActionType) -> bool:
    """Whether an actor's role may approve an action of this risk tier."""
    tier = assess_risk(action)
    if tier == RiskTier.BLOCKED:
        return False
    if tier == RiskTier.HIGH:
        return actor.role == "admin"
    if tier == RiskTier.MEDIUM:
        return actor.role in ("approver", "admin")
    return actor.role in ("operator", "approver", "admin")
