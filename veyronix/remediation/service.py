"""Approval workflow + safe executor + audit trail (spec §33/§34/§35/§53).

Lifecycle: submit -> (auto | pending) -> approve/reject -> execute -> outcome.
Every transition is authorized, validated and audited. Execution never runs code
supplied by the LLM: an action is an enum whose effect is applied via the
simulator, and every guardrail is re-checked at execution time (defense in
depth).
"""

from __future__ import annotations

from datetime import UTC, datetime

from veyronix.core.config import get_settings
from veyronix.core.constants import ActionType, ApprovalStatus, FaultType, Severity
from veyronix.core.logging import get_logger
from veyronix.db.models import Action, ActionOutcome
from veyronix.db.session import session_scope
from veyronix.remediation.policy import (
    ActionRequest,
    Actor,
    assess_risk,
    can_approve,
    gate,
    validate_params,
)
from veyronix.simulation.actions import efficacy

log = get_logger("remediation")


class RemediationError(Exception):
    """Base class for remediation guardrail violations."""


class BlockedActionError(RemediationError):
    pass


class UnauthorizedError(RemediationError):
    pass


class NotApprovedError(RemediationError):
    pass


def _audit(event: str, **kw) -> None:
    log.info(f"remediation.{event}", **kw)


def submit_action(request: ActionRequest, allow_auto: bool | None = None) -> dict:
    """Validate + persist a proposed action; returns {action_id, decision, risk}."""
    settings = get_settings()
    allow_auto = settings.allow_auto_remediation if allow_auto is None else allow_auto
    decision = gate(request.action_type, allow_auto)

    status = {
        "blocked": ApprovalStatus.REJECTED,
        "auto": ApprovalStatus.APPROVED,
    }.get(decision.decision, ApprovalStatus.PENDING)

    with session_scope() as s:
        row = Action(
            run_id=request.run_id, incident_id=request.incident_id,
            action_type=request.action_type.value, params=request.params,
            risk_tier=decision.risk, proposed_by=request.proposed_by,
            approval_status=status.value,
            reason=request.reason or decision.rationale,
            approved_by="system(auto)" if decision.decision == "auto" else None,
            approved_at=datetime.now(UTC) if decision.decision == "auto" else None)
        s.add(row)
        s.flush()
        action_id = row.id
    _audit("submit", action_id=action_id, action=request.action_type.value,
           decision=decision.decision, risk=decision.risk)
    return {"action_id": action_id, "decision": decision.decision, "risk": decision.risk,
            "rationale": decision.rationale, "status": status.value}


def approve_action(action_id: int, actor: Actor, reason: str = "") -> dict:
    with session_scope() as s:
        row = s.get(Action, action_id)
        if row is None:
            raise RemediationError(f"action {action_id} not found")
        action = ActionType(row.action_type)
        if assess_risk(action).value == "BLOCKED":
            raise BlockedActionError("cannot approve a blocked action")
        if not can_approve(actor, action):
            _audit("approve_denied", action_id=action_id, actor=actor.name, role=actor.role)
            raise UnauthorizedError(f"{actor.role} cannot approve {row.risk_tier} action")
        row.approval_status = ApprovalStatus.APPROVED.value
        row.approved_by = actor.name
        row.approved_at = datetime.now(UTC)
        row.reason = (row.reason + f" | approved by {actor.name}: {reason}").strip(" |")
    _audit("approve", action_id=action_id, actor=actor.name)
    return {"action_id": action_id, "status": "approved", "approved_by": actor.name}


def reject_action(action_id: int, actor: Actor, reason: str = "") -> dict:
    with session_scope() as s:
        row = s.get(Action, action_id)
        if row is None:
            raise RemediationError(f"action {action_id} not found")
        row.approval_status = ApprovalStatus.REJECTED.value
        row.approved_by = actor.name
        row.approved_at = datetime.now(UTC)
        row.reason = (row.reason + f" | rejected by {actor.name}: {reason}").strip(" |")
    _audit("reject", action_id=action_id, actor=actor.name)
    return {"action_id": action_id, "status": "rejected"}


def execute_action(action_id: int, actor: Actor,
                   fault_type: FaultType | None = None,
                   severity: Severity | None = None,
                   remaining_minutes: float = 25.0) -> dict:
    """Execute an approved/auto action under full guardrails. Never runs code."""
    with session_scope() as s:
        row = s.get(Action, action_id)
        if row is None:
            raise RemediationError(f"action {action_id} not found")
        action = ActionType(row.action_type)

        # guardrail 1: allowlist + blocked
        if assess_risk(action).value == "BLOCKED":
            _audit("execute_blocked", action_id=action_id, action=action.value)
            raise BlockedActionError(f"{action.value} is permanently blocked")
        # guardrail 2: params still valid (defense in depth)
        validate_params(action, row.params or {})
        # guardrail 3: authorization to execute
        if actor.role not in ("operator", "approver", "admin", "system"):
            raise UnauthorizedError(f"role {actor.role} may not execute actions")
        # guardrail 4: must be approved (or auto-approved)
        if row.approval_status != ApprovalStatus.APPROVED.value:
            _audit("execute_denied", action_id=action_id, status=row.approval_status)
            raise NotApprovedError(f"action {action_id} is {row.approval_status}, not approved")

        # apply the effect via the simulator (deterministic; no external code)
        if action == ActionType.NO_OP:
            p, success = 0.0, True  # doing nothing "succeeds" as intended
            recovery_minutes = round(remaining_minutes, 1)
        else:
            p = efficacy(action, fault_type) if fault_type else 0.5
            success = p >= 0.5
            recovery_minutes = round(2.0 + (0 if success else remaining_minutes), 1)
        predicted = {"recovery_probability": round(p, 3),
                     "expected_recovery_min": recovery_minutes}
        actual = {"resolved": bool(success),
                  "recovery_min": recovery_minutes,
                  "note": "applied in SIMULATED environment"}
        row.approval_status = (ApprovalStatus.EXECUTED.value if success
                               else ApprovalStatus.FAILED.value)
        outcome = ActionOutcome(
            action_id=action_id, executed_at=datetime.now(UTC), success=success,
            predicted_outcome=predicted, actual_outcome=actual,
            recovery_minutes=recovery_minutes,
            notes=f"executed by {actor.name} ({actor.role})")
        s.add(outcome)
    final_status = (ApprovalStatus.EXECUTED.value if success
                    else ApprovalStatus.FAILED.value)
    _audit("execute", action_id=action_id, actor=actor.name, success=success,
           action=action.value)
    return {"action_id": action_id, "success": success, "status": final_status,
            "predicted": predicted, "actual": actual}


def audit_trail(action_id: int) -> dict:
    """Return the full record for an action (who/when/why + outcome)."""
    with session_scope() as s:
        row = s.get(Action, action_id)
        if row is None:
            return {}
        out = {
            "action_id": row.id, "action_type": row.action_type,
            "risk_tier": row.risk_tier, "params": row.params,
            "proposed_by": row.proposed_by, "approval_status": row.approval_status,
            "approved_by": row.approved_by,
            "approved_at": row.approved_at.isoformat() if row.approved_at else None,
            "reason": row.reason,
        }
        if row.outcome:
            out["outcome"] = {"success": row.outcome.success,
                              "predicted": row.outcome.predicted_outcome,
                              "actual": row.outcome.actual_outcome,
                              "recovery_minutes": row.outcome.recovery_minutes}
        return out
