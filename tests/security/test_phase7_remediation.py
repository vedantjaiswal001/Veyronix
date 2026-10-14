"""Phase 7: safe remediation — approval workflow + security guarantees (spec §53).

The LLM/agent may PROPOSE actions but can never EXECUTE arbitrary ones. These
tests are the teeth of that guarantee.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from veyronix.core.constants import ActionType, FaultType
from veyronix.remediation.policy import ActionRequest, Actor
from veyronix.remediation.service import (
    BlockedActionError,
    NotApprovedError,
    UnauthorizedError,
    approve_action,
    audit_trail,
    execute_action,
    reject_action,
    submit_action,
)

pytestmark = [pytest.mark.security, pytest.mark.integration]

OPERATOR = Actor("alice", "operator")
APPROVER = Actor("bob", "approver")
ADMIN = Actor("carol", "admin")
GUEST = Actor("eve", "guest")


def _req(action, params=None, **kw):
    return ActionRequest(action_type=action, params=params or {}, **kw)


# ---- allowlist: the LLM cannot inject an arbitrary command ----
def test_arbitrary_command_is_not_a_valid_action():
    with pytest.raises(ValidationError):
        ActionRequest(action_type="rm -rf / --no-preserve-root")  # not in the enum
    with pytest.raises(ValidationError):
        ActionRequest(action_type="DROP TABLE incidents")


# ---- parameter validation ----
def test_invalid_or_missing_params_rejected():
    with pytest.raises(ValueError):
        _req(ActionType.RESTART_SERVICE, {})  # missing service
    with pytest.raises(ValueError):
        _req(ActionType.RESTART_SERVICE, {"service": "not_a_service"})
    with pytest.raises(ValueError):
        _req(ActionType.SCALE_UP_SERVICE, {"service": "order_service", "replicas": 999})
    with pytest.raises(ValueError):
        _req(ActionType.SCALE_UP_SERVICE, {"service": "order_service", "replicas": "three"})


# ---- BLOCKED stays blocked at every stage ----
def test_delete_database_is_permanently_blocked(db):
    sub = submit_action(_req(ActionType.DELETE_DATABASE), allow_auto=True)
    assert sub["decision"] == "blocked" and sub["status"] == "rejected"
    with pytest.raises(BlockedActionError):
        approve_action(sub["action_id"], ADMIN, "trying anyway")
    with pytest.raises(BlockedActionError):
        execute_action(sub["action_id"], ADMIN)


# ---- MEDIUM risk: full human-approval lifecycle ----
def test_medium_risk_requires_approval_then_executes(db):
    sub = submit_action(_req(ActionType.ROLLBACK_DEPLOYMENT,
                             {"service": "order_service"}, run_id="r1", incident_id=None))
    assert sub["decision"] == "requires_approval" and sub["status"] == "pending"
    aid = sub["action_id"]

    # cannot execute before approval
    with pytest.raises(NotApprovedError):
        execute_action(aid, OPERATOR, fault_type=FaultType.BAD_DEPLOYMENT)
    # operator role cannot approve a medium-risk action
    with pytest.raises(UnauthorizedError):
        approve_action(aid, OPERATOR, "self-approve")
    # approver can
    approve_action(aid, APPROVER, "looks correct")
    res = execute_action(aid, OPERATOR, fault_type=FaultType.BAD_DEPLOYMENT)
    assert res["success"] is True and res["status"] == "executed"

    trail = audit_trail(aid)
    assert trail["approved_by"] == "bob"
    assert trail["outcome"]["success"] is True
    assert "predicted" in trail["outcome"] and "actual" in trail["outcome"]


# ---- LOW risk gated by the global kill-switch ----
def test_low_risk_auto_only_when_enabled(db):
    pending = submit_action(_req(ActionType.RESTART_SERVICE, {"service": "user_service"}),
                            allow_auto=False)
    assert pending["decision"] == "requires_approval"

    auto = submit_action(_req(ActionType.RESTART_SERVICE, {"service": "user_service"}),
                         allow_auto=True)
    assert auto["decision"] == "auto" and auto["status"] == "approved"
    res = execute_action(auto["action_id"], Actor("veyronix", "system"),
                         fault_type=FaultType.SERVICE_CRASH)
    assert res["success"] is True


# ---- unauthorized execution role blocked ----
def test_unauthorized_role_cannot_execute(db):
    sub = submit_action(_req(ActionType.RESTART_SERVICE, {"service": "user_service"}),
                        allow_auto=True)
    with pytest.raises(UnauthorizedError):
        execute_action(sub["action_id"], GUEST, fault_type=FaultType.SERVICE_CRASH)


# ---- rejected actions never execute ----
def test_rejected_action_cannot_execute(db):
    sub = submit_action(_req(ActionType.ROLLBACK_DEPLOYMENT, {"service": "order_service"}))
    reject_action(sub["action_id"], APPROVER, "risky right now")
    with pytest.raises(NotApprovedError):
        execute_action(sub["action_id"], OPERATOR, fault_type=FaultType.BAD_DEPLOYMENT)


# ---- wrong action underperforms (outcome measured honestly) ----
def test_wrong_action_reports_failure(db):
    # restarting a service does not fix a database overload
    sub = submit_action(_req(ActionType.RESTART_SERVICE, {"service": "user_service"}),
                        allow_auto=True)
    res = execute_action(sub["action_id"], ADMIN, fault_type=FaultType.DATABASE_OVERLOAD)
    assert res["success"] is False and res["status"] == "failed"
