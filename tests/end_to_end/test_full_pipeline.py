"""End-to-end pipeline test (spec §43): fault -> telemetry -> prediction ->
investigation -> root cause -> simulation -> recommendation -> approval ->
action -> outcome, verified as one flow."""

from __future__ import annotations

import pytest

from veyronix.agents.memory import IncidentMemory
from veyronix.agents.orchestrator import Investigator
from veyronix.agents.tools import ToolContext, get_business_metrics
from veyronix.core.constants import ActionType, ServiceName
from veyronix.remediation.policy import ActionRequest, Actor
from veyronix.remediation.service import approve_action, execute_action, submit_action
from veyronix.simulation.counterfactual import build_incident_state
from veyronix.simulation.recommender import recommend
from veyronix.simulator.run import run_simulation

pytestmark = [pytest.mark.e2e, pytest.mark.integration]


def test_bad_deploy_cascade_full_loop(db):
    # 1. DATA — inject the flagship incident
    res = run_simulation(minutes=240, seed=5, scenario="bad_deploy")
    inc = res.incidents[0]
    ctx = ToolContext(run_id=res.run_id, metrics=res.metrics, logs=res.logs,
                      traces=res.traces, deployments=res.deployments,
                      business=res.business, memory=IncidentMemory())

    # 2. INVESTIGATE + ROOT CAUSE
    rep = Investigator().investigate(ctx, inc.start_min, inc.end_min, incident_id=inc.fault_id)
    assert rep.status == "done"
    assert rep.hallucination_flags == []
    assert rep.top_finding.cause_service == ServiceName.ORDER_SERVICE.value
    assert rep.recommended_action == ActionType.ROLLBACK_DEPLOYMENT.value

    # 3. SIMULATE + RECOMMEND
    biz = get_business_metrics(ctx, inc.start_min, inc.end_min)
    state = build_incident_state(inc.fault_type, inc.severity, biz,
                                 root_cause_confidence=rep.top_finding.confidence)
    recs = recommend(state)
    assert recs[0].action == ActionType.ROLLBACK_DEPLOYMENT.value
    assert recs[0].expected_benefit > 0

    # 4. APPROVE (medium risk -> needs approver)
    sub = submit_action(ActionRequest(
        action_type=ActionType.ROLLBACK_DEPLOYMENT,
        params={"service": rep.top_finding.cause_service}, run_id=res.run_id))
    assert sub["decision"] == "requires_approval"
    approve_action(sub["action_id"], Actor("bob", "approver"), "confirmed")

    # 5. ACT + MEASURE
    out = execute_action(sub["action_id"], Actor("alice", "operator"),
                         fault_type=inc.fault_type, severity=inc.severity)
    assert out["success"] is True and out["status"] == "executed"
    assert out["predicted"]["recovery_probability"] >= 0.85
