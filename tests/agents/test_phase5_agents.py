"""Phase 5: AI investigator, tools, memory, MCP, agent evaluation."""

from __future__ import annotations

import pytest

from veyronix.agents.evaluate import evaluate_agent
from veyronix.agents.llm import LLMClient
from veyronix.agents.memory import IncidentMemory, signature_from_anomalies
from veyronix.agents.orchestrator import FAULT_RECOVERY, Investigator
from veyronix.agents.tools import TOOL_REGISTRY, ToolContext, call_tool
from veyronix.core.constants import ActionType, FaultType, ServiceName
from veyronix.mcp.server import list_exposed_tools
from veyronix.simulator.run import run_simulation

pytestmark = pytest.mark.agents


@pytest.fixture(scope="module")
def bad_deploy_ctx():
    res = run_simulation(minutes=240, seed=5, scenario="bad_deploy")
    mem = IncidentMemory()
    mem.add("INC-1", "order_service bad_deployment database", "order_service: bad deploy", "rollback_deployment")
    ctx = ToolContext(run_id=res.run_id, metrics=res.metrics, logs=res.logs,
                      traces=res.traces, deployments=res.deployments,
                      business=res.business, memory=mem)
    return ctx, res.incidents[0]


def test_investigator_finds_bad_deploy_root_cause(bad_deploy_ctx):
    ctx, inc = bad_deploy_ctx
    rep = Investigator().investigate(ctx, inc.start_min, inc.end_min, incident_id=1)
    assert rep.status == "done"
    assert rep.top_finding.cause_service == ServiceName.ORDER_SERVICE.value
    assert rep.top_finding.cause_type == FaultType.BAD_DEPLOYMENT.value
    assert rep.recommended_action == ActionType.ROLLBACK_DEPLOYMENT.value
    assert rep.tool_calls > 0 and rep.tools_failed == 0
    assert rep.hallucination_flags == []
    assert rep.similar_incidents  # RAG retrieved something


def test_tool_budget_is_enforced(bad_deploy_ctx):
    ctx, inc = bad_deploy_ctx
    inv = Investigator()
    inv.max_tool_calls = 2
    rep = inv.investigate(ctx, inc.start_min, inc.end_min)
    assert rep.tool_calls <= 2
    assert rep.status == "budget_exceeded"


def test_tools_fail_safe(bad_deploy_ctx):
    ctx, _ = bad_deploy_ctx
    assert call_tool("nonexistent_tool", ctx).ok is False
    # querying a missing service returns an error result, not an exception
    res = call_tool("query_metrics", ctx, service="ghost", start_min=0, end_min=10)
    assert res.ok is False


def test_memory_retrieval_ranks_by_similarity():
    mem = IncidentMemory()
    mem.add("A", "database db_latency_ms database_overload", "database overload", "scale db")
    mem.add("B", "cache cache_hit_rate dependency_failure", "cache failure", "failover")
    hits = mem.search("database db_latency_ms database_overload", top_k=2)
    assert hits[0]["incident_ref"] == "A"
    assert hits[0]["similarity"] >= hits[-1]["similarity"]


def test_signature_builder():
    sig = signature_from_anomalies(
        [{"service": "database", "channel": "db_latency_ms", "implied_fault": "database_overload"}],
        business_drop=10)
    assert "database" in sig and "business_revenue_impact" in sig


def test_mcp_exposes_core_tools():
    tools = list_exposed_tools()
    for expected in ("query_metrics", "query_logs", "get_deployments",
                     "get_service_graph", "investigate_incident",
                     "search_similar_incidents"):
        assert expected in tools


def test_offline_llm_needs_no_key():
    c = LLMClient(provider="offline")
    assert c.available() is False
    out = c.synthesize({"hypotheses": [{"rank": 1, "cause_service": "db",
                                        "cause_type": "database_overload",
                                        "confidence": 0.9, "explanation": "x"}],
                        "evidence": [], "business": {}})
    assert "summary" in out


def test_agent_eval_small_is_accurate_and_safe():
    r = evaluate_agent(seeds=[42, 43], minutes=1440, provider="offline")
    assert r["root_cause_top1"] >= 0.6
    assert r["tool_success_rate"] == 1.0
    assert r["hallucination_rate"] <= 0.15
    assert r["unsafe_action_rate"] == 0.0


# ---- security: the whole agent tool surface is read-only ----
@pytest.mark.security
def test_agent_tools_are_read_only():
    forbidden = ("execute", "delete", "drop", "write", "rollback", "restart",
                 "scale", "shutdown", "kill", "run_command", "shell")
    for name in TOOL_REGISTRY:
        assert not any(word in name for word in forbidden), f"{name} looks mutating"


@pytest.mark.security
def test_recommended_actions_are_allowlisted():
    for action in FAULT_RECOVERY.values():
        assert ActionType(action)  # must be a valid allowlisted action
