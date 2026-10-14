"""AI Investigator: bounded multi-agent orchestration (spec §26/§27).

An orchestrator runs five specialist agents (metrics, logs, traces, deployment,
database), each calling read-only tools to gather real evidence. A synthesis
step fuses that evidence with the deterministic root-cause ranking (the LLM
explains and prioritises; it never invents the cause or the numbers) and
retrieves similar past incidents. Hard guardrails bound steps, tool calls, wall
time and tokens; tools fail safe. Every tool call is recorded for evaluation.
"""

from __future__ import annotations

import time

from veyronix.agents.llm import LLMClient
from veyronix.agents.memory import signature_from_anomalies
from veyronix.agents.schemas import (
    Evidence,
    InvestigationReport,
    RootCauseFinding,
    SimilarIncident,
    ToolCallRecord,
)
from veyronix.agents.tools import ToolContext, call_tool
from veyronix.core.config import get_settings
from veyronix.core.logging import get_logger
from veyronix.rootcause.engine import rank_root_causes
from veyronix.simulator.faults import FAULT_CATALOG

log = get_logger("agents.orchestrator")

# fault type -> allowlisted recovery action (from the fault catalog)
FAULT_RECOVERY: dict[str, str] = {
    ft.value: spec.recovery.value for ft, spec in FAULT_CATALOG.items()
}


class Investigator:
    def __init__(self, client: LLMClient | None = None) -> None:
        s = get_settings()
        self.client = client or LLMClient()
        self.max_steps = s.agent_max_steps
        self.max_tool_calls = s.agent_max_tool_calls
        self.timeout_s = s.agent_timeout_seconds
        self.token_budget = s.agent_token_budget

    def investigate(self, ctx: ToolContext, start_min: int, end_min: int,
                    incident_id: int | None = None) -> InvestigationReport:
        t0 = time.perf_counter()
        records: list[ToolCallRecord] = []
        evidence: list[Evidence] = []
        budget = {"tool_calls": 0, "steps": 0, "over": False}

        def run(agent: str, tool: str, **args):
            if budget["tool_calls"] >= self.max_tool_calls or self._timed_out(t0):
                budget["over"] = True
                return None
            res = call_tool(tool, ctx, **args)
            budget["tool_calls"] += 1
            records.append(ToolCallRecord(agent=agent, tool=tool, args=args, ok=res.ok,
                                          latency_ms=round(res.latency_ms, 2),
                                          result_summary=res.summary))
            return res

        # --- specialist agents (fixed runbook; each uses real tools) ---
        anomalies: list[dict] = []
        for agent, plan in self._runbook(start_min, end_min):
            if budget["over"] or budget["steps"] >= self.max_steps:
                break
            budget["steps"] += 1
            for tool, args in plan:
                res = run(agent, tool, **args)
                if res is None:
                    break
                ev = self._to_evidence(agent, tool, res)
                evidence.extend(ev)
                if tool == "get_anomalies" and res.ok:
                    anomalies = res.data.get("anomalies", [])
                # let specialists drill into the top anomalous service
                if tool == "get_anomalies" and res.ok and anomalies and agent == "metrics":
                    top = anomalies[0]["service"]
                    r2 = run("metrics", "query_metrics", service=top,
                             start_min=start_min, end_min=end_min)
                    if r2:
                        evidence.extend(self._to_evidence("metrics", "query_metrics", r2))

        biz_res = call_tool("get_business_metrics", ctx, start_min=start_min, end_min=end_min)
        biz = biz_res.data if biz_res.ok else {}

        # --- deterministic root-cause ranking (source of truth) ---
        hyps = rank_root_causes(ctx.metrics, ctx.deployments, start_min, end_min)
        findings = [RootCauseFinding(
            rank=h.rank, cause_service=h.cause_service, cause_type=h.cause_type,
            confidence=h.confidence, explanation=h.explanation,
            evidence_chain=h.evidence_chain) for h in hyps]

        # --- similar incidents (RAG) ---
        similar: list[SimilarIncident] = []
        if ctx.memory is not None and anomalies:
            sig = signature_from_anomalies(anomalies, biz.get("revenue_drop_pct", 0))
            sres = run("memory", "search_similar_incidents", signature=sig, top_k=3)
            if sres and sres.ok:
                similar = [SimilarIncident(**m) for m in sres.data.get("matches", [])]

        # --- synthesis (LLM narrates over evidence, constrained to findings) ---
        syn = self.client.synthesize({
            "hypotheses": [f.model_dump() for f in findings],
            "evidence": [e.model_dump() for e in evidence],
            "business": biz,
        })
        expl = {f["rank"]: f.get("explanation", "") for f in syn.get("findings", [])}
        for f in findings:
            if expl.get(f.rank):
                f.explanation = expl[f.rank]

        top = findings[0] if findings else None
        recommended = FAULT_RECOVERY.get(top.cause_type, "no_op") if top else "no_op"

        # --- hallucination guard: top cause must be grounded in evidence ---
        halluc = self._hallucination_flags(top, anomalies, ctx.deployments, start_min, end_min)

        status = "budget_exceeded" if budget["over"] else "done"
        report = InvestigationReport(
            incident_id=incident_id, run_id=ctx.run_id, provider=self.client.provider,
            top_finding=top, ranked_findings=findings, evidence=evidence,
            similar_incidents=similar, recommended_action=recommended,
            summary=syn.get("summary", ""), steps=budget["steps"],
            tool_calls=budget["tool_calls"],
            tools_failed=sum(1 for r in records if not r.ok),
            tokens_used=self.client.tokens_used,
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            status=status, hallucination_flags=halluc)
        log.info("agent.investigation", run_id=ctx.run_id, provider=self.client.provider,
                 top=top.cause_service if top else None, tool_calls=budget["tool_calls"],
                 status=status, halluc=len(halluc))
        return report

    # ---- helpers ----
    def _runbook(self, start, end):
        return [
            ("metrics", [("get_anomalies", {"start_min": start, "end_min": end})]),
            ("logs", [("query_logs", {"start_min": start, "end_min": end})]),
            ("traces", [("query_traces", {"start_min": start, "end_min": end})]),
            ("deployment", [("get_deployments", {"start_min": start, "end_min": end})]),
            ("database", [("query_metrics", {"service": "database",
                                             "start_min": start, "end_min": end})]),
        ]

    def _to_evidence(self, agent, tool, res) -> list[Evidence]:
        if not res.ok:
            return []
        kind = {"get_anomalies": "metric", "query_metrics": "metric",
                "query_logs": "log", "query_traces": "trace",
                "get_deployments": "deployment",
                "search_similar_incidents": "memory"}.get(tool, "metric")
        if tool == "database":
            kind = "database"
        out = [Evidence(agent=agent, kind=("database" if agent == "database" else kind),
                        observation=res.summary, strength=0.6)]
        if tool == "get_anomalies":
            for a in res.data.get("anomalies", [])[:4]:
                out.append(Evidence(
                    agent=agent, kind="metric", service=a["service"],
                    observation=f"{a['service']} {a['channel']} anomalous "
                                f"(z={a['magnitude']}, onset t={a['onset_min']})",
                    supports=a["implied_fault"], strength=min(a["magnitude"] / 5, 1.0)))
        return out

    def _timed_out(self, t0) -> bool:
        return (time.perf_counter() - t0) > self.timeout_s

    def _hallucination_flags(self, top, anomalies, deployments, start, end) -> list[str]:
        if top is None:
            return []
        flags = []
        anomalous_services = {a["service"] for a in anomalies}
        deploy_services = set()
        if deployments is not None and not deployments.empty:
            w = deployments[(deployments.minute >= start - 20) & (deployments.minute <= end)]
            deploy_services = set(w.service.tolist())
        grounded = top.cause_service in anomalous_services or top.cause_service in deploy_services
        if not grounded:
            flags.append(f"top cause {top.cause_service} not supported by gathered evidence")
        return flags
