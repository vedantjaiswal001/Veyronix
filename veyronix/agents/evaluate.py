"""AI-agent evaluation framework (spec §37).

Runs the investigator over a fixed set of simulated incidents and measures:
root-cause accuracy (top-1/top-3), tool-call success rate, average tool calls,
average investigation time, hallucination rate, and recommendation accuracy.
Incident memory is populated from *earlier* runs only, so retrieval is realistic
and never leaks the incident under investigation.
"""

from __future__ import annotations

import json

import numpy as np

from veyronix.agents.llm import LLMClient
from veyronix.agents.memory import IncidentMemory
from veyronix.agents.orchestrator import Investigator
from veyronix.agents.tools import ToolContext
from veyronix.core.config import ARTIFACT_DIR
from veyronix.core.logging import get_logger
from veyronix.simulator.run import run_simulation

log = get_logger("agents.eval")


def evaluate_agent(seeds: list[int] | None = None, minutes: int = 1440,
                   provider: str = "offline") -> dict:
    seeds = seeds or [42, 43, 44, 45, 46, 47, 48, 49]
    memory = IncidentMemory()
    client = LLMClient(provider=provider)
    inv = Investigator(client=client)

    rows: list[dict] = []
    for seed in seeds:
        res = run_simulation(minutes=minutes, seed=seed, run_id=f"ag-{seed}")
        ctx = ToolContext(run_id=res.run_id, metrics=res.metrics, logs=res.logs,
                          traces=res.traces, deployments=res.deployments,
                          business=res.business, memory=memory)
        for inc in res.incidents:
            rep = inv.investigate(ctx, inc.start_min, inc.end_min, incident_id=inc.fault_id)
            top = rep.top_finding
            rows.append({
                "fault": inc.fault_type.value,
                "truth_service": inc.root_cause_service.value,
                "truth_recovery": inc.recovery_action,
                "pred_service": top.cause_service if top else None,
                "top3": inc.root_cause_service.value in
                        [f.cause_service for f in rep.ranked_findings[:3]],
                "pred_recovery": rep.recommended_action,
                "tool_calls": rep.tool_calls,
                "tools_failed": rep.tools_failed,
                "duration_ms": rep.duration_ms,
                "hallucinated": bool(rep.hallucination_flags),
                "tokens": rep.tokens_used,
                "status": rep.status,
            })
        # after resolving, add this run's incidents to organizational memory
        for inc in res.incidents:
            memory.add(
                ref=f"{res.run_id}#{inc.fault_id}",
                signature=f"{inc.root_cause_service.value} {inc.fault_type.value} "
                          + " ".join(inc.affected_services),
                root_cause=f"{inc.root_cause_service.value}: {inc.fault_type.value}",
                resolution=inc.recovery_action)

    n = len(rows)
    tool_calls = sum(r["tool_calls"] for r in rows)
    tools_failed = sum(r["tools_failed"] for r in rows)
    result = {
        "provider": provider, "incidents": n,
        "root_cause_top1": round(np.mean([r["pred_service"] == r["truth_service"]
                                          for r in rows]), 4),
        "root_cause_top3": round(np.mean([r["top3"] for r in rows]), 4),
        "recommendation_accuracy": round(np.mean([r["pred_recovery"] == r["truth_recovery"]
                                                  for r in rows]), 4),
        "tool_success_rate": round(1 - tools_failed / max(tool_calls, 1), 4),
        "avg_tool_calls": round(tool_calls / max(n, 1), 2),
        "avg_investigation_ms": round(float(np.mean([r["duration_ms"] for r in rows])), 1),
        "hallucination_rate": round(np.mean([r["hallucinated"] for r in rows]), 4),
        "avg_tokens": round(float(np.mean([r["tokens"] for r in rows])), 1),
        "unsafe_action_rate": 0.0,  # recommender only proposes; execution is gated (Phase 7)
        "by_fault": {},
    }
    for ft in sorted({r["fault"] for r in rows}):
        fr = [r for r in rows if r["fault"] == ft]
        result["by_fault"][ft] = {
            "n": len(fr),
            "top1": round(np.mean([r["pred_service"] == r["truth_service"] for r in fr]), 3),
        }
    log.info("agents.eval.done", **{k: result[k] for k in
             ("incidents", "root_cause_top1", "hallucination_rate", "tool_success_rate")})
    return result


def main(provider: str = "offline") -> None:
    out = evaluate_agent(provider=provider)
    path = ARTIFACT_DIR / "ml_results" / "agent_eval.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "by_fault"}, indent=2))


if __name__ == "__main__":
    import typer

    typer.run(main)
