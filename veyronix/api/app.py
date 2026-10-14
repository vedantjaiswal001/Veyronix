"""VEYRONIX FastAPI application (spec §56).

Serves the AI Operations Center dashboard with real system output over a
SIMULATED environment: topology, system overview, predicted/active incidents,
per-incident timelines, on-demand AI investigation, counterfactual
recommendations, model health, agent activity, and the remediation workflow.
Exposes Prometheus metrics at /metrics and OpenAPI docs at /docs.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from veyronix.agents.orchestrator import Investigator
from veyronix.core.config import get_settings
from veyronix.core.logging import configure_logging, get_logger
from veyronix.core.topology import topology_summary
from veyronix.observability import metrics as M
from veyronix.observability.monitor import demo_model_drift
from veyronix.remediation import service as remediation
from veyronix.remediation.policy import ActionRequest, Actor
from veyronix.rootcause.engine import rank_root_causes

log = get_logger("api")
settings = get_settings()

app = FastAPI(title="VEYRONIX", version="0.1.0",
              description="Predictive AI Incident & Root-Cause Intelligence Platform")
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                   allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def _startup() -> None:
    configure_logging()
    M.SIM_HEALTH.set(1.0)


def _state():
    from veyronix.api.session import get_state
    return get_state()


def _incident(fault_id: int):
    for inc in _state().result.incidents:
        if inc.fault_id == fault_id:
            return inc
    raise HTTPException(404, f"incident {fault_id} not found")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "environment": settings.environment, "label": "SIMULATED"}


@app.get("/metrics")
def metrics() -> Response:
    payload, ctype = M.export_metrics()
    return Response(content=payload, media_type=ctype)


@app.get("/api/topology")
def topology() -> dict:
    return topology_summary()


@app.get("/api/overview")
def overview() -> dict:
    st = _state()
    cards = st.incident_cards
    pred = st.predictions
    latest_prob = float(pred.probability.iloc[-1]) if pred is not None and len(pred) else None
    if latest_prob is not None:
        M.set_incident_probability(latest_prob)
    by_sev = {}
    for c in cards:
        by_sev[c["severity"]] = by_sev.get(c["severity"], 0) + 1
    return {
        "run_id": st.run_id, "label": "SIMULATED ENVIRONMENT",
        "services": len(topology_summary()["nodes"]),
        "total_incidents": len(cards),
        "incidents_by_severity": by_sev,
        "predicted_incidents": sum(1 for c in cards if (c["predicted_probability"] or 0) >= 0.5),
        "model_loaded": st.model is not None,
        "latest_incident_probability": latest_prob,
    }


@app.get("/api/incidents")
def incidents() -> dict:
    return {"incidents": _state().incident_cards, "label": "SIMULATED"}


@app.get("/api/incidents/{fault_id}")
def incident_detail(fault_id: int) -> dict:
    st = _state()
    inc = _incident(fault_id)
    card = next((c for c in st.incident_cards if c["id"] == fault_id), {})
    hyps = rank_root_causes(st.result.metrics, st.result.deployments,
                            inc.start_min, inc.end_min, top_k=3)
    return {**card,
            "ranked_root_causes": [
                {"rank": h.rank, "cause_service": h.cause_service,
                 "cause_type": h.cause_type, "confidence": h.confidence,
                 "explanation": h.explanation, "evidence_chain": h.evidence_chain}
                for h in hyps]}


@app.get("/api/incidents/{fault_id}/timeline")
def incident_timeline(fault_id: int) -> dict:
    """Event timeline (spec §41): normal -> anomaly -> prediction -> RC -> recover."""
    st = _state()
    inc = _incident(fault_id)
    pred = st.predictions
    events = [{"minute": inc.start_min - 30, "kind": "normal", "detail": "baseline"}]
    if pred is not None:
        pre = pred[(pred.minute >= inc.start_min - 30) & (pred.minute < inc.start_min)]
        hit = pre[pre.probability >= 0.5]
        if not hit.empty:
            r = hit.iloc[0]
            events.append({"minute": int(r.minute), "kind": "prediction",
                           "detail": f"incident probability {r.probability:.0%}"})
    events += [
        {"minute": inc.start_min, "kind": "incident_start", "detail": inc.root_cause_desc},
        {"minute": inc.start_min + 1, "kind": "anomaly", "detail": "anomaly detected"},
        {"minute": inc.start_min + 2, "kind": "investigation", "detail": "AI investigation started"},
        {"minute": inc.end_min, "kind": "recovered", "detail": "telemetry back to baseline"},
    ]
    events.sort(key=lambda e: e["minute"])
    return {"incident_id": fault_id, "events": events}


@app.post("/api/incidents/{fault_id}/investigate")
def investigate(fault_id: int) -> dict:
    st = _state()
    inc = _incident(fault_id)
    with M.observe(M.AGENT_LATENCY, provider=settings.llm_provider):
        rep = Investigator().investigate(st.ctx, inc.start_min, inc.end_min, incident_id=fault_id)
    for _ in range(rep.tool_calls):  # reflect tool calls in metrics
        M.record_tool_call("investigation", True)
    M.record_tokens(rep.provider, rep.tokens_used)
    return rep.model_dump()


@app.get("/api/incidents/{fault_id}/recommendations")
def recommendations(fault_id: int) -> dict:
    from veyronix.agents.tools import get_business_metrics
    from veyronix.simulation.counterfactual import build_incident_state
    from veyronix.simulation.recommender import recommend

    st = _state()
    inc = _incident(fault_id)
    biz = get_business_metrics(st.ctx, inc.start_min, inc.end_min)
    hyps = rank_root_causes(st.result.metrics, st.result.deployments, inc.start_min, inc.end_min)
    conf = hyps[0].confidence if hyps else 0.5
    state = build_incident_state(inc.fault_type, inc.severity, biz, root_cause_confidence=conf)
    recs = recommend(state)
    return {"incident_id": fault_id,
            "recommendations": [r.__dict__ for r in recs],
            "note": "counterfactual estimates (what-if), not guaranteed outcomes"}


@app.get("/api/timeseries")
def timeseries(service: str, channel: str = "latency_p95_ms") -> dict:
    st = _state()
    g = st.result.metrics[st.result.metrics.service == service]
    if g.empty:
        raise HTTPException(404, f"unknown service {service}")
    return {"service": service, "channel": channel,
            "points": [{"minute": int(m), "value": round(float(v), 3)}
                       for m, v in zip(g.minute, g[channel], strict=True)]}


@app.get("/api/predictions")
def predictions() -> dict:
    st = _state()
    if st.predictions is None:
        return {"available": False, "note": "run `make train` to enable predictions"}
    p = st.predictions
    return {"available": True,
            "points": [{"minute": int(m), "probability": round(float(v), 4)}
                       for m, v in zip(p.minute, p.probability, strict=True)]}


@app.get("/api/business")
def business() -> dict:
    st = _state()
    b = st.result.business
    return {"points": [{"minute": int(r["minute"]), "revenue_per_min": r["revenue_per_min"],
                        "orders_per_min": r["orders_per_min"],
                        "checkout_error_rate": r["checkout_error_rate"]}
                       for _, r in b.iterrows()]}


@app.get("/api/model-health")
def model_health() -> dict:
    st = _state()
    if st.model is None:
        return {"available": False, "note": "run `make train` to enable model health"}
    return demo_model_drift()


@app.post("/api/actions/submit")
def submit_action(payload: dict) -> dict:
    req = ActionRequest(action_type=payload["action_type"], params=payload.get("params", {}),
                        run_id=payload.get("run_id"), incident_id=payload.get("incident_id"),
                        proposed_by=payload.get("proposed_by", "operator"),
                        reason=payload.get("reason", ""))
    return remediation.submit_action(req)


@app.post("/api/actions/{action_id}/approve")
def approve(action_id: int, payload: dict) -> dict:
    actor = Actor(payload.get("name", "operator"), payload.get("role", "approver"))
    try:
        return remediation.approve_action(action_id, actor, payload.get("reason", ""))
    except remediation.RemediationError as e:
        raise HTTPException(403, str(e)) from e


@app.post("/api/actions/{action_id}/execute")
def execute(action_id: int, payload: dict | None = None) -> dict:
    payload = payload or {}
    actor = Actor(payload.get("name", "operator"), payload.get("role", "operator"))
    try:
        return remediation.execute_action(action_id, actor)
    except remediation.RemediationError as e:
        raise HTTPException(403, str(e)) from e
