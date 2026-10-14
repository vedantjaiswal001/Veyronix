"""One-command flagship demo (spec §57/§58): the full VEYRONIX loop.

    DATA -> DETECT -> PREDICT -> INVESTIGATE -> ROOT CAUSE -> SIMULATE
         -> RECOMMEND -> APPROVE -> ACT -> MEASURE -> LEARN

Runs the Bad Deployment Cascade end to end and narrates every stage with real
system output. `--scenario unknown_cascade` and `--scenario model_drift` show
robustness and the MLOps drift path.
"""

from __future__ import annotations

from veyronix.agents.memory import IncidentMemory
from veyronix.agents.orchestrator import Investigator
from veyronix.agents.tools import ToolContext, get_anomalies, get_business_metrics
from veyronix.core.constants import ActionType
from veyronix.db.session import init_db
from veyronix.remediation.policy import ActionRequest, Actor
from veyronix.remediation.service import approve_action, execute_action, submit_action
from veyronix.simulation.counterfactual import build_incident_state
from veyronix.simulation.recommender import recommend
from veyronix.simulator.run import run_simulation


def _h(title: str) -> None:
    print(f"\n\033[1;36m{'─'*70}\n▶ {title}\033[0m")


def run_demo(scenario: str = "bad_deploy", seed: int = 5) -> dict:
    print("\033[1;35m" + "═" * 70)
    print("  VEYRONIX — AI OPERATIONS CENTER  ·  SIMULATED ENVIRONMENT")
    print("═" * 70 + "\033[0m")
    init_db()

    if scenario == "model_drift":
        return _demo_drift()

    _h("DATA · simulate a production incident")
    res = run_simulation(minutes=260, seed=seed, scenario=scenario)
    inc = res.incidents[0]
    print(f"  scenario        : {scenario}")
    print(f"  injected fault  : {inc.fault_type.value}  ({inc.root_cause_desc})")
    print(f"  window          : t={inc.start_min}..{inc.end_min} min  severity={inc.severity.value}")

    mem = IncidentMemory.load()
    ctx = ToolContext(run_id=res.run_id, metrics=res.metrics, logs=res.logs,
                      traces=res.traces, deployments=res.deployments,
                      business=res.business, memory=mem)

    _h("DETECT · anomalies (unsupervised, real detector)")
    anom = get_anomalies(ctx, inc.start_min, inc.end_min)
    for a in anom["anomalies"][:5]:
        print(f"  · {a['service']:16s} {a['channel']:16s} z={a['magnitude']:6.1f} onset t={a['onset_min']}")

    _h("PREDICT · incident probability (if model trained)")
    _predict(res, inc)

    _h("INVESTIGATE · bounded multi-agent AI investigator")
    rep = Investigator().investigate(ctx, inc.start_min, inc.end_min, incident_id=inc.fault_id)
    print(f"  provider={rep.provider}  tool_calls={rep.tool_calls}  "
          f"failed={rep.tools_failed}  time={rep.duration_ms:.0f}ms  "
          f"hallucinations={len(rep.hallucination_flags)}")
    print(f"  summary: {rep.summary}")

    _h("ROOT CAUSE · ranked hypothesis + evidence")
    top = rep.top_finding
    print(f"  → {top.cause_service}  ({top.cause_type})  confidence {top.confidence:.0%}")
    print(f"    {top.explanation}")
    for step in top.evidence_chain[:4]:
        print(f"    · [{step['type']}] {step.get('service','')}: {step.get('detail','')}")

    _h("SIMULATE + RECOMMEND · counterfactual what-if estimates")
    biz = get_business_metrics(ctx, inc.start_min, inc.end_min)
    state = build_incident_state(inc.fault_type, inc.severity, biz,
                                 root_cause_confidence=top.confidence)
    recs = recommend(state)
    for r in recs[:4]:
        print(f"  · {r.action:20s} recover {r.recovery_probability:.0%}  "
              f"downtime ~{r.expected_downtime_min:.0f}m  save ₹{r.expected_benefit:,.0f}  {r.risk}")
    best = recs[0]
    print(f"  ★ recommended: {best.action}  — {best.reason}")

    _h("APPROVE · human-gated, allowlisted remediation")
    sub = submit_action(ActionRequest(
        action_type=ActionType(best.action),
        params={"service": top.cause_service}, run_id=res.run_id,
        incident_id=None, proposed_by="recommender", reason=best.reason))
    print(f"  submitted #{sub['action_id']}  risk={sub['risk']}  decision={sub['decision']}")
    if sub["decision"] == "requires_approval":
        approve_action(sub["action_id"], Actor("on-call-approver", "approver"), "confirmed")
        print("  ✅ approved by on-call-approver")

    _h("ACT + MEASURE · safe execution + outcome")
    if sub["decision"] != "blocked":
        out = execute_action(sub["action_id"], Actor("operator", "operator"),
                             fault_type=inc.fault_type, severity=inc.severity)
        print(f"  executed: success={out['success']}  status={out['status']}")
        print(f"  predicted={out['predicted']}  actual={out['actual']}")

    _h("LEARN · store in incident memory (RAG)")
    mem.add(ref=f"{res.run_id}#{inc.fault_id}",
            signature=f"{top.cause_service} {inc.fault_type.value} "
                      + " ".join(inc.affected_services),
            root_cause=f"{top.cause_service}: {inc.fault_type.value}",
            resolution=best.action)
    mem.save()
    print(f"  incident memory now holds {len(mem.records)} resolved incident(s)")

    print("\n\033[1;32m✔ VEYRONIX loop complete — detect → predict → investigate → "
          "root cause → simulate → recommend → approve → act → measure → learn\033[0m\n")
    return {"scenario": scenario, "root_cause": top.cause_service,
            "recommended": best.action}


def _predict(res, inc):
    try:
        from veyronix.features.windows import build_feature_matrix
        from veyronix.ml.incident_prediction.train import load_active_model

        model = load_active_model()
        feats = build_feature_matrix(res.metrics, res.business)
        pre = feats[(feats.minute >= inc.start_min - model["horizon"]) & (feats.minute < inc.start_min)]
        if not pre.empty:
            prob = model["model"].predict_proba(pre[model["features"]])
            i = prob.argmax()
            m = int(pre.iloc[i].minute)
            print(f"  peak pre-incident probability {prob[i]:.1%} at t={m} "
                  f"→ early warning {inc.start_min - m} min before onset")
    except Exception:
        print("  (no trained model found — run `make train` to enable prediction)")


def _demo_drift():
    _h("MODEL DRIFT · MLOps monitoring (§60)")
    from veyronix.observability.monitor import demo_model_drift
    try:
        rep = demo_model_drift()
        fd = rep["feature_drift"]
        print(f"  status          : {rep['status'].upper()}  (health {rep['health_score']})")
        print(f"  drifted features: {fd['n_flagged']} / {fd['n_features']}  max PSI {fd['max_psi']}")
        print(f"  recommendation  : {rep['recommendation']}")
        return rep
    except Exception:
        print("  (run `make train` first to build the feature dataset)")
        return {}


def main(scenario: str = "bad_deploy") -> None:
    run_demo(scenario=scenario)


if __name__ == "__main__":
    import typer

    typer.run(main)
