"""Root-cause ranking engine (spec §23/§24/§25).

Ranks explainable hypotheses by combining four evidence sources:

1. **Intrinsic vs. symptom anomalies.** A deviation in an intrinsic resource
   channel (CPU, memory, DB latency, queue depth, cache hit-rate, packet loss)
   signals a *cause*; a deviation only in latency/error is frequently a
   *symptom* inherited from a degraded dependency.
2. **Sink-in-the-affected-subgraph.** The root is typically the affected service
   that does not itself depend on any other affected service — the source of the
   cascade rather than a victim of it.
3. **Deployment proximity.** A recent deployment whose downstream dependency is
   the intrinsic epicenter overrides the infrastructure hypothesis (a bad deploy
   causes DB latency; the *deploy* is the root cause, not the DB).
4. **Fleet-wide request-rate elevation** ⇒ a traffic spike at the ingress.

Correlation, temporal evidence and hypothesis are labelled distinctly; no causal
certainty is claimed from correlation alone (spec §25).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from veyronix.core.constants import FaultType, ServiceName
from veyronix.core.topology import downstream_dependencies, upstream_dependents
from veyronix.rootcause.detect import (
    INTRINSIC_CHANNELS,
    count_material_services,
    profile_services,
)


@dataclass
class Hypothesis:
    rank: int
    cause_service: str
    cause_type: str
    confidence: float
    score_breakdown: dict
    evidence_chain: list[dict] = field(default_factory=list)
    explanation: str = ""


def _recent_deployments(deployments: pd.DataFrame, start_min: int,
                        pre: int = 20, post: int = 5) -> pd.DataFrame:
    if deployments is None or deployments.empty:
        return pd.DataFrame()
    return deployments[(deployments.minute >= start_min - pre)
                       & (deployments.minute <= start_min + post)]


def rank_root_causes(
    metrics: pd.DataFrame, deployments: pd.DataFrame, start_min: int, end_min: int,
    top_k: int = 3,
) -> list[Hypothesis]:
    """Produce ranked root-cause hypotheses for an incident window."""
    profiles = profile_services(metrics, start_min, end_min)
    if not profiles:
        return []

    affected = set(profiles)
    intrinsic_services = {s for s, a in profiles.items()
                          if a.dominant_channel in INTRINSIC_CHANNELS}

    def onset_key(a):
        return (a.onset_min if a.onset_min is not None else 10_000, -a.magnitude)

    ordered = sorted(profiles.values(), key=onset_key)
    # epicenter = earliest-degrading intrinsic service (fallback: earliest overall)
    intrinsic_ordered = [a for a in ordered if a.service in intrinsic_services]
    epicenter = (intrinsic_ordered or ordered)[0].service

    scored: dict[tuple[str, str], dict] = {}

    def add(svc, ctype, comp):
        key = (svc, ctype)
        cur = scored.get(key)
        # keep the strongest hypothesis per (service, cause_type)
        if cur is None or comp["total"] > cur["total"]:
            scored[key] = comp

    # (1) fleet-wide request-rate elevation => traffic spike at the ingress.
    # Counts *material* request-rate rises across all services (not just where it
    # is the single dominant channel), since load also drives CPU/latency.
    rr_hits = count_material_services(metrics, start_min, end_min, "request_rate")
    if rr_hits >= 4:
        add(ServiceName.API_GATEWAY.value, FaultType.TRAFFIC_SPIKE.value,
            {"fleet_request_rate": 8.0, "services_hit": float(rr_hits),
             "total": 8.0 + rr_hits * 0.4})

    # (2) deployment-caused hypotheses. Recency is weighted by proximity to the
    # incident onset: a deployment right at onset is far more likely causal than
    # a routine deploy that merely happens to fall in the window (decoy).
    for _, d in _recent_deployments(deployments, start_min).iterrows():
        sd = d["service"]
        downstream = set(downstream_dependencies(sd))
        causal = bool(downstream & intrinsic_services) or sd in intrinsic_services or sd in affected
        dist = abs(int(d["minute"]) - start_min)
        recency = 4.0 * max(0.0, 1.0 - dist / 20.0)
        comp = {"deployment_recency": recency,
                "causal_to_epicenter": 5.0 if causal else 0.0,
                "downstream_affected": 1.0 * len(downstream & affected)}
        comp["total"] = sum(comp.values())
        add(sd, FaultType.BAD_DEPLOYMENT.value, comp)

    # (3) per-service infrastructure hypotheses
    for a in profiles.values():
        s = a.service
        affected_deps = affected & set(downstream_dependencies(s))
        is_sink = len(affected_deps) == 0
        intrinsic = a.dominant_channel in INTRINSIC_CHANNELS
        dependents = affected & set(upstream_dependents(s))
        comp = {
            "anomaly_magnitude": min(a.magnitude / 5.0, 3.0),
            "intrinsic_anomaly": 2.0 if intrinsic else 0.0,
            "is_source_sink": 3.0 if is_sink else 0.0,
            "temporal_precedence": 1.5 if s == epicenter else 0.0,
            "affected_dependents": 0.5 * len(dependents),
        }
        raw = sum(comp.values())
        if not intrinsic and not is_sink:  # pure inherited symptom
            raw *= 0.35
        comp["total"] = raw
        add(s, a.implied_fault, comp)

    items = sorted(scored.items(), key=lambda kv: kv[1]["total"], reverse=True)
    total = sum(max(v["total"], 0.0) for _, v in items) or 1.0
    deploys = _recent_deployments(deployments, start_min)
    hyps: list[Hypothesis] = []
    for i, ((svc, ctype), comp) in enumerate(items[:top_k], start=1):
        hyps.append(Hypothesis(
            rank=i, cause_service=svc, cause_type=ctype,
            confidence=round(comp["total"] / total, 4),
            score_breakdown={k: round(v, 3) for k, v in comp.items() if k != "total"},
            evidence_chain=_evidence_chain(svc, ctype, profiles, deploys, ordered),
            explanation=_explain(svc, ctype, profiles),
        ))
    return hyps


def _evidence_chain(svc, ctype, profiles, deploys, ordered) -> list[dict]:
    """Ordered, typed evidence from cause to impact (spec §25 edge tiers)."""
    chain: list[dict] = []
    if ctype == FaultType.BAD_DEPLOYMENT.value and not deploys.empty:
        d = deploys[deploys.service == svc]
        if not d.empty:
            row = d.iloc[0]
            chain.append({"type": "deployment", "service": svc,
                          "detail": f"deployment {row['version']} ({row['change_summary']})",
                          "minute": int(row["minute"])})
    for a in ordered:  # temporal chain, earliest degradation first
        chain.append({"type": "temporal", "service": a.service,
                      "detail": f"{a.dominant_channel} deviated (z={a.magnitude})",
                      "minute": a.onset_min})
    chain.append({"type": "hypothesis", "service": "business",
                  "detail": "checkout errors / revenue impacted downstream"})
    return chain


def _explain(svc, ctype, profiles) -> str:
    if ctype == FaultType.BAD_DEPLOYMENT.value:
        return (f"A recent deployment on {svc} precedes the incident and its "
                f"downstream dependency degraded — consistent with a bad deploy.")
    if ctype == FaultType.TRAFFIC_SPIKE.value:
        return ("Request rate rose across many services simultaneously — "
                "consistent with a traffic spike at the ingress.")
    a = profiles.get(svc)
    ch = a.dominant_channel if a else "metrics"
    return (f"{svc} shows an intrinsic anomaly on {ch} and sits upstream of the "
            f"other degraded services — consistent with {ctype.replace('_', ' ')}.")


def build_causal_edges(hyp: Hypothesis) -> list[dict]:
    """Serialize a hypothesis' evidence chain into causal_edges rows."""
    chain = hyp.evidence_chain
    return [{"source": a.get("service", "?"), "target": b.get("service", "?"),
             "edge_type": b.get("type", "hypothesis"), "lag_seconds": None}
            for a, b in zip(chain, chain[1:], strict=False)]
