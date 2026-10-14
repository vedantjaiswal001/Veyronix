"""Root-cause service layer: run analysis for an incident and persist the
ranked hypotheses + causal-graph edges (spec §10/§23/§25).

Used by the API and the AI investigator so the deterministic RCA result is the
system of record; the agent narrates it rather than inventing causes.
"""

from __future__ import annotations

import pandas as pd

from veyronix.core.logging import get_logger
from veyronix.db.models import CausalEdge, Incident, RootCauseHypothesis
from veyronix.db.session import session_scope
from veyronix.rootcause.engine import Hypothesis, build_causal_edges, rank_root_causes

log = get_logger("rootcause.service")


def investigate(metrics: pd.DataFrame, deployments: pd.DataFrame,
                start_min: int, end_min: int, top_k: int = 3) -> list[Hypothesis]:
    """Deterministic root-cause analysis for a window (pure, no DB)."""
    return rank_root_causes(metrics, deployments, start_min, end_min, top_k=top_k)


def store_hypotheses(run_id: str, incident_id: int | None,
                     hyps: list[Hypothesis]) -> list[int]:
    """Persist hypotheses (+ causal edges for the top one). Returns their ids."""
    ids: list[int] = []
    with session_scope() as s:
        for h in hyps:
            row = RootCauseHypothesis(
                run_id=run_id, incident_id=incident_id, rank=h.rank,
                cause_service=h.cause_service, cause_type=h.cause_type,
                confidence=h.confidence, score_breakdown=h.score_breakdown,
                evidence_chain=h.evidence_chain, explanation=h.explanation)
            s.add(row)
            s.flush()
            ids.append(row.id)
        if hyps:
            for e in build_causal_edges(hyps[0]):
                s.add(CausalEdge(run_id=run_id, incident_id=incident_id, **e))
    log.info("rootcause.stored", run_id=run_id, incident_id=incident_id, n=len(ids))
    return ids


def analyze_incident_id(metrics: pd.DataFrame, deployments: pd.DataFrame,
                        incident_id: int) -> list[Hypothesis]:
    """Look up an incident's window from the DB and analyze + store it."""
    with session_scope() as s:
        inc = s.get(Incident, incident_id)
        if inc is None:
            return []
        run_id = inc.run_id
        # minutes are relative to the run start; recompute from timestamps
        start_min = int((inc.start_time - metrics["ts"].min()).total_seconds() // 60)
        end_min = int((inc.end_time - metrics["ts"].min()).total_seconds() // 60)
    hyps = investigate(metrics, deployments, start_min, end_min)
    store_hypotheses(run_id, incident_id, hyps)
    return hyps
