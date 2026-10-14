"""In-memory application state for the API.

Holds one deterministic simulation run, the trained model (if present), incident
memory, and precomputed incident "cards" (prediction + root cause + recommended
action + business impact). Lazily initialised so the API starts fast. Everything
served is real system output over a SIMULATED environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import pandas as pd

from veyronix.agents.memory import IncidentMemory
from veyronix.agents.tools import ToolContext
from veyronix.core.constants import Severity
from veyronix.core.logging import get_logger
from veyronix.features.windows import build_feature_matrix
from veyronix.rootcause.engine import rank_root_causes
from veyronix.simulation.counterfactual import build_incident_state
from veyronix.simulation.recommender import recommend
from veyronix.simulator.models import SimulationResult
from veyronix.simulator.run import run_simulation

log = get_logger("api.session")


@dataclass
class AppState:
    result: SimulationResult
    ctx: ToolContext
    model: dict | None = None
    predictions: pd.DataFrame | None = None      # [minute, probability]
    incident_cards: list[dict] = field(default_factory=list)

    @property
    def run_id(self) -> str:
        return self.result.run_id


def _load_model() -> dict | None:
    try:
        from veyronix.ml.incident_prediction.train import load_active_model

        return load_active_model()
    except Exception:
        return None


def _predictions(result: SimulationResult, model: dict | None) -> pd.DataFrame | None:
    if model is None:
        return None
    feats = build_feature_matrix(result.metrics, result.business)
    prob = model["model"].predict_proba(feats[model["features"]])
    return pd.DataFrame({"minute": feats["minute"].to_numpy(), "probability": prob})


def _predicted_prob(pred: pd.DataFrame | None, start_min: int, horizon: int = 30):
    if pred is None:
        return None, None
    pre = pred[(pred.minute >= start_min - horizon) & (pred.minute < start_min)]
    if pre.empty:
        return None, None
    row = pre.loc[pre.probability.idxmax()]
    return round(float(row.probability), 4), int(start_min - row.minute)


def _biz_window(business: pd.DataFrame, start: int, end: int) -> dict:
    """Real baseline vs degraded business KPIs for an incident window."""
    win = business[(business.minute >= start) & (business.minute < end)]
    base = business[(business.minute >= start - 45) & (business.minute < start - 10)]
    if win.empty or base.empty:
        return {"kpis": {}}
    return {"kpis": {
        "revenue_per_min": {"baseline": float(base.revenue_per_min.mean()),
                            "window": float(win.revenue_per_min.mean())},
        "orders_per_min": {"baseline": float(base.orders_per_min.mean()),
                          "window": float(win.orders_per_min.mean())}}}


def _build_cards(result: SimulationResult, pred: pd.DataFrame | None) -> list[dict]:
    cards = []
    for inc in result.incidents:
        prob, eta = _predicted_prob(pred, inc.start_min)
        hyps = rank_root_causes(result.metrics, result.deployments,
                                inc.start_min, inc.end_min, top_k=3)
        top = hyps[0] if hyps else None
        biz = inc.expected_impact
        state = build_incident_state(
            inc.fault_type, inc.severity,
            _biz_window(result.business, inc.start_min, inc.end_min),
            root_cause_confidence=top.confidence if top else 0.5)
        rec = recommend(state)[0]
        cards.append({
            "id": inc.fault_id, "fault_type": inc.fault_type.value,
            "title": inc.root_cause_desc,
            "severity": inc.severity.value,
            "start_min": inc.start_min, "end_min": inc.end_min,
            "start_time": inc.start_time.isoformat(),
            "predicted_probability": prob, "eta_minutes": eta,
            "root_cause_service": top.cause_service if top else None,
            "root_cause_type": top.cause_type if top else None,
            "root_cause_confidence": top.confidence if top else None,
            "affected_services": inc.affected_services,
            "business_impact": biz,
            "recommended_action": rec.action,
            "recommendation_reason": rec.reason,
            "recovery_action_truth": inc.recovery_action,
        })
    cards.sort(key=lambda c: Severity(c["severity"]).rank, reverse=True)
    return cards


@lru_cache(maxsize=1)
def get_state(seed: int = 42, minutes: int = 1440) -> AppState:
    """Build (once) and cache the API's working state."""
    result = run_simulation(minutes=minutes, seed=seed, run_id="live")
    model = _load_model()
    pred = _predictions(result, model)
    ctx = ToolContext(run_id=result.run_id, metrics=result.metrics, logs=result.logs,
                      traces=result.traces, deployments=result.deployments,
                      business=result.business, memory=IncidentMemory.load())
    state = AppState(result=result, ctx=ctx, model=model, predictions=pred,
                     incident_cards=_build_cards(result, pred))
    log.info("api.state.ready", run_id=result.run_id, incidents=len(state.incident_cards),
             model=bool(model))
    return state
