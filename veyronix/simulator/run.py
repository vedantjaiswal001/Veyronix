"""High-level simulation orchestration: schedule faults, generate telemetry,
build ground-truth incidents + deployments, and compute expected impact."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from veyronix.core.constants import FaultType, ServiceName, Severity
from veyronix.core.logging import get_logger
from veyronix.core.topology import upstream_dependents
from veyronix.simulator.faults import FAULT_CATALOG
from veyronix.simulator.generator import SimulationEngine
from veyronix.simulator.models import (
    IncidentGroundTruth,
    SimulationResult,
)
from veyronix.simulator.scenarios import (
    build_bad_deploy_cascade,
    build_unknown_cascade,
    schedule_random_faults,
)

log = get_logger("simulator")

TELEMETRY = [s.value for s in ServiceName if s != ServiceName.CLIENT]


def _affected_services(target: ServiceName) -> list[str]:
    """Root-cause service plus everything that (transitively) depends on it."""
    affected = {target.value, *upstream_dependents(target.value)}
    return sorted(s for s in affected if s in TELEMETRY)


def run_simulation(
    minutes: int = 1440,
    seed: int = 42,
    run_id: str | None = None,
    scenario: str = "random",
    start_time: datetime | None = None,
) -> SimulationResult:
    """Generate a full labelled simulation run."""
    run_id = run_id or f"run-{uuid.uuid4().hex[:8]}"
    start_time = start_time or datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    rng = np.random.default_rng(seed)

    if scenario == "bad_deploy":
        faults = build_bad_deploy_cascade()
    elif scenario == "unknown_cascade":
        faults = build_unknown_cascade()
    else:
        faults = schedule_random_faults(minutes, seed)

    engine = SimulationEngine(run_id, start_time, minutes, seed)
    result = engine.run(faults)

    result.deployments = _build_deployments(run_id, start_time, minutes, faults, rng)
    result.incidents = _build_incidents(run_id, start_time, faults, result)
    result.config.update({"seed": seed, "scenario": scenario})
    log.info("simulation.done", **result.summary())
    return result


def _build_deployments(run_id, start_time, minutes, faults, rng) -> pd.DataFrame:
    rows: list[dict] = []
    dep_id = 900
    # bad deployments (ground-truth cause of BAD_DEPLOYMENT incidents)
    for f in faults:
        if f.fault_type == FaultType.BAD_DEPLOYMENT:
            ts = start_time + timedelta(minutes=f.start_min)
            rows.append({
                "dep_id": dep_id, "run_id": run_id, "ts": ts, "minute": f.start_min,
                "service": f.target.value, "version": f"v{dep_id}",
                "change_summary": "refactor order query (introduced N+1 / full scan)",
                "is_bad": True, "fault_id": f.fault_id,
            })
            dep_id += 1
    # benign deployments scattered as decoys
    n_benign = max(3, minutes // 240)
    for _ in range(n_benign):
        m = int(rng.integers(30, minutes - 30))
        svc = TELEMETRY[int(rng.integers(len(TELEMETRY)))]
        rows.append({
            "dep_id": dep_id, "run_id": run_id, "ts": start_time + timedelta(minutes=m),
            "minute": m, "service": svc, "version": f"v{dep_id}",
            "change_summary": "routine dependency bump / config change",
            "is_bad": False, "fault_id": None,
        })
        dep_id += 1
    df = pd.DataFrame(rows)
    return df.sort_values("minute").reset_index(drop=True) if not df.empty else df


def _build_incidents(run_id, start_time, faults, result) -> list[IncidentGroundTruth]:
    metrics = result.metrics
    business = result.business
    incidents: list[IncidentGroundTruth] = []
    dep_lookup = {}
    if not result.deployments.empty:
        for _, d in result.deployments[result.deployments["is_bad"]].iterrows():
            dep_lookup[d["fault_id"]] = int(d["dep_id"])

    for f in faults:
        spec = FAULT_CATALOG[f.fault_type]
        affected = _affected_services(f.target)
        window = metrics[(metrics["minute"] >= f.start_min) & (metrics["minute"] < f.end_min)]
        aw = window[window["service"].isin(affected)]
        peak_err = float(aw["error_rate"].max()) if not aw.empty else 0.0
        peak_p95 = float(aw["latency_p95_ms"].max()) if not aw.empty else 0.0

        biz_win = business[(business["minute"] >= f.start_min) & (business["minute"] < f.end_min)]
        # baseline revenue = median revenue at the same clock-time is complex; use
        # pre-incident window of equal length as the counterfactual baseline.
        pre = business[(business["minute"] >= f.start_min - f.duration_min)
                       & (business["minute"] < f.start_min)]
        rev_loss = 0.0
        lost_orders = 0.0
        if not biz_win.empty and not pre.empty:
            base_rev = float(pre["revenue_per_min"].mean())
            rev_loss = max(0.0, base_rev * len(biz_win) - float(biz_win["revenue_per_min"].sum()))
            base_ord = float(pre["orders_per_min"].mean())
            lost_orders = max(0.0, base_ord * len(biz_win) - float(biz_win["orders_per_min"].sum()))
        affected_users = round(float(biz_win["active_users"].mean()) if not biz_win.empty else 0.0, 0)

        incidents.append(IncidentGroundTruth(
            fault_id=f.fault_id,
            fault_type=f.fault_type,
            root_cause_service=f.target,
            root_cause_desc=spec.cause_desc.format(svc=f.target.value),
            start_time=start_time + timedelta(minutes=f.start_min),
            end_time=start_time + timedelta(minutes=f.end_min),
            start_min=f.start_min,
            end_min=f.end_min,
            severity=spec.severity,
            affected_services=affected,
            recovery_action=spec.recovery.value,
            expected_impact={
                "peak_error_rate": round(peak_err, 4),
                "peak_latency_p95_ms": round(peak_p95, 1),
                "estimated_revenue_loss": round(rev_loss, 2),
                "estimated_lost_orders": round(lost_orders, 1),
                "estimated_affected_users": affected_users,
                "currency": "INR",
                "note": "model estimate from SIMULATED environment, not financial truth",
            },
            trigger_deployment_id=dep_lookup.get(f.fault_id),
        ))
    return incidents


def incident_intervals(result: SimulationResult) -> list[tuple[int, int, Severity]]:
    """(start_min, end_min, severity) for each incident — used for labelling."""
    return [(i.start_min, i.end_min, i.severity) for i in result.incidents]
