"""Persist a SimulationResult to the operational DB and to parquet.

The DB copy powers the API, agent tools and MCP (queryable telemetry). The
parquet copy powers the ML/feature pipeline (fast columnar reads). Both are
derived from the same deterministic run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from veyronix.core.config import ARTIFACT_DIR
from veyronix.core.constants import BUSINESS_CRITICAL
from veyronix.core.logging import get_logger
from veyronix.core.topology import dependency_centrality
from veyronix.db.models import (
    BusinessMetricRow,
    Deployment,
    Incident,
    IncidentEvent,
    Log,
    Metric,
    Service,
    Trace,
)
from veyronix.db.session import session_scope
from veyronix.simulator.generator import METRIC_CHANNELS
from veyronix.simulator.models import SimulationResult
from veyronix.simulator.profiles import SERVICE_PROFILES

log = get_logger("simulator.persist")
RUN_ROOT = ARTIFACT_DIR / "runs"


def _upsert_services(session) -> None:
    cent = dependency_centrality()
    existing = {s.name for s in session.query(Service).all()}
    for name, prof in SERVICE_PROFILES.items():
        if name.value in existing:
            continue
        session.add(Service(
            name=name.value,
            business_critical=name in BUSINESS_CRITICAL,
            centrality=cent.get(name.value, 0.0),
            tier=prof.tier,
        ))


def persist_result(result: SimulationResult) -> dict:
    """Write the run to the DB. Returns row counts."""
    counts: dict[str, int] = {}
    with session_scope() as s:
        _upsert_services(s)

        metric_maps = [
            {"run_id": r["run_id"], "ts": r["ts"], "service": r["service"],
             "values": {c: r[c] for c in METRIC_CHANNELS}}
            for r in result.metrics.to_dict("records")
        ]
        s.bulk_insert_mappings(Metric, metric_maps)
        counts["metrics"] = len(metric_maps)

        biz_maps = [
            {"run_id": r["run_id"], "ts": r["ts"],
             "values": {k: r[k] for k in r if k not in ("run_id", "ts", "minute")}}
            for r in result.business.to_dict("records")
        ]
        s.bulk_insert_mappings(BusinessMetricRow, biz_maps)
        counts["business_metrics"] = len(biz_maps)

        if not result.logs.empty:
            s.bulk_insert_mappings(Log, [
                {"run_id": r["run_id"], "ts": r["ts"], "service": r["service"],
                 "level": r["level"], "template": r["template"],
                 "message": r["message"], "count": int(r["count"])}
                for r in result.logs.to_dict("records")
            ])
            counts["logs"] = len(result.logs)

        if not result.traces.empty:
            s.bulk_insert_mappings(Trace, [
                {"run_id": r["run_id"], "ts": r["ts"], "trace_id": r["trace_id"],
                 "root_service": r["root_service"], "duration_ms": float(r["duration_ms"]),
                 "error": bool(r["error"]), "spans": r["spans"]}
                for r in result.traces.to_dict("records")
            ])
            counts["traces"] = len(result.traces)

        dep_id_map: dict[int, int] = {}
        if not result.deployments.empty:
            for r in result.deployments.to_dict("records"):
                dep = Deployment(
                    run_id=r["run_id"], ts=r["ts"], service=r["service"],
                    version=r["version"], change_summary=r["change_summary"],
                    is_bad=bool(r["is_bad"]),
                )
                s.add(dep)
                s.flush()
                dep_id_map[int(r["dep_id"])] = dep.id
            counts["deployments"] = len(result.deployments)

        for inc in result.incidents:
            trig = dep_id_map.get(inc.trigger_deployment_id) if inc.trigger_deployment_id else None
            row = Incident(
                run_id=result.run_id, fault_type=inc.fault_type.value,
                root_cause_service=inc.root_cause_service.value,
                root_cause_desc=inc.root_cause_desc, start_time=inc.start_time,
                end_time=inc.end_time, severity=inc.severity.value,
                affected_services=inc.affected_services,
                expected_impact=inc.expected_impact,
                recovery_action=inc.recovery_action, trigger_deployment_id=trig,
            )
            row.events = [
                IncidentEvent(ts=inc.start_time, kind="incident_start",
                              detail=inc.root_cause_desc),
                IncidentEvent(ts=inc.end_time, kind="incident_recovered",
                              detail="telemetry returned to baseline"),
            ]
            s.add(row)
        counts["incidents"] = len(result.incidents)
    log.info("persist.db.done", run_id=result.run_id, **counts)
    return counts


def save_parquet(result: SimulationResult, base_dir: Path | None = None) -> Path:
    """Write the run to parquet for the ML/feature pipeline."""
    base = (base_dir or RUN_ROOT) / result.run_id
    base.mkdir(parents=True, exist_ok=True)
    result.metrics.to_parquet(base / "metrics.parquet", index=False)
    result.business.to_parquet(base / "business.parquet", index=False)
    if not result.logs.empty:
        result.logs.to_parquet(base / "logs.parquet", index=False)
    if not result.deployments.empty:
        result.deployments.to_parquet(base / "deployments.parquet", index=False)

    incidents = [{
        "fault_id": i.fault_id, "fault_type": i.fault_type.value,
        "root_cause_service": i.root_cause_service.value,
        "root_cause_desc": i.root_cause_desc,
        "start_min": i.start_min, "end_min": i.end_min,
        "start_time": i.start_time.isoformat(), "end_time": i.end_time.isoformat(),
        "severity": i.severity.value, "affected_services": i.affected_services,
        "recovery_action": i.recovery_action, "expected_impact": i.expected_impact,
        "trigger_deployment_id": i.trigger_deployment_id,
    } for i in result.incidents]
    (base / "incidents.json").write_text(json.dumps(incidents, indent=2))
    (base / "meta.json").write_text(json.dumps(
        {**result.summary(), **result.config,
         "start_time": result.start_time.isoformat()}, indent=2))

    # pointer to the most recent run for the pipeline to pick up
    (RUN_ROOT / "latest.txt").parent.mkdir(parents=True, exist_ok=True)
    (RUN_ROOT / "latest.txt").write_text(result.run_id)
    log.info("persist.parquet.done", run_id=result.run_id, path=str(base))
    return base


def load_latest_run_id() -> str | None:
    p = RUN_ROOT / "latest.txt"
    return p.read_text().strip() if p.exists() else None


def load_run_frames(run_id: str, base_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Load a persisted run's frames from parquet for ML use."""
    base = (base_dir or RUN_ROOT) / run_id
    out = {
        "metrics": pd.read_parquet(base / "metrics.parquet"),
        "business": pd.read_parquet(base / "business.parquet"),
        "incidents": pd.read_json(base / "incidents.json"),
    }
    for opt in ("logs", "deployments"):
        fp = base / f"{opt}.parquet"
        if fp.exists():
            out[opt] = pd.read_parquet(fp)
    return out
