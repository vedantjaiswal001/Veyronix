"""Dataclasses describing scheduled faults, ground-truth incidents, and the
full simulation result."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from veyronix.core.constants import FaultType, ServiceName, Severity


@dataclass
class FaultInstance:
    """A fault scheduled onto the timeline."""

    fault_id: int
    fault_type: FaultType
    target: ServiceName
    start_min: int            # labelled incident start (minute index)
    duration_min: int         # main-phase duration
    lead_min: int             # precursor length before start_min

    @property
    def onset_min(self) -> int:
        """First minute any perturbation is present (precursor begins)."""
        return self.start_min - self.lead_min

    @property
    def end_min(self) -> int:
        return self.start_min + self.duration_min

    @property
    def total_active(self) -> int:
        return self.lead_min + self.duration_min


@dataclass
class IncidentGroundTruth:
    """Labelled incident (spec §8) — the evaluation target."""

    fault_id: int
    fault_type: FaultType
    root_cause_service: ServiceName
    root_cause_desc: str
    start_time: datetime
    end_time: datetime
    start_min: int
    end_min: int
    severity: Severity
    affected_services: list[str]
    recovery_action: str
    expected_impact: dict = field(default_factory=dict)
    trigger_deployment_id: int | None = None


@dataclass
class SimulationResult:
    run_id: str
    start_time: datetime
    minutes: int
    metrics: pd.DataFrame          # wide: [ts, minute, service, <metric channels>]
    business: pd.DataFrame         # [ts, minute, <business channels>]
    logs: pd.DataFrame             # [ts, minute, service, level, template, message, count]
    traces: pd.DataFrame           # [ts, minute, trace_id, root_service, duration_ms, error, spans]
    deployments: pd.DataFrame      # [ts, minute, service, version, change_summary, is_bad]
    incidents: list[IncidentGroundTruth]
    config: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "run_id": self.run_id,
            "minutes": self.minutes,
            "metric_rows": len(self.metrics),
            "services": sorted(self.metrics["service"].unique().tolist()),
            "incidents": len(self.incidents),
            "fault_types": sorted({i.fault_type.value for i in self.incidents}),
            "deployments": len(self.deployments),
            "log_rows": len(self.logs),
            "trace_rows": len(self.traces),
        }
