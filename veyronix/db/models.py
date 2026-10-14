"""VEYRONIX operational data model (spec §10).

Design notes
------------
* Every time-series table indexes ``(service, ts)`` because virtually all reads
  are "give me channel X for service Y over window [t0, t1]".
* Telemetry uses a wide-per-sample row (one row per service per minute) rather
  than tall key/value rows: the simulator emits fixed channels, so a wide row
  keeps feature engineering vectorized and cheap.
* JSON columns hold open-ended structures (evidence, params, metric snapshots)
  where a rigid schema would be premature; typed columns are used everywhere a
  value is queried or filtered.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from veyronix.db.base import Base, TimestampMixin


# --------------------------------------------------------------------------- #
# Reference / identity
# --------------------------------------------------------------------------- #
class User(Base, TimestampMixin):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    email: Mapped[str | None] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(32), default="operator")  # operator|approver|admin


class Service(Base, TimestampMixin):
    __tablename__ = "services"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    business_critical: Mapped[bool] = mapped_column(Boolean, default=False)
    centrality: Mapped[float] = mapped_column(Float, default=0.0)
    tier: Mapped[str] = mapped_column(String(32), default="service")  # gateway|service|datastore|infra


# --------------------------------------------------------------------------- #
# Telemetry (metrics / logs / traces) — wide rows
# --------------------------------------------------------------------------- #
class Metric(Base):
    __tablename__ = "metrics"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    service: Mapped[str] = mapped_column(String(64), index=True)
    values: Mapped[dict[str, Any]] = mapped_column(JSON)  # {MetricName: float}
    __table_args__ = (
        Index("ix_metrics_run_service_ts", "run_id", "service", "ts"),
    )


class BusinessMetricRow(Base):
    __tablename__ = "business_metrics"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    values: Mapped[dict[str, Any]] = mapped_column(JSON)
    __table_args__ = (Index("ix_bizmetrics_run_ts", "run_id", "ts"),)


class Log(Base):
    __tablename__ = "logs"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    service: Mapped[str] = mapped_column(String(64), index=True)
    level: Mapped[str] = mapped_column(String(8), index=True)
    template: Mapped[str] = mapped_column(String(128), index=True)  # log template id
    message: Mapped[str] = mapped_column(Text)
    count: Mapped[int] = mapped_column(Integer, default=1)  # aggregated per minute
    __table_args__ = (Index("ix_logs_run_service_ts", "run_id", "service", "ts"),)


class Trace(Base):
    __tablename__ = "traces"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    trace_id: Mapped[str] = mapped_column(String(64), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    root_service: Mapped[str] = mapped_column(String(64), index=True)
    duration_ms: Mapped[float] = mapped_column(Float)
    error: Mapped[bool] = mapped_column(Boolean, default=False)
    spans: Mapped[list[dict[str, Any]]] = mapped_column(JSON)  # [{service, op, ms, error}]
    __table_args__ = (Index("ix_traces_run_ts", "run_id", "ts"),)


class Deployment(Base):
    __tablename__ = "deployments"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    service: Mapped[str] = mapped_column(String(64), index=True)
    version: Mapped[str] = mapped_column(String(32))
    change_summary: Mapped[str] = mapped_column(Text, default="")
    is_bad: Mapped[bool] = mapped_column(Boolean, default=False)  # ground-truth flag
    __table_args__ = (Index("ix_deploy_run_service_ts", "run_id", "service", "ts"),)


# --------------------------------------------------------------------------- #
# Incidents + ground truth
# --------------------------------------------------------------------------- #
class Incident(Base, TimestampMixin):
    __tablename__ = "incidents"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    fault_type: Mapped[str] = mapped_column(String(48), index=True)
    root_cause_service: Mapped[str] = mapped_column(String(64), index=True)
    root_cause_desc: Mapped[str] = mapped_column(Text, default="")
    start_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    end_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    severity: Mapped[str] = mapped_column(String(16), index=True)
    affected_services: Mapped[list[str]] = mapped_column(JSON, default=list)
    expected_impact: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    recovery_action: Mapped[str] = mapped_column(String(48), default="")
    trigger_deployment_id: Mapped[int | None] = mapped_column(ForeignKey("deployments.id"))
    is_ground_truth: Mapped[bool] = mapped_column(Boolean, default=True)

    events: Mapped[list[IncidentEvent]] = relationship(
        back_populates="incident", cascade="all, delete-orphan"
    )
    __table_args__ = (Index("ix_incident_run_start", "run_id", "start_time"),)


class IncidentEvent(Base):
    __tablename__ = "incident_events"
    id: Mapped[int] = mapped_column(primary_key=True)
    incident_id: Mapped[int] = mapped_column(ForeignKey("incidents.id"), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    kind: Mapped[str] = mapped_column(String(48))  # normal|latency_rising|anomaly|prediction|...
    detail: Mapped[str] = mapped_column(Text, default="")
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    incident: Mapped[Incident] = relationship(back_populates="events")


# --------------------------------------------------------------------------- #
# ML outputs
# --------------------------------------------------------------------------- #
class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    model_version_id: Mapped[int | None] = mapped_column(ForeignKey("model_versions.id"))
    incident_probability: Mapped[float] = mapped_column(Float)
    predicted_positive: Mapped[bool] = mapped_column(Boolean, index=True)
    horizon_minutes: Mapped[int] = mapped_column(Integer)
    predicted_severity: Mapped[str | None] = mapped_column(String(16))
    eta_minutes: Mapped[float | None] = mapped_column(Float)
    __table_args__ = (Index("ix_pred_run_ts", "run_id", "ts"),)


class Anomaly(Base):
    __tablename__ = "anomalies"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    service: Mapped[str] = mapped_column(String(64), index=True)
    metric: Mapped[str] = mapped_column(String(48), index=True)
    score: Mapped[float] = mapped_column(Float)
    severity: Mapped[str] = mapped_column(String(16))
    method: Mapped[str] = mapped_column(String(32))  # isolation_forest|zscore|ewma
    is_unknown: Mapped[bool] = mapped_column(Boolean, default=False)
    __table_args__ = (Index("ix_anom_run_service_ts", "run_id", "service", "ts"),)


# --------------------------------------------------------------------------- #
# Root cause + causal graph
# --------------------------------------------------------------------------- #
class RootCauseHypothesis(Base, TimestampMixin):
    __tablename__ = "root_cause_hypotheses"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    incident_id: Mapped[int | None] = mapped_column(ForeignKey("incidents.id"), index=True)
    rank: Mapped[int] = mapped_column(Integer)
    cause_service: Mapped[str] = mapped_column(String(64), index=True)
    cause_type: Mapped[str] = mapped_column(String(48))
    confidence: Mapped[float] = mapped_column(Float)
    score_breakdown: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    evidence_chain: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    explanation: Mapped[str] = mapped_column(Text, default="")


class CausalEdge(Base):
    __tablename__ = "causal_edges"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    incident_id: Mapped[int | None] = mapped_column(ForeignKey("incidents.id"), index=True)
    source: Mapped[str] = mapped_column(String(96))
    target: Mapped[str] = mapped_column(String(96))
    # Evidence tier: correlation < temporal < hypothesis < validated (spec §25).
    edge_type: Mapped[str] = mapped_column(String(24))
    weight: Mapped[float] = mapped_column(Float, default=0.0)
    lag_seconds: Mapped[float | None] = mapped_column(Float)


# --------------------------------------------------------------------------- #
# Actions / remediation
# --------------------------------------------------------------------------- #
class Action(Base, TimestampMixin):
    __tablename__ = "actions"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str | None] = mapped_column(String(64), index=True)
    incident_id: Mapped[int | None] = mapped_column(ForeignKey("incidents.id"), index=True)
    action_type: Mapped[str] = mapped_column(String(48), index=True)
    params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    risk_tier: Mapped[str] = mapped_column(String(16), index=True)
    proposed_by: Mapped[str] = mapped_column(String(32), default="recommender")
    approval_status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    approved_by: Mapped[str | None] = mapped_column(String(64))
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    reason: Mapped[str] = mapped_column(Text, default="")
    expected_benefit: Mapped[float | None] = mapped_column(Float)
    expected_cost: Mapped[float | None] = mapped_column(Float)
    confidence: Mapped[float | None] = mapped_column(Float)

    outcome: Mapped[ActionOutcome | None] = relationship(
        back_populates="action", cascade="all, delete-orphan", uselist=False
    )


class ActionOutcome(Base, TimestampMixin):
    __tablename__ = "action_outcomes"
    id: Mapped[int] = mapped_column(primary_key=True)
    action_id: Mapped[int] = mapped_column(ForeignKey("actions.id"), unique=True, index=True)
    executed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    success: Mapped[bool | None] = mapped_column(Boolean)
    predicted_outcome: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    actual_outcome: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    recovery_minutes: Mapped[float | None] = mapped_column(Float)
    notes: Mapped[str] = mapped_column(Text, default="")
    action: Mapped[Action] = relationship(back_populates="outcome")


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
class Model(Base, TimestampMixin):
    __tablename__ = "models"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    task: Mapped[str] = mapped_column(String(48))  # incident_prediction|anomaly|forecast|severity
    description: Mapped[str] = mapped_column(Text, default="")


class ModelVersion(Base, TimestampMixin):
    __tablename__ = "model_versions"
    id: Mapped[int] = mapped_column(primary_key=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("models.id"), index=True)
    version: Mapped[str] = mapped_column(String(32))
    algorithm: Mapped[str] = mapped_column(String(48))
    params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    artifact_path: Mapped[str | None] = mapped_column(String(255))
    feature_names: Mapped[list[str]] = mapped_column(JSON, default=list)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    __table_args__ = (UniqueConstraint("model_id", "version", name="uq_model_version"),)


# --------------------------------------------------------------------------- #
# Agent runs + evaluation
# --------------------------------------------------------------------------- #
class AgentRun(Base, TimestampMixin):
    __tablename__ = "agent_runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    incident_id: Mapped[int | None] = mapped_column(ForeignKey("incidents.id"), index=True)
    provider: Mapped[str] = mapped_column(String(24))
    model: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(24), default="running")  # running|done|failed|timeout
    steps: Mapped[int] = mapped_column(Integer, default=0)
    tool_calls: Mapped[int] = mapped_column(Integer, default=0)
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    duration_ms: Mapped[float | None] = mapped_column(Float)
    hypothesis: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    transcript: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)

    calls: Mapped[list[AgentToolCall]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class AgentToolCall(Base):
    __tablename__ = "agent_tool_calls"
    id: Mapped[int] = mapped_column(primary_key=True)
    agent_run_id: Mapped[int] = mapped_column(ForeignKey("agent_runs.id"), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    agent: Mapped[str] = mapped_column(String(32))
    tool: Mapped[str] = mapped_column(String(48), index=True)
    args: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    ok: Mapped[bool] = mapped_column(Boolean, default=True)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    result_summary: Mapped[str] = mapped_column(Text, default="")
    run: Mapped[AgentRun] = relationship(back_populates="calls")


class EvaluationRun(Base, TimestampMixin):
    __tablename__ = "evaluation_runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    suite: Mapped[str] = mapped_column(String(48), index=True)  # model_comparison|ablation|agent|...
    git_sha: Mapped[str | None] = mapped_column(String(40))
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    results: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


ALL_MODELS = [
    User, Service, Metric, BusinessMetricRow, Log, Trace, Deployment,
    Incident, IncidentEvent, Prediction, Anomaly, RootCauseHypothesis,
    CausalEdge, Action, ActionOutcome, Model, ModelVersion, AgentRun,
    AgentToolCall, EvaluationRun,
]
