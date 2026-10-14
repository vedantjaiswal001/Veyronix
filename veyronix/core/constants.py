"""Canonical enums and vocabulary shared across every VEYRONIX layer.

This module is the single source of truth for fault types, severities, action
types, risk tiers, metric names, and the service topology. Keeping them here
prevents string drift between the simulator, ML, root-cause, agents and UI.
"""

from __future__ import annotations

from enum import StrEnum


class Severity(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    @property
    def rank(self) -> int:
        return {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}[self.value]


class FaultType(StrEnum):
    """The 13 fault types the injection engine must support (spec §8)."""

    CPU_SPIKE = "cpu_spike"
    MEMORY_LEAK = "memory_leak"
    DATABASE_OVERLOAD = "database_overload"
    NETWORK_LATENCY = "network_latency"
    PACKET_LOSS = "packet_loss"
    TRAFFIC_SPIKE = "traffic_spike"
    SERVICE_CRASH = "service_crash"
    BAD_DEPLOYMENT = "bad_deployment"
    DEPENDENCY_FAILURE = "dependency_failure"
    QUEUE_BACKLOG = "queue_backlog"
    CASCADING_FAILURE = "cascading_failure"
    DATA_PIPELINE_FAILURE = "data_pipeline_failure"
    MODEL_DRIFT = "model_drift"


# Fault families used for the "unknown failure" experiment (spec §19): models
# train on these and are tested on CASCADING_FAILURE, which is held out.
KNOWN_FAULTS: tuple[FaultType, ...] = (
    FaultType.CPU_SPIKE,
    FaultType.MEMORY_LEAK,
    FaultType.DATABASE_OVERLOAD,
    FaultType.NETWORK_LATENCY,
    FaultType.PACKET_LOSS,
    FaultType.TRAFFIC_SPIKE,
    FaultType.SERVICE_CRASH,
    FaultType.BAD_DEPLOYMENT,
    FaultType.DEPENDENCY_FAILURE,
    FaultType.QUEUE_BACKLOG,
)
HELD_OUT_FAULTS: tuple[FaultType, ...] = (FaultType.CASCADING_FAILURE,)


class ActionType(StrEnum):
    """Allowlisted remediation actions. No free-form execution is ever allowed."""

    NO_OP = "no_op"
    ROLLBACK_DEPLOYMENT = "rollback_deployment"
    SCALE_UP_SERVICE = "scale_up_service"
    SCALE_UP_DATABASE = "scale_up_database"
    RESTART_SERVICE = "restart_service"
    CLEAR_QUEUE = "clear_queue"
    FLUSH_CACHE = "flush_cache"
    FAILOVER_DEPENDENCY = "failover_dependency"
    DELETE_DATABASE = "delete_database"  # deliberately present + BLOCKED (spec §33)


class RiskTier(StrEnum):
    LOW = "LOW"        # may auto-execute (if globally enabled)
    MEDIUM = "MEDIUM"  # requires human approval
    HIGH = "HIGH"      # requires elevated approval
    BLOCKED = "BLOCKED"  # never executes


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    FAILED = "failed"
    EXPIRED = "expired"


class MetricName(StrEnum):
    """Per-service telemetry channels emitted by the simulator (spec §7)."""

    CPU_PCT = "cpu_pct"
    MEMORY_PCT = "memory_pct"
    REQUEST_RATE = "request_rate"          # requests / second
    LATENCY_P50_MS = "latency_p50_ms"
    LATENCY_P95_MS = "latency_p95_ms"
    LATENCY_P99_MS = "latency_p99_ms"
    ERROR_RATE = "error_rate"              # fraction 0..1
    HTTP_5XX_RATE = "http_5xx_rate"
    HTTP_4XX_RATE = "http_4xx_rate"
    DB_LATENCY_MS = "db_latency_ms"
    DB_CONNECTIONS = "db_connections"
    DB_CPU_PCT = "db_cpu_pct"
    CACHE_HIT_RATE = "cache_hit_rate"
    QUEUE_DEPTH = "queue_depth"
    NETWORK_LATENCY_MS = "network_latency_ms"
    PACKET_LOSS_PCT = "packet_loss_pct"
    GC_PAUSE_MS = "gc_pause_ms"
    SATURATION = "saturation"              # resource saturation 0..1


# Business-level metrics (spec §22): technical health -> business outcome.
class BusinessMetric(StrEnum):
    ACTIVE_USERS = "active_users"
    CONVERSION_RATE = "conversion_rate"
    ORDERS_PER_MIN = "orders_per_min"
    CHECKOUT_ERROR_RATE = "checkout_error_rate"
    REVENUE_PER_MIN = "revenue_per_min"


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"


# ---- Service topology (spec §7) -------------------------------------------
# The dependency graph is DIRECTED: an edge (a -> b) means "a depends on b",
# so faults in `b` propagate upstream to `a`. Central services carry higher
# blast radius, which the severity + business-impact models use.

class ServiceName(StrEnum):
    CLIENT = "client"
    API_GATEWAY = "api_gateway"
    USER_SERVICE = "user_service"
    ORDER_SERVICE = "order_service"
    PAYMENT_SERVICE = "payment_service"
    DATABASE = "database"
    CACHE = "cache"
    MESSAGE_QUEUE = "message_queue"


# (source_depends_on_target) edges.
SERVICE_DEPENDENCIES: tuple[tuple[ServiceName, ServiceName], ...] = (
    (ServiceName.CLIENT, ServiceName.API_GATEWAY),
    (ServiceName.API_GATEWAY, ServiceName.USER_SERVICE),
    (ServiceName.API_GATEWAY, ServiceName.ORDER_SERVICE),
    (ServiceName.API_GATEWAY, ServiceName.PAYMENT_SERVICE),
    (ServiceName.USER_SERVICE, ServiceName.DATABASE),
    (ServiceName.USER_SERVICE, ServiceName.CACHE),
    (ServiceName.ORDER_SERVICE, ServiceName.DATABASE),
    (ServiceName.ORDER_SERVICE, ServiceName.MESSAGE_QUEUE),
    (ServiceName.ORDER_SERVICE, ServiceName.CACHE),
    (ServiceName.PAYMENT_SERVICE, ServiceName.DATABASE),
    (ServiceName.PAYMENT_SERVICE, ServiceName.MESSAGE_QUEUE),
)

# Services that emit application telemetry (the CLIENT is a load source only).
TELEMETRY_SERVICES: tuple[ServiceName, ...] = tuple(
    s for s in ServiceName if s is not ServiceName.CLIENT
)

# Business-critical services drive revenue; degradation here costs the most.
BUSINESS_CRITICAL: frozenset[ServiceName] = frozenset(
    {ServiceName.ORDER_SERVICE, ServiceName.PAYMENT_SERVICE, ServiceName.DATABASE}
)

# The label horizon: an observation at time t is positive if an incident STARTS
# within (t, t + horizon]. Set from Settings.prediction_horizon_minutes at use.
UNKNOWN_ANOMALY_LABEL = "UNKNOWN_ANOMALY"
