"""Fault catalog (spec §8/§9).

Design
------
A fault injects only the *root* perturbation (e.g. a bad deploy raises DB query
cost). The generator's dependency-coupling + error-propagation model then
produces the downstream cascade (DB latency -> API latency -> timeouts ->
checkout failures -> revenue). Incidents therefore have realistic *causal*
temporal signatures, not independent random spikes.

Timing vs. magnitude are separated: the engine computes an intensity envelope in
[0, 1] over the active window and each fault only declares the peak magnitudes it
applies at intensity = 1, plus ground-truth metadata (root-cause service,
severity, recovery action).

Precursors & early warning
--------------------------
Most faults ramp in over a ``lead`` window *before* the officially labelled
incident start, so telemetry shows subtle precursors an ML model can learn from
(this is what makes early warning possible without leakage). ``step`` faults
(traffic spike, hard crash) are deliberately abrupt — they are close to
unpredictable, which the evaluation reports honestly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from veyronix.core.constants import ActionType, FaultType, ServiceName, Severity

# Per-service mutable perturbation accumulator for one minute.
Drivers = dict[str, float]

PRECURSOR_FLOOR = 0.22  # relative intensity reached by the end of the lead window


def blank_drivers() -> Drivers:
    return {
        "req_mult": 1.0,
        "cpu_add": 0.0,
        "mem_drift": 0.0,
        "own_latency_mult": 1.0,
        "own_latency_add": 0.0,
        "error_add": 0.0,
        "db_latency_add": 0.0,
        "db_conn_add": 0.0,
        "db_cpu_add": 0.0,
        "network_add": 0.0,
        "packet_loss_add": 0.0,
        "gc_add": 0.0,
        "queue_add": 0.0,
        "cache_hit_mult": 1.0,
        "crash": 0.0,
        "model_drift": 0.0,
    }


@dataclass(frozen=True)
class FaultSpec:
    fault_type: FaultType
    default_target: ServiceName
    severity: Severity
    recovery: ActionType
    cause_desc: str
    shape: str                     # "plateau" | "ramp" | "grow" | "step"
    lead: int                      # precursor minutes before labelled start
    mags: dict[str, float]         # driver magnitudes at intensity = 1
    emits_deployment: bool = False
    crash_at_end: bool = False     # service crashes in the final minutes
    cache_kill: bool = False       # collapses cache hit-rate with intensity
    signature_metrics: tuple[str, ...] = field(default_factory=tuple)


def apply_fault(
    spec: FaultSpec, intensity: float, in_tail: bool,
    drivers_all: dict[str, Drivers], target: ServiceName, u: float,
) -> None:
    """Apply a fault to the appropriate services' drivers.

    Effects are routed to the service that *owns* the affected channel, which is
    not always the root-cause target. This is deliberate and central to making
    root-cause analysis non-trivial: e.g. a bad deployment's root cause is the
    service that deployed it, but its symptom (query latency) appears on the
    database. The dependency-coupling model then propagates the rest.
    """
    if intensity <= 0:
        return
    m = spec.mags
    jitter = 0.85 + 0.3 * u
    tgt = drivers_all[target.value]

    # effects local to the root-cause service
    tgt["cpu_add"] += m.get("cpu_add", 0.0) * intensity * jitter
    tgt["mem_drift"] += m.get("mem_drift_rate", 0.0)
    tgt["own_latency_mult"] *= 1.0 + m.get("own_lat_mult_extra", 0.0) * intensity
    tgt["own_latency_add"] += m.get("own_lat_add", 0.0) * intensity
    tgt["error_add"] += m.get("error_add", 0.0) * intensity
    tgt["network_add"] += m.get("network_add", 0.0) * intensity * jitter
    tgt["packet_loss_add"] += m.get("packet_loss_add", 0.0) * intensity
    tgt["gc_add"] += m.get("gc_add", 0.0) * intensity
    if m.get("model_drift", 0.0):
        tgt["model_drift"] = max(tgt["model_drift"], intensity)
    if spec.crash_at_end and in_tail:
        tgt["crash"] = 1.0
        tgt["error_add"] += 0.9

    # database-owned channels -> always land on the DATABASE service
    if any(k in m for k in ("db_latency_add", "db_conn_add", "db_cpu_add")):
        db = drivers_all[ServiceName.DATABASE.value]
        db["db_latency_add"] += m.get("db_latency_add", 0.0) * intensity * jitter
        db["db_conn_add"] += m.get("db_conn_add", 0.0) * intensity
        db["db_cpu_add"] += m.get("db_cpu_add", 0.0) * intensity

    # queue depth is owned by the message queue
    if m.get("queue_add", 0.0):
        drivers_all[ServiceName.MESSAGE_QUEUE.value]["queue_add"] += m["queue_add"] * intensity

    # cache-hit collapse lands on the cache
    if spec.cache_kill:
        c = drivers_all[ServiceName.CACHE.value]
        c["cache_hit_mult"] *= max(0.05, 1.0 - 0.9 * intensity)
        c["own_latency_mult"] *= 1.0 + 1.5 * intensity  # cache miss storm slows it

    # a traffic spike raises request rate fleet-wide (global load surge)
    if m.get("req_mult_extra", 0.0):
        for dr in drivers_all.values():
            dr["req_mult"] *= 1.0 + m["req_mult_extra"] * intensity * jitter


FAULT_CATALOG: dict[FaultType, FaultSpec] = {
    FaultType.CPU_SPIKE: FaultSpec(
        FaultType.CPU_SPIKE, ServiceName.USER_SERVICE, Severity.MEDIUM,
        ActionType.SCALE_UP_SERVICE, "CPU saturation on {svc}", "plateau", lead=8,
        mags={"cpu_add": 58, "own_lat_mult_extra": 1.6},
        signature_metrics=("cpu_pct", "latency_p95_ms"),
    ),
    FaultType.MEMORY_LEAK: FaultSpec(
        FaultType.MEMORY_LEAK, ServiceName.ORDER_SERVICE, Severity.HIGH,
        ActionType.RESTART_SERVICE,
        "Memory leak on {svc} causing GC thrash then crash", "ramp", lead=20,
        mags={"mem_drift_rate": 0.85, "gc_add": 55, "own_lat_mult_extra": 1.4},
        crash_at_end=True,
        signature_metrics=("memory_pct", "gc_pause_ms", "latency_p95_ms"),
    ),
    FaultType.DATABASE_OVERLOAD: FaultSpec(
        FaultType.DATABASE_OVERLOAD, ServiceName.DATABASE, Severity.HIGH,
        ActionType.SCALE_UP_DATABASE,
        "Database overload: connection pool + CPU saturated", "plateau", lead=10,
        mags={"db_conn_add": 220, "db_cpu_add": 55, "db_latency_add": 240},
        signature_metrics=("db_latency_ms", "db_cpu_pct", "db_connections"),
    ),
    FaultType.NETWORK_LATENCY: FaultSpec(
        FaultType.NETWORK_LATENCY, ServiceName.PAYMENT_SERVICE, Severity.MEDIUM,
        ActionType.FAILOVER_DEPENDENCY, "Elevated network latency at {svc}",
        "plateau", lead=8, mags={"network_add": 240, "own_lat_mult_extra": 0.4},
        signature_metrics=("network_latency_ms", "latency_p95_ms"),
    ),
    FaultType.PACKET_LOSS: FaultSpec(
        FaultType.PACKET_LOSS, ServiceName.API_GATEWAY, Severity.MEDIUM,
        ActionType.FAILOVER_DEPENDENCY, "Packet loss causing retransmits at {svc}",
        "plateau", lead=8, mags={"packet_loss_add": 12, "error_add": 0.06,
                                 "own_lat_mult_extra": 0.8},
        signature_metrics=("packet_loss_pct", "error_rate"),
    ),
    FaultType.TRAFFIC_SPIKE: FaultSpec(
        FaultType.TRAFFIC_SPIKE, ServiceName.API_GATEWAY, Severity.MEDIUM,
        ActionType.SCALE_UP_SERVICE, "Sudden traffic spike overwhelming {svc}",
        "step", lead=0, mags={"req_mult_extra": 2.3, "cpu_add": 20,
                              "queue_add": 8000},
        signature_metrics=("request_rate", "cpu_pct", "queue_depth"),
    ),
    FaultType.SERVICE_CRASH: FaultSpec(
        FaultType.SERVICE_CRASH, ServiceName.USER_SERVICE, Severity.HIGH,
        ActionType.RESTART_SERVICE, "{svc} process crashed / unavailable", "step",
        lead=0, mags={"error_add": 0.9, "own_lat_mult_extra": 3.0},
        signature_metrics=("error_rate", "http_5xx_rate"),
    ),
    FaultType.BAD_DEPLOYMENT: FaultSpec(
        FaultType.BAD_DEPLOYMENT, ServiceName.ORDER_SERVICE, Severity.CRITICAL,
        ActionType.ROLLBACK_DEPLOYMENT,
        "Bad deployment shipped an inefficient SQL query", "ramp", lead=18,
        mags={"db_latency_add": 340, "db_cpu_add": 45, "db_conn_add": 140},
        emits_deployment=True,
        signature_metrics=("db_latency_ms", "latency_p95_ms", "error_rate"),
    ),
    FaultType.DEPENDENCY_FAILURE: FaultSpec(
        FaultType.DEPENDENCY_FAILURE, ServiceName.CACHE, Severity.HIGH,
        ActionType.FAILOVER_DEPENDENCY, "Dependency {svc} failing", "plateau",
        lead=6, mags={"error_add": 0.45, "own_lat_mult_extra": 2.4},
        cache_kill=True,
        signature_metrics=("cache_hit_rate", "error_rate", "latency_p95_ms"),
    ),
    FaultType.QUEUE_BACKLOG: FaultSpec(
        FaultType.QUEUE_BACKLOG, ServiceName.MESSAGE_QUEUE, Severity.MEDIUM,
        ActionType.CLEAR_QUEUE, "Message queue backlog growing on {svc}", "grow",
        lead=12, mags={"queue_add": 52000, "own_lat_mult_extra": 1.8,
                       "error_add": 0.08},
        signature_metrics=("queue_depth", "latency_p95_ms"),
    ),
    FaultType.CASCADING_FAILURE: FaultSpec(
        FaultType.CASCADING_FAILURE, ServiceName.ORDER_SERVICE, Severity.CRITICAL,
        ActionType.ROLLBACK_DEPLOYMENT, "Cascading failure spreading across services",
        "ramp", lead=10, mags={"cpu_add": 46, "own_lat_mult_extra": 2.2,
                               "error_add": 0.25, "mem_drift_rate": 0.5},
        signature_metrics=("cpu_pct", "latency_p95_ms", "error_rate"),
    ),
    FaultType.DATA_PIPELINE_FAILURE: FaultSpec(
        FaultType.DATA_PIPELINE_FAILURE, ServiceName.MESSAGE_QUEUE, Severity.HIGH,
        ActionType.CLEAR_QUEUE, "Data pipeline stalled: consumers not draining queue",
        "grow", lead=15, mags={"queue_add": 62000, "error_add": 0.12,
                               "own_lat_mult_extra": 1.0},
        signature_metrics=("queue_depth", "error_rate"),
    ),
    FaultType.MODEL_DRIFT: FaultSpec(
        FaultType.MODEL_DRIFT, ServiceName.ORDER_SERVICE, Severity.MEDIUM,
        ActionType.NO_OP, "ML model input distribution drifted (degraded quality)",
        "ramp", lead=20, mags={"error_add": 0.02, "model_drift": 1.0},
        signature_metrics=("error_rate",),
    ),
}


def duration_bounds(fault: FaultType) -> tuple[int, int]:
    """Plausible (min, max) *main-phase* duration in minutes per fault type."""
    slow = {FaultType.MEMORY_LEAK, FaultType.BAD_DEPLOYMENT,
            FaultType.QUEUE_BACKLOG, FaultType.DATA_PIPELINE_FAILURE,
            FaultType.MODEL_DRIFT}
    return (35, 70) if fault in slow else (12, 30)
