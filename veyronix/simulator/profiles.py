"""Per-service baseline profiles and physical bounds for the simulator.

These constants define "healthy" steady-state behaviour. The generator evolves
metrics around these baselines with seasonality + noise, and the fault engine
perturbs them. Values are chosen to be *plausible* for a mid-size e-commerce
stack, not to model any specific real system.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from veyronix.core.constants import ServiceName


@dataclass(frozen=True)
class ServiceProfile:
    """Healthy steady-state parameters for one service."""

    name: ServiceName
    tier: str                      # gateway | service | datastore | infra
    base_cpu: float                # % at nominal load
    base_memory: float             # %
    base_latency_ms: float         # own processing latency (excl. dependencies)
    traffic_share: float           # fraction of global request rate it receives
    error_floor: float             # baseline error fraction
    # How strongly this service's latency is inflated by CPU saturation and by
    # its slowest dependency (0..1 coupling coefficients).
    cpu_latency_coupling: float = 0.8
    dep_latency_coupling: float = 0.6


# Global nominal request rate (requests/sec) at the daily peak.
PEAK_REQUEST_RATE = 1200.0
# Minimum fraction of peak at the daily trough (never zero traffic).
TROUGH_FRACTION = 0.28

SERVICE_PROFILES: dict[ServiceName, ServiceProfile] = {
    ServiceName.API_GATEWAY: ServiceProfile(
        ServiceName.API_GATEWAY, "gateway", base_cpu=32, base_memory=45,
        base_latency_ms=6, traffic_share=1.0, error_floor=0.002,
        cpu_latency_coupling=0.7, dep_latency_coupling=0.9,
    ),
    ServiceName.USER_SERVICE: ServiceProfile(
        ServiceName.USER_SERVICE, "service", base_cpu=28, base_memory=50,
        base_latency_ms=18, traffic_share=0.45, error_floor=0.003,
        dep_latency_coupling=0.85,
    ),
    ServiceName.ORDER_SERVICE: ServiceProfile(
        ServiceName.ORDER_SERVICE, "service", base_cpu=35, base_memory=55,
        base_latency_ms=26, traffic_share=0.35, error_floor=0.004,
        dep_latency_coupling=0.85,
    ),
    ServiceName.PAYMENT_SERVICE: ServiceProfile(
        ServiceName.PAYMENT_SERVICE, "service", base_cpu=30, base_memory=52,
        base_latency_ms=40, traffic_share=0.20, error_floor=0.005,
        dep_latency_coupling=0.85,
    ),
    ServiceName.DATABASE: ServiceProfile(
        ServiceName.DATABASE, "datastore", base_cpu=40, base_memory=60,
        base_latency_ms=9, traffic_share=0.9, error_floor=0.001,
        cpu_latency_coupling=1.1, dep_latency_coupling=0.0,
    ),
    ServiceName.CACHE: ServiceProfile(
        ServiceName.CACHE, "infra", base_cpu=15, base_memory=35,
        base_latency_ms=1.2, traffic_share=0.8, error_floor=0.0005,
        cpu_latency_coupling=0.3, dep_latency_coupling=0.0,
    ),
    ServiceName.MESSAGE_QUEUE: ServiceProfile(
        ServiceName.MESSAGE_QUEUE, "infra", base_cpu=20, base_memory=40,
        base_latency_ms=3, traffic_share=0.5, error_floor=0.001,
        cpu_latency_coupling=0.4, dep_latency_coupling=0.0,
    ),
}

# Physical clamps applied to every metric after generation.
METRIC_BOUNDS: dict[str, tuple[float, float]] = {
    "cpu_pct": (0.0, 100.0),
    "memory_pct": (0.0, 100.0),
    "request_rate": (0.0, 100_000.0),
    "latency_p50_ms": (0.1, 60_000.0),
    "latency_p95_ms": (0.1, 60_000.0),
    "latency_p99_ms": (0.1, 60_000.0),
    "error_rate": (0.0, 1.0),
    "http_5xx_rate": (0.0, 1.0),
    "http_4xx_rate": (0.0, 1.0),
    "db_latency_ms": (0.0, 60_000.0),  # 0 == not-applicable for non-DB services
    "db_connections": (0.0, 5_000.0),
    "db_cpu_pct": (0.0, 100.0),
    "cache_hit_rate": (0.0, 1.0),
    "queue_depth": (0.0, 1_000_000.0),
    "network_latency_ms": (0.0, 10_000.0),
    "packet_loss_pct": (0.0, 100.0),
    "gc_pause_ms": (0.0, 10_000.0),
    "saturation": (0.0, 1.0),
}

# Baseline business KPIs at nominal (healthy) conversion.
BUSINESS_BASELINE = {
    "conversion_rate": 0.062,        # 6.2% of active users convert
    "orders_per_min_at_peak": 340.0,
    "avg_order_value": 1850.0,       # currency units (INR-scale, per spec ₹ examples)
    "checkout_error_floor": 0.006,
}


@dataclass
class ServiceState:
    """Mutable per-minute state for one service, seeded from its profile."""

    profile: ServiceProfile
    memory_pct: float = 0.0            # persists across steps (leaks accumulate)
    crashed_until: int = -1            # minute index the service stays crashed to
    extra: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.memory_pct == 0.0:
            self.memory_pct = self.profile.base_memory
