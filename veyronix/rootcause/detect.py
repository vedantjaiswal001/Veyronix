"""Per-service anomaly profiling for root-cause analysis.

For an incident window we compute, per service: the magnitude of deviation from
its own pre-incident baseline, the onset minute (when it first deviated), and
the dominant deviating channel. Temporal ordering of onsets underpins causal
reasoning — the service that degraded *first* is a stronger cause than one that
degraded later as a downstream symptom.

Materiality gates (per-channel absolute + relative floors) are essential: several
channels have tiny baseline variance (network ~4 ms, packet loss 0), so a raw
z-score would explode on trivial noise. A channel only counts as anomalous when
the deviation is both statistically large *and* materially large.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from veyronix.core.constants import FaultType, ServiceName

# channel -> most-implied fault type
CHANNEL_FAULT = {
    "cpu_pct": FaultType.CPU_SPIKE,
    "memory_pct": FaultType.MEMORY_LEAK,
    "gc_pause_ms": FaultType.MEMORY_LEAK,
    "db_latency_ms": FaultType.DATABASE_OVERLOAD,
    "db_cpu_pct": FaultType.DATABASE_OVERLOAD,
    "db_connections": FaultType.DATABASE_OVERLOAD,
    "queue_depth": FaultType.QUEUE_BACKLOG,
    "network_latency_ms": FaultType.NETWORK_LATENCY,
    "packet_loss_pct": FaultType.PACKET_LOSS,
    "request_rate": FaultType.TRAFFIC_SPIKE,
    "cache_hit_rate": FaultType.DEPENDENCY_FAILURE,
    "error_rate": FaultType.SERVICE_CRASH,
    "latency_p95_ms": FaultType.CASCADING_FAILURE,
}

# per-channel (abs_floor, rel_floor, direction). direction -1 => a drop is bad.
CHANNEL_CFG = {
    "cpu_pct": (15.0, 1.4, +1),
    "memory_pct": (12.0, 1.25, +1),
    "gc_pause_ms": (25.0, 2.0, +1),
    "db_latency_ms": (25.0, 2.0, +1),
    "db_cpu_pct": (15.0, 1.4, +1),
    "db_connections": (50.0, 1.5, +1),
    "queue_depth": (3000.0, 3.0, +1),
    "network_latency_ms": (50.0, 2.5, +1),
    "packet_loss_pct": (2.0, 0.0, +1),
    "request_rate": (150.0, 1.6, +1),
    "cache_hit_rate": (0.15, 0.0, -1),
    "error_rate": (0.05, 2.0, +1),
    "latency_p95_ms": (80.0, 1.8, +1),
}

# Intrinsic resource/state channels (a deviation here signals a *cause*), vs
# symptom channels (latency/error) which are frequently inherited downstream.
INTRINSIC_CHANNELS = frozenset({
    "cpu_pct", "memory_pct", "db_latency_ms", "db_cpu_pct", "db_connections",
    "queue_depth", "network_latency_ms", "packet_loss_pct", "gc_pause_ms",
    "request_rate", "cache_hit_rate",
})


@dataclass
class ServiceAnomaly:
    service: str
    magnitude: float
    onset_min: int | None
    dominant_channel: str
    implied_fault: str
    peak_values: dict


def profile_services(
    metrics: pd.DataFrame, start_min: int, end_min: int,
    baseline_lead: int = 45, z_thresh: float = 3.5,
) -> dict[str, ServiceAnomaly]:
    """Return an anomaly profile per service over the incident window."""
    out: dict[str, ServiceAnomaly] = {}
    base_lo, base_hi = start_min - baseline_lead, start_min - 12
    for svc in (s.value for s in ServiceName if s != ServiceName.CLIENT):
        g = metrics[metrics.service == svc].sort_values("minute")
        base = g[(g.minute >= base_lo) & (g.minute < base_hi)]
        win = g[(g.minute >= start_min - 12) & (g.minute <= end_min)]
        if win.empty or base.empty:
            continue

        best = None  # (magnitude, channel, onset)
        for ch, (abs_floor, rel_floor, direction) in CHANNEL_CFG.items():
            if ch not in g.columns:
                continue
            b = base[ch].to_numpy(float)
            w = win[ch].to_numpy(float)
            b_mean = float(np.mean(b))
            # scale floor is tied to the channel's materiality so small-range
            # channels (e.g. cache hit-rate 0..1) are not swamped by a fixed floor
            scale = max(float(np.std(b)), abs_floor / 8.0)
            if direction > 0:
                extreme = float(np.nanmax(w))
                dev = extreme - b_mean
                rel = extreme / (abs(b_mean) + 1e-6)
            else:  # a drop (e.g. cache hit rate)
                extreme = float(np.nanmin(w))
                dev = b_mean - extreme
                rel = 999.0  # rel gate not used for drops
            if dev < abs_floor:
                continue
            if rel_floor > 0 and rel < rel_floor:
                continue
            z = dev / scale
            if z < z_thresh:
                continue
            # onset: first minute the per-sample deviation crosses the floor
            if direction > 0:
                crossed = win.minute.to_numpy()[w - b_mean >= abs_floor]
            else:
                crossed = win.minute.to_numpy()[b_mean - w >= abs_floor]
            onset = int(crossed.min()) if len(crossed) else None
            if best is None or z > best[0]:
                best = (z, ch, onset, extreme)

        if best is not None:
            z, ch, onset, extreme = best
            out[svc] = ServiceAnomaly(
                service=svc, magnitude=round(z, 2), onset_min=onset,
                dominant_channel=ch, implied_fault=CHANNEL_FAULT[ch].value,
                peak_values={ch: round(extreme, 3)})
    return out


def count_material_services(
    metrics: pd.DataFrame, start_min: int, end_min: int, channel: str,
    baseline_lead: int = 45,
) -> int:
    """How many services show a *material* rise in ``channel`` during the window.

    Used to detect fleet-wide patterns (e.g. a traffic spike raises request rate
    across many services at once)."""
    abs_floor, rel_floor, _ = CHANNEL_CFG[channel]
    base_lo, base_hi = start_min - baseline_lead, start_min - 12
    n = 0
    for svc in (s.value for s in ServiceName if s != ServiceName.CLIENT):
        g = metrics[metrics.service == svc]
        if channel not in g.columns:
            continue
        base = g[(g.minute >= base_lo) & (g.minute < base_hi)][channel].to_numpy(float)
        win = g[(g.minute >= start_min) & (g.minute <= end_min)][channel].to_numpy(float)
        if len(base) == 0 or len(win) == 0:
            continue
        b_mean = float(np.mean(base))
        peak = float(np.nanmax(win))
        if peak - b_mean >= abs_floor and (rel_floor <= 0 or peak / (abs(b_mean) + 1e-6) >= rel_floor):
            n += 1
    return n
