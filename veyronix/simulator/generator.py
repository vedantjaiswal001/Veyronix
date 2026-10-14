"""The generative simulation engine.

Advances a per-minute clock. Each service's metrics evolve around healthy
baselines with daily seasonality + smooth AR(1) noise; the fault engine perturbs
the *root* drivers and a dependency-coupling + error-propagation model produces
the downstream cascade. Business KPIs are derived from order/payment health.
Deterministic given a seed.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import networkx as nx
import numpy as np
import pandas as pd

from veyronix.core.constants import MetricName, ServiceName
from veyronix.core.topology import service_graph
from veyronix.simulator.faults import (
    FAULT_CATALOG,
    PRECURSOR_FLOOR,
    apply_fault,
    blank_drivers,
)
from veyronix.simulator.models import FaultInstance, SimulationResult
from veyronix.simulator.profiles import (
    BUSINESS_BASELINE,
    METRIC_BOUNDS,
    PEAK_REQUEST_RATE,
    SERVICE_PROFILES,
    TROUGH_FRACTION,
    ServiceState,
)

METRIC_CHANNELS: list[str] = [m.value for m in MetricName]
BASE_USERS_PEAK = 5500.0
BASE_NETWORK_MS = 4.0


def _clamp(name: str, value: float) -> float:
    lo, hi = METRIC_BOUNDS[name]
    return float(min(max(value, lo), hi))


def _seasonal(minute_of_day: int) -> float:
    """Diurnal traffic multiplier in [TROUGH_FRACTION, 1.0], peak ~15:00."""
    phase = 2 * np.pi * ((minute_of_day - 15 * 60) / (24 * 60))
    wave = (np.cos(phase) + 1) / 2
    return TROUGH_FRACTION + (1.0 - TROUGH_FRACTION) * wave


def _intensity(shape: str, t_rel: int, lead: int, duration: int) -> tuple[float, bool]:
    """Fault intensity in [0,1] and whether we are in the crash tail.

    Precursor phase [0, lead) ramps 0 -> PRECURSOR_FLOOR (subtle early signal);
    main phase [lead, lead+duration) rises to 1 per shape then falls.
    """
    total = lead + duration
    if t_rel < 0 or t_rel >= total:
        return 0.0, False
    if t_rel < lead:  # precursor
        return PRECURSOR_FLOOR * (t_rel + 1) / max(lead, 1), False
    mt = t_rel - lead  # minutes into main phase
    prog = mt / max(duration, 1)
    if shape == "ramp":
        s = prog
    elif shape == "grow":
        s = prog ** 2
    elif shape == "step":
        s = 1.0
    else:  # plateau
        rise = min(1.0, (mt + 1) / 2)
        fall = min(1.0, (duration - mt) / 3)
        s = max(0.0, min(rise, fall))
    intensity = PRECURSOR_FLOOR + (1.0 - PRECURSOR_FLOOR) * s
    in_tail = mt >= duration - 2
    return float(min(1.0, intensity)), in_tail


class SimulationEngine:
    def __init__(self, run_id: str, start_time: datetime, minutes: int, seed: int):
        self.run_id = run_id
        self.start_time = start_time
        self.minutes = minutes
        self.rng = np.random.default_rng(seed)
        self.graph = service_graph()
        # dependency-first order (a dependency is computed before its dependents)
        order = list(nx.topological_sort(self.graph.reverse(copy=True)))
        self.order: list[str] = [s for s in order if s != ServiceName.CLIENT.value]
        self.direct_deps: dict[str, list[str]] = {
            s: list(self.graph.successors(s)) for s in self.order
        }
        self.state: dict[str, ServiceState] = {
            s: ServiceState(profile=SERVICE_PROFILES[ServiceName(s)]) for s in self.order
        }
        self.noise: dict[str, float] = dict.fromkeys(self.order, 0.0)
        self._global_noise = 0.0

    # ---- per-service metric computation -----------------------------------
    def _service_metrics(
        self, svc: str, dr: dict, traffic_mult: float,
        eff_latency: dict[str, float], err: dict[str, float],
    ) -> dict[str, float]:
        p = SERVICE_PROFILES[ServiceName(svc)]
        st = self.state[svc]
        # smooth AR(1) noise per service
        self.noise[svc] = 0.8 * self.noise[svc] + 0.2 * float(self.rng.normal())
        n = self.noise[svc]

        load_ratio = traffic_mult * dr["req_mult"]
        req = PEAK_REQUEST_RATE * p.traffic_share * load_ratio * (1 + 0.03 * n)

        cpu = p.base_cpu * (0.55 + 0.55 * load_ratio) + dr["cpu_add"] + 2.5 * n
        cpu = _clamp("cpu_pct", cpu)

        # memory = baseline + load component + accumulated leak. Leaked memory
        # does NOT mean-revert (that is what makes a leak a leak); it is only
        # cleared by a restart (crash). Non-leaking services stay near baseline.
        leaked = st.extra.get("leaked", 0.0) + dr["mem_drift"]
        leaked *= 0.995  # negligible natural reclamation
        load_mem = max(0.0, load_ratio - 0.5) * 10.0
        mem = _clamp("memory_pct", p.base_memory + load_mem + leaked + 1.2 * n)
        st.extra["leaked"] = leaked
        st.memory_pct = mem

        gc = 4.0 + dr["gc_add"] + max(0.0, mem - 80) * 2.0 + max(0.0, n)
        cpu_sat = min(max((cpu - 70) / 30, 0.0), 1.5)

        network = BASE_NETWORK_MS + dr["network_add"] + max(0.0, n)
        own_latency = (p.base_latency_ms * dr["own_latency_mult"] + dr["own_latency_add"])
        own_latency *= 1 + p.cpu_latency_coupling * cpu_sat
        own_latency += gc * 0.3 + network

        # datastore-specific channels
        db_latency = db_conn = db_cpu = 0.0
        if svc == ServiceName.DATABASE.value:
            db_cpu = _clamp("db_cpu_pct", p.base_cpu * (0.5 + 0.6 * load_ratio)
                            + dr["db_cpu_add"] + 2.0 * n)
            db_conn = _clamp("db_connections", 60 * (0.5 + load_ratio)
                             + dr["db_conn_add"] + 3 * n)
            db_cpu_sat = min(max((db_cpu - 70) / 30, 0.0), 1.5)
            db_latency = _clamp("db_latency_ms", p.base_latency_ms
                                * (1 + p.cpu_latency_coupling * db_cpu_sat)
                                + dr["db_latency_add"])
            own_latency = db_latency  # DB's effective latency IS its query latency

        # queue-specific
        queue_depth = 0.0
        if svc == ServiceName.MESSAGE_QUEUE.value:
            queue_depth = _clamp("queue_depth", 150 * load_ratio + dr["queue_add"]
                                 + 10 * max(0.0, n))
            own_latency += queue_depth * 0.0004

        # cache-specific
        cache_hit = 1.0
        if svc == ServiceName.CACHE.value:
            cache_hit = _clamp("cache_hit_rate", 0.94 * dr["cache_hit_mult"] + 0.01 * n)

        # dependency latency coupling
        deps = self.direct_deps[svc]
        dep_lat = max((eff_latency[d] for d in deps), default=0.0)
        eff = own_latency + p.dep_latency_coupling * dep_lat
        eff_latency[svc] = eff

        # error rate + propagation
        err_own = p.error_floor + dr["error_add"]
        err_own += cpu_sat * 0.04 + max(0.0, mem - 90) / 10 * 0.15
        err_own += dr["packet_loss_add"] / 100 * 0.5
        dep_err = max((err[d] for d in deps), default=0.0)
        e = err_own + p.dep_latency_coupling * 0.6 * dep_err
        # timeout-induced errors: very high effective latency -> requests time out.
        # This is what turns a latency cascade (e.g. slow DB) into checkout failures.
        e += min(0.6, max(0.0, (eff - 150.0) / 1000.0))
        if dr["crash"] >= 1.0:
            e = max(e, 0.98)
        e = _clamp("error_rate", e)
        err[svc] = e

        p50 = eff
        p95 = _clamp("latency_p95_ms", p50 * (1.7 + 1.5 * cpu_sat + 3.0 * e))
        p99 = _clamp("latency_p99_ms", p50 * (2.4 + 2.5 * cpu_sat + 5.0 * e))

        # restart clears leaked memory after a crash
        if dr["crash"] >= 1.0:
            st.extra["leaked"] = 0.0

        return {
            "cpu_pct": cpu,
            "memory_pct": mem,
            "request_rate": _clamp("request_rate", req),
            "latency_p50_ms": _clamp("latency_p50_ms", p50),
            "latency_p95_ms": p95,
            "latency_p99_ms": p99,
            "error_rate": e,
            "http_5xx_rate": _clamp("http_5xx_rate", e * 0.85),
            "http_4xx_rate": _clamp("http_4xx_rate", 0.008 + 0.004 * abs(n)),
            "db_latency_ms": db_latency,
            "db_connections": db_conn,
            "db_cpu_pct": db_cpu,
            "cache_hit_rate": cache_hit,
            "queue_depth": queue_depth,
            "network_latency_ms": _clamp("network_latency_ms", network),
            "packet_loss_pct": _clamp("packet_loss_pct", dr["packet_loss_add"]),
            "gc_pause_ms": _clamp("gc_pause_ms", gc),
            "saturation": _clamp("saturation", max(cpu, mem) / 100.0),
        }

    # ---- full run ---------------------------------------------------------
    def run(self, faults: list[FaultInstance]) -> SimulationResult:
        start_minute_of_day = self.start_time.hour * 60 + self.start_time.minute
        metric_rows: list[dict] = []
        business_rows: list[dict] = []
        log_rows: list[dict] = []
        trace_rows: list[dict] = []

        for t in range(self.minutes):
            ts = self.start_time + timedelta(minutes=t)
            mod = (start_minute_of_day + t) % 1440
            self._global_noise = 0.85 * self._global_noise + 0.15 * float(self.rng.normal())
            traffic_mult = float(_seasonal(mod)) * (1 + 0.04 * self._global_noise)
            traffic_mult = max(0.05, traffic_mult)

            drivers = {s: blank_drivers() for s in self.order}
            for f in faults:
                t_rel = t - f.onset_min
                if t_rel < 0 or t_rel >= f.total_active:
                    continue
                spec = FAULT_CATALOG[f.fault_type]
                inten, in_tail = _intensity(spec.shape, t_rel, f.lead_min, f.duration_min)
                apply_fault(spec, inten, in_tail, drivers, f.target, float(self.rng.random()))

            eff_latency: dict[str, float] = {}
            err: dict[str, float] = {}
            minute_metrics: dict[str, dict] = {}
            for svc in self.order:
                mvals = self._service_metrics(svc, drivers[svc], traffic_mult, eff_latency, err)
                minute_metrics[svc] = mvals
                row = {"run_id": self.run_id, "ts": ts, "minute": t, "service": svc}
                row.update(mvals)
                metric_rows.append(row)

            business_rows.append(self._business(t, ts, traffic_mult, minute_metrics, drivers))
            log_rows.extend(self._logs(t, ts, minute_metrics, drivers))
            if t % 1 == 0:  # sample traces every minute
                trace_rows.extend(self._traces(t, ts, minute_metrics, err))

        metrics = pd.DataFrame(metric_rows)
        business = pd.DataFrame(business_rows)
        logs = pd.DataFrame(log_rows) if log_rows else pd.DataFrame(
            columns=["run_id", "ts", "minute", "service", "level", "template", "message", "count"])
        traces = pd.DataFrame(trace_rows)
        return SimulationResult(
            run_id=self.run_id, start_time=self.start_time, minutes=self.minutes,
            metrics=metrics, business=business, logs=logs, traces=traces,
            deployments=pd.DataFrame(), incidents=[],
            config={"channels": METRIC_CHANNELS},
        )

    # ---- business layer ---------------------------------------------------
    def _business(self, t, ts, traffic_mult, mm, drivers) -> dict:
        order_m = mm[ServiceName.ORDER_SERVICE.value]
        pay_m = mm[ServiceName.PAYMENT_SERVICE.value]
        model_drift = drivers[ServiceName.ORDER_SERVICE.value].get("model_drift", 0.0)

        active_users = BASE_USERS_PEAK * traffic_mult
        checkout_err = (
            BUSINESS_BASELINE["checkout_error_floor"]
            + 0.55 * order_m["error_rate"] + 0.45 * pay_m["error_rate"]
        )
        # latency penalty when payment/order p95 far above nominal
        lat_pen = min(0.4, max(0.0, (pay_m["latency_p95_ms"] - 150) / 2000)
                      + max(0.0, (order_m["latency_p95_ms"] - 150) / 2000))
        checkout_err = float(min(1.0, checkout_err + lat_pen * 0.2))

        conv_penalty = min(0.92, 3.0 * checkout_err + 0.35 * model_drift + lat_pen)
        conversion = BUSINESS_BASELINE["conversion_rate"] * (1 - conv_penalty)
        orders = max(0.0, active_users * conversion)
        revenue = orders * BUSINESS_BASELINE["avg_order_value"] * (1 - checkout_err)
        return {
            "run_id": self.run_id, "ts": ts, "minute": t,
            "active_users": round(active_users, 1),
            "conversion_rate": round(conversion, 5),
            "orders_per_min": round(orders, 2),
            "checkout_error_rate": round(checkout_err, 5),
            "revenue_per_min": round(revenue, 2),
        }

    # ---- logs -------------------------------------------------------------
    def _logs(self, t, ts, mm, drivers) -> list[dict]:
        rows: list[dict] = []
        for svc in self.order:
            m = mm[svc]
            dr = drivers[svc]
            p = SERVICE_PROFILES[ServiceName(svc)]
            rpm = m["request_rate"] * 60
            if t % 5 == 0:  # INFO heartbeat
                rows.append(self._log(ts, t, svc, "INFO", "request_processed",
                                      "processed requests", int(rpm)))
            if m["error_rate"] > 2.5 * p.error_floor:
                errc = int(m["error_rate"] * rpm)
                if dr["crash"] >= 1.0:
                    rows.append(self._log(ts, t, svc, "ERROR", "service_unavailable",
                                          "service unavailable / not responding", max(1, errc)))
                if dr["db_latency_add"] > 0 or svc == ServiceName.DATABASE.value:
                    rows.append(self._log(ts, t, svc, "WARN", "db_slow_query",
                                          "slow query detected", max(1, int(rpm * 0.1))))
                if m["memory_pct"] > 90:
                    rows.append(self._log(ts, t, svc, "ERROR", "gc_overhead",
                                          "GC overhead limit / OOM risk", max(1, int(rpm * 0.05))))
                if dr["packet_loss_add"] > 0:
                    rows.append(self._log(ts, t, svc, "WARN", "connection_timeout",
                                          "upstream connection timeout", max(1, int(rpm * 0.08))))
                rows.append(self._log(ts, t, svc, "ERROR", "http_5xx",
                                      "returned 5xx response", max(1, int(m["http_5xx_rate"] * rpm))))
        return rows

    def _log(self, ts, t, svc, level, template, message, count) -> dict:
        return {"run_id": self.run_id, "ts": ts, "minute": t, "service": svc,
                "level": level, "template": template, "message": message, "count": count}

    # ---- traces -----------------------------------------------------------
    def _traces(self, t, ts, mm, err) -> list[dict]:
        rows: list[dict] = []
        paths = [
            [ServiceName.API_GATEWAY, ServiceName.ORDER_SERVICE, ServiceName.DATABASE],
            [ServiceName.API_GATEWAY, ServiceName.PAYMENT_SERVICE, ServiceName.MESSAGE_QUEUE],
        ]
        for i, path in enumerate(paths):
            spans = []
            total = 0.0
            has_err = False
            for svc in path:
                base = mm[svc.value]["latency_p50_ms"]
                ms = float(base * (0.8 + 0.5 * self.rng.random()))
                span_err = bool(self.rng.random() < err[svc.value])
                has_err = has_err or span_err
                spans.append({"service": svc.value, "op": f"{svc.value}.handle",
                              "ms": round(ms, 2), "error": span_err})
                total += ms
            rows.append({
                "run_id": self.run_id, "ts": ts, "minute": t,
                "trace_id": f"{self.run_id}-{t}-{i}", "root_service": path[0].value,
                "duration_ms": round(total, 2), "error": has_err, "spans": spans,
            })
        return rows
