"""Prometheus metrics for VEYRONIX self-observability (spec §54).

VEYRONIX monitors itself: API/model/agent latency, tool calls, LLM tokens,
errors, DB latency, simulator health, live incident probability and model drift.
A dedicated registry keeps these isolated and testable; the API exposes them at
``/metrics`` for Prometheus to scrape.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

REGISTRY = CollectorRegistry()

API_LATENCY = Histogram("veyronix_api_request_seconds", "API request latency",
                        ["endpoint", "method"], registry=REGISTRY)
MODEL_LATENCY = Histogram("veyronix_model_inference_seconds", "Model inference latency",
                          ["model"], registry=REGISTRY)
AGENT_LATENCY = Histogram("veyronix_agent_investigation_seconds", "Agent investigation latency",
                          ["provider"], registry=REGISTRY)
DB_LATENCY = Histogram("veyronix_db_query_seconds", "DB query latency", ["op"],
                       registry=REGISTRY)

TOOL_CALLS = Counter("veyronix_tool_calls_total", "Agent tool calls", ["tool", "ok"],
                     registry=REGISTRY)
LLM_TOKENS = Counter("veyronix_llm_tokens_total", "LLM tokens used", ["provider"],
                     registry=REGISTRY)
ERRORS = Counter("veyronix_errors_total", "Errors by component", ["component"],
                 registry=REGISTRY)
INCIDENTS = Counter("veyronix_simulator_incidents_total", "Injected incidents", ["fault_type"],
                    registry=REGISTRY)

INCIDENT_PROB = Gauge("veyronix_incident_probability", "Latest incident probability",
                      registry=REGISTRY)
MODEL_DRIFT = Gauge("veyronix_model_feature_drift", "Feature drift score (PSI)",
                    ["feature"], registry=REGISTRY)
MODEL_HEALTH = Gauge("veyronix_model_health", "Model health (1 healthy,0.5 warning,0 degraded)",
                     registry=REGISTRY)
SIM_HEALTH = Gauge("veyronix_simulator_health", "Simulator health (1 ok)", registry=REGISTRY)


@contextmanager
def observe(hist: Histogram, **labels) -> Iterator[None]:
    """Time a block into a histogram."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        hist.labels(**labels).observe(time.perf_counter() - t0)


def record_tool_call(tool: str, ok: bool) -> None:
    TOOL_CALLS.labels(tool=tool, ok=str(ok).lower()).inc()


def record_tokens(provider: str, n: int) -> None:
    if n:
        LLM_TOKENS.labels(provider=provider).inc(n)


def record_error(component: str) -> None:
    ERRORS.labels(component=component).inc()


def set_incident_probability(p: float) -> None:
    INCIDENT_PROB.set(p)


def export_metrics() -> tuple[bytes, str]:
    """Return (payload, content_type) for the /metrics endpoint."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
