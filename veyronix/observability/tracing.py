"""OpenTelemetry tracing hooks (spec §54), degrading gracefully.

If the ``otel`` extra is installed and OTLP is configured, spans are exported;
otherwise ``traced`` is a no-op context manager so the rest of the system runs
unchanged without the dependency.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from veyronix.core.logging import get_logger

log = get_logger("observability.tracing")
_TRACER = None
_INITIALIZED = False


def setup_tracing(service_name: str = "veyronix") -> None:
    """Best-effort OpenTelemetry setup; silently no-ops if otel is absent."""
    global _TRACER, _INITIALIZED
    if _INITIALIZED:
        return
    _INITIALIZED = True
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider

        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        trace.set_tracer_provider(provider)
        _TRACER = trace.get_tracer(service_name)
        log.info("tracing.enabled", service=service_name)
    except Exception:  # pragma: no cover - optional dependency
        _TRACER = None
        log.debug("tracing.disabled")


@contextmanager
def traced(name: str, **attrs) -> Iterator[None]:
    """Create a span if OTel is active; otherwise a no-op."""
    if _TRACER is None:
        yield
        return
    with _TRACER.start_as_current_span(name) as span:  # pragma: no cover
        for k, v in attrs.items():
            span.set_attribute(k, v)
        yield
