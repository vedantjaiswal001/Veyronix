"""Structured logging via structlog.

Renders human-friendly console logs locally and JSON in docker/ci/prod so logs
are machine-parseable in deployment. Import ``get_logger`` everywhere; call
``configure_logging`` once at process start (API startup, CLI entrypoint).
"""

from __future__ import annotations

import logging
import sys

import structlog

_CONFIGURED = False


def configure_logging(json_logs: bool | None = None, level: str = "INFO") -> None:
    """Configure structlog + stdlib logging. Safe to call more than once."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    if json_logs is None:
        # Late import to avoid a hard config dependency at module import time.
        from veyronix.core.config import get_settings

        json_logs = get_settings().environment in {"docker", "ci", "prod"}

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    renderer = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(level)
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=level)
    _CONFIGURED = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger, configuring logging on first use."""
    if not _CONFIGURED:
        configure_logging()
    return structlog.get_logger(name)
