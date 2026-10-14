"""Build a ToolContext from persisted runs, for the MCP server and API.

Guarantees data exists: if no run has been persisted yet, it generates and saves
a deterministic one so the agent/MCP surface is always usable.
"""

from __future__ import annotations

from veyronix.agents.memory import IncidentMemory
from veyronix.agents.tools import ToolContext
from veyronix.simulator.persist import (
    RUN_ROOT,
    load_latest_run_id,
    load_run_frames,
    save_parquet,
)
from veyronix.simulator.run import run_simulation


def context_for_run(run_id: str, with_memory: bool = True) -> ToolContext:
    frames = load_run_frames(run_id)
    mem = IncidentMemory.load() if with_memory else None
    return ToolContext(
        run_id=run_id, metrics=frames["metrics"],
        logs=frames.get("logs"), deployments=frames.get("deployments"),
        business=frames["business"], memory=mem)


def default_context(seed: int = 42, minutes: int = 1440) -> ToolContext:
    """Latest persisted run, or generate + persist one deterministically."""
    run_id = load_latest_run_id()
    if run_id and (RUN_ROOT / run_id / "metrics.parquet").exists():
        return context_for_run(run_id)
    res = run_simulation(minutes=minutes, seed=seed, run_id=f"ctx-{seed}")
    save_parquet(res)
    return ToolContext(run_id=res.run_id, metrics=res.metrics, logs=res.logs,
                       traces=res.traces, deployments=res.deployments,
                       business=res.business, memory=IncidentMemory())
