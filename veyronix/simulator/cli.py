"""`veyronix simulate` — generate a labelled run, persist to DB + parquet."""

from __future__ import annotations

import json

import typer

from veyronix.core.logging import get_logger
from veyronix.db.session import init_db
from veyronix.simulator.persist import persist_result, save_parquet
from veyronix.simulator.run import run_simulation

log = get_logger("cli.simulate")


def simulate(
    minutes: int = typer.Option(1440, help="Minutes of telemetry to generate"),
    seed: int = typer.Option(42, help="Random seed (reproducible)"),
    scenario: str = typer.Option("random", help="random | bad_deploy | unknown_cascade"),
    run_id: str = typer.Option(None, help="Explicit run id"),
    to_db: bool = typer.Option(True, help="Persist to the operational database"),
    parquet: bool = typer.Option(True, help="Persist parquet for the ML pipeline"),
) -> None:
    """Generate synthetic telemetry + labelled incidents."""
    result = run_simulation(minutes=minutes, seed=seed, scenario=scenario, run_id=run_id)
    if to_db:
        init_db()  # ensure schema exists
        persist_result(result)
    if parquet:
        save_parquet(result)
    typer.echo(json.dumps(result.summary(), indent=2))
