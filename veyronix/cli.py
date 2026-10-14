"""VEYRONIX command-line interface.

Thin Typer wrapper over the library so every capability is reproducible from
one command surface (used by the Makefile: install/simulate/train/evaluate/...).
Subcommands are wired up phase by phase; each imports lazily so the CLI loads
fast and does not require optional extras to be installed.
"""

from __future__ import annotations

import typer

from veyronix.core.logging import configure_logging, get_logger

app = typer.Typer(add_completion=False, help="VEYRONIX control plane")
log = get_logger("cli")


@app.command()
def version() -> None:
    """Print the VEYRONIX version."""
    from veyronix import __version__

    typer.echo(f"VEYRONIX {__version__}")


@app.command("init-db")
def init_db_cmd(drop: bool = typer.Option(False, help="Drop existing tables first")) -> None:
    """Create the operational database schema."""
    from veyronix.db.session import init_db

    init_db(drop=drop)
    typer.echo("database initialized")


@app.command()
def topology() -> None:
    """Print the service dependency topology and centrality."""
    import json

    from veyronix.core.topology import topology_summary

    typer.echo(json.dumps(topology_summary(), indent=2))


# ---- phase commands (registered as they are implemented) ----
try:  # simulator (Phase 1)
    from veyronix.simulator.cli import simulate as _simulate  # type: ignore

    app.command("simulate")(_simulate)
except Exception:  # pragma: no cover - not yet implemented
    pass


def main() -> None:
    configure_logging()
    app()


if __name__ == "__main__":
    main()
