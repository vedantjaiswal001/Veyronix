"""Shared pytest fixtures. Isolates tests onto a temp DB so runs never touch
the developer's data/ store."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Point the operational + feature stores at a temp dir BEFORE settings load.
_TMP = Path(tempfile.mkdtemp(prefix="veyronix-tests-"))
os.environ.setdefault("VEYRONIX_ENVIRONMENT", "ci")
os.environ["VEYRONIX_DATABASE_URL"] = f"sqlite:///{_TMP/'test.db'}"
os.environ["VEYRONIX_DUCKDB_PATH"] = str(_TMP / "test.duckdb")

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def db():
    """Session-scoped initialized database."""
    from veyronix.db.session import init_db

    init_db(drop=True)
    yield
