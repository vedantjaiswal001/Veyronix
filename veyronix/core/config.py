"""Central configuration via pydantic-settings.

All configuration is environment-driven (12-factor). Secrets are never
hard-coded and never committed; see ``.env.example`` for the full surface.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root = two levels up from this file (veyronix/core/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
ARTIFACT_DIR = DATA_DIR / "artifacts"


class Settings(BaseSettings):
    """Runtime settings for every VEYRONIX component."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="VEYRONIX_",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- Environment ----
    environment: Literal["local", "docker", "ci", "prod"] = "local"
    debug: bool = True

    # ---- Determinism / reproducibility ----
    random_seed: int = 42

    # ---- Storage ----
    # Operational store (SQLAlchemy). SQLite by default so the system runs with
    # zero infrastructure; Docker Compose overrides this with PostgreSQL.
    database_url: str = f"sqlite:///{DATA_DIR / 'veyronix.db'}"
    # Analytical / feature store (columnar). DuckDB file by default.
    duckdb_path: str = str(DATA_DIR / "veyronix.duckdb")
    redis_url: str = "redis://localhost:6379/0"

    # ---- Simulator ----
    sim_seed: int = 7
    sim_step_seconds: int = 60           # one telemetry sample per simulated minute
    sim_default_minutes: int = 60 * 24   # 24h of telemetry by default

    # ---- ML ----
    prediction_horizon_minutes: int = 30   # predict incident within next N minutes
    lookback_minutes: int = 30             # feature window length
    train_frac: float = 0.6
    val_frac: float = 0.2
    # test_frac is the remainder (0.2), preserved chronologically.
    mlflow_tracking_uri: str = f"file://{ARTIFACT_DIR / 'mlruns'}"

    # ---- AI / LLM ----
    # Provider abstraction. "offline" uses the deterministic reasoner and needs
    # no API keys, so the full pipeline runs without secrets.
    llm_provider: Literal["offline", "anthropic", "openai"] = "offline"
    llm_model: str = "claude-3-5-sonnet-latest"
    anthropic_api_key: str | None = Field(default=None, repr=False)
    openai_api_key: str | None = Field(default=None, repr=False)

    # ---- Agent guardrails ----
    agent_max_steps: int = 12
    agent_timeout_seconds: int = 90
    agent_max_tool_calls: int = 20
    agent_token_budget: int = 20_000

    # ---- Remediation safety ----
    # Global kill-switch: when False, no action may auto-execute regardless of
    # its risk tier. Defaults to False so nothing acts without explicit opt-in.
    allow_auto_remediation: bool = False

    # ---- API ----
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    @field_validator("train_frac", "val_frac")
    @classmethod
    def _fraction(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError("fraction must be in (0, 1)")
        return v

    @property
    def test_frac(self) -> float:
        return round(1.0 - self.train_frac - self.val_frac, 6)

    def ensure_dirs(self) -> None:
        """Create data/artifact directories if missing (idempotent)."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached Settings instance."""
    settings = Settings()
    settings.ensure_dirs()
    return settings
