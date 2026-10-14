"""Assemble the modelling dataset: features + labels + time-aware splits.

Each simulation run is an independent timeline; we split it chronologically
(train = earliest 60%, val = next 20%, test = latest 20%) with an embargo gap
so a train row's forward-looking label can never overlap a val/test row's
backward-looking features. The assembled dataset is stored to parquet and
registered in DuckDB (the analytical/feature store).
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

from veyronix.core.config import ARTIFACT_DIR, get_settings
from veyronix.core.logging import get_logger
from veyronix.features.labels import build_labels
from veyronix.features.windows import build_feature_matrix
from veyronix.simulator.models import SimulationResult
from veyronix.simulator.run import run_simulation

log = get_logger("features.build")
FEATURE_DIR = ARTIFACT_DIR / "features"
DATASET_PATH = FEATURE_DIR / "dataset.parquet"
INCIDENTS_PATH = FEATURE_DIR / "incidents.parquet"
META_PATH = FEATURE_DIR / "dataset_meta.json"

# Columns that are labels/targets or identifiers — never model inputs.
LABEL_COLUMNS = [
    "incident_within_h", "current_incident", "eta_minutes",
    "upcoming_fault_type", "upcoming_severity", "upcoming_incident_id",
    "active_fault_type", "active_severity", "active_incident_id",
]
ID_COLUMNS = ["run_id", "minute", "ts", "split"]


def model_feature_columns(df: pd.DataFrame) -> list[str]:
    """Feature columns for models: everything except identifiers and labels."""
    exclude = set(ID_COLUMNS) | set(LABEL_COLUMNS)
    return [c for c in df.columns if c not in exclude]


def incidents_to_df(result: SimulationResult) -> pd.DataFrame:
    return pd.DataFrame([{
        "run_id": result.run_id, "fault_id": i.fault_id,
        "fault_type": i.fault_type.value,
        "root_cause_service": i.root_cause_service.value,
        "start_min": i.start_min, "end_min": i.end_min,
        "severity": i.severity.value,
    } for i in result.incidents])


def _tag_splits(df: pd.DataFrame, embargo: int) -> pd.DataFrame:
    s = get_settings()
    out = []
    for _, g in df.groupby("run_id", sort=False):
        g = g.sort_values("minute").copy()
        tmax = g["minute"].max()
        t1 = int(s.train_frac * tmax)
        t2 = int((s.train_frac + s.val_frac) * tmax)
        split = pd.Series("drop", index=g.index)
        split[g["minute"] < t1] = "train"
        split[(g["minute"] >= t1 + embargo) & (g["minute"] < t2)] = "val"
        split[g["minute"] >= t2 + embargo] = "test"
        g["split"] = split.values
        out.append(g)
    return pd.concat(out).reset_index(drop=True)


def build_dataset(
    seeds: list[int] | None = None, minutes: int = 1440, horizon: int | None = None,
    persist: bool = True,
) -> pd.DataFrame:
    """Build the full features+labels dataset across several simulation runs."""
    s = get_settings()
    seeds = seeds or [42, 43, 44, 45, 46, 47, 48, 49]
    horizon = horizon or s.prediction_horizon_minutes
    embargo = max(horizon, s.lookback_minutes)

    parts: list[pd.DataFrame] = []
    inc_parts: list[pd.DataFrame] = []
    for seed in seeds:
        res = run_simulation(minutes=minutes, seed=seed, run_id=f"ds-{seed}")
        inc_df = incidents_to_df(res)
        feats = build_feature_matrix(res.metrics, res.business)
        labels = build_labels(feats, inc_df, horizon)
        merged = feats.merge(labels.drop(columns=["ts"]), on=["run_id", "minute"])
        parts.append(merged)
        inc_parts.append(inc_df)
        log.info("features.run", seed=seed, rows=len(merged),
                 positives=int(merged["incident_within_h"].sum()),
                 active=int(merged["current_incident"].sum()))

    df = pd.concat(parts).reset_index(drop=True)
    df = _tag_splits(df, embargo)
    df = df[df["split"] != "drop"].reset_index(drop=True)
    incidents = pd.concat(inc_parts).reset_index(drop=True)

    if persist:
        _persist(df, incidents, seeds, minutes, horizon, embargo)
    return df


def _persist(df, incidents, seeds, minutes, horizon, embargo) -> None:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATASET_PATH, index=False)
    incidents.to_parquet(INCIDENTS_PATH, index=False)

    feat_cols = model_feature_columns(df)
    meta = {
        "rows": len(df), "n_features": len(feat_cols),
        "seeds": seeds, "minutes_per_run": minutes, "horizon": horizon,
        "embargo": embargo, "feature_columns": feat_cols,
        "splits": {k: int(v) for k, v in df["split"].value_counts().items()},
        "positive_rate_overall": round(float(df["incident_within_h"].mean()), 4),
        "positive_rate_pred_pool": round(float(
            df[df.current_incident == 0]["incident_within_h"].mean()), 4),
        "active_incident_rows": int(df["current_incident"].sum()),
    }
    META_PATH.write_text(json.dumps(meta, indent=2))

    # register in DuckDB (analytical store). DuckDB disallows bound params in
    # CREATE VIEW, so we inline our own (trusted) path via its string literal.
    con = duckdb.connect(get_settings().duckdb_path)
    safe_path = str(DATASET_PATH).replace("'", "''")
    con.execute(f"CREATE OR REPLACE VIEW features AS SELECT * FROM read_parquet('{safe_path}')")
    con.close()
    log.info("features.persisted", **{k: meta[k] for k in
             ("rows", "n_features", "splits", "positive_rate_pred_pool")})


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    return pd.read_parquet(path or DATASET_PATH)


def load_incidents() -> pd.DataFrame:
    return pd.read_parquet(INCIDENTS_PATH)


def load_meta() -> dict:
    return json.loads(META_PATH.read_text())


def main(seed: int = 42) -> None:
    """`make features` entrypoint."""
    build_dataset()
    meta = load_meta()
    print(json.dumps({k: meta[k] for k in
                      ("rows", "n_features", "splits", "positive_rate_pred_pool",
                       "active_incident_rows")}, indent=2))


if __name__ == "__main__":
    import typer

    typer.run(main)
