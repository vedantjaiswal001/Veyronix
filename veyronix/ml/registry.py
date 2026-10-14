"""Model registry: persist trained model versions + metrics (spec §10/§36).

Writes to the ``models`` / ``model_versions`` tables and optionally mirrors to
MLflow when it is installed and configured. Keeping our own registry means the
API and monitoring can read the active model + its metrics without MLflow being
a hard dependency.
"""

from __future__ import annotations

from datetime import UTC, datetime

from veyronix.core.logging import get_logger
from veyronix.db.models import Model, ModelVersion
from veyronix.db.session import init_db, session_scope

log = get_logger("ml.registry")


def register_model_version(
    name: str, task: str, algorithm: str, params: dict, metrics: dict,
    feature_names: list[str], artifact_path: str | None = None,
    make_active: bool = False,
) -> int:
    """Create (or extend) a model and add a new version. Returns version id."""
    init_db()
    version = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    with session_scope() as s:
        model = s.query(Model).filter_by(name=name).one_or_none()
        if model is None:
            model = Model(name=name, task=task, description=f"{task} — {algorithm}")
            s.add(model)
            s.flush()
        if make_active:
            for mv in s.query(ModelVersion).filter_by(model_id=model.id, is_active=True):
                mv.is_active = False
        mv = ModelVersion(
            model_id=model.id, version=version, algorithm=algorithm,
            params=params, metrics=metrics, feature_names=feature_names,
            artifact_path=artifact_path, is_active=make_active,
        )
        s.add(mv)
        s.flush()
        vid = mv.id
    log.info("registry.version", model=name, algorithm=algorithm, version=version,
             active=make_active)
    return vid


def try_mlflow_log(experiment: str, run_name: str, params: dict, metrics: dict) -> None:
    """Best-effort MLflow logging; silently skips if MLflow is unavailable."""
    try:
        import mlflow

        from veyronix.core.config import get_settings

        mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({k: v for k, v in params.items()
                               if isinstance(v, (int, float, str, bool))})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()
                                if isinstance(v, (int, float))})
    except Exception as exc:  # pragma: no cover - optional dependency
        log.debug("mlflow.skip", reason=str(exc))
