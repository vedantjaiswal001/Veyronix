"""Train + evaluate the whole deterministic ML layer and record results.

Runs: incident prediction (baselines + models), anomaly detection (IsolationForest
+ statistical), severity classification, and short-horizon forecasting. Writes a
single measured results summary to data/artifacts/ml_results/summary.json. Every
number here is produced by these runs — none is hard-coded.
"""

from __future__ import annotations

import json

import joblib

from veyronix.core.config import ARTIFACT_DIR
from veyronix.core.logging import get_logger
from veyronix.features.build import DATASET_PATH, build_dataset, load_dataset, load_meta
from veyronix.ml.anomaly.detectors import (
    IsolationForestDetector,
    StatisticalDetector,
    evaluate_detector_scores,
)
from veyronix.ml.forecasting.forecasters import forecast_eval
from veyronix.ml.incident_prediction.train import train_all
from veyronix.ml.registry import register_model_version
from veyronix.ml.severity.model import train_severity

log = get_logger("ml.training")
SUMMARY_PATH = ARTIFACT_DIR / "ml_results" / "summary.json"
MODEL_DIR = ARTIFACT_DIR / "models"


def _anomaly_eval(df, feat_cols, seed) -> dict:
    normal = df[(df.split == "train") & (df.current_incident == 0)
               & (df.incident_within_h == 0)]
    test = df[df.split == "test"]
    y = test.current_incident.to_numpy(int)

    iso = IsolationForestDetector(seed=seed).fit(normal[feat_cols])
    iso_scores = iso.score(test[feat_cols])
    iso_eval = evaluate_detector_scores(iso_scores, y)

    stat = StatisticalDetector()
    stat_scores = stat.score_frame(test)
    stat_eval = evaluate_detector_scores(stat_scores, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": iso, "features": feat_cols},
                MODEL_DIR / "anomaly_isolation_forest.joblib")
    register_model_version(
        "anomaly.isolation_forest", "anomaly", "isolation_forest",
        {"contamination": iso.contamination}, iso_eval, feat_cols,
        artifact_path=str(MODEL_DIR / "anomaly_isolation_forest.joblib"),
        make_active=True)
    return {"isolation_forest": iso_eval, "statistical": stat_eval}


def run_all(seed: int = 42, rebuild: bool = True) -> dict:
    if rebuild or not DATASET_PATH.exists():
        build_dataset(minutes=1440, persist=True)
    df = load_dataset()
    meta = load_meta()
    feat_cols = meta["feature_columns"]

    log.info("ml.training.start", rows=len(df), features=len(feat_cols))
    prediction = train_all(seed=seed)
    anomaly = _anomaly_eval(df, feat_cols, seed)
    severity = train_severity(df, feat_cols, seed)
    if severity.get("status") == "ok" and "model" in severity:
        joblib.dump({"model": severity.pop("model"), "features": feat_cols},
                    MODEL_DIR / "severity.joblib")
        register_model_version(
            "severity.random_forest", "severity", "random_forest", {},
            {k: v for k, v in severity.items() if isinstance(v, (int, float))},
            feat_cols, artifact_path=str(MODEL_DIR / "severity.joblib"),
            make_active=True)
    forecasting = forecast_eval(df, horizon=10, seed=seed)

    summary = {
        "dataset": {k: meta[k] for k in ("rows", "n_features", "horizon",
                                         "splits", "positive_rate_pred_pool")},
        "incident_prediction": {
            name: {k: m[k] for k in ("pr_auc", "roc_auc", "precision", "recall",
                                     "f1", "fp_rate", "brier", "recall_at_fpr10")
                   if k in m} | {"warning_median":
                   m["early_warning"]["warning_minutes"]["median"],
                   "detection_rate": m["early_warning"]["detection_rate"]}
            for name, m in prediction["models"].items()},
        "active_prediction_model": prediction["active_model"],
        "anomaly_detection": anomaly,
        "severity": severity,
        "forecasting": forecasting,
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=float))
    log.info("ml.training.done", path=str(SUMMARY_PATH))
    return summary


def main(seed: int = 42) -> None:
    summary = run_all(seed=seed)
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    import typer

    typer.run(main)
