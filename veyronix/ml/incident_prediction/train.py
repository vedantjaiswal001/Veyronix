"""Train + evaluate incident-prediction models on the time-aware dataset.

Trains every model on the *prediction pool* (rows with no active incident, so
the task is genuinely predicting a FUTURE onset), tunes the decision threshold on
validation (favouring recall for early warning), and reports test metrics plus
warning time. Produces real, measured numbers — nothing is hard-coded.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from veyronix.core.config import ARTIFACT_DIR
from veyronix.core.logging import get_logger
from veyronix.features.build import load_dataset, load_incidents, load_meta
from veyronix.ml.evaluation.early_warning import early_warning
from veyronix.ml.evaluation.metrics import (
    best_threshold,
    calibration_curve,
    classification_metrics,
    recall_at_fpr,
)
from veyronix.ml.incident_prediction.models import MODEL_REGISTRY
from veyronix.ml.registry import register_model_version, try_mlflow_log

log = get_logger("ml.incident_prediction")
MODEL_DIR = ARTIFACT_DIR / "models"
RESULTS_PATH = ARTIFACT_DIR / "ml_results" / "incident_prediction.json"
TARGET = "incident_within_h"


def _pool(df: pd.DataFrame, split: str) -> pd.DataFrame:
    # predict FUTURE incidents from non-incident states only
    return df[(df.split == split) & (df.current_incident == 0)]


def _test_incidents(df: pd.DataFrame, incidents: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Incidents whose pre-alert window lies within each run's test region."""
    rows = []
    for run_id, g in df[df.split == "test"].groupby("run_id"):
        tmin, tmax = int(g.minute.min()), int(g.minute.max())
        inc = incidents[incidents.run_id == run_id]
        rows.append(inc[(inc.start_min > tmin) & (inc.start_min <= tmax)])
    return pd.concat(rows).reset_index(drop=True) if rows else incidents.iloc[0:0]


def train_all(
    seed: int = 42, make_active: str = "random_forest",
    df: pd.DataFrame | None = None, meta: dict | None = None,
    incidents: pd.DataFrame | None = None, register: bool = True,
) -> dict:
    df = load_dataset() if df is None else df
    meta = load_meta() if meta is None else meta
    incidents = load_incidents() if incidents is None else incidents
    horizon = meta["horizon"]
    feat_cols = meta["feature_columns"]

    tr, va, te = _pool(df, "train"), _pool(df, "val"), _pool(df, "test")
    Xtr, ytr = tr[feat_cols], tr[TARGET].to_numpy(int)
    Xva, yva = va[feat_cols], va[TARGET].to_numpy(int)
    Xte, yte = te[feat_cols], te[TARGET].to_numpy(int)
    test_inc = _test_incidents(df, incidents, horizon)

    log.info("train.pool", train=len(tr), val=len(va), test=len(te),
             train_pos=int(ytr.sum()), test_pos=int(yte.sum()))

    results: dict[str, dict] = {}
    fitted: dict[str, object] = {}
    for name, cls in MODEL_REGISTRY.items():
        model = cls(seed=seed) if name != "naive_threshold" else cls()
        model.fit(Xtr, ytr, Xva, yva)
        # Operating point tuned on validation to a fixed false-alarm budget
        # (max 5% FPR) — how alert thresholds are actually set in ops. We also
        # report the balanced-F1 point and recall at a 10% FPR cap.
        pva = model.predict_proba(Xva)
        thr = best_threshold(yva, pva, objective="recall_at_fpr", max_fpr=0.05)
        pte = model.predict_proba(Xte)
        m = classification_metrics(yte, pte, thr)
        m["calibration"] = calibration_curve(yte, pte)
        m["f1_operating_point"] = classification_metrics(
            yte, pte, best_threshold(yva, pva, objective="f1"))["f1"]
        m["recall_at_fpr10"] = recall_at_fpr(yte, pte, max_fpr=0.10)

        # early warning on the test region
        preds = te[["run_id", "minute"]].copy()
        preds["pred"] = (pte >= thr).astype(int)
        m["early_warning"] = early_warning(preds, test_inc, horizon)

        results[name] = m
        fitted[name] = model
        try_mlflow_log("incident_prediction", name,
                       {"algorithm": name, "threshold": thr},
                       {k: v for k, v in m.items() if isinstance(v, (int, float))})
        log.info("train.model", model=name, pr_auc=m["pr_auc"], recall=m["recall"],
                 precision=m["precision"], fp_rate=m["fp_rate"],
                 warn_median=m["early_warning"]["warning_minutes"]["median"])

    # persist the chosen active model + register all versions
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best = fitted[make_active]
    best_thr = results[make_active]["threshold"]
    artifact = MODEL_DIR / "incident_prediction.joblib"
    joblib.dump({"model": best, "threshold": best_thr, "features": feat_cols,
                 "horizon": horizon, "algorithm": make_active}, artifact)

    if register:
        for name, m in results.items():
            register_model_version(
                name=f"incident_prediction.{name}", task="incident_prediction",
                algorithm=name, params={"threshold": m["threshold"], "horizon": horizon},
                metrics={k: v for k, v in m.items() if isinstance(v, (int, float))},
                feature_names=feat_cols,
                artifact_path=str(artifact) if name == make_active else None,
                make_active=(name == make_active),
            )

    fi = fitted[make_active].feature_importances()
    out = {"horizon": horizon, "n_features": len(feat_cols),
           "pool_sizes": {"train": len(tr), "val": len(va), "test": len(te)},
           "test_positives": int(yte.sum()), "models": results,
           "top_features": fi, "active_model": make_active}
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(out, indent=2, default=float))
    return out


def load_active_model(path: Path | None = None) -> dict:
    return joblib.load(path or (MODEL_DIR / "incident_prediction.joblib"))
