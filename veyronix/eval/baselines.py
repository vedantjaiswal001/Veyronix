"""Baseline comparison (spec §51).

Compares, on the SAME time-aware test split and a fixed false-alarm budget:
  1. static threshold alerts   (the traditional monitor)
  2. ML-only prediction        (Random Forest)
  3. anomaly-only detection    (Isolation Forest)
  4. ML + anomaly              (union)
  5. complete VEYRONIX         (ML + anomaly for detection; + causal RCA for
                                attribution)

The goal is to show each layer adds operational value, not that "AI is cool".
"""

from __future__ import annotations

import numpy as np

from veyronix.core.logging import get_logger
from veyronix.features.build import load_dataset, load_incidents, load_meta
from veyronix.ml.anomaly.detectors import IsolationForestDetector
from veyronix.ml.evaluation.early_warning import early_warning
from veyronix.ml.evaluation.metrics import best_threshold, classification_metrics
from veyronix.ml.incident_prediction.models import NaiveThreshold, RandomForestModel

log = get_logger("eval.baselines")
TARGET = "incident_within_h"
FPR_BUDGET = 0.05


def _pool(df, split):
    return df[(df.split == split) & (df.current_incident == 0)]


def _norm(x, ref):
    lo, hi = np.min(ref), np.max(ref)
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)


def run_baselines(seed: int = 42) -> dict:
    df = load_dataset()
    meta = load_meta()
    incidents = load_incidents()
    horizon = meta["horizon"]
    feat = meta["feature_columns"]

    tr, va, te = _pool(df, "train"), _pool(df, "val"), _pool(df, "test")
    ytr, yva, yte = (g[TARGET].to_numpy(int) for g in (tr, va, te))

    # models
    rf = RandomForestModel(seed).fit(tr[feat], ytr)
    naive = NaiveThreshold().fit(tr[feat], ytr)
    normal = df[(df.split == "train") & (df.current_incident == 0) & (df[TARGET] == 0)]
    iso = IsolationForestDetector(seed=seed).fit(normal[feat])

    # per-config probability/score on val + test
    def scores(frame):
        rf_p = rf.predict_proba(frame[feat])
        nv_p = naive.predict_proba(frame[feat])
        an_raw = iso.score(frame[feat])
        return rf_p, nv_p, an_raw

    rf_va, nv_va, an_va_raw = scores(va)
    rf_te, nv_te, an_te_raw = scores(te)
    an_va = _norm(an_va_raw, an_va_raw)
    an_te = _norm(an_te_raw, an_va_raw)

    configs = {
        "static_threshold": (nv_va, nv_te),
        "ml_only": (rf_va, rf_te),
        "anomaly_only": (an_va, an_te),
        "ml_plus_anomaly": (np.maximum(rf_va, an_va), np.maximum(rf_te, an_te)),
    }
    configs["veyronix_full"] = configs["ml_plus_anomaly"]  # same detector union

    # scores on ALL test rows (incl. active-incident rows) for in-incident
    # detection — anomaly detection detects *during* an incident, not before.
    te_all = df[df.split == "test"]
    rf_all = rf.predict_proba(te_all[feat])
    nv_all = naive.predict_proba(te_all[feat])
    an_all = _norm(iso.score(te_all[feat]), an_va_raw)
    all_scores = {
        "static_threshold": nv_all, "ml_only": rf_all, "anomaly_only": an_all,
        "ml_plus_anomaly": np.maximum(rf_all, an_all),
        "veyronix_full": np.maximum(rf_all, an_all),
    }
    all_minutes = te_all[["run_id", "minute"]].reset_index(drop=True)

    # test incidents in-region for warning-time
    test_inc = []
    for run_id, g in te.groupby("run_id"):
        inc = incidents[incidents.run_id == run_id]
        test_inc.append(inc[(inc.start_min > g.minute.min()) & (inc.start_min <= g.minute.max())])
    import pandas as pd
    test_inc = pd.concat(test_inc).reset_index(drop=True) if test_inc else incidents.iloc[0:0]

    def incident_detection(score_all, thr):
        flagged = all_minutes.copy()
        flagged["hit"] = (score_all >= thr).astype(int)
        det = 0
        for _, inc in test_inc.iterrows():
            w = flagged[(flagged.run_id == inc.run_id)
                        & (flagged.minute >= inc.start_min - horizon)
                        & (flagged.minute <= inc.end_min)]
            det += int(w.hit.any())
        return round(det / max(len(test_inc), 1), 4)

    results = {}
    for name, (pva, pte) in configs.items():
        thr = best_threshold(yva, pva, objective="recall_at_fpr", max_fpr=FPR_BUDGET)
        m = classification_metrics(yte, pte, thr)
        preds = te[["run_id", "minute"]].copy()
        preds["pred"] = (pte >= thr).astype(int)
        ew = early_warning(preds, test_inc, horizon)
        results[name] = {
            "precision": m["precision"], "recall": m["recall"], "fp_rate": m["fp_rate"],
            "pr_auc": m["pr_auc"],
            "early_warning_rate": ew["detection_rate"],
            "incident_detection_rate": incident_detection(all_scores[name], thr),
            "warning_median": ew["warning_minutes"]["median"],
        }
        log.info("eval.baseline", config=name, recall=m["recall"], fpr=m["fp_rate"],
                 warn=ew["detection_rate"])
    return {"fpr_budget": FPR_BUDGET, "horizon": horizon, "configs": results,
            "test_incidents": len(test_inc)}
