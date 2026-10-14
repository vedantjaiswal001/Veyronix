"""Classification metrics for incident prediction (spec §14).

Emphasis, per the spec, on PR-AUC / recall / false-positive rate rather than
accuracy, because incidents are rare and a false alarm is costly. Includes
calibration (reliability) and Brier score so probabilities — not just the
decision — are evaluated.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5,
) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    pos, neg = int(y_true.sum()), int((y_true == 0).sum())
    pr_auc = float(average_precision_score(y_true, y_prob)) if pos and neg else float("nan")
    roc_auc = float(roc_auc_score(y_true, y_prob)) if pos and neg else float("nan")
    brier = float(brier_score_loss(y_true, y_prob)) if pos and neg else float("nan")

    return {
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(f1, 4), "pr_auc": round(pr_auc, 4), "roc_auc": round(roc_auc, 4),
        "brier": round(brier, 4), "fp_rate": round(fpr, 4),
        "accuracy": round((tp + tn) / max(len(y_true), 1), 4),
        "threshold": round(float(threshold), 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn, "n": int(len(y_true)), "positives": pos,
    }


def best_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, objective: str = "f1",
    min_recall: float = 0.0, max_fpr: float | None = None,
) -> float:
    """Pick a decision threshold on a (validation) set.

    objective="f1" maximizes F1 (balanced operating point); "fbeta2" favours
    recall (beta=2); "recall_at_fpr" maximizes recall subject to FPR<=max_fpr
    (the ops-realistic "cap the false alarms" policy). ``min_recall`` optionally
    constrains any objective.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    grid = np.unique(np.quantile(y_prob, np.linspace(0.01, 0.99, 99)))
    best_t, best_score = 0.5, -1.0
    for t in grid:
        m = classification_metrics(y_true, y_prob, t)
        if m["recall"] < min_recall:
            continue
        if max_fpr is not None and m["fp_rate"] > max_fpr:
            continue
        if objective == "fbeta2":
            b2 = 4
            denom = b2 * m["precision"] + m["recall"]
            score = (1 + b2) * m["precision"] * m["recall"] / denom if denom else 0.0
        elif objective == "recall_at_fpr":
            score = m["recall"]
        else:
            score = m["f1"]
        if score > best_score:
            best_score, best_t = score, float(t)
    return best_t


def recall_at_fpr(y_true: np.ndarray, y_prob: np.ndarray, max_fpr: float = 0.1) -> dict:
    """Best achievable recall while keeping the false-positive rate <= max_fpr."""
    t = best_threshold(y_true, y_prob, objective="recall_at_fpr", max_fpr=max_fpr)
    m = classification_metrics(y_true, y_prob, t)
    return {"max_fpr": max_fpr, "recall": m["recall"], "precision": m["precision"],
            "fp_rate": m["fp_rate"], "threshold": m["threshold"]}


def calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> dict:
    """Reliability curve: mean predicted vs observed frequency per probability bin."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(y_prob, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        rows.append({
            "bin": b, "count": int(mask.sum()),
            "mean_pred": round(float(y_prob[mask].mean()), 4),
            "obs_freq": round(float(y_true[mask].mean()), 4),
        })
    ece = sum(r["count"] * abs(r["mean_pred"] - r["obs_freq"]) for r in rows) / max(len(y_true), 1)
    return {"bins": rows, "ece": round(ece, 4)}
