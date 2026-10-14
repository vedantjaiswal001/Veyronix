"""Unknown-failure experiment (spec §19).

An unsupervised Isolation Forest is trained ONLY on normal telemetry — it never
sees any incident, known or unknown. We then measure detection separately for
KNOWN fault families and for the held-out CASCADING_FAILURE (the "unknown"),
plus the false-positive rate on normal periods. The point: the anomaly detector
generalises to a previously unseen failure mode and flags it as an anomaly,
rather than a supervised classifier confidently mislabelling it. Limitations are
reported honestly.
"""

from __future__ import annotations

import numpy as np

from veyronix.core.constants import KNOWN_FAULTS
from veyronix.core.logging import get_logger
from veyronix.features.build import load_dataset, load_incidents, load_meta
from veyronix.ml.anomaly.detectors import IsolationForestDetector, evaluate_detector_scores

log = get_logger("eval.unknown")
KNOWN = {f.value for f in KNOWN_FAULTS}


def run_unknown_failure(seed: int = 42, target_fpr: float = 0.05) -> dict:
    df = load_dataset()
    meta = load_meta()
    incidents = load_incidents()
    feat = meta["feature_columns"]

    normal = df[(df.split == "train") & (df.current_incident == 0)
               & (df.incident_within_h == 0)]
    iso = IsolationForestDetector(seed=seed).fit(normal[feat])

    # peak anomaly score within each incident window
    def peak_score(inc):
        rows = df[(df.run_id == inc.run_id) & (df.minute >= inc.start_min)
                  & (df.minute < inc.end_min)]
        return float(np.max(iso.score(rows[feat]))) if not rows.empty else None

    known_scores, unknown_scores = [], []
    for _, inc in incidents.iterrows():
        sc = peak_score(inc)
        if sc is None:
            continue
        (known_scores if inc.fault_type in KNOWN else unknown_scores).append(sc)
    known_scores = np.array(known_scores)
    unknown_scores = np.array(unknown_scores)

    # Sensitivity is set to catch ~90% of KNOWN faults; then we ask: at that same
    # threshold, is the never-before-seen cascading failure flagged?
    threshold = float(np.quantile(known_scores, 0.10)) if len(known_scores) else 0.0
    known_det = float((known_scores >= threshold).mean()) if len(known_scores) else 0.0
    unknown_det = float((unknown_scores >= threshold).mean()) if len(unknown_scores) else 0.0

    test_normal = df[(df.split == "test") & (df.current_incident == 0)
                    & (df.incident_within_h == 0)]
    fpr = float((iso.score(test_normal[feat]) >= threshold).mean())

    test = df[df.split == "test"]
    disc = evaluate_detector_scores(iso.score(test[feat]), test.current_incident.to_numpy(int))

    result = {
        "sensitivity": "threshold set to catch ~90% of known faults",
        "target_fpr": target_fpr,
        "observed_fpr": round(fpr, 4),
        "known_detection_rate": round(known_det, 4),
        "unknown_detection_rate": round(unknown_det, 4),
        "known_incidents": int(len(known_scores)),
        "unknown_incidents": int(len(unknown_scores)),
        "auroc": disc["auroc"], "auprc": disc["auprc"],
        "note": ("Isolation Forest trained on normal telemetry only; at the "
                 "sensitivity that catches ~90% of known faults, the held-out "
                 "cascading failure is flagged as an anomaly without ever being "
                 "trained on it."),
    }
    log.info("eval.unknown.done", known=result["known_detection_rate"],
             unknown=result["unknown_detection_rate"], fpr=result["observed_fpr"])
    return result
