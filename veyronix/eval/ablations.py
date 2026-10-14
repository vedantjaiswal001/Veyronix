"""Ablation study (spec §50).

Adds one capability at a time and reports what each layer buys:
  Exp1  ML only                     -> prediction (detection + warning)
  Exp2  ML + anomaly                -> better detection / unseen patterns
  Exp3  ML + anomaly + causal       -> + root-cause attribution
  Exp4  ML + anomaly + AI investigator -> + explanation, recommendation, safety

Reuses the baseline detection numbers and the root-cause / agent evaluations so
every figure is measured, not asserted.
"""

from __future__ import annotations

from veyronix.core.logging import get_logger

log = get_logger("eval.ablations")


def run_ablations(base_configs: dict, rc: dict, agent_result: dict | None = None) -> dict:
    ml = base_configs["ml_only"]
    mla = base_configs["ml_plus_anomaly"]

    exps = {
        "exp1_ml_only": {
            "early_warning_rate": ml["early_warning_rate"],
            "incident_detection_rate": ml["incident_detection_rate"],
            "warning_median": ml["warning_median"],
            "root_cause_top1": None, "recommendation_accuracy": None,
            "adds": "future-incident prediction with early warning",
        },
        "exp2_ml_plus_anomaly": {
            "early_warning_rate": mla["early_warning_rate"],
            "incident_detection_rate": mla["incident_detection_rate"],
            "warning_median": mla["warning_median"],
            "root_cause_top1": None, "recommendation_accuracy": None,
            "adds": "unsupervised anomaly coverage (incl. unseen patterns)",
        },
        "exp3_ml_anomaly_causal": {
            "early_warning_rate": mla["early_warning_rate"],
            "incident_detection_rate": mla["incident_detection_rate"],
            "warning_median": mla["warning_median"],
            "root_cause_top1": rc["top1_accuracy"], "recommendation_accuracy": None,
            "adds": "causal root-cause attribution (symptom vs cause)",
        },
        "exp4_ml_anomaly_investigator": {
            "early_warning_rate": mla["early_warning_rate"],
            "incident_detection_rate": mla["incident_detection_rate"],
            "warning_median": mla["warning_median"],
            "root_cause_top1": (agent_result or {}).get("root_cause_top1", rc["top1_accuracy"]),
            "recommendation_accuracy": (agent_result or {}).get("recommendation_accuracy"),
            "hallucination_rate": (agent_result or {}).get("hallucination_rate"),
            "adds": "evidence-grounded explanation + recommendation + safety gate",
        },
    }
    log.info("eval.ablations.done", exps=len(exps))
    return exps
