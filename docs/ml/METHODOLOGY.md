# VEYRONIX — ML Methodology & Measured Results (Phase 3)

All numbers below are **produced by `make train`** on the deterministic 8-run
synthetic dataset (seeds 42–49, 1 simulated day each; horizon H = 30 min) and
written to `data/artifacts/ml_results/summary.json`. They are re-measured on
every run — nothing here is hard-coded (spec §45/§68). Your exact figures may
shift slightly with dataset size/seed; regenerate to refresh.

## Task & data

* **Primary task:** predict `P(incident starts within the next 30 min)` from the
  last 30 min of telemetry — a *future* onset, not a current incident (spec §13).
* **Dataset:** 11,040 per-minute rows × 579 leakage-safe features across 8 runs.
* **Prediction pool:** rows with no *currently active* incident (so the model
  predicts onsets, it doesn't detect ongoing ones). Positive rate ≈ 24%.
* **Split:** chronological 60/20/20 per run with a 30-min embargo. Verified
  **leak-free** by truncation-invariance (Phase 2).

## Incident prediction — model comparison (test split, operating point ≤5% FPR)

| Model | PR-AUC | ROC-AUC | Precision | Recall | FP rate | Brier | Recall @ ≤10% FPR | Median warning |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Naive threshold | 0.404 | 0.787 | 0.44 | 0.077 | 0.019 | 0.335 | 0.391 | 16.5 min |
| Logistic regression | 0.235 | 0.534 | 0.30 | 0.338 | 0.151 | 0.316 | 0.313 | 17.5 min |
| **Random forest** ✅ | **0.496** | 0.678 | **0.85** | 0.299 | **0.010** | **0.123** | **0.394** | 14.0 min |
| XGBoost | 0.457 | 0.727 | 0.69 | 0.225 | 0.020 | 0.127 | 0.352 | 10.0 min |

**Reading of the result (honest):** at a realistic **≤5% false-alarm budget**,
Random Forest catches ~30% of imminent incidents with **85% precision** and a
median **14-minute** early warning, versus the traditional static threshold
catching only **7.7%**. Random Forest is the best model on PR-AUC and
calibration — it beats XGBoost here, a reminder that the most complex model is
not automatically best (spec §14). Top predictive signals include business
conversion-rate trend, checkout-error rate, system max p95 latency, and
order-service latency trend.

**Why recall isn't higher:** several fault types (sudden traffic spikes, hard
crashes) have *no precursor* by construction and are close to unpredictable —
this caps achievable recall and is reported honestly rather than hidden (see
`docs/evaluation/FAILURES.md`, Phase 10).

## Anomaly detection (unsupervised; test split, vs. active-incident ground truth)

| Detector | AUROC | AUPRC |
|----------|:---:|:---:|
| Isolation Forest | 0.895 | 0.477 |
| Statistical (robust rolling z-score) | 0.737 | 0.390 |

Isolation Forest, trained only on *normal* telemetry, separates incident from
healthy states strongly (AUROC 0.90) and can flag **previously unseen** patterns
— the basis of the unknown-failure experiment (spec §19, Phase 10).

## Severity classification (active incidents; test split)

Accuracy 0.45, macro-F1 0.31, **within-one-severity-band 1.00**. The environment
produces MEDIUM/HIGH/CRITICAL incidents; exact-band accuracy is modest but every
prediction lands within one band of truth. Severity remains a hard sub-problem —
reported transparently.

## Forecasting (10-min ahead; MAE, test split)

| Target | Naive | Moving-avg | ML (XGBoost) |
|--------|:---:|:---:|:---:|
| DB query latency | 5.14 | 7.98 | **3.48** |
| API p95 latency | **18.9** | 26.5 | 32.6 |
| Order-service CPU | **2.35** | 2.94 | 3.46 |
| API request rate | **38.0** | 60.3 | 162.2 |

ML forecasting **wins clearly where the signal has learnable structure** (DB
latency during incidents: MAE 3.48 vs 5.14, RMSE 23.3 vs 33.5). For smoother or
noisier signals, **naive persistence is a strong baseline and ML does not beat
it** — so we don't claim it does (spec §20/§62).

## Root-cause analysis (Phase 4; 71 incidents across 8 runs)

The engine ranks hypotheses by combining intrinsic-vs-symptom anomaly
classification, sink-in-the-affected-subgraph topology, deployment proximity,
and fleet-wide request-rate detection.

| Metric | Value |
|--------|:---:|
| **Top-1 service accuracy** | **0.90** |
| **Top-3 service accuracy** | **0.92** |
| True cause in evidence chain | 0.92 |
| Fault-type accuracy | 0.49 |

Per-fault Top-1 is ~1.00 for 12 of 13 fault types (bad deploy, DB overload,
cpu, memory leak, queue backlog, data-pipeline, network, packet-loss,
dependency failure, service crash, cascading; traffic spike 0.75). The one
systematic miss is **model drift** (0.00) — by design it has no infrastructure
signature and is surfaced by the drift monitor (Phase 8/§60), not the infra
root-cause engine. The flagship discrimination works: in the bad-deployment
cascade the database shows the largest symptom, but the engine correctly blames
the **deploying service** via deployment evidence, not the database.

## AI investigator — agent evaluation (Phase 5; 71 incidents, offline reasoner)

The bounded multi-agent investigator (5 specialists + synthesis) runs a
read-only tool runbook, grounds its conclusion in the deterministic root-cause
ranking, and retrieves similar past incidents from memory (RAG).

| Metric | Value |
|--------|:---:|
| Root-cause accuracy (top-1) | 0.90 |
| Root-cause accuracy (top-3) | 0.92 |
| Recommendation accuracy | 0.73 |
| Tool-call success rate | 1.00 |
| Avg tool calls / investigation | 6.8 |
| Avg investigation time | ~72 ms |
| **Hallucination rate** | **0.00** |
| Unsafe-action rate | 0.00 |

Hallucination rate is zero by construction: the top cause must be grounded in
gathered evidence (an anomalous service or a deploy), and the LLM may only
narrate/prioritise the deterministic candidates — it cannot invent a cause or a
number. These numbers are for the offline reasoner (no API key); a live
Anthropic/OpenAI model can be enabled for richer narration without changing what
is measured.

## Reproduce

```bash
make train          # rebuilds dataset, trains everything, writes summary.json
```
Models + metrics are registered in the `model_versions` table and (if MLflow is
installed) mirrored to MLflow.
