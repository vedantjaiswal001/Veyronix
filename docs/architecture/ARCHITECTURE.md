# VEYRONIX — Architecture

> Predictive AI Incident & Root-Cause Intelligence Platform.
> A deterministic Data/ML layer owns every number; a separate AI reasoning
> layer investigates, explains, and recommends. The LLM is never the source of
> truth for a prediction.

This document is the durable answer to spec §70 ("first response requirement"):
final architecture, repo structure, data flow, ML pipeline, agent architecture,
testing strategy, hardest risks, and build order.

---

## 1. Design principles

1. **Deterministic core, probabilistic edge.** Metrics, features, anomaly
   scores, forecasts, incident probabilities, root-cause rankings and every
   evaluation number are produced by code and models we can re-run and test.
   The LLM layer reasons *over retrieved evidence* and may propose actions, but
   it computes no numbers and executes nothing on its own.
2. **Ground truth everywhere.** The simulator emits labelled incidents
   (`fault_type`, `root_cause_service`, `start/end`, `affected_services`,
   `severity`, `recovery_action`). Every ML/RCA/agent claim is scored against
   it. No metric is hard-coded (spec §45/§68); unmeasured values render as
   `NOT YET MEASURED` / `NA`.
3. **Predict the future, not the present.** The primary task is
   `P(incident starts within the next H minutes)` — not classifying an ongoing
   incident (spec §13).
4. **Leakage is the enemy.** Feature windows use only information available at
   or before time *t*; labels look strictly forward. Time-aware splits, no
   shuffling. Automated leakage tests (spec §15/§16).
5. **Safety is structural, not advisory.** Actions are an allowlisted enum with
   risk tiers; a global kill-switch defaults off; MEDIUM/HIGH require human
   approval; `DELETE_DATABASE` is permanently `BLOCKED`. The LLM cannot emit an
   action outside the enum (spec §33/§53).
6. **Every technology earns its place** (spec §62). One SQL store + one
   columnar store, three agents not ten, gradient boosting over deep learning
   unless DL measurably wins, in-process event bus with a Kafka-shaped seam.

---

## 2. Layered architecture

```
SIMULATED PRODUCTION ENVIRONMENT  (veyronix/simulator)
   │  metrics · logs · traces · deployments · business KPIs · ground-truth incidents
   ▼
DATA PLATFORM  (veyronix/ingestion, veyronix/db, DuckDB feature store)
   │  ingest → validate (data quality) → store → materialize features
   ▼
ML LAYER  (veyronix/ml)
   │  incident prediction · anomaly detection · forecasting · severity · drift
   ▼
INCIDENT ENGINE  (veyronix/rootcause + ml outputs)
   │  fuse predictions + anomalies + deployments + topology
   ▼
AI INVESTIGATOR  (veyronix/agents + veyronix/mcp)
   │  orchestrator → metrics/logs/traces/deploy/db agents → synthesis
   ▼
ROOT-CAUSE ENGINE → CAUSAL GRAPH  (veyronix/rootcause)
   ▼
SIMULATION / COUNTERFACTUAL ENGINE  (veyronix/simulation)
   ▼
ACTION RECOMMENDER → RISK POLICY → HUMAN APPROVAL / (guarded) AUTO
   │  (veyronix/remediation)
   ▼
OUTCOME → POST-INCIDENT LEARNING → EVALUATION  (veyronix/eval)
```

Cross-cutting: `veyronix/core` (config, logging, constants, topology) and
`veyronix/observability` (Prometheus metrics, OTel spans, model/feature drift).

---

## 3. Repository structure

A single installable package `veyronix/` (clean imports, one test target) plus a
`web/` React app — the same components as the spec §42 tree, organized as a
Python package because it is materially more testable (spec §42 permits this).

```
veyronix/                      # installable package
  core/        config, logging, constants, topology         (Phase 0)
  db/          SQLAlchemy models, session, migrations target (Phase 0)
  datasets/    LogHub/GAIA/AIOpsArena ingestion adapters      (Phase 2)
  simulator/   microservice sim + fault injection + ground truth (Phase 1)
  ingestion/   telemetry ingest + data-quality validation     (Phase 2)
  features/    leakage-safe temporal feature engineering       (Phase 2)
  ml/          anomaly, forecasting, incident_prediction, severity, evaluation, training (Phase 3)
  rootcause/   service graph, temporal reasoning, ranking      (Phase 4)
  agents/      orchestrator + 5 specialist agents + synthesis  (Phase 5)
  mcp/         MCP server exposing VEYRONIX tools              (Phase 5)
  simulation/  action simulator + counterfactuals + recommender (Phase 6)
  remediation/ allowlist, risk policy, approval, audit         (Phase 7)
  observability/ prometheus, otel, drift monitors              (Phase 8)
  api/         FastAPI app                                     (Phase 3+)
  eval/        baseline/ablation/agent suites + demo           (Phase 10)
web/           React + TS + Vite + Tailwind + Recharts         (Phase 9)
infrastructure/ docker, prometheus, grafana, kubernetes        (Phase 0/11)
migrations/    Alembic                                         (Phase 0)
tests/         unit, integration, ml, agents, security, end_to_end
docs/          architecture, ml, evaluation, incidents, deployment
```

---

## 4. Data flow

1. **Generate.** The simulator advances a per-minute clock. Each service's
   metrics evolve from a baseline via an AR(1)-style process with daily
   seasonality plus service-coupled noise. Fault injections perturb the causal
   chain (e.g. bad deploy → DB latency → API latency → errors → checkout
   failures → revenue), producing realistic *temporal signatures* rather than
   random noise (spec §9).
2. **Ingest & validate.** Rows pass through a data-quality gate (missing,
   duplicate, invalid timestamp, impossible value, schema change, distribution
   shift, delayed event). Problems are logged and quarantined, not silently
   cleaned (spec §11).
3. **Store.** Operational store = SQLAlchemy (SQLite local / Postgres in
   Docker). Analytical/feature store = DuckDB (columnar, fast windowed reads).
4. **Featurize.** Per service/minute we compute rolling mean/std/min/max,
   rate-of-change, %-change, lags, EWMA, volatility, moving quantiles, trend
   slope, seasonality index, error ratios, latency percentiles, saturation, and
   topology-aware dependency-health features — all backward-looking only.
5. **Predict / detect / forecast.** Models emit incident probability + ETA,
   anomaly scores, forecasts and severity. Outputs are persisted to
   `predictions`, `anomalies`, etc.
6. **Investigate.** When probability/anomaly cross thresholds the incident
   engine opens an investigation; the agent orchestrator gathers evidence via
   tools (which only read the deterministic stores).
7. **Rank causes → simulate → recommend → approve → act → measure → learn.**

---

## 5. ML pipeline

- **Target:** binary `incident_within_H` at each timestamp; H default 30 min.
- **Split:** chronological 60/20/20 train/val/test — never shuffled (spec §15).
- **Baselines → models:** naive threshold, logistic regression, random forest,
  gradient boosting (XGBoost) (spec §14). Complexity must beat simplicity to win.
- **Metrics:** precision, recall, F1, PR-AUC, ROC-AUC, Brier, calibration,
  FP rate — PR-AUC/recall/FP/warning-time emphasized (spec §14).
- **Early-warning metric:** for each detected incident, `actual_start −
  first_correct_alert`; report mean/median/P25/P75/min/max (spec §17).
- **Anomaly detection:** IsolationForest + statistical (rolling z-score / EWMA);
  designed to flag *unseen* patterns as `UNKNOWN_ANOMALY` (spec §18/§19).
- **Forecasting:** naive + seasonal-naive baselines vs gradient boosting for
  CPU/mem/latency/rate/error/queue; MAE/RMSE/MAPE (spec §20).
- **Severity:** LOW/MEDIUM/HIGH/CRITICAL from blast radius (topology
  centrality), error/latency, business exposure (spec §21).
- **Registry + monitoring:** MLflow + `models`/`model_versions` tables; drift
  monitors for feature/prediction/calibration (spec §36).

---

## 6. Agent architecture

Bounded, tool-using, evidence-first (spec §26–§29):

- **Orchestrator** plans the investigation and enforces budgets: max steps,
  wall-clock timeout, tool-call budget, token budget.
- **Specialists** (metrics, logs, traces, deployment, database) each own a
  slice of evidence and call *read-only* tools against the deterministic stores.
- **Synthesis** fuses evidence into a ranked root-cause hypothesis with an
  explicit evidence chain — reconciled against the deterministic RCA engine so
  the LLM narrates, it does not fabricate.
- **Provider abstraction:** `offline` (deterministic reasoner, no keys),
  `anthropic`, `openai`. Same tool interface; the offline path keeps the whole
  system runnable and testable without secrets.
- **MCP server** exposes the same tools (`query_metrics`, `query_logs`,
  `query_traces`, `get_incidents`, `get_deployments`, `get_service_graph`,
  `get_business_impact`, `simulate_action`, `get_similar_incidents`) to any MCP
  client, cleanly separated from the core app (spec §28).
- **Incident memory (RAG):** resolved incidents are embedded/summarized and
  retrieved by similarity to inform new investigations (spec §29).

---

## 7. Testing strategy

- **unit** — feature math, anomaly math, policy, business impact, simulator
  invariants, topology.
- **integration** — API, DB, ML inference, agent tools, MCP.
- **ml** — split integrity, **leakage tests**, metric correctness, calibration.
- **agents** — tool success, budget enforcement, structured-output validity.
- **security** — LLM cannot execute arbitrary commands; BLOCKED stays blocked;
  invalid/unauthorized/malformed actions fail safely (spec §43).
- **end_to_end** — fault → telemetry → prediction → investigation → root cause →
  simulation → recommendation → approval → action → outcome (spec §43).

Everything reproducible via fixed seeds and `make` targets (spec §44).

---

## 8. Hardest technical risks (and mitigations)

| # | Risk | Mitigation |
|---|------|------------|
| 1 | **Data leakage** silently inflating scores | Strict backward-only features; forward-only labels; automated leakage tests that fail CI; time-aware split enforced in one place. |
| 2 | **Unrealistic simulator** → models learn artifacts, not ops | Causal generation with AR + seasonality + coupled propagation; statistical realism tests (autocorrelation, seasonality, pre-incident drift is detectable but not trivially separable). |
| 3 | **Class imbalance** (incidents are rare) | PR-AUC-first evaluation, threshold tuning on validation, class weighting/`scale_pos_weight`, report FP rate + warning time not just accuracy. |
| 4 | **Root-cause ambiguity** in cascades | Combine temporal-precedence + topology blame-propagation + deployment proximity; output *ranked* hypotheses with evidence, Top-1/Top-3 scored honestly. |
| 5 | **LLM hallucination / unsafe actions** | Evidence-only tools, structured outputs, allowlisted actions, approval gates, audit; hallucination-rate + unsafe-action-rate in agent eval. |
| 6 | **Overconfidence on unknown failures** | Held-out cascading fault; anomaly-based `UNKNOWN_ANOMALY` path; calibrated probabilities; honest limitation reporting (spec §19/§52). |
| 7 | **Scope/complexity** | Strict phase gates: a phase is "done" only when its tests pass and it integrates. |

---

## 9. Build order

Phase 0 (this) → 1 simulator → 2 data pipeline → 3 ML+eval → 4 root cause →
5 agents+MCP → 6 simulation → 7 remediation → 8 observability → 9 frontend →
10 evaluation → 11 deployment/demo. Each gate: implement → test → fix → verify
integration → document → next (spec §63/§70).
