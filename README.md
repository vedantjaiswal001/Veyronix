<div align="center">

# VEYRONIX

### Predictive AI Incident & Root-Cause Intelligence Platform

*Detect the incident before it happens · explain why · simulate the fix · act safely.*

</div>

> **Status:** all 11 phases complete. Every
> quantitative result is produced by real code (`make evaluate`) and written to
> `docs/evaluation/`; anything not yet run is shown as **`NOT YET MEASURED`** —
> never fabricated (spec §45/§68).

---

## 1. Overview

VEYRONIX watches a **simulated production microservice environment** and runs a
full operational-intelligence loop:

```
DATA → DETECT → PREDICT → INVESTIGATE → ROOT CAUSE → SIMULATE
     → RECOMMEND → APPROVE → ACT → MEASURE → LEARN
```

It behaves like an **AI Operations Engineer**: it forecasts telemetry, predicts
whether an incident will occur within the next *N* minutes, detects anomalies
(including previously unseen ones), infers a ranked root cause with an evidence
chain, estimates business impact, simulates candidate interventions with
uncertainty, recommends one, and — only through a human-gated, allowlisted
safety layer — can execute a remediation and measure the outcome.

## Results at a glance

Measured by `make evaluate` on the time-aware test split (full tables:
[docs/evaluation/RESULTS.md](docs/evaluation/RESULTS.md), honest failure analysis:
[docs/evaluation/FAILURES.md](docs/evaluation/FAILURES.md)):

| Capability | Result |
|-----------|--------|
| Incident prediction (best: Random Forest) | PR-AUC **0.50**, 85% precision @ 1% FPR, median **14-min** early warning |
| vs. traditional static threshold | catches 7.7% of incidents at the same false-alarm budget; ML catches 30% |
| Anomaly detection (Isolation Forest) | AUROC **0.90** |
| **Unknown-failure generalization** (held-out cascading) | **84%** detected, never trained on, 8.5% FPR |
| Root-cause attribution | Top-1 **0.90**, Top-3 0.92 over 71 incidents |
| AI investigator | root-cause 0.90, **hallucination rate 0.0%**, tool success 100%, ~70 ms |
| Safety | LLM cannot execute arbitrary actions; `DELETE_DATABASE` permanently blocked |

~6,900 lines of Python + a React/TS dashboard, **111 tests**, lint-clean.

## 2. The problem

Modern systems emit metrics, logs, traces, deployment/infra events, DB stats,
business KPIs and ML-model telemetry. Traditional monitoring alerts *after*
something breaks. VEYRONIX answers, in order: *Is an incident coming? What is
probably causing it? What happens if we do nothing? Which intervention is best?
Can we execute it safely?*

## 3. Why existing monitoring isn't enough

Threshold alerts fire late and noisily, don't predict, don't explain causation,
don't estimate business impact, and can't reason about interventions. VEYRONIX
adds prediction, unsupervised anomaly detection, causal root-cause ranking,
counterfactual simulation, and a controlled remediation layer — and proves each
layer adds value via ablations (spec §50/§51).

## 4. Architecture

Deterministic Data/ML layer owns every number; a separate AI reasoning layer
investigates and recommends. Full details: **[docs/architecture/ARCHITECTURE.md](docs/architecture/ARCHITECTURE.md)**.

## 5. Quickstart (local, zero infra)

```bash
make install        # create .venv and install core + ml + dev deps
make init-db        # create the schema (SQLite by default)
make simulate       # generate labelled synthetic telemetry        (Phase 1)
make features       # build the leakage-safe feature matrix        (Phase 2)
make train          # train + evaluate all models                  (Phase 3)
make evaluate       # full benchmark suite -> docs/evaluation/      (Phase 10)
make run            # API at http://localhost:8000 (docs at /docs)
make web            # dashboard at http://localhost:5173            (Phase 9)
make demo           # one-command flagship incident demo           (Phase 11)
make test           # run the test suite
```

Runs entirely offline with **no API keys** (LLM provider defaults to `offline`).
Set `VEYRONIX_LLM_PROVIDER=anthropic|openai` + a key to use a live model.

### Full stack (Postgres, Redis, Prometheus, Grafana)

```bash
cp .env.example .env
docker compose up --build
# web :3000 · api :8000 (/docs) · grafana :3001 · prometheus :9090
```

## 6.–16. Data generation · ingestion · features · models · evaluation · root cause · agents · MCP · simulation · safety

Documented per phase under `docs/`. See the build status below for what is live.

## 17. Screenshots

**AI Operations Center** — system health timeline (incident probability with
WARN/CRIT bands), live dependency graph, and incidents ranked by severity with
prediction, ETA, root cause, business impact and recommended action:

![Operations Center](docs/screenshots/operations_center.png)

**Incident investigation** — root-cause chain, event timeline, the DB-latency
cascade signal, the grounded AI investigation, and the rollback approval flow:

![Incident investigation](docs/screenshots/incident_investigation.png)

> These are live renders of the running app against the FastAPI backend — real
> system output over the simulated environment.

## 18. Demo instructions

`make demo` runs the **Bad Deployment Cascade** end to end (Phase 11). Two more
scenarios — **Unknown Cascading Failure** and **Model Drift** — demonstrate
robustness and MLOps coverage (spec §58–§60).

## 19. Limitations

Honestly catalogued in `docs/evaluation/FAILURES.md` (Phase 10): false alarms,
missed incidents, wrong root causes, ambiguous evidence, unknown failures. The
environment is **SIMULATED**; business-impact figures are **model estimates**,
not financial truth (spec §22/§47).

## 20. Future work

Real-telemetry adapters (OTel/Prometheus scrape), online learning, richer
causal discovery, and a production remediation backend.

---

## Build status

| Phase | Component | State |
|------:|-----------|-------|
| 0 | Architecture, packaging, config, data model, migrations, Docker, CI scaffold | ✅ done |
| 1 | Simulator + fault injection + ground truth | ✅ done |
| 2 | Data pipeline: ingestion, validation, features | ✅ done |
| 3 | ML: prediction, anomaly, forecasting, severity + eval | ✅ done |
| 4 | Root-cause engine + causal graph | ✅ done |
| 5 | AI investigator, multi-agent, MCP, incident memory | ✅ done |
| 6 | Simulation + counterfactual + recommender | ✅ done |
| 7 | Safe remediation + approval + audit | ✅ done |
| 8 | Observability + model monitoring | ✅ done |
| 9 | Frontend AI Operations Center | ✅ done |
| 10 | Evaluation: baselines, ablations, RESULTS/FAILURES | ✅ done |
| 11 | Deployment + demo | ✅ done |

## Reproducibility

Python 3.11 · Node 22 · Postgres 16 · fixed seeds throughout. `make` targets are
deterministic and idempotent. Model versions + metrics are recorded in the
`model_versions` table and MLflow.

## License

MIT © 2026 Vedant Jaiswal — see [LICENSE](LICENSE).
