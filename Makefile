# VEYRONIX — reproducible control surface.
# Every command is deterministic (fixed seeds) and safe to re-run.

VENV      ?= .venv
PY        ?= $(VENV)/bin/python
PIP       ?= $(VENV)/bin/pip
UVICORN   ?= $(VENV)/bin/uvicorn
PYTEST    ?= $(VENV)/bin/pytest
SEED      ?= 42

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	 awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

$(VENV):
	python3 -m venv $(VENV)

.PHONY: install
install: $(VENV) ## Create venv and install core + ml + dev deps
	$(PIP) install -U pip wheel
	$(PIP) install -e ".[ml,dev]"

.PHONY: install-all
install-all: $(VENV) ## Install everything (core, ml, ai, db, otel, dev)
	$(PIP) install -U pip wheel
	$(PIP) install -e ".[ml,ai,db,otel,dev]"

.PHONY: init-db
init-db: ## Create the operational database schema
	$(PY) -m veyronix.cli init-db --drop

.PHONY: simulate
simulate: ## Generate synthetic telemetry + incidents (Phase 1)
	$(PY) -m veyronix.cli simulate --minutes 1440 --seed $(SEED)

.PHONY: features
features: ## Build leakage-safe feature matrix (Phase 2)
	$(PY) -m veyronix.features.build --seed $(SEED)

.PHONY: train
train: ## Train all ML models (Phase 3)
	$(PY) -m veyronix.ml.training.run --seed $(SEED)

.PHONY: evaluate
evaluate: ## Run full evaluation suite -> docs/evaluation (Phase 10)
	$(PY) -m veyronix.eval.run --seed $(SEED)

.PHONY: test
test: ## Run the test suite
	$(PYTEST)

.PHONY: test-fast
test-fast: ## Run only fast unit tests
	$(PYTEST) -m unit

.PHONY: lint
lint: ## Ruff lint
	$(VENV)/bin/ruff check veyronix tests

.PHONY: typecheck
typecheck: ## mypy type-check
	$(VENV)/bin/mypy veyronix

.PHONY: run
run: ## Run the API (http://localhost:8000, docs at /docs)
	$(UVICORN) veyronix.api.app:app --host 0.0.0.0 --port 8000 --reload

.PHONY: mcp
mcp: ## Run the MCP server
	$(PY) -m veyronix.mcp.server

.PHONY: web
web: ## Run the frontend dev server (http://localhost:5173)
	cd web && npm install && npm run dev

.PHONY: demo
demo: ## One-command flagship demo (bad-deploy cascade end-to-end)
	$(PY) -m veyronix.eval.demo --scenario bad_deploy

.PHONY: demo-unknown
demo-unknown: ## Demo: unknown cascading failure (robustness)
	$(PY) -m veyronix.eval.demo --scenario unknown_cascade

.PHONY: demo-drift
demo-drift: ## Demo: model drift detection (MLOps)
	$(PY) -m veyronix.eval.demo --scenario model_drift

.PHONY: docker-up
docker-up: ## Start the full stack with Docker Compose
	docker compose up --build

.PHONY: docker-down
docker-down: ## Stop the stack
	docker compose down -v

.PHONY: clean
clean: ## Remove caches and regenerable data
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	rm -f data/*.db data/*.duckdb
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
