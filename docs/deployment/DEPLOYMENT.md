# VEYRONIX — Deployment

Local development is trivial; a production path is documented for Docker Compose,
Kubernetes and CI/CD.

## 1. Local (zero infrastructure)

```bash
make install        # venv + deps
make init-db        # SQLite schema
make train          # dataset + models + evaluation summary
make run            # API  -> http://localhost:8000  (/docs, /metrics)
make web            # dashboard -> http://localhost:5173
make demo           # flagship Bad Deployment Cascade, end to end
make evaluate       # full benchmark suite -> docs/evaluation/
make test           # test suite
```

Runs entirely offline with no API keys (`VEYRONIX_LLM_PROVIDER=offline`).

## 2. Docker Compose (production-shaped)

```bash
cp .env.example .env
docker compose up --build
```

| Service | URL |
|---------|-----|
| Dashboard | http://localhost:3000 |
| API + docs | http://localhost:8000 / http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3001 (anon enabled; dashboard auto-provisioned) |

Postgres replaces SQLite; Redis, Prometheus and Grafana come up automatically.
The API exposes `/metrics`; Prometheus scrapes it; Grafana loads the VEYRONIX
self-observability dashboard from `infrastructure/grafana/provisioning`.

## 3. Kubernetes

```bash
# build & push images
docker build -f infrastructure/docker/api.Dockerfile -t ghcr.io/OWNER/veyronix-api:latest .
docker build -f infrastructure/docker/web.Dockerfile -t ghcr.io/OWNER/veyronix-web:latest ./web
docker push ghcr.io/OWNER/veyronix-api:latest && docker push ghcr.io/OWNER/veyronix-web:latest

# edit image names in infrastructure/kubernetes/veyronix.yaml, then:
kubectl apply -f infrastructure/kubernetes/veyronix.yaml
```

The manifests include a namespace, Postgres, API (2 replicas, health probes,
Prometheus scrape annotations, resource limits), web (LoadBalancer) and an
Ingress. Secrets (DB URL, optional LLM key) are in a `Secret`.

## 4. CI/CD

`.github/workflows/ci.yml` runs lint + tests on the backend and builds the
frontend on every push/PR. Extend it with an image build-and-push job and a
`kubectl apply` (or Helm) step gated on the `main` branch to complete continuous
deployment.

## 5. Configuration

All configuration is environment-driven (`veyronix/core/config.py`, prefix
`VEYRONIX_`). Secrets are never committed; see `.env.example`. The remediation
kill-switch (`VEYRONIX_ALLOW_AUTO_REMEDIATION`) is **off by default** — no action
auto-executes unless explicitly enabled, and MEDIUM+/blocked actions always
require the approval workflow regardless.

## 6. Reproducibility

Python 3.11 · Node 22 · Postgres 16. Fixed seeds throughout; `make` targets are
deterministic and idempotent. Model versions + metrics are recorded in the
`model_versions` table (and MLflow when installed).
