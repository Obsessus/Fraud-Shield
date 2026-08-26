# Fraud Intelligence Platform

A production-oriented fraud detection system built on the **IEEE-CIS Fraud Detection**
dataset. The goal is to demonstrate a full ML engineering lifecycle: from raw
transaction/identity data to a monitored, containerized inference service.

## Problem

Online payment transactions must be scored for fraud risk in near-real-time and
converted into an operational decision (allow / review / block) while controlling
false positives and false negatives.

## Objective

Build a reproducible system that ingests and validates data, engineers documented
features, trains a defensible baseline and a stronger tabular model, evaluates with
fraud-appropriate metrics using temporal validation, exposes predictions via an API,
containerizes and deploys the service, and monitors it for health and data/model drift.

## Planned capabilities

- Ingest + validate raw transaction/identity CSVs (DVC-versioned)
- Reusable, documented feature engineering
- Baseline (Logistic Regression) and candidate (XGBoost) models
- Class-imbalance handling without leakage
- Temporal train / validation / hold-out validation
- Operational decision layer (LOW RISK / REVIEW / HIGH RISK) from a tuned threshold
- Per-prediction explanations (SHAP feature contributions)
- Experiment + model-version tracking (MLflow)
- Production-style API (FastAPI + Pydantic)
- Containerization (Docker) + CI (GitHub Actions)
- Deployment to a simple PaaS/VM
- System + ML/data monitoring (Prometheus + Evidently) and drift detection

## High-level architecture

Raw CSVs → DVC → ingest/join → validate → features (time-safe aggregates) →
temporal split → train (LR/XGBoost) → evaluate/threshold → explain → MLflow registry
→ FastAPI service → Docker → PaaS. Prometheus scrapes `/metrics`; Evidently runs
periodic drift checks.

## Technology stack

Python 3.11 · pandas · NumPy · scikit-learn · XGBoost · MLflow · DVC · FastAPI ·
Pydantic · Docker · GitHub Actions · Evidently · Prometheus

## Current status

**Complete.** The full lifecycle is implemented and verified:

- Data ingestion + validation (DVC), **temporal** 70/15/15 train/val/hold-out split
- Leakage-safe feature engineering (416 features; expanding-window aggregates exclude
  the current row)
- Baseline **Logistic Regression** and candidate **XGBoost** (hold-out PR-AUC 0.195 →
  **0.511**)
- Promotion gate → champion registered in MLflow (`fraud-intel-champion`, alias
  `champion`) at operating threshold **0.798**
- Per-prediction SHAP risk factors (XGBoost native TreeSHAP)
- FastAPI service + Docker image + `docker compose` deployment (read-only, hardened)
- Monitoring: Prometheus `/metrics` + Evidently data-drift checks
- Reproducible **model card** (`scripts/build_report.py` →
  `data/artifacts/final_evaluation.md`)

See `DESIGN.md` / `ARCHITECTURE.md` for design, `ROADMAP.md` for stage history,
`DECISIONS.md` for the decision log, and `docs/final_report.md` for the evaluation
writeup.

## Quickstart

```bash
# Environment
python -m venv .venv && .\.venv\Scripts\Activate.ps1
pip install -e ".[base,ml,serving,mlops,dev]"

# Reproduce the whole pipeline + regenerate the model card
dvc repro
python scripts/build_report.py

# Quality gate
ruff check src tests scripts
python -m mypy src
python -m pytest -q

# Serve
docker compose up -d            # image: fraud-intel-api:local
# GET  /health
# POST /predict  { "transaction_id": "...", "features": { ... } }
# GET  /metrics  (Prometheus)
```

> Requires the IEEE-CIS dataset pulled into `data/raw` (see `DESIGN.md`) and a
> `dvc pull` (or local run) to populate `data/artifacts`.

