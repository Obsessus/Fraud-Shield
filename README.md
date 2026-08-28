# Fraud Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2-EC6198)](https://xgboost.ai)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)

**A fraud-detection system for online payments, built end to end.**  
Live demo: https://fraud-shield-gnec.onrender.com/

I built this as a portfolio project to practice taking a fraud model from raw data all the way
to a deployed, monitored service — not just training a model and stopping there. It uses the
real-world IEEE-CIS dataset (~590K transactions, where fraud is only ~3.5% of cases, which is
most of the challenge).

## What it does
You load a transaction, the model scores it for fraud, and you get back a risk score, a decision
(allow / review), and the few factors that drove that decision. The live demo lets you poke at
real transactions and watch the score change.

## How it's built
- **Leakage-safe features** — aggregates computed with an expanding window that excludes the
  current row, so validation isn't quietly cheating.
- **Temporal split** — train / validate / hold-out split by *time*, not randomly, to get honest
  metrics.
- **Baseline vs. XGBoost** — a Logistic Regression baseline to set expectations, then XGBoost
  with imbalance handling.
- **Explainability** — per-prediction SHAP (TreeSHAP) so every score comes with its reasons.
- **Serving + monitoring** — a FastAPI service in Docker, with Prometheus metrics and Evidently
  drift checks.

## Results
| Model                       | Hold-out PR-AUC |
|-----------------------------|-----------------|
| Logistic Regression (baseline) | 0.195       |
| XGBoost (champion)             | **0.511**   |

The champion runs at a tuned threshold of **0.798**. PR-AUC matters more than accuracy here
because of the heavy class imbalance.

## Try it
Open **https://fraud-shield-gnec.onrender.com/** — click **New random transaction**, edit a
field, hit **Check fraud**. Hover any term like `card1` for a plain-English meaning. Heads up:
it's on a free tier, so the first load after a quiet period takes ~30s to wake up.

## Run it yourself
```bash
python -m venv .venv && .\.venv\Scripts\Activate.ps1
pip install -e ".[base,ml,serving,mlops,dev]"
docker compose up -d
```
The trained model is baked into the image, so you don't need the dataset to run the service.
(`dvc repro` will rebuild the whole pipeline if you do have the data.) See `ARCHITECTURE.md` for
 the full design.

## Stack
Python 3.11 · pandas · scikit-learn · XGBoost · MLflow · DVC · FastAPI · Docker · Prometheus · Evidently
