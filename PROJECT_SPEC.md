# PROJECT_SPEC.md

## Problem definition

Predict whether an online payment transaction is fraudulent, using the IEEE-CIS
Fraud Detection dataset (transaction + identity tables). The business need is an
operational risk decision, not just a probability.

## Target

Binary label `isFraud` ∈ {0,1} from `train_transaction`. Positive rate ≈ **3.5%**
(severe class imbalance). Not present in the Kaggle test set.

## Project goals

1. Reproducible ingest + validation of raw data.
2. Reusable, documented feature engineering.
3. Baseline and stronger tabular models compared under one framework.
4. Class-imbalance handling without leakage.
5. Fraud-appropriate evaluation (PR-AUC, precision, recall, F1, confusion matrix,
   threshold analysis).
6. Temporal validation instead of random splitting.
7. Operational decision layer tuned from validation data.
8. Per-prediction explanations (feature contribution, not causal claims).
9. Experiment + model-version tracking.
10. Production-style API.
11. Containerization.
12. Automated tests + CI.
13. Deployment.
14. System + ML/data monitoring and drift detection.
15. Interview-grade documentation.

## Intended users

- ML engineering owner (this agent + human reviewer).
- Fraud operations / risk analysts (consume decisions + explanations).
- Technical interviewer (evaluate architecture + reasoning).

## ML requirements

- Supervised binary classification, gradient-boosted trees + interpretable baseline.
- Features must be leakage-free (see validation requirement).
- Models compared with identical pipeline + split.

## Evaluation philosophy

- **Primary metric: PR-AUC** (robust to imbalance).
- Secondary: precision, recall, F1, confusion matrix, threshold/PR curves.
- **Accuracy is not a headline metric.**
- Slice evaluation (ProductCD, identity-present vs absent, time period).

## Temporal validation requirement

Fraud is temporal. Split the labeled train set chronologically by `TransactionDT`
(reference date ≈ 2017-11-30): TRAIN (~70%) → VALIDATION (~15%, threshold tuning)
→ HOLD-OUT (~15%, final labeled "future"). The Kaggle test set is a later disjoint
period used only for production simulation / monitoring, never for metric tuning.

## Imbalance considerations

≈3.5% positives. Use class weighting (`class_weight` / `scale_pos_weight`); compare
against unweighted via controlled experiments. No random oversampling (leakage risk).
PR-AUC and threshold analysis are the guardrails.

## Explainability requirement

SHAP feature contributions per prediction. Clearly distinguish *feature contribution*
from *causal reasoning*. API returns top risk factors.

## API requirement

FastAPI service exposing at least: `POST /predict`, `GET /health`, `GET /model`,
`GET /metrics`. Pydantic request/response schemas. Response includes risk score,
decision, model version, explanation. No internal implementation details exposed.

## MLOps requirement

MLflow tracks: dataset version, feature config, model type, hyperparameters, validation
strategy, metrics, threshold, artifacts, model version. Promotion gate beyond "best
score" (see `DECISIONS.md`).

## Monitoring requirement

- System: request count, latency, errors, throughput (Prometheus).
- ML/data: missing values, feature distributions, prediction distribution, drift
  (Evidently), performance when labels arrive.
- Every dashboard panel must have a monitoring purpose.

## Deployment requirement

First deploy: single Dockerized FastAPI service on a simple PaaS/VM. No Kubernetes.
Reproducible build + smoke test in CI.

## Constraints

- No Kaggle credentials committed; dataset via DVC local remote.
- Python pinned to 3.11.
- No unnecessary infrastructure (K8s, cloud SDKs, Grafana, heavy orchestration).
- Keep it portfolio-explainable.

## Definition of done

A stage is complete only when: implementation works, failure cases considered, tests
exist where appropriate, results reproducible, documentation current, assumptions
recorded, and the next stage has a clear starting point.
