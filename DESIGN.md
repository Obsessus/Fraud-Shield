# Fraud Intelligence Platform — Engineering Design Document

> Stage: Discovery & Architecture (no models trained yet)
> Status: Draft for human review
> Author: Engineering agent (technical owner)
> Last updated: 2026-08-19

This document is the agreed foundation for building a production-oriented fraud
intelligence platform on the IEEE-CIS Fraud Detection dataset. It is deliberately
decision-heavy: every significant choice records its rationale and the alternative
that was rejected.

---

## 1. Problem definition

Online payment transactions must be scored for fraud risk in near-real-time. We
are given historical transactions (and partial identity data) labeled `isFraud`.
The business need is not "predict a probability" but "make an operational decision"
(allow / review / block) while controlling false positives (blocked good
customers, lost revenue, friction) and false negatives (fraud losses).

The ML problem is **supervised binary classification under severe class imbalance
(≈3.5% positives) with a temporal dimension** (fraud patterns and data
distribution shift over time).

## 2. System goals

1. Ingest + validate raw transaction/identity CSVs reproducibly.
2. Produce reusable, documented features.
3. Train a defensible baseline and a stronger tabular model.
4. Handle class imbalance without train/test leakage.
5. Evaluate with fraud-appropriate metrics (PR-AUC, precision, recall, F1,
   confusion matrix, threshold analysis).
6. Use temporal validation, not only random splits.
7. Convert score → operational decision via a validated threshold.
8. Explain individual predictions (feature contributions, not causal claims).
9. Track experiments + model versions (MLflow).
10. Serve predictions via a production-style API (FastAPI).
11. Containerize the service (Docker).
12. Automated tests + CI (GitHub Actions).
13. Deploy the inference service.
14. Monitor API health + ML/data behavior.
15. Detect meaningful distribution drift.
16. Document well enough for a technical interview.

## 3. Functional requirements

- `ingest`: download/register raw CSVs (DVC), load + join by `TransactionID`.
- `validate`: schema, null-ratio, range, and temporal-order checks; fail loudly.
- `features`: deterministic transforms producing a feature frame + metadata.
- `train`: fit baseline (Logistic Regression) and candidate (XGBoost) models.
- `evaluate`: temporal train/val/holdout scoring + threshold analysis.
- `explain`: per-prediction SHAP risk factors.
- `serve`: `/predict`, `/health`, `/model`, `/metrics`.
- `monitor`: emit system metrics (Prometheus) + periodic drift checks (Evidently).
- `track`: every run logged to MLflow (params, metrics, artifacts, model version).

## 4. Non-functional requirements

- **Reproducibility**: fixed seeds, pinned deps, DVC-tracked data, MLflow runs.
- **Portability**: single Docker image runs locally and in the cloud.
- **Testability**: pytest suite covering transforms, validation, API, edge cases.
- **Explainability**: every `/predict` returns risk factors.
- **Simplicity**: no Kubernetes/orchestration unless a later stage justifies it.
- **Cost awareness**: cheap PaaS/VM; no always-on heavy compute for serving.

## 5. Dataset understanding

Source: IEEE-CIS Fraud Detection (Vesta Corp), via Kaggle. Files:

- `train_transaction.csv` (~590,540 rows) + `train_identity.csv` (subset of
  transactions have identity rows)
- `test_transaction.csv` (~506,691 rows) + `test_identity.csv` (labels withheld)
- Joined on `TransactionID`; **not every transaction has identity data**.

Known feature families:

| Family | Columns | Notes |
|---|---|---|
| ID | `TransactionID` | unique, not a feature |
| Time | `TransactionDT` | seconds from ~2017-11-30 reference; use for temporal split + derived time features |
| Amount | `TransactionAMT` | USD; decimal (cents) portion is informative |
| Product | `ProductCD` | categorical |
| Card | `card1`–`card6` | card1 ≈ hashed card/account id; high-value grouping key |
| Address | `addr1`, `addr2` | billing region / country |
| Email | `P_emaildomain`, `R_emaildomain` | purchaser / recipient domain |
| Counts | `C1`–`C14` | counts (e.g., addresses linked to card) |
| Deltas | `D1`–`D15` | time deltas in days (D1 ≈ "days since first seen"), partly missing |
| Match | `M1`–`M9` | match flags, >50% missing |
| Vesta | `V1`–`V339` | masked, highly correlated engineered features |
| Distance | `dist1`, `dist2` | distances (IP/addr/phone) |
| Identity | `DeviceType`, `DeviceInfo`, `id_12`–`id_38` | device + identity-derived |

**Important**: `V*` are masked/undisclosed; we treat them as opaque numeric
features and do not attempt to interpret them causally. Identity columns exist
only for a subset → models must handle their absence at inference.

## 6. Target definition

- `isFraud ∈ {0,1}` from `train_transaction`.
- Positive rate ≈ **3.5%** → accuracy is meaningless; PR-AUC / precision / recall
  are primary.
- `isFraud` **not** present in test → the labeled train set is our only ground
  truth; the Kaggle test set is used purely as a *future-period inference +
  monitoring* demonstration (and optionally for an external leaderboard sanity
  check, not for internal metric optimization).

## 7. Proposed temporal validation strategy

Fraud is temporal. We split the **labeled train set chronologically** by
`TransactionDT` (converted to a synthetic datetime from the reference date):

```
Reference datetime ≈ 2017-11-30
TransactionDT (sec) → DT = reference + timedelta(seconds=TransactionDT)

TRAIN     : earliest ~70% of time  (≈ Dec 2017 – Mar 2018)
VALIDATION: next    ~15% of time  (≈ Apr 2018)   ← threshold tuning here
HOLD-OUT  : latest  ~15% of time  (≈ May 2018)   ← final labeled "future" test
Kaggle TEST: disjoint later period (no labels) ← production simulation / monitoring
```

Rationale: training on the past, tuning on a near-future window, and measuring on
the most recent labeled window realistically mirrors production (we only ever know
the past when making a decision). We **never** tune thresholds on the hold-out.

Leakage controls enforced by this design:
- **Temporal leakage**: chronological split; no future rows in train.
- **Target leakage**: identity-only columns carry no label; `isFraud` never a feature.
- **Feature (aggregation) leakage**: entity aggregates (fraud rate / frequency of
  `card1`, `addr1`, email, device) are computed from **train-only** history and
  joined forward; never global target means applied to all rows.
- **Preprocessing leakage**: encoders / imputers / scalers fit on TRAIN only.
- **Duplication**: `TransactionID` is unique; verify no row appears in two splits.

## 8. Initial feature groups (each with a documented reason)

1. **Transaction characteristics**: `TransactionAMT` (log1p), amount cents,
   `ProductCD`. Reason: amount/context are core fraud signals.
2. **Temporal behavior**: derived `DT_M` (month), `DT_W` (week), `DT_D` (day),
   hour-of-day, day-of-week, holiday flag. Reason: fraud rate varies by time.
3. **Frequency / velocity**: `card1`, `addr1`, `P_emaildomain`, `DeviceInfo`
   frequency counts; inter-transaction time gaps. Reason: rapid reuse of a card/
   device is a classic fraud velocity signal.
4. **Entity reuse**: counts of distinct emails/devices/addresses per `card1`.
   Reason: shared infrastructure across accounts is suspicious.
5. **Identity relationships**: `DeviceType`, `DeviceInfo`, `id_*` presence flags,
   parsed OS/browser from `id_30`/`id_31`. Reason: device/identity context helps
   separate real vs synthetic sessions.
6. **Historical aggregation (train-only, time-safe)**: rolling mean/std of
   `TransactionAMT` and `isFraud` rate per `card1`/`addr1`/`emaildomain` computed
   on past data. Reason: entity reputation is predictive but must be leakage-free.
7. **Transaction context**: `dist1`/`dist2`, `C1`–`C14`, `M1`–`M9`, `V1`–`V339`.
   Reason: supplied context + Vesta features; kept but monitored for drift.
8. **Graph-inspired (no GNN)**: simple relational flags (e.g., "card seen with
   multiple devices", "device shared across cards"). Reason: relationship density
   is a fraud indicator; GNN deferred unless it proves necessary.

We will **not** create hundreds of arbitrary features; each group above is gated by
whether it improves PR-AUC on validation in a controlled experiment.

## 9. Baseline model

- **Baseline**: Logistic Regression (with class weighting / balanced). Rationale:
  cheap, interpretable, sets a floor; also a natural SHAP/explanation sanity check.
- **Candidate**: XGBoost (gradient-boosted trees), scale-pos-weight for imbalance.
- Models are compared under the **same** feature pipeline + temporal split.
  "Stronger model" is not auto-promoted (see §10).

## 10. Primary evaluation metrics

- **Primary**: PR-AUC (robust to imbalance).
- **Secondary**: precision, recall, F1, confusion matrix at the operating
  threshold, and threshold/PR curves.
- **Explicitly not** accuracy as a headline metric.
- Slice evaluation: by `ProductCD`, `card1` country proxy, identity-present vs
  absent, time period — to catch hidden degradation.

## 11. Proposed architecture

```
                ┌──────────────────────────────┐
   raw CSVs ───▶│ DVC (data versioning)        │
                └──────────────┬───────────────┘
                               ▼
                ┌──────────────────────────────┐
                │ ingest + join (TransactionID)│
                └──────────────┬───────────────┘
                               ▼
                ┌──────────────────────────────┐
                │ validate (schema/nulls/time)  │ ── fail build if violated
                └──────────────┬───────────────┘
                               ▼
                ┌──────────────────────────────┐
                │ feature engineering (train-   │
                │ only fit, time-safe aggregates)│
                └──────────────┬───────────────┘
                               ▼
        ┌──────────────┬───────┴────────┬───────────────┐
        ▼              ▼                 ▼               ▼
  ┌──────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐
  │ train    │  │ evaluate    │  │ explain     │  │ MLflow track │
  │ (LR/XGB) │  │ (temporal)  │  │ (SHAP)      │  │ (exp/version)│
  └────┬─────┘  └──────┬──────┘  └──────┬──────┘  └──────┬───────┘
       └───────────────┴────────────────┴────────────────┘
                               ▼
                   promoted model registry (MLflow)
                               ▼
                ┌──────────────────────────────┐
                │ FastAPI inference service     │
                │ /predict /health /model /metrics│
                └──────────────┬───────────────┘
                               ▼
          ┌────────────────────────┬────────────────────────┐
          ▼                        ▼                         ▼
   ┌─────────────┐        ┌──────────────┐         ┌──────────────┐
   │ Prometheus  │        │ Evidently    │         │ Docker image │
   │ system metr.│        │ drift monit. │         │ deploy (PaaS)│
   └─────────────┘        └──────────────┘         └──────────────┘
                               ▼
                   GitHub Actions (lint/test/CI)
```

## 12. Proposed repository structure

```
fraud-intelligence-platform/
├── README.md
├── DESIGN.md                # this document
├── pyproject.toml           # deps + tool config (single source of truth)
├── requirements.txt         # lockable install (generated)
├── .dvc/  +  dvc.yaml       # data + pipeline versioning
├── .github/workflows/ci.yml # lint, type-check, tests
├── data/
│   ├── raw/                 # DVC-tracked CSVs (not committed to git)
│   ├── interim/             # joined/validated frames
│   ├── processed/           # final feature matrices
│   └── splits/              # temporal split manifests (commit-friendly)
├── configs/                 # YAML: features, training, thresholds
├── src/fraudintel/
│   ├── __init__.py
│   ├── config.py            # load YAML configs
│   ├── data/ingest.py
│   ├── data/validate.py
│   ├── features/build.py
│   ├── features/entities.py # time-safe aggregations
│   ├── models/train.py
│   ├── models/evaluate.py
│   ├── models/threshold.py
│   ├── models/explain.py
│   ├── tracking/mlflow_utils.py
│   └── serving/api.py       # FastAPI app
├── notebooks/               # EDA + experiments ( NOT production logic)
│   └── 01_eda.ipynb
├── tests/                   # mirrors src layout
├── monitoring/
│   ├── prometheus.py        # /metrics instrumentation
│   └── drift.py             # Evidently drift job
├── docker/ Dockerfile
└── docs/                    # interview-facing architecture writeups
```

Notebooks stay separate from `src/`; all reusable logic lives in modules and is
unit-tested. No premature abstraction layers.

## 13. Dependency strategy

- **Single source of truth**: `pyproject.toml` (PEP 621) + `pip-compile`/
  `uv` lockfile. Avoid environment drift.
- **Python**: pin **3.11** (the dev box currently has 3.14, which is too new for
  some ML wheels — we will use a 3.11 venv/conda env for the project).
- Pinned major libs: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `mlflow`,
  `fastapi`, `uvicorn`, `pydantic`, `dvc`, `evidently`, `shap`, `prometheus-client`,
  `prometheus-fastapi-instrumentator`, `pytest`, `ruff`, `mypy`.
- Versions chosen for compatibility, not novelty; revisit only with justification.

## 14. Testing strategy

pytest suite (behavior-focused, not coverage-chasing):
- **features**: known transforms produce expected values; time-safe aggregates
  never use future rows; missing identity handled.
- **validation**: rejects bad schema, out-of-range, broken temporal order.
- **models**: training runs; predict returns scores in [0,1]; threshold logic.
- **API**: schema validation (Pydantic), `/health`, malformed/missing input,
  unknown categories, extreme numeric values, invalid requests.
- **edge cases**: empty batch, all-missing identity row, single-row inference.
- CI runs `ruff`, `mypy`, `pytest` on every push/PR.

## 15. MLOps strategy

- **Experiment tracking**: MLflow. Each run logs dataset version (DVC rev), feature
  config hash, model type, hyperparams, validation strategy, metrics, threshold,
  artifacts, and a model version tag.
- **Pipeline**: `dvc.yaml` stages (ingest → validate → features → train → eval)
  for reproducible re-runs.
- **Model lifecycle / promotion rule** (candidate → production):
  1. Outperforms current production on **PR-AUC (validation)**
  2. Satisfies **minimum precision ≥ P_min** and **recall ≥ R_min** at operating
     threshold (values set after first threshold study)
  3. Passes all tests + data validation
  4. Holds on key slices (identity-present/absent, top `ProductCD`s)
  → only then registered as `Production` in MLflow. No auto-promotion on score alone.

## 16. Deployment strategy

- **First deploy**: single Dockerized FastAPI service on a simple PaaS
  (Railway / Render / Fly.io) or a small cloud VM. No K8s.
- Reproducible build; CI builds the image and runs a **smoke test**
  (`/health` + sample `/predict`) before any deploy.
- Model loaded from MLflow `Production` alias at container start.

## 17. Monitoring strategy

Separate the two concerns:
- **System** (Prometheus + fastapi-instrumentator): request count, latency
  (p50/p95/p99), error rate, throughput. Answers "is the service healthy?"
- **ML / data** (Evidently, periodic batch job): missing-value rate, feature
  distribution shift (PSI / KS), prediction-distribution drift, and
  performance when labels arrive. Answers "is the model still valid?"
- Both feed a dashboard with a *purpose* per panel; no vanity metrics.

## 18. Major risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Temporal leakage | Medium | Critical | chronological split + time-safe aggregates |
| Aggregation/target leakage | High | Critical | train-only stats; code review + tests |
| Severe class imbalance | Certain | High | PR-AUC; class weighting; threshold tuning |
| Identity features absent at inference | High | Medium | train with missingness; serve without them |
| `V*` drift / masked semantics | Medium | Medium | Evidently drift monitoring; keep model robust |
| Model degradation over time | High | High | scheduled retraining + hold-out + drift alerts |
| API failure / latency | Medium | Medium | health checks, smoke tests, Prometheus alerts |
| Reproducibility gaps | Medium | Medium | seeds, pinned deps, DVC, MLflow |
| Environment Python 3.14 too new | Medium | Medium | pin project to Python 3.11 |

## 19. Open technical questions

1. **Dataset access**: Kaggle requires an account + API token; raw CSVs cannot be
   committed. Decision needed: DVC remote = local cache vs S3/GCS bucket.
   *Recommendation: local DVC remote for now; cloud remote if collaboration grows.*
2. **Threshold business costs**: exact false-positive vs false-negative cost ratio
   is unknown. We will first tune for F1 / PR, then let the human supply a
   cost matrix to refine LOW/REVIEW/HIGH bands.
3. **Minimum precision/recall constraints** for promotion: set after the first
   threshold study (Stage 4).
4. **PCA on `V*`**: defer; only if dimensionality hurts training time/materially
   hurts metrics, and only inside a controlled experiment.
5. **Cloud target**: confirm PaaS choice (Railway/Render/Fly) vs VM at deploy stage.

## 20. Recommended implementation order

1. **Repo foundation** — structure, `pyproject.toml`, ruff/mypy, DVC init, CI skeleton.
2. **Data layer** — ingest + join + validation + temporal split manifest.
3. **EDA notebook** — confirm dataset facts, leakage checks, time distribution.
4. **Feature engineering** — groups 1–7 with tests; time-safe aggregates.
5. **Baseline + evaluation** — Logistic Regression, temporal metrics, threshold study.
6. **Candidate model** — XGBoost + class weighting experiment; compare on PR-AUC.
7. **Explainability** — SHAP risk factors in API.
8. **MLflow + promotion rule** — wire tracking + registry.
9. **API service** — FastAPI + Pydantic + `/metrics`.
10. **Docker + deploy + smoke test**.
11. **Monitoring** — Prometheus + Evidently drift job.
12. **Docs + interview writeup**.

---

## A. Architecture proposal (data & inference flow)

**Training / offline flow**
```
Kaggle CSVs ─DVC─▶ raw/ ─▶ ingest(join by TransactionID) ─▶ interim/
   ─▶ validate(schema/nulls/time) ─▶ features(train-only fit, time-safe agg)
   ─▶ temporal split (train/val/hold-out manifests in data/splits)
   ─▶ train(LR + XGBoost, scale-pos-weight) ─▶ evaluate(PR-AUC/precision/recall/F1)
   ─▶ threshold study(LOW/REVIEW/HIGH) ─▶ explain(SHAP) ─▶ MLflow(log+registry)
```

**Inference / online flow**
```
POST /predict {transaction+identity json}
   ─▶ validate request (Pydantic)
   ─▶ feature transform (same fitted pipeline, missing-identity safe)
   ─▶ load MLflow Production model ─▶ score p(fraud)
   ─▶ decision(threshold bands) + SHAP risk factors
   ─▶ response {risk_score, decision, model_version, explanation}
   ─▶ Prometheus counters/latency recorded
```

**Monitoring flow**
```
periodic batch: reference(interim train) vs current(live requests/logged)
   ─▶ Evidently drift report ─▶ alert if PSI/KS exceeds threshold
system: Prometheus scrapes /metrics ─▶ dashboard/alerts
```

## B. Decision log

| # | Decision | Reason | Alternative considered | Why rejected |
|---|---|---|---|---|
| D1 | Temporal split by `TransactionDT`, not random | Fraud is temporal; random split inflates metrics and leaks future | Random 80/20 split | Hides temporal drift; unrealistic for production |
| D2 | Entity aggregates computed train-only, joined forward | Prevents target leakage from group means | Global target-mean encoding on all rows | Leaks label info into features → invalid results |
| D3 | PR-AUC as primary metric | Robust under 3.5% positives; accuracy misleading | Accuracy / ROC-AUC only | Masks poor rare-class performance |
| D4 | Logistic Regression as baseline, XGBoost as candidate | Interpretable floor + strong tabular model; same pipeline for fair compare | Deep NN / LightGBM first | NN adds complexity; compare XGB first, add others only if justified |
| D5 | Single Dockerized FastAPI on PaaS, no K8s | Simplest deploy satisfying requirements | Kubernetes cluster | Over-engineering for this scale/cost |
| D6 | Promotion gate beyond "best score" | Prevents shipping a model that fails slices/precision/recall | Auto-promote highest PR-AUC | Ignores operational constraints + slice safety |
| D7 | Pin Python 3.11 (not dev-box 3.14) | 3.14 too new for some ML wheels; reproducibility | Use system Python 3.14 | Dependency install/runtime risk |
| D8 | `V*` treated as opaque features | Masked semantics; no causal interpretation possible | Attempt to reverse-engineer Vesta features | Not feasible/valid; risk of false claims |
| D9 | DVC local remote initially | Dataset can't be committed; versioning needed without cloud cost | S3 remote from day 1 | Unneeded complexity now; easy to add later |
| D10 | Time-derived features from `TransactionDT` allowed; raw ID/DT not model inputs | Time context is legitimate; unique ID is noise/leak risk | Feed TransactionDT directly | Unique/monotonic; no generalization |

## C. Risk register

See §18 (tabulated). Top critical risks: **temporal leakage**, **aggregation/target
leakage**, **class imbalance**, **model degradation**, **identity-missing at
inference**. Each has a concrete mitigation already designed in.

## D. Implementation roadmap (small, independently committable stages)

- **S0 — Foundation**: repo layout, `pyproject.toml`, ruff/mypy, DVC init, CI skeleton, `.gitignore`. → commit.
- **S1 — Data layer**: `ingest.py` (join), `validate.py`, temporal split manifest. → tests + commit.
- **S2 — EDA**: `notebooks/01_eda.ipynb` confirming time distribution, missingness, leakage probes. → commit.
- **S3 — Features**: `build.py` + `entities.py` (time-safe). → unit tests + commit.
- **S4 — Baseline + eval + threshold**: LR, metrics, threshold bands. → MLflow run + commit.
- **S5 — Candidate**: XGBoost + class-weight experiment; comparison. → MLflow + commit.
- **S6 — Explain**: SHAP integration. → tests + commit.
- **S7 — Tracking + promotion**: MLflow registry + rule. → commit.
- **S8 — API**: FastAPI + Pydantic + `/metrics`. → tests + commit.
- **S9 — Docker + deploy + smoke**: image + PaaS deploy + CI smoke. → commit.
- **S10 — Monitoring**: Prometheus + Evidently drift job. → commit.
- **S11 — Docs**: interview writeup + README. → commit.

## E. Recommended next action

**Implement Stage S0 (Foundation) + S1 (Data layer) next**, because every later
stage depends on a reproducible, validated data path and no modeling should begin
until the temporal split and leakage controls exist (per the project's explicit
"do not build the model yet" rule).

Concretely, the immediate steps are:
1. Scaffold the repository structure and `pyproject.toml` with pinned deps (Python 3.11).
2. Initialize DVC and define the raw-data stage (download/register IEEE-CIS CSVs via Kaggle API; *requires your Kaggle credentials* — see open question #1).
3. Implement `ingest.py` (join transaction+identity) and `validate.py` (schema/null/time checks).
4. Implement the chronological temporal split and write split manifests to `data/splits/`.
5. Add pytest tests for ingest/validate/split and a CI workflow running ruff + mypy + pytest.

I will proceed with S0/S1 once you confirm the **Kaggle dataset access method**
(local download + DVC local remote vs a cloud remote) so I do not block on
credentials. Everything else in this design I will implement as routine work
without further questions.
