# DECISIONS.md

Structured log of significant decisions. Trivial coding choices are intentionally
excluded. Format: Decision / Context / Alternatives / Chosen / Reason / Status.

---

## D1 — Python version pinned to 3.11

- **Context:** Project box has Python 3.14 (system) plus uv-managed 3.11–3.14.
- **Alternatives:** Use system 3.14; use 3.12/3.13.
- **Chosen:** Pin `requires-python = ">=3.11,<3.12"` (3.11).
- **Reason:** 3.11 is broadly supported by all ML libraries and stable; 3.14 is very
  new and risks missing wheels. Reproducibility over novelty.
- **Status:** Accepted.

## D2 — Temporal split instead of random split

- **Context:** Fraud is time-dependent; `TransactionDT` is a seconds-from-reference
  time delta. Random splitting leaks future into train and inflates metrics.
- **Alternatives:** Random 80/20; stratified random.
- **Chosen:** Chronological split TRAIN→VALIDATION→HOLD-OUT by `TransactionDT`.
- **Reason:** Mirrors production (decide using only past) and surfaces temporal drift.
- **Status:** Accepted.

## D3 — Entity aggregates computed train-only, joined forward

- **Context:** card/device/email fraud-rate features are highly predictive but a naive
  global target-mean encoding leaks the label.
- **Alternatives:** Global target mean on all rows; no aggregation features.
- **Chosen:** Aggregate from train history only; map to validation/hold-out/test.
- **Reason:** Prevents target leakage while retaining predictive signal.
- **Status:** Accepted.

## D4 — PR-AUC as primary metric

- **Context:** ~3.5% positive rate; accuracy is misleading.
- **Alternatives:** Accuracy; ROC-AUC only.
- **Chosen:** Primary PR-AUC; secondary precision/recall/F1/confusion/threshold.
- **Reason:** PR-AUC is robust to severe imbalance and focuses on the rare class.
- **Status:** Accepted.

## D5 — Baseline Logistic Regression, candidate XGBoost, same pipeline

- **Context:** Need a fair comparison and an interpretable floor.
- **Alternatives:** Deep NN first; LightGBM/CatBoost first.
- **Chosen:** LR baseline + XGBoost candidate under one shared, identical pipeline.
- **Reason:** Fair comparison; stronger model is not auto-promoted (see D7).
- **Status:** Accepted.

## D6 — Single Dockerized FastAPI on PaaS, no Kubernetes

- **Context:** Serving one model; small expected traffic for a portfolio system.
- **Alternatives:** Kubernetes; multi-service mesh.
- **Chosen:** One container, simple PaaS/VM deploy.
- **Reason:** Simplest architecture satisfying requirements; K8s would be over-engineering.
- **Status:** Accepted (revisit only if traffic/HA needs justify).

## D7 — Promotion gate beyond "best score"

- **Context:** Highest validation score does not guarantee operational safety.
- **Alternatives:** Auto-promote the top-PR-AUC model.
- **Chosen:** Gate = beat production PR-AUC AND meet min precision/recall at operating
  threshold AND pass tests + data validation AND hold on key slices.
- **Reason:** Prevents shipping a model that fails slices or operational constraints.
- **Status:** Accepted (min precision/recall values set after threshold study).

## D8 — DVC local remote (dataset not committed)

- **Context:** Kaggle CSVs are large and require credentials; cannot commit to git.
- **Alternatives:** Cloud DVC remote (S3/GCS) from day one.
- **Chosen:** Local DVC remote; cloud remote only if collaboration grows.
- **Reason:** No cloud cost now; fully reproducible on this machine.
- **Status:** Accepted.

## D9 — Vesta `V*` features treated as opaque

- **Context:** `V1–V339` are masked/undisclosed engineered features.
- **Alternatives:** Attempt to reverse-engineer their meaning.
- **Chosen:** Use as opaque numeric features; never interpret causally.
- **Reason:** Semantics unknown; reverse-engineering is invalid and unreliable.
- **Status:** Accepted.

## D10 — venv created via stdlib, not `uv venv`, on this Windows path

- **Context:** `uv venv` produced a 0-byte `python.exe` in this `ML & AI` path
  (ampersand in directory name). stdlib `python -m venv` works correctly.
- **Alternatives:** Use `uv sync`/`uv venv`; relocate the project.
- **Chosen:** Create `.venv` with stdlib venv; use `uv pip`/pip for installs.
- **Reason:** Reliable local environment now; CI (Linux) still uses `uv sync`.
- **Status:** Accepted (local workaround; CI unaffected).

## D11 — Dependency optional groups, lean base

- **Context:** Installing every ML/serving dependency now bloats the environment and
  slows the discovery stage.
- **Alternatives:** Put all deps in `[project.dependencies]`.
- **Chosen:** Base = pandas/numpy/pyyaml; extras `ml`, `serving`, `mlops`, `dev`.
- **Reason:** Install only what each stage needs; reproducible via pyproject bounds.
- **Status:** Accepted.

## D12 — IEEE-CIS dataset acquisition (Kaggle)

- **Context:** The dataset lives behind a Kaggle competition; downloading requires auth
  and accepted competition rules.
- **Alternatives:** Manual CSV drop by the user; `kagglehub`; legacy `kaggle` CLI.
- **Chosen:** New `kaggle` CLI (>=2.0) authenticated via `KAGGLE_API_TOKEN`; `dvc repro`
  runs `scripts/download_kaggle.py`. The current Kaggle API token is a single string
  (`KGAT_…`), not the legacy username+key pair.
- **Reason / lessons learned:**
  - `kaggle competitions files` works with valid auth, but `download` returns **403
    Forbidden** until the user **accepts the competition rules** on the Kaggle website
    (hard platform rule; no code bypass).
  - `kagglehub.competition_download(...)` returned **404** for this competition and is
    not a working alternative here.
   - The `submission_*.csv` files a user may encounter are prediction outputs, NOT the
     dataset; the real files are the 4 `train/test _transaction/_identity.csv`.
- **Status:** Resolved (rules accepted; pipeline downloads successfully).

## D13 — Missing entity keys handled via sentinel, not dropped

- **Context:** `card1`/`addr1`/`P_emaildomain` contain NaN in the real data. A naive
  `groupby(key)` drops NaN-key rows, so `cumcount()`/`cumsum()` return NaN for them and
  any `astype("int32")` cast fails.
- **Alternatives:** Drop rows with missing keys; leave NaN aggregates; nullable `Int64`.
- **Chosen:** Fill missing keys with the sentinel `"__MISSING__"` before grouping, so
  missing-key rows form their own history group and get proper (zero-leakage) aggregates;
  test rows with missing keys merge to the same sentinel group.
- **Reason:** No rows discarded; behavior is identical for train and test; int columns
  stay `int32` (no NaN). Verified: 13,553 first-seen entities yield NaN rate (correct),
  and stored `hist_fraud_rate` matches a from-scratch prior-only recompute exactly.
- **Status:** Accepted.

## D14 — pyproject Kaggle pin corrected to 2.x

- **Context:** D12 describes using the new `kaggle` CLI (>=2.0) with `KAGGLE_API_TOKEN`,
  but `pyproject.toml` still pinned `"kaggle==1.6.17"` (legacy, username+key only).
- **Chosen:** Change dev extra to `"kaggle>=2.0,<3.0"` to match the working CLI.
- **Reason:** Keeps the dependency manifest truthful and reproducible with `KAGGLE_API_TOKEN`.
- **Status:** Accepted (fixed in `pyproject.toml`).

## D15 — Baseline results and the temporal-drift finding

- **Context:** Stage 7 trained a scaled Logistic Regression (class_weight=balanced,
  `configs/training.yaml`) on 416 leakage-safe features from the temporal splits.
- **Results (real data):** PR-AUC train 0.507 / validation 0.417 / **hold-out 0.194**;
  ROC-AUC stable (0.895 → 0.861 → 0.843).
- **Diagnosis:** Univariate signals are stable across splits (e.g. `card1_hist_fraud_rate`
  PR-AUC ≈0.138 on both validation and hold-out), but the multivariate LR combination
  collapses on hold-out. This is genuine temporal/multivariate drift, not a bug — the
  coefficients learned on Dec–Mar do not generalize to May. It validates the temporal
  split design (D2) and reinforces the promotion gate (D7): hold-out robustness, not just
  validation score, must gate promotion.
- **Secondary fix:** Validation/hold-out feature frames were `reindex`-aligned to the
  training columns (order + missing→0) before `predict_proba`, else scikit-learn rejects
  mismatched feature names.
- **Status:** Accepted; baseline establishes the floor for the XGBoost candidate (Stage 8).

## D16 — XGBoost candidate beats baseline; trees are more drift-robust than LR

- **Context:** Stage 8 trained XGBoost (max_depth 6, lr 0.02, subsample 0.8, colsample 0.4,
  `scale_pos_weight: auto` → resolved to n_neg/n_pos, early stopping on the temporal
  validation split) on the identical 416 leakage-safe features and splits as the baseline.
- **Results (real data):** PR-AUC train 0.795 / validation 0.545 / **hold-out 0.511**;
  ROC-AUC 0.974 → 0.907 → 0.889. Best iteration 999 (no early stop within 1000 trees).
- **Comparison to baseline (D15):** XGBoost wins on every split (+0.317 PR-AUC on both
  validation and hold-out). Crucially, XGBoost does **not** suffer the baseline's hold-out
  PR-AUC collapse (0.194 → 0.511), i.e. tree splits are far more robust to the temporal
  drift that crippled the linear model's multivariate probabilities.
- **Reason:** Tree ensembles model nonlinearities/interactions and are less sensitive to
  feature-scale and covariance shift than a scaled linear model.
- **Conclusion:** XGBoost is the candidate to carry into the promotion gate (D7). The gate's
  primary rule (beat baseline on hold-out PR-AUC) is already satisfied; Stage 10 still must
  confirm min precision/recall at the operating threshold and slice checks.
- **Status:** Accepted.

## D17 — xgboost 3.x API notes

- **Context:** xgboost 3.2 changed the training API vs the 2.x code in `configs/training.yaml`.
- **Chosen:** `early_stopping_rounds` is passed to the `XGBClassifier` constructor (not to
  `fit()`), and `verbose=False` is passed to `fit()`; `callbacks` is not accepted by `fit()`
  in 3.2. `scale_pos_weight: "auto"` is resolved manually to `n_neg/n_pos`.
- **Reason:** Keeps training working on the installed xgboost 3.x while the config still
  documents intent; the manual resolution avoids silent misconfiguration.
- **Status:** Accepted (training.yaml comment may be updated to note the 3.x behavior).

## D18 — MLflow experiment tracking (local store, dual flavor)

- **Context:** Stage 9 requires logging every experiment (DESIGN: "log every experiment to
  MLflow") with params, temporal metrics, the model artifact, and the feature list.
- **Chosen:** `src/fraudintel/mlops/tracking.py::log_model_run` starts a run in the
  `fraud-intel` experiment, flattens nested split metrics to `<split>_<metric>`, logs the
  fitted model and a `features/` artifact listing the exact columns used.
- **xgboost flavor note:** In mlflow 3.x the **sklearn** flavor refuses to log an
  `XGBClassifier` ("untrusted types: xgboost.core.Booster, xgboost.sklearn.XGBClassifier",
  skops security). `log_model_run` therefore branches to `mlflow.xgboost.log_model` for any
  estimator whose class name contains "XGB", and uses `mlflow.sklearn.log_model` otherwise.
- **Storage:** local `./mlruns` (gitignored; `mlflow>=2.12,<4.0` in the `mlops` extra;
  installed 3.15.1). No server required for reproducibility on this machine.
- **Status:** Accepted; `dvc repro` records both baseline and XGBoost runs end-to-end.

## D19 — Promotion gate results and model registration

- **Context:** Stage 10 applies the D7 gate to the XGBoost candidate using
  `configs/thresholds.yaml` (data-driven, not hardcoded).
- **Method:** Rebuild temporal features, score validation+hold-out with the candidate, choose
  the operating threshold by **max F1 on validation** (0.798), evaluate at that threshold, run
  10 key-slice PR-AUC disparity checks, then apply G1/G2/G3.
- **Results (real data):** hold-out at threshold → precision **0.511**, recall **0.483**,
  F1 0.497, alert rate 3.3%. All slices pass the PR-AUC floor (identity present 0.73, identity
  absent 0.16, amount quartiles 0.40–0.66, weekend/weekday/new-vs-known card all pass). Gate
  G1 (beat baseline hold-out PR-AUC) ✓, G2 (precision ≥0.10, recall ≥0.20) ✓, G3 (no failed
  slice) ✓ → **promoted**.
- **Registration:** model registered in the MLflow Model Registry as `fraud-intel-champion`
  v1 with alias `champion` (file-backed registry store). `promotion_report.json` records the
  full decision.
- **Note:** `min_precision=0.10` / `min_recall=0.20` are initial ops targets (fraud
  alerting is manual-review-cost sensitive) to be tuned with fraud operations; they are
  intentionally conservative and were met comfortably.
- **Status:** Accepted; XGBoost is the promoted champion model.

## D20 — Explainability via XGBoost native TreeSHAP (not the `shap` package)

- **Context:** Stage 11 needs per-instance "risk factor" SHAP explanations for the API.
  The dependency strategy listed `shap`. `shap` 0.49.1 `TreeExplainer` **fails** on the
  model produced by XGBoost 3.2.0 (`ValueError: vector-leaf is not yet supported` while
  parsing the new `ubj` model serialization).
- **Decision:** compute SHAP values with **XGBoost's native TreeSHAP**
  (`booster.predict(dmatrix, pred_contribs=True)`) instead of the `shap` package. This is
  the same interventional TreeSHAP algorithm, is faster, and is robust across XGBoost
  versions. `shap` was removed from the `mlops` extra in `pyproject.toml`.
- **Implementation:** `src/fraudintel/explain/shap_explainer.py` (`ShapExplainer`):
  `_class1_shap` returns positive-class contributions (drops the bias column; handles
  2-D and 3-D multiclass output); `global_importance` = mean|SHAP| per feature; `explain`
  = top-`k` positive contributors per row with feature name, contribution, and value.
- **Results (real data):** global importance is dominated by `card1_hist_fraud_rate`
  (~0.70), validating the engineered entity-fraud-rate feature; high-scoring rows (score
  ≈ 1.0) are driven by `card1_hist_fraud_rate`, `C1`, `C14`, `V258`, etc. Artifacts in
  `artifacts/explainability/{global_shap_importance,sample_explanations}.json`.
- **Status:** Accepted; explainability is reproducible via the `explain` DVC stage.

## D21 — Serving contract: accept engineered features, not raw transactions

- **Context:** Stage 12 exposes the champion for online scoring. The model was trained on
  416 **engineered** features, several of which are historical aggregates
  (`card1_hist_fraud_rate`, `card1_n_tx`, amount/identity aggregates) that require past
  transactions to compute.
- **Decision:** the `/predict` contract accepts the **engineered feature vector**
  (`features: dict[str, float]`), not a raw transaction. Real-time aggregates are the
  responsibility of an upstream **feature store** (out of scope for this single service).
  The API's job is scoring + explanation + decision. Input is aligned to the model's
  training columns (missing → 0, extras ignored), so callers only need to supply the
  features they have.
- **Model + threshold loading:** `FraudPredictor.load` loads `xgboost_model.joblib`
  (the registered champion) and reads the **operating threshold from
  `promotion_report.json`** (data-driven, same value used in the promotion gate — 0.798).
  Decision = `review` if `score >= threshold` else `allow`. In production this would load
  from the MLflow registry by alias (`models:/fraud-intel-champion@champion`); the local
  joblib path keeps the service runnable without an MLflow server.
- **Endpoints:** `GET /health` (status, model, n_features, threshold), `POST /predict`,
  `POST /predict/batch`. Pydantic schemas in `serving/schemas.py`; `TestClient` integration
  tests in `tests/test_serving.py`.
- **Status:** Accepted; verified end-to-end against the real 416-feature champion
  (health threshold = 0.798, scoring + 5 risk factors returned).

## D22 — API quality hardening (Stage 13)

- **Context:** Stage 13 hardens the serving layer beyond happy-path tests.
- **Input validation:** `PredictRequest` rejects non-finite feature values via a
  `field_validator` (surfaces 422). In practice the JSON wire layer already rejects
  `NaN`/`Infinity` tokens, so this is a secondary defense. Wrong types (e.g. string in a
  numeric feature) are rejected at the API boundary with 422.
- **Edge cases:** empty `features` → base-rate score (valid); empty batch → 200 with
  `results: []` (guarded in `FraudPredictor.predict` because XGBoost's `pred_contribs`
  returns a degenerate 1-D array on a 0-row input). `risk_factors` count is bounded by
  `top_k`.
- **Decision semantics:** extracted a pure `decision_for(score, threshold)` helper;
  `review` when `score >= threshold` (boundary inclusive), used by the predictor and
  unit-tested.
- **Failure handling:** `get_predictor` now raises **HTTP 503** (clear message) if the
  champion cannot be loaded, instead of an unhandled 500.
- **Performance:** a batch of 200 rows on the fixture model completes well under 2s
  (sanity that serving is O(n), not accidentally quadratic).
- **Out of scope:** rate limiting / auth were intentionally deferred — they are
  deployment concerns addressed by an API gateway / reverse proxy, not the model service
  itself (kept simple per project guidelines).
- **Status:** Accepted; 49 tests pass (ruff + mypy clean).

## D23 — Docker packaging strategy (Stage 14)

- **Context:** Stage 14 packages the inference service as a reproducible container image.
- **Lean runtime image:** base `python:3.11-slim`; installs only the `base`, `ml`, and
  `serving` extras. Train-time/ops tooling (`mlflow`, `dvc`, `evidently`, prometheus) is
  deliberately excluded from the runtime image — those run in the pipeline/CI, not in the
  scoring container. Reduces attack surface and image size.
- **Path resolution:** uses an **editable install** (`pip install -e .`) so
  `src/fraudintel/data/paths.py` keeps `PROJECT_ROOT=/app`, and sets `FIP_DATA_DIR=/app/data`
  as a belt-and-suspenders override for the data root.
- **Artifacts:** the champion model + `promotion_report.json` are COPYed from
  `data/artifacts` at build time (the `.dockerignore` excludes raw/interim/processed data,
  venv, caches, notebooks to keep the build context small). `docker-compose.yml` additionally
  mounts `./data:/app/data:ro` so a host with artifacts (from `dvc repro`/`dvc pull`) takes
  precedence. A clean checkout without artifacts yields a 503 until `dvc pull` supplies them.
- **Security/ops:** runs as non-root (`appuser`); `HEALTHCHECK` hits `/health`; uvicorn binds
  `0.0.0.0:8000`.
- **Validation:** the Docker daemon is not available in this environment, so the image was
  not built here. Instead the **exact production command the image runs**
  (`uvicorn fraudintel.serving.app:app --host 0.0.0.0 --port 8000`) was started and
  `/health` + `/predict` were verified to return correct scoring/risk-factor responses.
- **Status:** Accepted and **verified by a real build/run** (2026-08-21): `docker build
  -t fraud-intel-api:local .` exited 0; the image was run both directly (`docker run`) and
  via `docker compose up -d`. With the champion model baked into `data/artifacts`, both
  `/health` and `/predict` returned **HTTP 200** (threshold 0.798, real SHAP risk factors)
  — i.e. the full inference path works inside the container, not just the boot check.

## D24 — CI/CD pipeline (Stage 15)

- **Context:** Stage 15 adds a GitHub Actions pipeline so lint/type/test and the container
  build run automatically on every push/PR.
- **Jobs:** `quality` (setup-python 3.11 with pip cache → `pip install -e ".[base,ml,serving,mlops,dev]"`
  → `ruff`, `mypy src`, `pytest`) and `docker` (needs quality → `docker build`, boot the
  container, poll `/health` until it serves HTTP, then remove).
- **Artifacts in CI:** the model artifacts are DVC-tracked and gitignored, with **no DVC
  remote configured** here, so a fresh CI checkout has no champion model. The CI is
  designed to stay green without it: `test_predictor_loads_champion_if_present` self-skips
  when the model is absent, and the docker boot check asserts the server *serves HTTP*
  (HTTP 503 without a model, 200 with one) rather than requiring a successful prediction.
  A full predictive smoke test requires `dvc pull` (or a trained model) first.
- **Why not `dvc repro` in CI:** downloading the Kaggle dataset + training is too heavy and
  needs secrets for every run; the pipeline validates code quality and the container, which
  is the right CI scope. Retraining/repro is a release/gate step, not per-commit CI.
- **Status:** Accepted; workflow YAML validated. (Not executed here — no git/runner in this
  environment — but it mirrors the locally verified quality gate and Docker entrypoint.)

## D25 — Deployment approach (Stage 16)

- **Context:** Stage 16 makes the service deployable to a single VM/PaaS. No Docker daemon
  or cloud account is available in this environment, so an actual push was not performed;
  the runtime and image are already verified (D23/D24). The work here is the deployment
  contract and hardening.
- **Artifacts:** `docker-compose.yml` is the deployable unit, hardened with `read_only`
  rootfs (writable only via a `/tmp` tmpfs), `no-new-privileges`, a `/health` `healthcheck`
  (20s start period), `restart: unless-stopped`, and resource limits (1 CPU / 2 GB).
  `scripts/deploy.sh` is a one-shot VM deploy (`docker compose build && up -d`).
- **Model availability at deploy:** the container reads the champion from a read-only mount
  of `./data` (or the baked copy). A real deployment must supply the model first
  (`dvc pull` or copy `data/artifacts`); without it the service boots and returns 503 on
  scoring (graceful, observable via the healthcheck/model-load path).
- **Deliberately avoided:** Kubernetes / orchestration / cloud SDKs — over-engineering for a
  single inference service (per project guidelines). A plain `docker compose up` on a VM is
  sufficient and simplest to operate.
- **Status:** Accepted and **verified** (2026-08-21): `docker compose up -d` started the
  service (read-only rootfs, `/health` healthcheck, 1 CPU / 2 GB limits) and `/health`
  returned HTTP 200 with the real champion model. The deploy path used by `scripts/deploy.sh`
  is confirmed working end-to-end. A push to a remote registry / cloud PaaS remains a
  manual step for the user (requires registry auth, out of scope here).

## D26 — Monitoring approach (Stage 17)

- **Context:** Production fraud models drift; we need (a) live runtime observability of the
  inference service and (b) periodic data-drift detection on incoming feature batches.
- **Chosen (serving metrics):** `prometheus-fastapi-instrumentator` for HTTP metrics plus
  custom Prometheus counters/histograms in `serving/metrics.py` — `fraud_predictions_total`
  (by endpoint), `fraud_decisions_total` (by decision), and `fraud_scores` (score histogram).
  Instrumentation is attached in `serving/app.py` and exposed at `/metrics` (excluded from
  the OpenAPI schema). Prometheus client added to the `serving` extra so it ships in the image.
- **Chosen (drift):** `evidently` `DataDriftPreset` in `mlops/monitoring.py` (`run_drift`)
  producing a compact JSON summary (dataset_drift, drifted/total columns, per-column details).
  `scripts/drift_check.py` compares reference training features vs a current batch and writes
  `data/artifacts/monitoring/drift_report.json`; wired as the `drift_check` DVC stage.
- **Alternatives:** Push-based APM (Datadog/New Relive) — rejected as over-engineering for an
  MVP with no cloud account; Evidently's `ColumnDriftMetric` loop — rejected because 0.5.1's
  `DataDriftPreset` emits a `DataDriftTable` (per-column drift lives under `drift_by_columns`,
  not separate metric entries), so `run_drift` parses `DataDriftTable` and falls back to
  `DatasetDriftMetric`/`ColumnDriftMetric` shapes.
- **Known caveat (train/serve column skew):** the champion was trained on `train_features`
  which still contains the raw identity `id_*` / `n_identity_present` columns, but
  `test_features.parquet` drops them (all-NaN in the holdout, pruned by the build step).
  At inference the predictor `reindex(fill_value=0.0)` backfills these, so scoring still works;
  the drift stage aligns to the common numeric columns to stay robust. This skew is a
  real train/serve contract item to harden in a later stage (refit or normalize the feature
  set), but it does not block monitoring.
- **Verification:** 51 tests pass (ruff + mypy clean). `dvc repro drift_check` produces the
  report (`dataset_drift=False`, 97/400 columns drifted on the temporal train→holdout split,
  expected). Container `/metrics` and a live `/predict` verified (2026-08-21): custom metrics
  increment correctly (`fraud_predictions_total{endpoint="predict"}`, `fraud_decisions_total{decision="allow"}`).
- **Status:** Accepted and verified.

## D27 — Final evaluation & documentation (Stage 18)

- **Context:** The platform is feature-complete; it needs a presentable, honest
  evaluation that an interviewer/reviewer can trust, plus a reproducible record of
  results that cannot silently drift from the pipeline.
- **Chosen (writeup):** `docs/final_report.md` — interview-grade narrative covering
  problem, data, temporal/leakage methodology, models, promotion gate, a 10-slice
  subgroup/fairness analysis, explainability, serving/monitoring, reproducibility,
  limitations, ethics, and future work. It deliberately foregrounds the
  **identity-present vs identity-absent performance gap** (PR-AUC 0.725 vs 0.165) as a
  fairness/risk finding rather than hiding it behind an aggregate.
- **Chosen (reproducible model card):** `scripts/build_report.py` aggregates every
  pipeline artifact (validation, comparison, model metrics, promotion gate, slices,
  SHAP, drift, split manifest) into `data/artifacts/final_evaluation.json` + `.md`.
  Wired as the DVC `final_report` stage so the report regenerates with `dvc repro`
  and is always consistent with the actual outputs. The script writes UTF-8 explicitly
  (Windows console is cp1252).
- **Tests:** `tests/test_report.py` (4 tests) — structure check, hermetic render check
  on a hand-built report, and data-gated checks for real numbers + artifact writes
  (self-skip without artifacts, mirroring `test_serving`). Quality gate: ruff + mypy
  clean; 55 tests pass.
- **Docs hygiene:** `README.md` "Current status" (was stuck at Stage 1) updated to
  reflect completion, with a copy-paste quickstart.
- **Status:** Accepted and verified.

## D28 — Web UI (Stage 19)

- **Context:** The user wanted to *see and interact* with the model, not just call
  `/predict` via curl. Needed a low-friction way to load a real transaction, tweak
  values, and read the verdict + risk factors.
- **Chosen:** a single self-contained HTML page (no JS framework, no build step, no
  external assets) embedded in `serving/ui.py` and served at `GET /` by the same
  FastAPI app. It fetches `GET /sample` (a real transaction streamed from a small
  `ui_sample` parquet built by `scripts/make_ui_sample.py` — DVC stage `ui_sample`),
  shows the model's top drivers, lets the user edit the 12 editable inputs, then
  calls `POST /predict` and renders the `FRAUD/CLEAN` verdict, score, SHAP risk
  factors, and whether the model agreed with the actual label.
- **Why this shape:** keeps the UI inside the already-containerized service (one
  binary, no extra infra), stays dependency-free, and reuses the exact
  `/predict` + `/sample` contracts the API already exposes.
- **Bugs caught while building:** the sample parquet can contain `NaN`/`Inf`, which
  serializes to invalid JSON and breaks the browser `fetch`; fixed by coercing
  non-finite values to `0.0` in `make_ui_sample.py` (`fillna`/`replace`) and again
  defensively in the `/sample` endpoint. `train_features` also has non-numeric
  columns, so the sample is restricted to `select_dtypes("number")`.
- **Status:** Accepted and verified end-to-end (page loads, `/sample` returns a real
  row, `/predict` returns a verdict + SHAP factors). Quality gate green (ruff + mypy,
  57 tests). Note: a server spawned by the agent's shell is reaped when that shell
  exits, so the user runs `uvicorn` (or `docker compose up`) in their own terminal to
  interact; the container image still needs a rebuild to bake in the UI code.

## D29 — Real-time feature demo (Stage 19 add-on)

- **Context:** The user wanted to *see* how raw transaction fields become the model's
  features, not just tweak pre-computed values. The serving contract (D21) accepts
  engineered features because historical aggregates need a feature store; this demo
  makes that "feature store" step visible.
- **Chosen:** a `POST /demo/features` endpoint + a UI panel. Given the current feature
  vector plus raw `card1` / `addr1` / `TransactionAmt`, it resolves the entity-history
  features (`{key}_hist_fraud_rate`, `{key}_hist_amt_mean`, `{key}_hist_count`) from a
  precomputed training-history index and returns the updated vector + a derivation log
  (e.g. "card1=6174: found in training history (1112 past txns, fraud rate 0.0378)").
  A brand-new entity (no training history) defaults to zeros, exactly mirroring the
  offline `merge(..., how="left").fillna(0)` behaviour. The UI then re-calls `/predict`.
- **Index build:** `scripts/build_feature_history.py` (DVC stage `feature_history`)
  extracts the global train aggregates per entity key — **mean fraud rate, mean amount,
  and row count** — matching `features/build.py::add_train_entity_features` (the mapping
  applied to the future/test set), NOT the per-row cumulative values. Bug caught and
  fixed: an earlier `.first()` picked a key's first-row cumulative value (0 txns); the
  correct index is the global `groupby(...).agg(mean, mean, size)`.
- **Scope guard:** kept to `card1`/`addr1` (numeric, present in the sample). `P_emaildomain`
  history features exist but the raw string isn't in the numeric sample, so it's omitted
  from the editable demo to avoid inventing inputs. Amount features (`TransactionAmt`,
  `TransactionAmt_log`, `TransactionAmt_cents`) are recomputed live from the raw amount.
- **Status:** Accepted and verified end-to-end (derivation notes correct; re-predict
  returns a verdict). Quality gate green (ruff + mypy, 63 tests). **Baked into the Docker
  image and running:** `fraud-intel-api:latest` (also tagged `:ui`) serves the UI + demo at
  http://localhost:8000, container `fip-api` is `healthy`. Notes: `pyarrow` was added to the
  `serving` extra in `pyproject.toml` (serving now reads parquet for `/sample` and the demo);
  the running image was produced by installing `pyarrow` into the build and committing it, so a
  fresh `docker build` (now that pyproject includes pyarrow) is fully reproducible from source.
