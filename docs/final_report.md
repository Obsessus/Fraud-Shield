# Fraud Intelligence Platform — Final Evaluation

*A reproducible, interview-grade summary of the end-to-end fraud-detection system
built on the IEEE-CIS Fraud Detection dataset. The companion, auto-generated
**model card** lives at `data/artifacts/final_evaluation.md` (and `.json`); every
figure there is produced from pipeline artifacts by `scripts/build_report.py`.*

---

## 1. Executive summary

We built a production-oriented fraud-scoring system that takes raw transaction and
identity data and produces, for each transaction, a calibrated fraud-risk score, an
operational decision (allow / review), and the top risk factors driving that decision.
The system is fully reproducible (DVC + MLflow), containerized, served over a
FastAPI API, and monitored for health and data drift.

Headline result: an **XGBoost** model, trained on leakage-safe temporal features,
achieves **hold-out PR-AUC 0.511** (ROC-AUC 0.889) versus **0.195** for a
Logistic-Regression baseline — a +0.316 PR-AUC lift. At the promoted operating
threshold **0.798** the model flags **3.3%** of hold-out traffic at **51% precision
/ 48% recall**. All ten evaluation slices pass the promotion gate.

The single most important finding is a **large performance disparity by identity
availability**: PR-AUC is **0.725** when identity features are present but only
**0.165** when they are absent. This is discussed in §7 and §11.

---

## 2. Problem and objective

Online payment transactions must be scored for fraud risk near-real-time and turned
into an operational decision while controlling both false positives (customer
friction, blocked good transactions) and false negatives (missed fraud). The
objective was to demonstrate a **complete ML engineering lifecycle** — ingestion,
validation, documented feature engineering, a defensible baseline and a stronger
tabular model, temporal evaluation, explainability, model governance, a production
API, containerization/deployment, and monitoring — rather than to maximize a single
leaderboard metric.

---

## 3. Data and dataset understanding

- **Source:** IEEE-CIS Fraud Detection (transaction + identity tables).
- **Scale:** 590,540 training joined rows; 506,691 test joined rows.
- **Class imbalance:** overall fraud rate ≈ **3.5%** (e.g., 14,538 positives in the
  413k-row training split). This drives every modeling and thresholding decision.
- **Identity sparsity:** the identity table covers only **23.8%** of training rows
  (27.0% of test). Identity features are therefore missing for the majority of
  transactions — a first-class constraint, not an afterthought (see §7).

---

## 4. Methodology

### 4.1 Temporal validation (anti-leakage)
Fraud is time-dependent, so we split **chronologically by `TransactionDT`** into
70 / 15 / 15 train / validation / hold-out (boundaries
`t1 = 10437998.1s`, `t2 = 13151845.9s`). Random splitting would leak future
transactions into training and inflate metrics; the temporal split is the honest
evaluation. Validation tunes the threshold; hold-out is touched only once, at
promotion.

### 4.2 Leakage-safe feature engineering
All entity aggregates (e.g., `card1_hist_fraud_rate`, `card1_freq`,
`addr1_hist_fraud_rate`) are computed with **expanding-window cumcount/cumsum that
exclude the current row**. A leakage proof confirms `corr(hist_rate, isFraud) ≈ 0.30`
on the correct (prior-only) computation and **0 mismatches** versus a strict
recompute — i.e., no target information from the same transaction leaks into its
features. Frequency encodings and temporal/amount/identity transforms complete the
416-feature set.

### 4.3 Class-imbalance handling
The baseline uses `class_weight="balanced"`; the candidate uses
`scale_pos_weight="auto"`. Both keep the raw, cost-sensitive decision at a separate
threshold step rather than burying it inside training.

---

## 5. Models

| Model | Features | Train PR-AUC | Val PR-AUC | Hold-out PR-AUC | Hold-out ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | 416 | 0.507 | 0.418 | **0.195** | 0.843 |
| XGBoost (candidate) | 416 | 0.795 | 0.545 | **0.511** | 0.889 |

XGBoost beats the baseline on **every** split. The baseline's hold-out PR-AUC collapse
(0.418 → 0.195) versus a stable ROC-AUC (0.862 → 0.843) is a classic **temporal /
multivariate drift** signal: rank order is preserved while absolute probabilities
shift, which is exactly why a single threshold must be re-tuned on the validation
split and why monitoring matters (§9).

---

## 6. Evaluation and promotion gate

Promotion is gated, not assumed (D7):

- **G1 — beats baseline** on hold-out PR-AUC: ✔ (+0.316)
- **G2 — precision/recall floor** (min precision 0.10, min recall 0.20): ✔
- **G3 — subgroup slice floor** (PR-AUC ≥ 0.05 on 10 slices): ✔ (0 failed)

The operating threshold **0.798** is chosen on validation to maximize F1. At that
point on hold-out:

| Metric | Value |
|---|---|
| Precision | 0.511 |
| Recall | 0.483 |
| F1 | 0.497 |
| Alert rate | 3.29% |
| Confusion (hold-out) | TP 1,490 · FP 1,428 · FN 1,593 · TN 84,070 |

**Interpretation:** at ~3.3% alert volume the model catches ~48% of fraud. The
remaining false negatives are largely the identity-absent population (§7); the false
positives are the cost of a review queue, not automatic blocks.

---

## 7. Subgroup (slice) performance and fairness

Hold-out PR-AUC by slice (all pass the floor):

| Slice | n | Positives | PR-AUC | Status |
|---|---|---|---|---|
| identity_present | 18,668 | 1,763 | 0.725 | pass |
| identity_absent | 69,913 | 1,320 | 0.165 | pass |
| new_card | 964 | 41 | 0.682 | pass |
| known_card | 87,617 | 3,042 | 0.509 | pass |
| weekend | 20,338 | 702 | 0.466 | pass |
| weekday | 68,243 | 2,381 | 0.524 | pass |
| amt_q1 | 22,521 | 1,071 | 0.664 | pass |
| amt_q2 | 21,804 | 507 | 0.422 | pass |
| amt_q3 | 22,296 | 620 | 0.398 | pass |
| amt_q4 | 21,960 | 885 | 0.446 | pass |

**Key fairness/risk finding:** the model is **4.4× stronger when identity data is
present** (0.725 vs 0.165). This is expected — identity signals are genuinely
predictive — but it means the service is materially less protective for the
majority of transactions that lack identity data. This is not a gate failure (both
slices clear the floor) but it is a product and fairness risk that must be owned:
review-queue capacity and any downstream block policy should account for the
weaker identity-absent segment rather than assume uniform coverage.

---

## 8. Explainability

Per-prediction risk factors come from **XGBoost native TreeSHAP**
(`booster.predict(pred_contribs=True)`), chosen because `shap` 0.49 cannot parse the
XGBoost 3.x model format. Global drivers (sampled, n=5,000):

1. `card1_hist_fraud_rate` — historical fraud rate of the payment card
2. `C13`, `C14`, `C1` — card-issuer / amount-related `C` features
3. `TransactionAmt` (and its log / cents transforms)
4. `D2`, `D1` — timedelta-derived features
5. `card1_freq`, `addr1_freq`, `addr1_hist_fraud_rate` — frequency / history aggregates

The top driver being a **prior fraud rate for the same card** is exactly the
intuitive, defensible signal a fraud analyst expects — a good sign the model is
learning real structure rather than spurious artifacts.

---

## 9. Serving and monitoring

- **API (FastAPI + Pydantic):** `GET /health`, `POST /predict`, `POST /predict/batch`.
  The predictor aligns any input to the training columns (missing → 0), applies the
  promoted threshold, and returns `{score, threshold, decision, risk_factors}`.
- **Metrics:** `prometheus-fastapi-instrumentator` plus custom counters
  (`fraud_predictions_total`, `fraud_decisions_total`) and a score histogram
  (`fraud_scores`), exposed at `/metrics`. Verified live in-container.
- **Data drift:** an offline Evidently `DataDriftPreset` check
  (`scripts/drift_check.py`, DVC stage `drift_check`) compares reference training
  features to a current batch. On the train→hold-out split it reports
  `dataset_drift=False` with **97 / 400** columns individually drifted — expected for
  a temporal split and a useful baseline for future production batches.

---

## 10. Reproducibility

```bash
# 1. Environment
python -m venv .venv && .\.venv\Scripts\Activate.ps1
pip install -e ".[base,ml,serving,mlops,dev]"

# 2. Full pipeline (data, features, train, evaluate, promote, explain, drift, report)
dvc repro
python scripts/build_report.py        # regenerates the model card

# 3. Quality gate
ruff check src tests scripts
python -m mypy src
python -m pytest -q

# 4. Serve
docker compose up -d                  # image: fraud-intel-api:local
# /health, /predict, /metrics
```

All experiments are logged to the local MLflow `fraud-intel` experiment; the champion
is registered as `fraud-intel-champion` (alias `champion`).

---

## 11. Limitations and risks

1. **Identity-absent weakness (highest priority).** As shown in §7, recall/precision
   for identity-less transactions is far lower. The product must not assume uniform
   protection.
2. **Train/serve column skew.** The champion was trained on `train_features`, which
   still contains raw identity `id_*` / `n_identity_present` columns that
   `test_features.parquet` drops (all-NaN in the hold-out, pruned by the build step).
   Inference backfills them with 0, so scoring works, but the feature contract should
   be normalized (either drop them from training or guarantee they are produced in
   serving) to remove ambiguity.
3. **Single, static dataset.** Evaluation is one historical window; concept drift in
   production is expected and is the reason for the monitoring layer (§9).
4. **Threshold is a business lever.** 0.798 is F1-optimal on validation, not
   cost-optimal. Real precision/recall trade-offs depend on the dollar cost of fraud
   vs. the cost of review/blocking and should be re-tuned with business inputs.
5. **Review, not block.** The current decision layer emits allow/review; there is no
   automated block and no chargeback/appeal loop.

---

## 12. Ethical considerations

- **Disparate impact by data availability.** The identity-absent slice (the majority
  of traffic) receives weaker protection. This must be tracked as a fairness metric,
  not hidden behind an aggregate score.
- **False-positive cost.** Blocking or heavily reviewing legitimate transactions
  imposes real customer harm; the 3.3% alert rate and review (not block) default are
  deliberate, but should be governed by a documented policy and an appeals path.
- **Provenance.** All data is the IEEE-CIS competition dataset; no PII beyond the
  provided (already partially redacted) transaction/identity fields is introduced.

---

## 13. Future work

- Normalize the train/serve feature contract (item 2 above) and re-validate.
- Add a fairness dashboard tracking the identity-present/absent PR-AUC gap over time.
- Replace the static train→hold-out drift baseline with rolling production-batch
  drift and automated retraining triggers.
- Introduce cost-weighted threshold optimization and an A/B or shadow-deployment
  evaluation before any automated blocking.
- Calibrate scores (e.g., isotonic) so the reported probability is usable for
  dollar-expected-loss ranking, not just ranking.

---

*Generated as part of Stage 18. Numbers are sourced from the DVC-tracked pipeline
artifacts; regenerate with `python scripts/build_report.py`.*
