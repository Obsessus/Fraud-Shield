---
title: Fraud Shield
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
short_description: Real-time card fraud risk scoring with explanations and a live feature-engineering demo
---

# Fraud Shield

A production-style fraud detection service built on the IEEE-CIS dataset.

- Scores an online payment transaction for fraud risk in real time.
- Returns a risk score, an operational decision (LOW RISK / REVIEW / HIGH RISK),
  and the top risk factors driving the prediction.
- Includes an interactive demo: edit `card1` / `addr1` / `TransactionAmt` and see
  the historical aggregates and the re-scored prediction update live.

The model, API, UI, and monitoring were built as a full ML engineering lifecycle
(ingest → leakage-safe features → temporal validation → XGBoost champion →
FastAPI service → containerized deployment).
