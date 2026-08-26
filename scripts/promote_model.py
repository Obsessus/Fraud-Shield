"""Model selection / promotion gate (Stage 10).

Usage: python scripts/promote_model.py

Loads the candidate (XGBoost) model, rebuilds temporal features, scores the validation
and hold-out sets, performs a validation-based threshold study, evaluates at the operating
threshold, runs key-slice disparity checks, and applies the promotion gate
(DECISIONS D7). If promoted, the model is registered in the MLflow Model Registry
(aliased "champion"). Writes artifacts/models/promotion_report.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import yaml

from fraudintel.data.paths import artifacts_dir, interim_dir, splits_dir
from fraudintel.features.build import build_temporal_splits
from fraudintel.mlops.selection import (
    evaluate_at_threshold,
    evaluate_gate,
    run_slice_checks,
    select_threshold_by_f1,
)
from fraudintel.models.baseline import load_model, select_features

CONFIG_PATH = Path("configs/thresholds.yaml")
MODEL_PATH = artifacts_dir() / "models" / "xgboost_model.joblib"
BASE_METRICS = artifacts_dir() / "models" / "baseline_metrics.json"
CAND_METRICS = artifacts_dir() / "models" / "xgboost_metrics.json"
REPORT_PATH = artifacts_dir() / "models" / "promotion_report.json"
REGISTRY_NAME = "fraud-intel-champion"


def _build_slices(ho: pd.DataFrame) -> dict[str, object]:
    slices: dict[str, object] = {
        "identity_present": (ho["n_identity_present"] > 0).to_numpy(),
        "identity_absent": (ho["n_identity_present"] == 0).to_numpy(),
        "new_card": (ho["card1_freq"] == 0).to_numpy(),
        "known_card": (ho["card1_freq"] > 0).to_numpy(),
        "weekend": (ho["is_weekend"] == 1).to_numpy(),
        "weekday": (ho["is_weekend"] == 0).to_numpy(),
    }
    amt_q = pd.qcut(ho["TransactionAmt"], 4, labels=["q1", "q2", "q3", "q4"], duplicates="drop")
    for lab in amt_q.cat.categories:
        slices[f"amt_{lab}"] = (amt_q == lab).to_numpy()
    return slices


def _latest_run_id(experiment_name: str, run_name: str) -> str | None:
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        return None
    runs = client.search_runs(experiment_ids=[exp.experiment_id], order_by=["start_time DESC"])
    for r in runs:
        if r.info.run_name == run_name:
            return r.info.run_id
    return None


def main() -> None:
    cfg = yaml.safe_load(Path(CONFIG_PATH).read_text())["promotion"]

    manifest = json.loads((splits_dir() / "split_manifest.json").read_text())
    t1 = float(manifest["thresholds_seconds"]["t1"])
    t2 = float(manifest["thresholds_seconds"]["t2"])

    train = pd.read_parquet(interim_dir() / "train_joined.parquet")
    tr_feat, va_feat, ho_feat = build_temporal_splits(train, t1, t2)
    X_tr, y_tr = select_features(tr_feat)
    X_va, y_va = select_features(va_feat)
    X_ho, y_ho = select_features(ho_feat)
    X_va = X_va.reindex(columns=X_tr.columns, fill_value=0)
    X_ho = X_ho.reindex(columns=X_tr.columns, fill_value=0)

    model = load_model(MODEL_PATH)
    val_scores = model.predict_proba(X_va)[:, 1]
    ho_scores = model.predict_proba(X_ho)[:, 1]

    study = select_threshold_by_f1(y_va, val_scores)
    threshold = study["threshold"]
    val_eval = evaluate_at_threshold(y_va, val_scores, threshold)
    ho_eval = evaluate_at_threshold(y_ho, ho_scores, threshold)

    slices = _build_slices(ho_feat)
    slice_results = run_slice_checks(
        y_ho,
        ho_scores,
        slices,
        min_positives=int(cfg["slice_min_positives"]),
        pr_auc_floor=float(cfg["slice_pr_auc_floor"]),
    )

    base_metrics = json.loads(BASE_METRICS.read_text())
    cand_metrics = json.loads(CAND_METRICS.read_text())
    gate = evaluate_gate(
        candidate_holdout_pr_auc=cand_metrics["holdout"]["pr_auc"],
        baseline_holdout_pr_auc=base_metrics["holdout"]["pr_auc"],
        holdout_eval=ho_eval,
        slice_results=slice_results,
        cfg=cfg,
    )

    registration: dict[str, Any] = {"attempted": False, "status": "not_promoted"}
    if gate["promoted"]:
        registration["attempted"] = True
        try:
            run_id = _latest_run_id("fraud-intel", "xgboost")
            if run_id:
                mv = mlflow.register_model(f"runs:/{run_id}/model", REGISTRY_NAME)
                registration.update(
                    {"status": "registered", "run_id": run_id, "version": mv.version}
                )
                client = mlflow.tracking.MlflowClient()
                try:
                    client.set_registered_model_alias(REGISTRY_NAME, "champion", mv.version)
                    registration["alias"] = "champion"
                except Exception as exc:  # alias API differences across mlflow versions
                    registration["alias_error"] = str(exc)
            else:
                registration["status"] = "run_not_found"
        except Exception as exc:
            registration.update({"status": "error", "error": str(exc)})

    report = {
        "model": "xgboost",
        "operating_threshold": threshold,
        "threshold_study": {k: study[k] for k in ("precision", "recall", "f1")},
        "validation_at_threshold": val_eval,
        "holdout_at_threshold": ho_eval,
        "slice_checks": slice_results,
        "gate": gate,
        "registration": registration,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print(f"[promote] operating threshold={threshold:.4f}  "
          f"hold-out P={ho_eval['precision']:.3f} R={ho_eval['recall']:.3f}")
    print(f"[promote] gate: G1={gate['g1_beat_baseline']} G2={gate['g2_precision_recall']} "
          f"G3={gate['g3_slices']} -> promoted={gate['promoted']}")
    print(f"[promote] registration: {registration['status']}")
    print(f"[promote] -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
