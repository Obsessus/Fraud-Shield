"""Train the XGBoost candidate and evaluate on temporal splits.

Usage: python scripts/train_xgboost.py
Mirrors scripts/train_baseline.py so the two models are compared on identical features/splits.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from fraudintel.data.paths import artifacts_dir, interim_dir, splits_dir
from fraudintel.features.build import build_temporal_splits
from fraudintel.mlops.tracking import log_model_run
from fraudintel.models.candidate import evaluate, save_model, select_features, train_xgboost

CONFIG_PATH = Path("configs/training.yaml")
MODEL_PATH = artifacts_dir() / "models" / "xgboost_model.joblib"
METRICS_PATH = artifacts_dir() / "models" / "xgboost_metrics.json"


def main() -> None:
    cfg = yaml.safe_load(Path(CONFIG_PATH).read_text())
    seed = int(cfg.get("random_seed", 42))
    model_cfg = cfg["models"]["xgboost"]
    params = dict(model_cfg.get("params", {}))

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

    clf = train_xgboost(X_tr, y_tr, X_val=X_va, y_val=y_va, seed=seed, params=params)

    metrics = {
        "config": {"model": "xgboost", "params": params, "seed": seed},
        "n_features": int(X_tr.shape[1]),
        "best_iteration": int(getattr(clf, "best_iteration", -1)),
        "train": evaluate(clf, X_tr, y_tr),
        "validation": evaluate(clf, X_va, y_va),
        "holdout": evaluate(clf, X_ho, y_ho),
    }

    save_model(clf, MODEL_PATH)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    log_model_run(
        model_name="xgboost",
        model=clf,
        params={"seed": seed, "n_features": metrics["n_features"], **params},
        metrics={
            "train": metrics["train"],
            "validation": metrics["validation"],
            "holdout": metrics["holdout"],
        },
        feature_columns=list(X_tr.columns),
        signature_sample=X_tr.head(5),
        tags={"model_type": "xgboost", "stage": "candidate"},
    )

    print(f"[xgboost] features={metrics['n_features']}  best_iter={metrics['best_iteration']}")
    print(f"[xgboost] PR-AUC  val={metrics['validation']['pr_auc']:.4f}  "
          f"holdout={metrics['holdout']['pr_auc']:.4f}")
    print(f"[xgboost] model -> {MODEL_PATH}")
    print(f"[xgboost] metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
