"""Compute SHAP explanations for the champion model (Stage 11).

Usage: python scripts/compute_explanations.py

Loads the XGBoost champion, rebuilds the hold-out features, computes global SHAP feature
importance (on a representative sample) and example per-instance risk factors for the
highest- and lowest-risk hold-out rows. Writes JSON artifacts under
``data/artifacts/explainability/``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from fraudintel.data.paths import artifacts_dir, interim_dir, splits_dir
from fraudintel.explain.shap_explainer import ShapExplainer
from fraudintel.features.build import build_temporal_splits
from fraudintel.models.baseline import load_model, select_features

MODEL_PATH = artifacts_dir() / "models" / "xgboost_model.joblib"
GLOBAL_OUT = artifacts_dir() / "explainability" / "global_shap_importance.json"
SAMPLE_OUT = artifacts_dir() / "explainability" / "sample_explanations.json"
SAMPLE_ROWS = 5000
N_EXAMPLES = 5
TOP_K = 5


def main() -> None:
    manifest = json.loads((splits_dir() / "split_manifest.json").read_text())
    t1 = float(manifest["thresholds_seconds"]["t1"])
    t2 = float(manifest["thresholds_seconds"]["t2"])

    train = pd.read_parquet(interim_dir() / "train_joined.parquet")
    _, _, ho_feat = build_temporal_splits(train, t1, t2)
    X_ho, _ = select_features(ho_feat)

    model = load_model(MODEL_PATH)
    expected = list(model.feature_names_in_)
    X_ho = X_ho.reindex(columns=expected, fill_value=0)
    explainer = ShapExplainer(model, expected)

    sample = X_ho.sample(n=min(SAMPLE_ROWS, len(X_ho)), random_state=42)
    global_imp = explainer.global_importance(sample, top_n=30)
    GLOBAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    GLOBAL_OUT.write_text(
        json.dumps({"n_sample": int(len(sample)), "top_features": global_imp}, indent=2)
    )

    scores = model.predict_proba(X_ho)[:, 1]
    order = np.argsort(-scores)
    hi_idx = order[:N_EXAMPLES]
    lo_idx = order[-N_EXAMPLES:]

    def explain_rows(idx_list):
        sub = X_ho.iloc[idx_list]
        explained = explainer.explain(sub, top_k=TOP_K)
        return [
            {"row_index": int(idx), "score": float(scores[idx]), "risk_factors": factors}
            for idx, factors in zip(idx_list, explained, strict=False)
        ]

    sample_explanations = {
        "highest_risk": explain_rows(list(hi_idx)),
        "lowest_risk": explain_rows(list(lo_idx)),
    }
    SAMPLE_OUT.write_text(json.dumps(sample_explanations, indent=2))

    print(f"[explain] global importance top5: {[g['feature'] for g in global_imp[:5]]}")
    print(f"[explain] examples explained: {N_EXAMPLES} high-risk + {N_EXAMPLES} low-risk")
    print(f"[explain] -> {GLOBAL_OUT}")
    print(f"[explain] -> {SAMPLE_OUT}")


if __name__ == "__main__":
    main()
