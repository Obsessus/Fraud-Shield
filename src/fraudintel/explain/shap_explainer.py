"""SHAP-based explainability for the Fraud Intelligence Platform.

Provides global feature importance and per-instance "risk factor" explanations for the
champion model, to be surfaced in the API response (Stage 11/12).

We use XGBoost's **native** TreeSHAP (``booster.predict(..., pred_contribs=True)``) rather
than the ``shap`` package, because ``shap`` 0.49's ``TreeExplainer`` cannot parse the
XGBoost 3.x model serialization (``vector-leaf``/ubjson). Both implement the same
interventional TreeSHAP algorithm; the native path is also faster and version-robust.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xgboost as xgb


class ShapExplainer:
    """Per-instance and global SHAP explanations for a fitted XGBoost classifier.

    Explanations are produced for the **positive (fraud) class**: a feature's contribution
    is its effect on the model output relative to the base (bias) value. Positive
    contributions are risk-increasing.
    """

    def __init__(self, model: Any, feature_names: list[str]) -> None:
        self.model = model
        self.feature_names = list(feature_names)
        self.booster = model.get_booster()

    def _class1_shap(self, X: Any) -> np.ndarray[Any, Any]:
        dmat = xgb.DMatrix(np.asarray(X, dtype="float64"), feature_names=self.feature_names)
        contribs = self.booster.predict(
            dmat, pred_contribs=True
        )  # (n, n_f+1) or (n, n_f+1, n_class)
        contribs = np.asarray(contribs, dtype="float64")
        if contribs.ndim == 3:
            contribs = contribs[:, :, 1]  # positive class
        return contribs[:, :-1]  # drop bias column

    def global_importance(self, X: Any, top_n: int | None = None) -> list[dict[str, Any]]:
        """Mean absolute SHAP contribution per feature, descending."""
        sv = self._class1_shap(X)
        imp = np.abs(sv).mean(axis=0)
        order = np.argsort(-imp)
        out = [{"feature": self.feature_names[i], "importance": float(imp[i])} for i in order]
        return out[:top_n] if top_n else out

    def explain(self, X: Any, top_k: int = 5) -> list[list[dict[str, Any]]]:
        """Per-instance top-``k`` risk factors (largest positive contributions)."""
        sv = self._class1_shap(X)
        Xv = np.asarray(X, dtype="float64")
        rows: list[list[dict[str, Any]]] = []
        for i in range(sv.shape[0]):
            contribs = sv[i]
            order = np.argsort(-contribs)
            items = [
                {
                    "feature": self.feature_names[j],
                    "contribution": float(contribs[j]),
                    "value": float(Xv[i, j]),
                }
                for j in order[:top_k]
            ]
            rows.append(items)
        return rows
