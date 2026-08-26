"""Model selection and promotion-gate logic (DECISIONS D7/D10).

Pure, testable functions for choosing an operating threshold, scoring a model at a
threshold, checking key slices for disparity, and applying the promotion gate. The
concrete gate thresholds live in ``configs/thresholds.yaml`` (data-driven, not hardcoded).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def select_threshold_by_f1(y_true: Any, y_score: Any) -> dict[str, float]:
    """Pick the threshold maximizing F1 on the given scores (validation study).

    Returns ``{threshold, precision, recall, f1}`` at the chosen operating point.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.sum() == 0:
        return {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = np.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall + 1e-12),
        0.0,
    )
    # `thresholds` has one fewer entry than precision/recall; argmax over the thresholded range.
    idx = int(np.nanargmax(f1[:-1]))
    return {
        "threshold": float(thresholds[idx]),
        "precision": float(precision[idx]),
        "recall": float(recall[idx]),
        "f1": float(f1[idx]),
    }


def evaluate_at_threshold(y_true: Any, y_score: Any, threshold: float) -> dict[str, float]:
    """Hard-classify at ``threshold`` and return precision/recall/F1 and counts."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pred = (y_score >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-12) if (precision + recall) else 0.0
    alert_rate = (tp + fp) / len(y_true) if len(y_true) else 0.0
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "alert_rate": float(alert_rate),
        "n": int(len(y_true)),
    }


def slice_pr_auc(y_true: Any, y_score: Any) -> float:
    """PR-AUC for a slice; NaN when the slice has no positives (undefined)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def run_slice_checks(
    y_true: Any,
    y_score: Any,
    slices: dict[str, Any],
    min_positives: int = 20,
    pr_auc_floor: float = 0.05,
) -> dict[str, dict[str, Any]]:
    """Per-slice PR-AUC disparity check.

    Slices with fewer than ``min_positives`` positives are marked ``insufficient_data``
    (not failed), since PR-AUC is too noisy to judge on sparse fraud.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    results: dict[str, dict[str, Any]] = {}
    for name, mask in slices.items():
        m = np.asarray(mask, dtype=bool)
        sub_y, sub_s = y_true[m], y_score[m]
        n_pos = int(sub_y.sum())
        if n_pos < min_positives:
            results[name] = {
                "n": int(m.sum()),
                "positives": n_pos,
                "pr_auc": None,
                "status": "insufficient_data",
            }
            continue
        pa = slice_pr_auc(sub_y, sub_s)
        results[name] = {
            "n": int(m.sum()),
            "positives": n_pos,
            "pr_auc": pa,
            "status": "pass" if pa >= pr_auc_floor else "fail",
        }
    return results


def evaluate_gate(
    candidate_holdout_pr_auc: float,
    baseline_holdout_pr_auc: float,
    holdout_eval: dict[str, float],
    slice_results: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Apply the promotion gate (DECISIONS D7).

    G1: candidate beats baseline on hold-out PR-AUC (by ``min_pr_auc_improvement``).
    G2: hold-out precision/recall meet the configured floors.
    G3: no slice fails the PR-AUC floor.
    """
    min_improve = float(cfg.get("min_pr_auc_improvement", 0.0))
    g1 = float(candidate_holdout_pr_auc) > float(baseline_holdout_pr_auc) + min_improve
    g2 = (holdout_eval["precision"] >= float(cfg["min_precision"])) and (
        holdout_eval["recall"] >= float(cfg["min_recall"])
    )
    failed_slices = [k for k, v in slice_results.items() if v.get("status") == "fail"]
    g3 = len(failed_slices) == 0
    return {
        "g1_beat_baseline": bool(g1),
        "g2_precision_recall": bool(g2),
        "g3_slices": bool(g3),
        "failed_slices": failed_slices,
        "promoted": bool(g1 and g2 and g3),
    }
