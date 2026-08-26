"""Tests for model selection / promotion-gate logic."""

from __future__ import annotations

import numpy as np
import pytest

from fraudintel.mlops.selection import (
    evaluate_at_threshold,
    evaluate_gate,
    run_slice_checks,
    select_threshold_by_f1,
)


def test_select_threshold_by_f1_perfect():
    y = [0, 0, 1, 1]
    score = [0.1, 0.2, 0.8, 0.9]
    res = select_threshold_by_f1(y, score)
    assert 0.2 <= res["threshold"] <= 0.9
    assert res["precision"] == pytest.approx(1.0)
    assert res["recall"] == pytest.approx(1.0)
    assert res["f1"] == pytest.approx(1.0)


def test_evaluate_at_threshold_counts():
    y = [0, 0, 1, 1]
    score = [0.1, 0.2, 0.8, 0.9]
    ev = evaluate_at_threshold(y, score, 0.5)
    assert ev["tp"] == 2 and ev["fp"] == 0 and ev["fn"] == 0 and ev["tn"] == 2
    assert ev["precision"] == pytest.approx(1.0)
    assert ev["recall"] == pytest.approx(1.0)
    assert ev["f1"] == pytest.approx(1.0)
    assert ev["alert_rate"] == pytest.approx(0.5)


def test_run_slice_checks_pass_and_fail_and_sparse():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 0])
    score = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.5])
    slices = {
        "good": np.array([True, True, True, True, False, False, False, False]),
        "bad": np.array([False, False, False, False, True, True, True, True]),
        "poor": np.array([True, False, False, False, True, False, False, False]),
    }
    res = run_slice_checks(y, score, slices, min_positives=2, pr_auc_floor=0.1)
    assert res["good"]["status"] == "pass"
    assert res["bad"]["status"] == "insufficient_data"
    # poor slice has 1 positive but >= min_positives? it has 1 pos < 2 -> insufficient
    assert res["poor"]["status"] == "insufficient_data"


def test_evaluate_gate_promotes_only_when_all_pass():
    cfg = {"min_precision": 0.1, "min_recall": 0.2}
    # all pass
    g = evaluate_gate(
        candidate_holdout_pr_auc=0.5, baseline_holdout_pr_auc=0.2,
        holdout_eval={"precision": 0.3, "recall": 0.4}, slice_results={}, cfg=cfg,
    )
    assert g["promoted"] is True
    # fails G2: precision below floor
    g2 = evaluate_gate(
        candidate_holdout_pr_auc=0.5, baseline_holdout_pr_auc=0.2,
        holdout_eval={"precision": 0.05, "recall": 0.4}, slice_results={}, cfg=cfg,
    )
    assert g2["promoted"] is False and g2["g2_precision_recall"] is False
    # fails G1: does not beat baseline
    g1 = evaluate_gate(
        candidate_holdout_pr_auc=0.1, baseline_holdout_pr_auc=0.2,
        holdout_eval={"precision": 0.3, "recall": 0.4}, slice_results={}, cfg=cfg,
    )
    assert g1["promoted"] is False and g1["g1_beat_baseline"] is False


def test_evaluate_gate_fails_on_slice():
    cfg = {"min_precision": 0.1, "min_recall": 0.2}
    slices = {"s1": {"status": "fail"}}
    res = evaluate_gate(
        candidate_holdout_pr_auc=0.5, baseline_holdout_pr_auc=0.2,
        holdout_eval={"precision": 0.3, "recall": 0.4}, slice_results=slices, cfg=cfg,
    )
    assert res["g3_slices"] is False
    assert res["failed_slices"] == ["s1"]
    assert res["promoted"] is False
