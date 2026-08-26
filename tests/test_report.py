"""Tests for the Stage 18 final-evaluation aggregation.

The script reads DVC-tracked pipeline artifacts. Tests that require those artifacts
self-skip when they are absent (e.g., in CI without a DVC remote), mirroring the
champion-load test in `test_serving.py`.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

from fraudintel.data.paths import artifacts_dir

_REPORT_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "build_report.py"
_spec = importlib.util.spec_from_file_location("build_report_test_mod", _REPORT_SCRIPT)
build_report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_report)

ARTIFACTS_PRESENT = (artifacts_dir() / "models" / "promotion_report.json").exists()


def test_build_final_evaluation_has_expected_sections():
    report = build_report.build_final_evaluation()
    for key in (
        "dataset",
        "temporal_split",
        "models",
        "comparison",
        "promotion_gate",
        "explainability",
        "monitoring",
        "reproduction",
    ):
        assert key in report


def test_render_markdown_structured_on_minimal_report():
    # Render a hand-built report so this is hermetic (no pipeline artifacts needed).
    fake = {
        "project": "Fraud Intelligence Platform",
        "dataset": {
            "source": "IEEE-CIS",
            "train_rows": 100,
            "test_rows": 50,
            "train_fraud_rate": 0.03,
            "identity_coverage_train": 0.2,
            "identity_coverage_test": 0.25,
        },
        "temporal_split": {
            "policy": "chronological 70/15/15",
            "thresholds_seconds": {"t1": 1, "t2": 2},
            "train": {"rows": 70, "fraud_rate": 0.03, "date_min": "a", "date_max": "b"},
            "validation": {"rows": 15, "fraud_rate": 0.03, "date_min": "a", "date_max": "b"},
            "holdout": {"rows": 15, "fraud_rate": 0.03, "date_min": "a", "date_max": "b"},
        },
        "models": {
            "baseline_logistic_regression": {
                "params": {"C": 1.0},
                "n_features": 416,
                "metrics": {
                    "train": {"pr_auc": 0.5, "roc_auc": 0.89},
                    "validation": {"pr_auc": 0.4, "roc_auc": 0.86},
                    "holdout": {"pr_auc": 0.2, "roc_auc": 0.84},
                },
            },
            "candidate_xgboost": {
                "params": {"max_depth": 6},
                "n_features": 416,
                "best_iteration": 999,
                "metrics": {
                    "train": {"pr_auc": 0.79, "roc_auc": 0.97},
                    "validation": {"pr_auc": 0.54, "roc_auc": 0.90},
                    "holdout": {"pr_auc": 0.51, "roc_auc": 0.88},
                },
            },
        },
        "comparison": {
            "splits": {
                "validation": {
                    "baseline_pr_auc": 0.4,
                    "candidate_pr_auc": 0.5,
                    "pr_auc_delta": 0.1,
                },
                "holdout": {
                    "baseline_pr_auc": 0.2,
                    "candidate_pr_auc": 0.5,
                    "pr_auc_delta": 0.3,
                },
            }
        },
        "promotion_gate": {
            "operating_threshold": 0.798,
            "holdout_at_threshold": {
                "precision": 0.51,
                "recall": 0.48,
                "f1": 0.5,
                "alert_rate": 0.03,
            },
            "gate": {
                "g1_beat_baseline": True,
                "g2_precision_recall": True,
                "g3_slices": True,
                "promoted": True,
            },
            "registration": {
                "status": "registered",
                "alias": "champion",
                "version": 1,
                "run_id": "x",
            },
            "slice_checks": {
                "identity_present": {"n": 100, "positives": 10, "pr_auc": 0.7, "status": "pass"},
            },
        },
        "explainability": {
            "method": "TreeSHAP",
            "n_sample": 5000,
            "top_features": [{"feature": "card1_hist_fraud_rate", "importance": 0.7}],
        },
        "monitoring": {
            "drift_dataset_drift": False,
            "drift_drifted_columns": 97,
            "drift_total_columns": 400,
            "serving_metrics": ["fraud_predictions_total"],
        },
        "reproduction": {
            "pipeline": "dvc repro",
            "quality_gate": "pytest",
            "serve": "docker compose up",
        },
    }
    md = build_report.render_markdown(fake)
    assert "# Fraud Intelligence Platform" in md
    assert "## 5. Promotion gate" in md
    assert "card1_hist_fraud_rate" in md


@pytest.mark.skipif(not ARTIFACTS_PRESENT, reason="pipeline artifacts not present")
def test_build_final_evaluation_pulls_real_numbers():
    report = build_report.build_final_evaluation()
    assert report["models"]["candidate_xgboost"]["n_features"] == 416
    assert report["promotion_gate"]["operating_threshold"] is not None
    assert report["monitoring"]["drift_total_columns"] == 400


@pytest.mark.skipif(not ARTIFACTS_PRESENT, reason="pipeline artifacts not present")
def test_main_writes_artifacts(tmp_path, monkeypatch):
    import shutil

    real = artifacts_dir()
    rels = [
        "validation_report.json",
        "models/comparison.json",
        "models/xgboost_metrics.json",
        "models/baseline_metrics.json",
        "models/promotion_report.json",
        "explainability/global_shap_importance.json",
        "monitoring/drift_report.json",
    ]
    for rel in rels:
        src = real / rel
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
    monkeypatch.setattr(build_report, "artifacts_dir", lambda: tmp_path)
    build_report.main()
    assert (tmp_path / "final_evaluation.json").exists()
    assert (tmp_path / "final_evaluation.md").exists()
