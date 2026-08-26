"""Final evaluation aggregation (Stage 18).

Reads the pipeline artifacts and produces a single reproducible "model card"
(``data/artifacts/final_evaluation.json``) plus a Markdown render
(``data/artifacts/final_evaluation.md``). Every number comes from an existing
DVC output, so the report is regenerable with ``python scripts/build_report.py``
and never drifts from the actual pipeline results.
"""

from __future__ import annotations

import json
from pathlib import Path

from fraudintel.data.paths import artifacts_dir, splits_dir


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def build_final_evaluation() -> dict:
    validation = _load(artifacts_dir() / "validation_report.json")
    comparison = _load(artifacts_dir() / "models" / "comparison.json")
    xgb = _load(artifacts_dir() / "models" / "xgboost_metrics.json")
    base = _load(artifacts_dir() / "models" / "baseline_metrics.json")
    promotion = _load(artifacts_dir() / "models" / "promotion_report.json")
    shap = _load(artifacts_dir() / "explainability" / "global_shap_importance.json")
    drift = _load(artifacts_dir() / "monitoring" / "drift_report.json")
    split = _load(splits_dir() / "split_manifest.json")

    train_stats = (validation.get("train", {}) or {}).get("stats", {})
    test_stats = (validation.get("test", {}) or {}).get("stats", {})

    return {
        "project": "Fraud Intelligence Platform",
        "dataset": {
            "source": "IEEE-CIS Fraud Detection",
            "train_rows": train_stats.get("rows"),
            "test_rows": test_stats.get("rows"),
            "train_fraud_rate": train_stats.get("fraud_rate"),
            "identity_coverage_train": train_stats.get("identity_coverage"),
            "identity_coverage_test": test_stats.get("identity_coverage"),
        },
        "temporal_split": {
            "policy": "chronological 70/15/15 by TransactionDT",
            "thresholds_seconds": split.get("thresholds_seconds"),
            "train": split.get("summary", {}).get("train"),
            "validation": split.get("summary", {}).get("validation"),
            "holdout": split.get("summary", {}).get("holdout"),
        },
        "models": {
            "baseline_logistic_regression": {
                "params": (base.get("config", {}) or {}).get("params"),
                "n_features": base.get("n_features"),
                "metrics": {k: base[k] for k in base if k != "config"},
            },
            "candidate_xgboost": {
                "params": (xgb.get("config", {}) or {}).get("params"),
                "n_features": xgb.get("n_features"),
                "best_iteration": xgb.get("best_iteration"),
                "metrics": {k: xgb[k] for k in xgb if k != "config"},
            },
        },
        "comparison": comparison,
        "promotion_gate": {
            "operating_threshold": promotion.get("operating_threshold"),
            "validation_at_threshold": promotion.get("validation_at_threshold"),
            "holdout_at_threshold": promotion.get("holdout_at_threshold"),
            "slice_checks": promotion.get("slice_checks"),
            "gate": promotion.get("gate"),
            "registration": promotion.get("registration"),
        },
        "explainability": {
            "method": "XGBoost native TreeSHAP (predict pred_contribs=True)",
            "n_sample": shap.get("n_sample"),
            "top_features": (shap.get("top_features") or [])[:10],
        },
        "monitoring": {
            "drift_dataset_drift": bool(drift.get("dataset_drift", False)),
            "drift_drifted_columns": drift.get("drifted_columns"),
            "drift_total_columns": drift.get("total_columns"),
            "serving_metrics": [
                "fraud_predictions_total",
                "fraud_decisions_total",
                "fraud_scores",
            ],
        },
        "reproduction": {
            "pipeline": "dvc repro",
            "quality_gate": "ruff + mypy clean; pytest",
            "serve": "docker compose up -d (image: fraud-intel-api:local)",
        },
    }


def render_markdown(report: dict) -> str:
    lines: list[str] = []
    lines.append("# Fraud Intelligence Platform — Model Card")
    lines.append("")
    lines.append(
        "Auto-generated from pipeline artifacts by `scripts/build_report.py`. "
        "All figures are reproducible from the DVC outputs."
    )
    lines.append("")

    ds = report["dataset"]
    lines.append("## 1. Dataset")
    lines.append("")
    lines.append(f"- Source: **{ds['source']}**")
    lines.append(
        f"- Training joined rows: **{ds['train_rows']:,}** "
        f"(fraud rate {ds['train_fraud_rate']:.2%})"
    )
    lines.append(f"- Test joined rows: **{ds['test_rows']:,}**")
    lines.append(
        f"- Identity-table coverage: train {ds['identity_coverage_train']:.1%} / "
        f"test {ds['identity_coverage_test']:.1%}"
    )
    lines.append("")

    sp = report["temporal_split"]
    lines.append("## 2. Validation strategy")
    lines.append("")
    lines.append(f"- Policy: **{sp['policy']}** (prevents future→past leakage).")
    for name in ("train", "validation", "holdout"):
        s = sp[name] or {}
        lines.append(
            f"- {name.capitalize()}: {s.get('rows'):,} rows, "
            f"fraud rate {s.get('fraud_rate'):.2%}, "
            f"{s.get('date_min')} → {s.get('date_max')}"
        )
    lines.append("")

    models = report["models"]
    lines.append("## 3. Models")
    lines.append("")
    for key, label in (
        ("baseline_logistic_regression", "Baseline — Logistic Regression"),
        ("candidate_xgboost", "Candidate — XGBoost"),
    ):
        m = models[key]
        lines.append(f"### {label}")
        lines.append("")
        lines.append(f"- Features: **{m['n_features']}**")
        if m.get("params"):
            parts = ", ".join(f"{k}={v}" for k, v in m["params"].items())
            lines.append(f"- Params: {parts}")
        if m.get("best_iteration"):
            lines.append(f"- Best iteration: {m['best_iteration']}")
        metr = m["metrics"]
        for split_name in ("train", "validation", "holdout"):
            sm = metr.get(split_name, {})
            if sm:
                lines.append(
                    f"  - {split_name}: PR-AUC {sm.get('pr_auc'):.3f}, "
                    f"ROC-AUC {sm.get('roc_auc'):.3f}"
                )
        lines.append("")

    comp = report["comparison"].get("splits", {})
    lines.append("## 4. Baseline vs candidate")
    lines.append("")
    for split_name in ("validation", "holdout"):
        c = comp.get(split_name, {})
        lines.append(
            f"- {split_name.capitalize()}: baseline PR-AUC {c.get('baseline_pr_auc'):.3f} → "
            f"candidate {c.get('candidate_pr_auc'):.3f} "
            f"(Δ {c.get('pr_auc_delta'):+.3f})"
        )
    lines.append("")

    gate = report["promotion_gate"]
    lines.append("## 5. Promotion gate (operating point)")
    lines.append("")
    lines.append(f"- Operating threshold: **{gate['operating_threshold']:.4f}**")
    ho = gate["holdout_at_threshold"] or {}
    lines.append(
        f"- Hold-out at threshold: precision {ho.get('precision'):.3f}, "
        f"recall {ho.get('recall'):.3f}, F1 {ho.get('f1'):.3f}, "
        f"alert rate {ho.get('alert_rate'):.2%}"
    )
    g = gate.get("gate", {})
    lines.append(
        f"- Gate: beat-baseline={g.get('g1_beat_baseline')}, "
        f"precision/recall={g.get('g2_precision_recall')}, "
        f"slices={g.get('g3_slices')}, promoted={g.get('promoted')}"
    )
    reg = gate.get("registration", {})
    lines.append(
        f"- Registry: {reg.get('status')} as `{reg.get('alias')}` "
        f"(version {reg.get('version')}, run {reg.get('run_id')})"
    )
    lines.append("")

    lines.append("## 6. Subgroup (slice) performance — hold-out PR-AUC")
    lines.append("")
    slices = gate.get("slice_checks", {})
    lines.append("| Slice | n | Positives | PR-AUC | Status |")
    lines.append("|---|---|---|---|---|")
    for name, s in slices.items():
        lines.append(
            f"| {name} | {s.get('n'):,} | {s.get('positives'):,} | "
            f"{s.get('pr_auc'):.3f} | {s.get('status')} |"
        )
    lines.append("")

    ex = report["explainability"]
    lines.append("## 7. Explainability (global drivers)")
    lines.append("")
    lines.append(f"- Method: {ex['method']}")
    lines.append(f"- Sample size: {ex.get('n_sample')}")
    lines.append("")
    lines.append("| Rank | Feature | Mean |SHAP| |")
    lines.append("|---|---|---|")
    for i, f in enumerate(ex["top_features"], 1):
        lines.append(f"| {i} | {f['feature']} | {f['importance']:.4f} |")
    lines.append("")

    mon = report["monitoring"]
    lines.append("## 8. Monitoring")
    lines.append("")
    lines.append(
        f"- Data drift (train→holdout): dataset_drift={mon['drift_dataset_drift']}, "
        f"{mon['drift_drifted_columns']}/{mon['drift_total_columns']} columns drifted."
    )
    lines.append(
        "- Serving metrics exposed at `/metrics`: "
        + ", ".join(f"`{m}`" for m in mon["serving_metrics"])
        + "."
    )
    lines.append("")

    rep = report["reproduction"]
    lines.append("## 9. Reproduction")
    lines.append("")
    lines.append(f"- Pipeline: `{rep['pipeline']}`")
    lines.append(f"- Quality gate: {rep['quality_gate']}")
    lines.append(f"- Serve: `{rep['serve']}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    report = build_final_evaluation()
    out_json = artifacts_dir() / "final_evaluation.json"
    out_md = artifacts_dir() / "final_evaluation.md"
    out_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")
    print(f"[report] wrote {out_json}")
    print(f"[report] wrote {out_md}")


if __name__ == "__main__":
    main()
