"""Compare baseline vs candidate model metrics for the promotion gate (DECISIONS D7).

Usage: python scripts/compare_models.py
Reads baseline_metrics.json and xgboost_metrics.json, writes comparison.json, and prints
which model wins on each temporal split by PR-AUC (primary) and ROC-AUC (secondary).
"""

from __future__ import annotations

import json

from fraudintel.data.paths import artifacts_dir

METRICS_DIR = artifacts_dir() / "models"
OUT = METRICS_DIR / "comparison.json"


def _load(name: str) -> dict:
    return json.loads((METRICS_DIR / name).read_text())


def main() -> None:
    base = _load("baseline_metrics.json")
    cand = _load("xgboost_metrics.json")

    comparison: dict = {
        "models": {"baseline": base["config"], "candidate": cand["config"]},
        "splits": {},
    }
    for split in ("validation", "holdout"):
        b_pr = base[split]["pr_auc"]
        c_pr = cand[split]["pr_auc"]
        b_ro = base[split]["roc_auc"]
        c_ro = cand[split]["roc_auc"]
        comparison["splits"][split] = {
            "baseline_pr_auc": b_pr,
            "candidate_pr_auc": c_pr,
            "pr_auc_delta": c_pr - b_pr,
            "baseline_roc_auc": b_ro,
            "candidate_roc_auc": c_ro,
            "roc_auc_delta": c_ro - b_ro,
            "candidate_beats_baseline_pr_auc": bool(c_pr > b_pr),
        }

    # Promotion gate (D7) primary rule: candidate must beat baseline on hold-out PR-AUC.
    ho = comparison["splits"]["holdout"]
    comparison["candidate_promotes_over_baseline"] = bool(ho["candidate_beats_baseline_pr_auc"])

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(comparison, indent=2))

    print(f"[compare] validation  PR-AUC  base={base['validation']['pr_auc']:.4f}  "
          f"xgb={cand['validation']['pr_auc']:.4f}  (delta={ho['pr_auc_delta']:+.4f})")
    print(f"[compare] hold-out   PR-AUC  base={base['holdout']['pr_auc']:.4f}  "
          f"xgb={cand['holdout']['pr_auc']:.4f}  (delta={ho['pr_auc_delta']:+.4f})")
    print(f"[compare] candidate promotes over baseline (hold-out PR-AUC rule): "
          f"{comparison['candidate_promotes_over_baseline']}")
    print(f"[compare] -> {OUT}")


if __name__ == "__main__":
    main()
