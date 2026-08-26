# Dependency strategy

Single source of truth: `pyproject.toml`. Python pinned to **3.11** via
`.python-version` and `requires-python`.

## Principle

Install a dependency only when a real requirement exists. No Kubernetes, cloud SDKs,
Grafana, heavy orchestration, unnecessary LLM libs, or databases at this stage.

## Groups

| Group | Packages | Why / when installed |
|---|---|---|
| base | pandas, numpy, pyyaml | Data handling from ingestion onward; needed in discovery for dataset inspection |
| ml | scikit-learn, xgboost | Modeling stage (baseline + candidate) |
| serving | fastapi, uvicorn, pydantic | API stage |
| mlops | mlflow, dvc, evidently, prometheus-client, prometheus-fastapi-instrumentator | Tracking, versioning, monitoring stages; SHAP explanations use XGBoost native TreeSHAP (not the `shap` package — see D20) |
| dev | ruff, mypy, pytest, pytest-cov, types-* | Lint/type/test in every stage; installed in discovery |

## Version policy

- Lower bound = first version known to work; upper bound = next major, to avoid
  surprise breaking changes while staying current (2026 reality: pandas 3.x,
  numpy 2.x, ruff 0.16.x, mypy 2.x, pytest 9.x are current).
- ML/serving/mlops groups are **not** installed during discovery; they are added with
  `uv sync --extra <group>` (or pip into the venv) when their stage begins.
- Local environment note: on this Windows path `uv venv` misbehaves, so `.venv` is
  created with stdlib `python -m venv` and packages installed via pip/`uv pip`.
  CI (Linux) uses `uv sync`.

## Reproducibility

- `pyproject.toml` bounds + `.python-version` define the environment.
- A lockfile (`uv.lock`) will be generated when the first ML stage begins.
- Data is pinned via DVC; experiments via MLflow runs.
