# AGENTS.md — Repository Operating Manual

This file is the persistent operating manual for any agent (human or automated)
working in this repository. Follow it before modifying anything.

## Core working rules

- **Inspect before modifying.** Read the relevant module, tests, configs, and
  `DESIGN.md` before changing code. Do not assume structure.
- **Make routine decisions independently.** Module names, function decomposition,
  reasonable library versions, test organization, logging, config, and ordinary
  dependency choices are your call. Do not ask for permission on routine work.
- **Verify assumptions.** Do not claim success without running the relevant check
  (test, lint, type-check, or a real data/behavior validation). "It should work"
  is not evidence.
- **Prefer simple solutions.** Introduce a technology only when it solves a real
  problem. No Kubernetes, cloud SDKs, orchestration, or heavy frameworks unless a
  later stage demonstrates a genuine need.
- **Write tests for meaningful behavior.** Cover feature transforms, validation
  logic, model input handling, prediction behavior, API schemas, and important
  edge cases (malformed input, missing values, unknown categories, extreme numerics).
  Do not chase coverage percentages.
- **Document important decisions.** Add an entry to `DECISIONS.md` for every
  significant architectural or scope choice, using the structured format there.
- **Keep notebooks for exploration only.** All reusable production logic lives in
  `src/` Python modules and is unit-tested. Notebooks under `notebooks/` must not
  contain the only copy of important logic.
- **Avoid unnecessary dependencies.** Every dependency must have a documented reason
  (see `docs/dependency-strategy.md` and `pyproject.toml`). Do not add packages
  speculatively.
- **Do not silently change project scope.** If a task appears to expand scope,
  surface it. Escalate only for major architectural, cost, security, validity, or
  scope decisions — and when you do, provide the problem, options considered,
  tradeoffs, and your recommendation.
- **Reproducibility is mandatory.** Pin Python (3.11) and dependencies; use DVC for
  data; log every experiment to MLflow; never hard-code thresholds or splits that
  should be data-driven.

## Repository layout

```
src/            production code (packages, no notebooks)
tests/          pytest suite, mirrors src/
notebooks/      EDA / exploration only
configs/        YAML: features, training, thresholds
scripts/        runnable pipeline entrypoints (used by DVC stages)
docs/           design + strategy docs
data/           DVC-tracked raw/interim/processed (gitignored)
artifacts/      models, reports, plots (gitignored)
.github/        CI workflows
```

## Definition of done for a stage

Implementation works · important failure cases considered · tests exist where
appropriate · results reproducible · documentation reflects the implementation ·
assumptions recorded · next stage has a clear starting point.
