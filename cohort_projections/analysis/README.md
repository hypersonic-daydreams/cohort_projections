# cohort_projections/analysis/

Status: current · Last Updated: 2026-06-23

Orientation README for the analysis **library** package. This package contains the
importable implementation of the benchmarking, evaluation, and Projection Observatory
systems. It is the *library half* of a deliberate library-vs-CLI split: the runnable
command-line entry points live in `scripts/analysis/`, and they import their logic from
here.

## What lives here (library code)

- `benchmarking.py` — core benchmarking helpers: run-id construction, method-profile and
  alias loading, comparison-to-champion, summary scorecards, prediction intervals, manifest
  writing, and benchmark/promotion index maintenance. The package `__init__.py` re-exports
  this module's public surface.
- `benchmark_contract.py` — the contract/version guard for benchmark bundles.
- `evaluation/` — the forecast-evaluation framework: metrics, forecast-accuracy,
  scorecards, structural-realism checks, sensitivity, HTML reporting, and the
  `EvaluationRunner`.
- `evaluation_policy.py`, `experiment_log.py` — evaluation-gate policy and the experiment
  log/ledger used by the sweep/experiment workflows.
- `observatory/` — the Projection Observatory: variant catalog, candidate generation,
  comparator, recommender, decision support, deep/autonomous search (`search_controller.py`,
  `search_policy.py`, `deep_search.py`), results store, runtime contract, AI synthesis,
  workspace reset, and the `dashboard/` building blocks.

## Relationship to `scripts/analysis/` (the CLI half)

`scripts/analysis/` holds the **thin CLI runners** — `observatory.py`,
`observatory_dashboard.py`, `run_benchmark_suite.py`, `run_experiment.py`,
`run_experiment_sweep.py`, `compare_benchmark_runs.py`, `promote_method.py`,
`generate_evaluation_report.py`, `walk_forward_validation.py`, and so on. These scripts
parse arguments, wire up paths, and then call into this package
(`from cohort_projections.analysis.benchmarking import ...`,
`from cohort_projections.analysis.observatory.workspace_reset import ...`,
`from cohort_projections.analysis.evaluation.runner import ...`).

The split exists so that the analysis logic is importable, unit-testable, and reusable
without invoking a CLI: tests in `tests/test_analysis/` (including the nested
`test_evaluation/` suite and the Observatory runtime-contract tests) exercise this package
directly. When changing behavior, edit the implementation here; touch `scripts/analysis/`
only for argument handling, orchestration, and output wiring.

## Neighbors

This package sits alongside the projection-engine packages (`core/`, `data/`,
`geographic/`, `output/`, `utils/`) under `cohort_projections/`. It *consumes* their
outputs (projection runs, rates) to score and compare methods; it does not produce the
projections themselves. Method/config profiles it loads live in
`config/method_profiles/` (with `aliases.yaml` naming the champion/candidate/reference
roles).
