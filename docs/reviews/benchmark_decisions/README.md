# Benchmark Decisions

Decision records comparing challenger methods against the current champion.

These records are generated from benchmark bundles in:

- `data/analysis/benchmark_history/`

and serve as the human-review layer between:

- automated benchmark execution, and
- champion alias promotion.

## Workflow

1. Run `scripts/analysis/run_benchmark_suite.py`
2. Generate the draft record with `scripts/analysis/compare_benchmark_runs.py`
3. Review the metrics and fill in the decision rationale
4. Mark the record `Approved` only if the challenger should be eligible for promotion
5. If approved, update the champion alias with `scripts/analysis/promote_method.py`

## Naming Convention

Files use:

- `YYYY-MM-DD-<challenger>-vs-<champion>.md`

## Current Records

| Date | Decision | Run ID | Status |
|------|----------|--------|--------|
| 2026-03-09 | [m2026r1 vs m2026](./2026-03-09-m2026r1-vs-m2026.md) | `br-20260309-160948-m2026r1-ecb4498` | Draft |
