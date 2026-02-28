# RB-003 / RB-004 Remediation Results

**Date (UTC):** 2026-02-27T18:58:26Z  
**Scope:** Close residual risks `RB-003` (pipeline dry-run coverage) and `RB-004` (full-repo lint/type debt)

## Summary

1. Implemented explicit `--dry-run` support for:
   - `scripts/pipeline/01a_compute_residual_migration.py`
   - `scripts/pipeline/01b_compute_convergence.py`
   - `scripts/pipeline/01c_compute_mortality_improvement.py`
2. Updated `scripts/pipeline/run_complete_pipeline.sh` so steps 3-5 run in dry-run mode instead of being skipped.
3. Executed a dedicated lint/type cleanup wave:
   - Autofix + manual cleanup for outstanding Ruff issues.
   - Type fixes in `cohort_projections/utils/__init__.py` and `cohort_projections/data/load/base_population_loader.py`.
   - Validation patch for one test regression introduced during autofix.

## Validation Evidence

### Dry-run coverage

- Command: `bash scripts/pipeline/run_complete_pipeline.sh --dry-run`
- Result: `RC=0`
- Evidence: [dry-pipeline-rb003-remediation-postedit.txt](./dry-pipeline-rb003-remediation-postedit.txt)
- Key outcome: stages `01a`, `01b`, and `01c` execute dry-run validation paths and are no longer skipped.

### Lint/type gates

- Command: `ruff check .`
- Result: `RC=0`
- Evidence: [dry-lint-type-rb004-remediation-ruff.txt](./dry-lint-type-rb004-remediation-ruff.txt)

- Command: `mypy .`
- Result: `RC=0`
- Evidence: [dry-lint-type-rb004-remediation-mypy.txt](./dry-lint-type-rb004-remediation-mypy.txt)

### Tests

- Command: `pytest tests/ -q`
- Result: `1258 passed, 5 skipped`
- Notes: one intermediate regression in `tests/test_data/test_county_race_distributions.py` was fixed and full suite rerun to green.

## Outcome

- `RB-003`: **closed** (deterministic dry-run policy now covers all seven pipeline stages).
- `RB-004`: **closed** (full-repo lint/type baseline passes).
