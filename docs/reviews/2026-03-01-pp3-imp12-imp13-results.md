# 2026-03-01 PP3 IMP-12/IMP-13 Implementation Results

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-03-01 |
| **Scope** | PP-003 IMP-12 (QA artifacts) + IMP-13 (consistency constraints) |
| **Status** | Completed; human approval recorded 2026-03-01 |
| **Related Contract** | `docs/reviews/2026-02-28-pp3-s06-output-contract.md` |
| **Approval Record** | `docs/reviews/2026-03-01-pp3-imp12-imp13-approval-gate.md` |

## Summary

IMP-12 and IMP-13 are implemented in the place projection orchestrator.

- QA artifact generation now writes all four S06 Section 5 files to `data/projections/{scenario}/place/qa/`.
- Hard constraints are enforced with loud failures.
- Soft constraints are surfaced through QA artifacts/log warnings without blocking the run.

## Implementation Details

### 1. QA Artifact Generation (IMP-12)

Implemented in `cohort_projections/data/process/place_projection_orchestrator.py`:

- `qa_tier_summary.csv`
- `qa_share_sum_validation.csv`
- `qa_outlier_flags.csv`
- `qa_balance_of_county.csv`

Contract-aligned checks included:

- Tier summary always emits `HIGH`, `MODERATE`, `LOWER`.
- Outlier flag types are validated against the 5 allowed S06 types.
- QA files are always emitted (even when any given table is empty).

### 2. Hard Constraint Enforcement (IMP-13)

Implemented hard checks:

1. Share bound: `0 <= place_share <= 1.0` (tolerance-aware).
2. County share sum: `sum(place_shares) <= 1.0`.
3. Place-county consistency: `sum(place_populations) <= county_total`.
4. Non-negative population checks.
5. Output universe exact match to projected crosswalk (no missing/orphan place FIPS).
6. State scenario ordering: `restricted_growth <= baseline <= high_growth` validated from county outputs when all required scenarios are available.

### 3. Soft Constraint QA Signaling

- Balance-of-county negatives are surfaced in `qa_balance_of_county.csv` and warning logs.
- Share-stability outliers are flagged in `qa_outlier_flags.csv`.
- Tier-band extreme annual growth outliers are flagged in `qa_outlier_flags.csv`.

## Tests Added

- `tests/test_data/test_place_qa_artifacts.py`
- `tests/test_data/test_place_consistency_constraints.py`

Test coverage includes:

- QA artifact schema and row-shape checks.
- Hard-constraint violation raising behavior.
- Soft-constraint non-blocking behavior with QA evidence.
- Scenario-ordering violation detection.

## Verification Commands and Results

- `uv run ruff check cohort_projections/data/process/place_projection_orchestrator.py cohort_projections/data/process/__init__.py tests/test_data/test_place_qa_artifacts.py tests/test_data/test_place_consistency_constraints.py` -> pass
- `uv run mypy cohort_projections/data/process/place_projection_orchestrator.py tests/test_data/test_place_qa_artifacts.py tests/test_data/test_place_consistency_constraints.py` -> pass
- `uv run pytest tests/test_data/test_place_projection_orchestrator.py tests/test_data/test_place_qa_artifacts.py tests/test_data/test_place_consistency_constraints.py` -> `8 passed`
- `uv run pytest tests/test_integration/test_place_pipeline_stage.py` -> `3 passed`
- `uv run python scripts/pipeline/02a_run_place_projections.py --scenarios baseline` -> successful baseline stage run (`90` places, `46` county balances, `31` outlier flags)
- `uv run python scripts/pipeline/02a_run_place_projections.py` -> successful full stage run for `baseline`, `restricted_growth`, and `high_growth`; each scenario completed with `90` places, `46` county balances, and QA artifact output.

## Baseline Runtime Snapshot

From the 2026-03-01 baseline stage rerun:

- `qa_tier_summary.csv`: `3` rows
- `qa_share_sum_validation.csv`: `1,426` rows (46 counties x 31 years)
- `qa_balance_of_county.csv`: `1,426` rows
- `qa_outlier_flags.csv`: `31` rows
- Outlier type counts: `SHARE_RESCALED=20`, `SHARE_REVERSAL=11`

## Human Review Disposition (2026-03-01)

Formal gate decisions are recorded in:

- `docs/reviews/2026-03-01-pp3-imp12-imp13-approval-gate.md`

Recorded disposition summary:

- Overall: **Approved with notes**
- Gates 1/2/3/7/8: **Approved**
- Gates 4/5/6: **Approved with notes**
- Non-blocking follow-up retained: `IMP-13A` (reconciliation magnitude QA visibility)
