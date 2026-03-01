# 2026-03-01 PP3 IMP-13A Implementation Results

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-03-01 |
| **Scope** | PP-003 IMP-13A: reconciliation-magnitude QA visibility (Gate 4 follow-up) |
| **Status** | Completed |
| **Related Artifacts** | `docs/reviews/2026-03-01-pp3-gate4-rescaling-review-note.md`, `docs/reviews/2026-03-01-pp3-human-review-package.html` |

## Summary

Implemented the non-blocking Gate 4 follow-up approved in the IMP-12/IMP-13 review cycle:

- Added a new county-year QA artifact, `qa_reconciliation_magnitude.csv`.
- Integrated reconciliation-magnitude diagnostics into the PP3 human-review package Gate 4 tab.
- Added invariant-style tests for the new artifact schema and reconciliation relationships.

## Implementation Details

### 1. New QA Artifact: `qa_reconciliation_magnitude.csv`

Implemented in `cohort_projections/data/process/place_projection_orchestrator.py`:

- Artifact path: `data/projections/{scenario}/place/qa/qa_reconciliation_magnitude.csv`
- Emitted fields:
  - `county_fips`, `county_name`, `year`
  - `total_before_adjustment`
  - `total_after_adjustment`
  - `reconciliation_adjustment`
  - `reconciliation_flag`
  - `reconciliation_flag_threshold`
  - `rescaling_applied`

County-year consistency guards were added to raise if reconciliation metadata is inconsistent within a county-year slice.

### 2. Gate 4 Human-Review Package Integration

Implemented in `scripts/reviews/build_pp3_human_review_package.py`:

- Loads `qa_reconciliation_magnitude.csv` as required input.
- Gate 1 completeness now includes reconciliation artifact row counts.
- Gate 4 now includes:
  - Reconciliation adjustment distribution visual:
    `reconciliation_adjustment_distribution`
  - Adjustment summary metrics (mean, median, p95, max, threshold counts)
  - Top county-years by adjustment table

### 3. Tests Added/Updated

Updated `tests/test_data/test_place_qa_artifacts.py` with invariant checks for:

- New artifact schema.
- `reconciliation_adjustment >= 0`.
- `reconciliation_adjustment ~= abs(total_before_adjustment - 1.0)`.
- `total_after_adjustment ~= 1.0`.
- `reconciliation_flag == (reconciliation_adjustment > reconciliation_flag_threshold)`.

## Verification Commands and Results

- `source .venv/bin/activate && ruff check cohort_projections/data/process/place_projection_orchestrator.py scripts/reviews/build_pp3_human_review_package.py tests/test_data/test_place_qa_artifacts.py`
  - `All checks passed!`
- `source .venv/bin/activate && mypy cohort_projections/data/process/place_projection_orchestrator.py scripts/reviews/build_pp3_human_review_package.py tests/test_data/test_place_qa_artifacts.py`
  - `Success: no issues found in 3 source files`
- `source .venv/bin/activate && pytest tests/test_data/test_place_qa_artifacts.py tests/test_data/test_place_consistency_constraints.py tests/test_data/test_place_projection_orchestrator.py tests/test_integration/test_place_pipeline_stage.py`
  - `11 passed`
- `source .venv/bin/activate && python scripts/pipeline/02a_run_place_projections.py`
  - Successful stage execution for `baseline`, `restricted_growth`, `high_growth`
- `source .venv/bin/activate && python scripts/reviews/build_pp3_human_review_package.py --refresh-latest`
  - Successfully regenerated versioned package and refreshed canonical latest HTML/manifest

## Runtime Snapshot (Post-Implementation)

From regenerated QA artifacts:

- `qa_reconciliation_magnitude.csv` rows per scenario: `1,426` (`46` counties × `31` years)
- Max `reconciliation_adjustment` (all scenarios): `0.048411`
- `reconciliation_flag=true` rows (all scenarios): `0` with current threshold settings

