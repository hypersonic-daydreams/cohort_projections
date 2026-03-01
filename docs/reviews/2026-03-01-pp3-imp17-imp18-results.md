# 2026-03-01 PP3 IMP-17/IMP-18 Implementation Results

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-03-01 |
| **Scope** | PP-003 IMP-17 (methodology text update) + IMP-18 (integration tests) |
| **Status** | Implemented; IMP-17 pending human wording/tone sign-off |
| **Related Contract** | `docs/plans/pp3-s08-implementation-kickoff.md` |

## Summary

Completed the next planned PP-003 integration steps after IMP-16:

- **IMP-17**: Updated shared export methodology constants and publication methodology text with explicit place-model details, including winner variant `B-II` (`wls` + `cap_and_redistribute`) and county-constrained accounting language.
- **IMP-18**: Added an end-to-end synthetic integration test that runs the stage entrypoint and export pipeline (`--places`) with real orchestrator logic and validates QA constraints plus workbook output.

## Implementation Details

### 1. IMP-17 Methodology Text Update

Updated:

- `scripts/exports/_methodology.py`
  - Added ADR-033 to source traceability.
  - Expanded `PLACE_METHODOLOGY_LINE` to include:
    - method name (`share-of-county trending`)
    - winner variant (`B-II`, `wls` + `cap_and_redistribute`)
    - tier definitions
    - county constraint statement (`place + balance = county` by year)
- `docs/methodology.md`
  - Reworked Section **7.5 Place Projection Orchestration** to include:
    - logit-linear share specification
    - explicit winner variant statement
    - confidence-tier thresholds and output granularity
    - current QA artifact set, including `qa_reconciliation_magnitude.csv`
    - hard-constraint enforcement framing
  - Updated ADR table cross-reference for ADR-033 section pointer (`7.5`).

### 2. IMP-18 Integration Test

Added:

- `tests/test_integration/test_place_projection_integration.py`

The new test:

1. Creates synthetic county projection inputs (subset scope).
2. Builds synthetic place share history through `compute_historical_shares(...)` (IMP-03 module path).
3. Runs `scripts/pipeline/02a_run_place_projections.py` with a synthetic winner payload (`B-II`).
4. Verifies generated place outputs and all required QA artifacts.
5. Confirms invariant relationships:
   - `sum_place_shares + balance_share == 1.0`
   - `sum_of_places + balance_of_county == county_total`
   - `total_after_adjustment == 1.0`
6. Runs `scripts/pipeline/03_export_results.py --places`.
7. Verifies exported place summary, workbook creation, converted CSV outputs, and data dictionary files.

## Verification Commands and Results

- `source .venv/bin/activate && ruff check scripts/exports/_methodology.py tests/test_output/test_place_workbook.py tests/test_integration/test_place_projection_integration.py`
  - `All checks passed!`
- `source .venv/bin/activate && mypy scripts/exports/_methodology.py tests/test_output/test_place_workbook.py tests/test_integration/test_place_projection_integration.py`
  - `Success: no issues found in 3 source files`
- `source .venv/bin/activate && pytest tests/test_output/test_place_workbook.py tests/test_integration/test_place_projection_integration.py`
  - `2 passed`

## Notes

- IMP-17 kickoff guidance marks final wording/tone review as a human gate; this implementation is ready for that sign-off pass.
