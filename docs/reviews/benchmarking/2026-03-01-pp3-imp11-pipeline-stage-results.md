# 2026-03-01 PP-003 IMP-11 Pipeline Stage Results

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-03-01 |
| **Scope** | IMP-11 implementation and execution of place projection pipeline stage |
| **Status** | Completed |
| **Related ADR** | ADR-033 |
| **Stage Script** | `scripts/pipeline/02a_run_place_projections.py` |

## 1. Implementation Summary

- Added new pipeline stage script:
  - `scripts/pipeline/02a_run_place_projections.py`
- Script behavior:
  - Loads accepted backtest winner from `data/backtesting/place_backtest_results/backtest_winner.json`
  - Resolves scenarios from config active set (or CLI override)
  - Runs `run_place_projections(...)` per scenario
  - Supports `--dry-run` dependency validation
- Wired stage into complete pipeline runner:
  - `scripts/pipeline/run_complete_pipeline.sh`
  - Step count updated `7 -> 8` with new Step 7: `Running Place Projections`

## 2. Validation Evidence

### 2.1 Static checks

- `source .venv/bin/activate && ruff check scripts/pipeline/02a_run_place_projections.py tests/test_integration/test_place_pipeline_stage.py`
  - `All checks passed!`
- `source .venv/bin/activate && mypy scripts/pipeline/02a_run_place_projections.py`
  - `Success: no issues found in 1 source file`

### 2.2 Integration tests

- `source .venv/bin/activate && pytest tests/test_integration/test_place_pipeline_stage.py`
  - `3 passed`
  - Coverage assertions include:
    - `--dry-run` dependency validation path
    - Active-scenario execution (`baseline`, `restricted_growth`, `high_growth`)
    - Explicit `--scenarios` override behavior

### 2.3 Runtime stage execution

- Dry-run:
  - `source .venv/bin/activate && python scripts/pipeline/02a_run_place_projections.py --dry-run`
  - Result: winner payload + county input directories validated for all active scenarios
- Full run:
  - `source .venv/bin/activate && python scripts/pipeline/02a_run_place_projections.py`
  - Result:
    - `baseline`: `90` places, `46` balance rows
    - `restricted_growth`: `90` places, `46` balance rows
    - `high_growth`: `90` places, `46` balance rows

## 3. Output Verification

For each active scenario, stage outputs are present under:

- `data/projections/baseline/place/`
- `data/projections/restricted_growth/place/`
- `data/projections/high_growth/place/`

Each directory includes scenario-level contract outputs:

- `places_summary.csv`
- `places_metadata.json`
- per-place parquet/metadata/summary files (`nd_place_{place_fips}_projection_2025_2055_{scenario}.*`)

## 4. Notes

- Stage consumes approved winner `B-II` (recorded in `backtest_winner.json` and IMP-10 approval narrative).
- No methodology changes were introduced in IMP-11; this is pipeline integration/wiring.
