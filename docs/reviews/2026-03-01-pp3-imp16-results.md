# 2026-03-01 PP3 IMP-16 Implementation Results

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-03-01 |
| **Scope** | PP-003 IMP-16: export pipeline `--places` flag wiring |
| **Status** | Completed |
| **Related Contract** | `docs/plans/pp3-s08-implementation-kickoff.md` |

## Summary

Implemented `--places` wiring in `scripts/pipeline/03_export_results.py` so the export pipeline now performs PP-003 place artifact generation when place level is selected (`--places` or `--all`):

- Copies scenario-level `places_summary.csv` from `data/projections/{scenario}/place/` into export outputs.
- Builds standalone place workbook via IMP-14 builder integration.
- Honors dry-run semantics for place artifacts (logs only, no place files written).
- Extends integration test coverage for `--places`, `--all`, and dry-run behavior.

## Implementation Details

### 1. Export Pipeline Wiring

Updated `scripts/pipeline/03_export_results.py`:

- Added `export_place_outputs(...)` for per-scenario place export artifacts.
- Added `_build_place_workbook(...)` helper to invoke `scripts.exports.build_place_workbook.build_workbook`.
- Integrated place export step into `export_all_results(...)` when `"place"` is in selected levels.
- Updated main flow to skip writing `export_report_*.json` during `--dry-run`, ensuring no dry-run file artifacts.

### 2. New Integration Tests

Added `tests/test_integration/test_export_places.py`:

- `test_places_flag_exports_place_summary_and_workbook`
  - Verifies `--places` emits place summary CSV + workbook artifact.
- `test_all_flag_includes_place_export`
  - Verifies `--all` includes place level in export path and emits place summary output.
- `test_places_dry_run_creates_no_files`
  - Verifies `--dry-run --places` does not create place files or report artifacts.

## Verification Commands and Results

- `source .venv/bin/activate && ruff check scripts/pipeline/03_export_results.py tests/test_integration/test_export_places.py`
  - `All checks passed!`
- `source .venv/bin/activate && mypy scripts/pipeline/03_export_results.py tests/test_integration/test_export_places.py`
  - `Success: no issues found in 2 source files`
- `source .venv/bin/activate && pytest tests/test_integration/test_export_places.py`
  - `3 passed`
- `source .venv/bin/activate && pytest tests/test_integration/test_export_places.py tests/test_output/test_place_workbook.py tests/test_output/test_provisional_workbook_places.py`
  - `5 passed, 1 warning`
- `source .venv/bin/activate && pytest`
  - `1440 passed, 5 skipped, 1 warning in 263.68s (0:04:23)`

## Notes

- Full-suite warning is pre-existing pandas `FutureWarning` in `scripts/exports/build_provisional_workbook.py` (`pivot_table observed` default change); IMP-16 did not modify this module.
