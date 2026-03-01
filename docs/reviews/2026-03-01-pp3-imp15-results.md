# 2026-03-01 PP3 IMP-15 Implementation Results

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-03-01 |
| **Scope** | PP-003 IMP-15: provisional workbook `Places` sheet wiring |
| **Status** | Completed |
| **Related Contract** | `docs/reviews/2026-02-28-pp3-s06-output-contract.md` |

## Summary

Implemented IMP-15 by extending the provisional workbook generator with scenario-specific place summary sheets:

- Added `Places — Baseline`, `Places — Restricted Growth`, and `Places — High Growth` tabs.
- Each tab includes projected place rows plus balance-of-county rows.
- Columns include place identifiers, county/tier labels, key-year populations, and growth metrics.
- Legacy provisional workbook sheets (`State*`, `Counties*`, `Growth Rankings`, `Age Structure`) remain intact.

## Implementation Details

### 1. Provisional Workbook Place-Sheet Builder

Updated `scripts/exports/build_provisional_workbook.py`:

- Added place-level loaders:
  - `load_place_summaries()`
  - `load_place_yearly_totals()`
  - `load_balance_yearly_totals()`
- Added `build_places_detail(...)`:
  - creates `Places — {scenario_short}` sheet per scenario
  - merges place summary metadata with key-year totals from per-place parquet outputs
  - pulls balance-of-county key-year totals from `qa/qa_balance_of_county.csv`
  - writes formatted table with:
    - `Place FIPS`, `Place`, `County`, `Row Type`, `Tier`
    - key-year columns (`2025, 2030, 2035, 2040, 2045, 2050, 2055`)
    - total change and percent change (`2025–2055`)
- Added new `Places` entries to workbook TOC metadata and build flow.

### 2. New Contract Test Coverage

Added `tests/test_output/test_provisional_workbook_places.py`:

- Builds a synthetic provisional workbook end-to-end.
- Verifies `Places` sheets exist for all active scenarios.
- Verifies table headers/columns include IMP-15 required fields.
- Verifies each scenario tab includes place rows and balance rows.
- Verifies legacy county/state sheet set remains present and unchanged in count.

## Verification Commands and Results

- `source .venv/bin/activate && ruff check scripts/exports/build_provisional_workbook.py tests/test_output/test_provisional_workbook_places.py`
  - `All checks passed!`
- `source .venv/bin/activate && mypy scripts/exports/build_provisional_workbook.py tests/test_output/test_provisional_workbook_places.py`
  - `Success: no issues found in 2 source files`
- `source .venv/bin/activate && pytest tests/test_output/test_provisional_workbook_places.py tests/test_output/test_place_workbook.py`
  - `2 passed`
- `source .venv/bin/activate && python scripts/exports/build_provisional_workbook.py`
  - Generated `data/exports/nd_population_projections_provisional_20260301.xlsx`
- `source .venv/bin/activate && pytest`
  - `1437 passed, 5 skipped` in `266.13s`

## Runtime Snapshot

From `data/exports/nd_population_projections_provisional_20260301.xlsx`:

- Workbook sheet count: `12`
- New place sheets:
  - `Places — Baseline`: `136` rows (`90` place + `46` balance)
  - `Places — Restricted Growth`: `136` rows (`90` place + `46` balance)
  - `Places — High Growth`: `136` rows (`90` place + `46` balance)
