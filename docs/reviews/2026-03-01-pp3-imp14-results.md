# 2026-03-01 PP3 IMP-14 Implementation Results

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-03-01 |
| **Scope** | PP-003 IMP-14: standalone place workbook builder |
| **Status** | Completed |
| **Related Contract** | `docs/reviews/2026-02-28-pp3-s06-output-contract.md` |

## Summary

Implemented a new standalone place workbook builder that satisfies S06 Section 7.2:

- New script: `scripts/exports/build_place_workbook.py`
- Output pattern: `data/exports/nd_projections_{scenario}_places_{datestamp}.xlsx`
- Workbook structure per scenario:
  - `Table of Contents`
  - `9` HIGH-tier place sheets
  - `9` MODERATE-tier place sheets
  - `1` combined `LOWER Tier` sheet
  - `1` `Methodology` sheet

The builder reads canonical PP-003 outputs from `data/projections/{scenario}/place/` and writes workbook exports using configured key years.

## Implementation Details

### 1. New Place Workbook Builder Script

Implemented in `scripts/exports/build_place_workbook.py`:

- Reads active scenarios from config (`--scenarios` override supported).
- Loads `places_summary.csv` and filters to projected place rows (`row_type == "place"`).
- Loads per-place parquet outputs and renders tier-specific sheets:
  - HIGH: `18` five-year age groups x `2` sex columns at each key year
  - MODERATE: `6` broad age groups x `2` sex columns at each key year
  - LOWER: one combined table for all `72` places with total population at key years
- Adds TOC hyperlinks for all projected places.
- Adds a prominent LOWER-tier uncertainty caveat header.
- Writes one workbook per scenario in `pipeline.export.output_dir`.

### 2. Shared Methodology Constant

Updated `scripts/exports/_methodology.py`:

- Added `PLACE_METHODOLOGY_LINE` constant:
  - Share-of-county trending method reference (ADR-033)
  - Tier definitions (HIGH/MODERATE/LOWER)
  - County-constraint statement

The new constant is rendered on the place workbook `Methodology` sheet.

### 3. Test Coverage

Added `tests/test_output/test_place_workbook.py` with synthetic-data contract validation:

- Verifies sheet count and tier-sheet composition (`21` sheets total).
- Verifies HIGH and MODERATE table structures (expected age rows and key-year sex columns).
- Verifies LOWER combined sheet has `72` place rows and caveat text styling.
- Verifies TOC hyperlinks for all `90` projected places.
- Verifies methodology sheet includes place-specific method text.
- Verifies output filename pattern.

## Verification Commands and Results

- `source .venv/bin/activate && ruff check scripts/exports/build_place_workbook.py scripts/exports/_methodology.py tests/test_output/test_place_workbook.py`
  - `All checks passed!`
- `source .venv/bin/activate && mypy scripts/exports/build_place_workbook.py tests/test_output/test_place_workbook.py`
  - `Success: no issues found in 2 source files`
- `source .venv/bin/activate && pytest tests/test_output/test_place_workbook.py`
  - `1 passed`
- `source .venv/bin/activate && python scripts/exports/build_place_workbook.py`
  - Generated scenario workbooks for `baseline`, `restricted_growth`, and `high_growth`

## Runtime Snapshot

From 2026-03-01 execution:

- `nd_projections_baseline_places_20260301.xlsx`: `21` sheets (`9` HIGH + `9` MODERATE + `1` LOWER + TOC + Methodology)
- `nd_projections_restricted_growth_places_20260301.xlsx`: `21` sheets
- `nd_projections_high_growth_places_20260301.xlsx`: `21` sheets

All scenarios used the configured key-year set:
`2025, 2030, 2035, 2040, 2045, 2050, 2055`.
