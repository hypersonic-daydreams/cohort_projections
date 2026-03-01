# PP-003 IMP-19 End-to-End Validation

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-03-01 |
| **Scope** | PP-003 IMP-19 full end-to-end validation across active scenarios |
| **Status** | Approved (human sign-off completed 2026-03-01) |
| **Related Plan** | `docs/plans/pp3-s08-implementation-kickoff.md` |
| **Approval Record** | `docs/reviews/2026-03-01-pp3-imp19-approval-gate.md` |

## Execution Summary

Full place pipeline execution was run for all active scenarios using project scripts:

1. `source .venv/bin/activate && python scripts/pipeline/02a_run_place_projections.py --scenarios baseline restricted_growth high_growth`
2. `source .venv/bin/activate && python scripts/pipeline/03_export_results.py --places --scenarios baseline restricted_growth high_growth --formats parquet --no-package`

Observed stage completion for each scenario:

- `baseline`: `90` places, `46` balance-of-county rows, `31` QA outlier flags
- `restricted_growth`: `90` places, `46` balance-of-county rows, `31` QA outlier flags
- `high_growth`: `90` places, `46` balance-of-county rows, `31` QA outlier flags

Observed export completion:

- Export pipeline summary: `10` components, `10 successful`, `0 failed`, `14 files exported`
- Place workbook artifacts generated for all three scenarios

## IMP-19 Validation Checks

| Check | Result | Evidence |
|------|------|------|
| 1. All 90 projected places produce output for all three scenarios | **PASS** | Per scenario: `90` place parquet + `90` metadata JSON + `90` per-place summary CSV files |
| 2. All hard constraints pass (S06 §6.1) | **PASS** | `qa_share_sum_validation.csv`: all `constraint_satisfied=True`; max absolute share-sum error `2.22e-16`; `qa_balance_of_county.csv` shows `sum_of_places <= county_total` for all county-years; no negative populations (`0` negative rows) |
| 3. QA artifacts generated and reviewed | **PASS** | All required files exist per scenario: `qa_tier_summary.csv`, `qa_share_sum_validation.csv`, `qa_outlier_flags.csv`, `qa_balance_of_county.csv`, `qa_reconciliation_magnitude.csv` |
| 4. Scenario ordering holds at state level (`restricted <= baseline <= high`) | **PASS** | Checked across years `2025-2055` (`31` years); ordering true for all years |
| 5. Place workbooks generated for all scenarios | **PASS** | `data/exports/nd_projections_baseline_places_20260301.xlsx`, `...restricted_growth...`, `...high_growth...` |
| 6. `places_summary.csv` contains 90 place rows + balance rows | **PASS** | Per scenario row counts: `place=90`, `balance_of_county=46` (total `136`) |
| 7. `places_metadata.json` contains correct counts | **PASS** | Per scenario: `num_geographies=90`, `successful=90`, `failed=0`, `by_tier={HIGH:9, MODERATE:9, LOWER:72}` |
| 8. Place totals vs county totals (`sum(places) <= county`) | **PASS** | Validated from `qa_balance_of_county.csv` across all county-years; no violations |

## Metric Snapshot

- QA county-year rows per scenario: `1,426` (`46 counties x 31 years`)
- Outlier flag rows per scenario: `31`
- Outlier type mix per scenario:
  - `SHARE_RESCALED=20`
  - `SHARE_REVERSAL=11`
- Reconciliation hard-flag count: `0`
- Reconciliation rescaling applied count: `530`
- Minimum projected place population value: `12.612609089445753`

## Soft-Flag Notes (Non-Blocking)

1. Share-stability soft flags are present (31 rows/scenario), dominated by expected rescaling/reversal diagnostics (`SHARE_RESCALED`, `SHARE_REVERSAL`); these are QA signals, not hard failures.
2. Export pipeline logs warnings when attempting age/sex/race/dependency summaries for place-level files that do not expose those columns uniformly (tier-driven schema differences). Export run still completed successfully with place artifacts and workbooks.
3. State scenario-ordering minimum margins are `0.0` in at least one year (equality case), which is acceptable under non-strict ordering (`<=`).

## Disposition

IMP-19 implementation acceptance checks are satisfied on this run. Human sign-off is now recorded as **Approved** in `docs/reviews/2026-03-01-pp3-imp19-approval-gate.md`, and PP-003 proceeds to publication-facing closure actions (IMP-20, IMP-21, IMP-22 final status updates).
