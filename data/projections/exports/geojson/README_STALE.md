# STALE GEOSPATIAL EXPORTS — DO NOT USE

**Status:** Quarantined (PUB-2026 finality remediation, Stage 0.3)
**Last Updated:** 2026-06-10

## What this directory contains

GeoJSON county/place exports for **all three** scenario subtrees
(`baseline/`, `high_growth/`, `restricted_growth/`), written by the
geospatial export components of `scripts/pipeline/03_export_results.py`
in the export run of 2026-03-02 UTC
(`data/exports/export_report_20260302_033303.json`, components
`geospatial_geojson_{baseline,restricted_growth,high_growth}`).

## Why it is stale

All files here were built from the projection outputs on disk as of
2026-03-01/02 — i.e., the **2026-02-26 county/state runs under the
pre-ADR-065/066 method and base population**. The production baseline was
regenerated on 2026-05-27 (ADR-065 CBO-adjusted public baseline, ADR-066
Vintage 2025 PEP base refresh), and the 2026-05-27 export run
(`data/exports/export_report_20260527_195826.json`) did **not** include a
geospatial component. Therefore:

- `high_growth/` and `restricted_growth/` here mirror the quarantined stale
  scenario outputs (see the `README_STALE.md` in each of
  `data/projections/high_growth/` and `data/projections/restricted_growth/`).
- Even `baseline/` here predates the 2026-05-27 regeneration and is **NOT
  comparable to the current public baseline**
  (`data/projections/baseline/`, run of 2026-05-27).

## Disposition

- Retained pending **Stage 3.4** of
  `docs/plans/2026-public-projection-release-handoff/finality-remediation-plan.md`
  (regenerate under the locked config **or** formally retire), with the
  baseline GeoJSON to be rebuilt from the final locked run (Stage 3.3/5.2).
- **Do not use in exports, comparisons, or dashboards.**
- Files were intentionally left in place (no moves, renames, or deletions).

Note: data files in this directory are gitignored
(`data/projections/**/*.{csv,parquet,xlsx,png}`), and the current rclone
bisync filter (`~/.config/rclone/cohort_projections-bisync-filter.txt`)
excludes `*.md` and does not whitelist `data/projections/**`, so this marker
exists per machine until Stage 3.4 resolves the directory's disposition.
