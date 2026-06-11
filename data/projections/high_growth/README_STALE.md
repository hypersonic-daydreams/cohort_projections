# STALE OUTPUTS — DO NOT USE

**Status:** Quarantined (PUB-2026 finality remediation, Stage 0.3)
**Last Updated:** 2026-06-10

## What this directory contains

`high_growth` scenario outputs generated **2026-02-26** under the
**pre-ADR-065/066 method and base population** (county/state run
`metadata/projection_run_20260226_150927.json`, processing date
2026-02-26T15:09:27Z; the `place/` subtree was derived from those county
outputs on 2026-03-01/02). State total: 799,358 (2025) → 1,078,346 (2055).

## Why it is stale

ADR-065 (2026-05-27) made the CBO-adjusted `baseline` the only active
production scenario and retained `high_growth` only as an **inactive internal
elevated-growth sensitivity**. ADR-066 (2026-05-27) corrected the county
base-population source to Census PEP Vintage 2025, and the baseline was
regenerated on 2026-05-27.

These on-disk outputs predate both ADRs: they embed the pre-CF-001-merge
configuration and the pre-Vintage-2025-refresh base allocation, and are
**NOT comparable to the current public baseline**
(`data/projections/baseline/`, run of 2026-05-27, 799,358 → 876,479 by 2055).
Any "scenario range" built from this directory against the current baseline
mixes incompatible method vintages.

## Disposition

- Retained pending **Stage 3.4** of
  `docs/plans/2026-public-projection-release-handoff/finality-remediation-plan.md`
  (regenerate under the locked config **or** formally retire).
- **Do not use in exports, comparisons, or dashboards.**
- Files were intentionally left in place (no moves, renames, or deletions).

Note: data files in this directory are gitignored
(`data/projections/**/*.{csv,parquet,xlsx,png}`), and the current rclone
bisync filter (`~/.config/rclone/cohort_projections-bisync-filter.txt`)
excludes `*.md` and does not whitelist `data/projections/**`, so this marker
exists per machine until Stage 3.4 resolves the directory's disposition.
