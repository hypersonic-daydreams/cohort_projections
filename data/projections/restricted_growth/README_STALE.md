# STALE OUTPUTS — DO NOT USE

**Status:** Quarantined (PUB-2026 finality remediation, Stage 0.3)
**Last Updated:** 2026-06-10

## What this directory contains

`restricted_growth` scenario outputs generated **2026-02-26** under the
**pre-ADR-065/066 method and base population** (county/state run
`metadata/projection_run_20260226_150733.json`, processing date
2026-02-26T15:07:33Z; the `place/` subtree was derived from those county
outputs on 2026-03-01/02).

## Why it is stale

ADR-065 (2026-05-27) redefined the production `baseline` as the CBO-adjusted
public projection and retained `restricted_growth` only as a **deprecated,
inactive compatibility alias** for that baseline. ADR-066 (2026-05-27)
corrected the county base-population source to Census PEP Vintage 2025, and
the baseline was regenerated on 2026-05-27.

These on-disk outputs predate both ADRs and are **NOT comparable to the
current public baseline** (`data/projections/baseline/`, run of 2026-05-27):

| Year | This directory (2026-02-26 run) | Current baseline (2026-05-27 run) | Difference |
|------|---------------------------------|-----------------------------------|------------|
| 2045 | 823,275                         | 847,960                           | ~24,684    |
| 2055 | 828,470                         | 876,479                           | ~48,010    |

(State totals from `state/nd_state_38_projection_2025_2055_restricted_growth_summary.csv`
and `../baseline/state/nd_state_38_projection_2025_2055_baseline_summary.csv`.)

So the "deprecated alias = equivalent to baseline" description in ADR-065
does **not** hold for these files — they reflect the older method and config,
not the current baseline.

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
