# `data/exports/` — Published Deliverables Index

| Attribute | Value |
|-----------|-------|
| **Status** | current (orientation index) |
| **Last Updated** | 2026-06-23 |
| **Purpose** | Explain that exports are generated FROM `data/projections/`, identify the canonical dated set, and flag older dated exports + run logs as provisional/superseded pending archival. |
| **Owner** | cohort_projections |

> **Note:** `data/` is gitignored. This README is an on-disk orienting marker;
> git tracking is handled separately.

---

## What this directory is

`data/exports/` holds the **published deliverables generated FROM**
`data/projections/` — formatted Excel workbooks, summary CSVs, HTML reports,
packaged `.zip` bundles, data dictionaries, and geospatial exports. The
derivation is **one-way: projections → exports**, produced by
`scripts/exports/...` and `scripts/pipeline/03_export_results.py`.

Per **Decision 7** of `docs/REPOSITORY_HYGIENE_PROPOSAL_2026-06-23.md`,
`data/exports/` and `data/projections/` are **kept as separate trees**; this
README and `data/projections/README.md` together document how they relate. See
`data/projections/README.md` for the projection-side classification and the
identity of the current locked run (`baseline/`, `m2026r1`,
`cfg-20260611-production-lock`).

---

## Which dated set is canonical

Export files are dated in their filenames (`..._YYYYMMDD....`) and each export
run writes an `export_report_YYYYMMDD_HHMMSS.json` log.

- **Canonical = the latest dated set, `2026-05-27`** (the most recent full
  export run; `export_report_20260527_195826.json`, 309 files, 5/5 components).
  A small number of files were regenerated later as one-offs and carry their own
  later dates (e.g. `nd_projections_baseline_detail_20260616.xlsx`,
  `nd_methodology_comparison_sdc_20260616.html`).
- **Provisional / superseded = everything dated before `2026-05-27`** — e.g.
  `nd_population_projections_provisional_2026{0217,0223,0301}.xlsx`,
  `nd_*_20260302.html`, the `20251228`–`20260301` package zips — **plus** the
  ~20 accumulated `export_report_*.json` run logs from Dec 2025 → Mar 2026.
  These predate the current public numbers and are **slated for
  `data/exports/archive/`**.

> **⚠️ Important currency caveat.** Even the canonical `2026-05-27` export set
> was generated from the **provisional** baseline, and predates the locked
> production run (2026-06-13) **and** its ADR-068 correction (2026-06-15/16).
> The projection engine outputs in `data/projections/baseline/` are current
> (corrected), but the exports here have **not yet been regenerated** against the
> corrected run. Before publication, re-run the export pipeline against the
> corrected `baseline/` and treat the resulting set as the new canonical one.
> Do not publish numbers straight from the `2026-05-27` exports without that
> regeneration.

## Subdirectories

- `baseline/`, `CBO/`, `methodology_comparison/` — per-scenario / reference
  formatted exports (`county/`, `place/`, `state/`, `summaries/`).
- `packages/` — packaged `.zip` bundles by date; the `20260527` set is the
  latest, older dated zips are superseded.
- `sensitivity/`, `sdc_method_new_data/` — internal analytical export tables.
- `pyramid_explorer/` — the interactive pyramid explorer HTML.
- `exports/geojson/` lives under `data/projections/exports/` (not here) and is
  marked stale (`README_STALE.md`).
- `high_growth/`, `restricted_growth/` here mirror the deprecated stale
  scenarios; their projection-side sources have been moved to
  `data_archives/projections/`.

## Before moving anything to `data/exports/archive/`

**Confirm the date → run mapping against the `export_report_*.json` files
first.** The hygiene proposal flags this as **[verify before acting]**: each
report log records the run's start/end time, components, and exported files —
use it to confirm a dated export truly belongs to a superseded run before
relocating it. Do not move files on filename date alone.
