# config/

Status: current · Last Updated: 2026-06-23

Orientation README for the repository-root `config/` directory. This directory holds
the **runtime YAML configuration** that drives projections, evaluation, and the
Observatory. It is the authoritative configuration for *what the pipeline does* — not to
be confused with the in-package `cohort_projections/config/` (Python constants; see below).

## What lives here

- `projection_config.yaml` — **the master projection config.** Geography scope, base year,
  horizon, method dispatch, and per-method parameters. Loaded by
  `cohort_projections/utils/config_loader.py`, which resolves it as
  `<project_root>/config/projection_config.yaml` by default. This is the file the
  production pipeline reads.
- `method_profiles/` — versioned **method × config profiles** used by the benchmarking /
  Observatory machinery. Filenames follow `<method_id>__<config_id>.yaml`
  (e.g. `m2026r1__cfg-20260611-production-lock.yaml` is the current locked public run).
  `aliases.yaml` maps logical roles (`county_champion`, `county_candidate`,
  `county_reference`) to a specific `method_id` + `config_id` so tooling can refer to
  "the champion" without hard-coding a profile filename.
- `observatory_search_packs/` — search-pack YAML (e.g. `cf001.yaml`) that scopes an
  Observatory autonomous-search session to a curated set of variants/parameters.
- Observatory + evaluation configs: `observatory_config.yaml`, `observatory_variants.yaml`,
  `observatory_recipes.yaml`, `observatory_search_policy.yaml`, `evaluation_config.yaml`,
  `benchmark_evaluation_policy.yaml`, `experiment_spec_schema.yaml`,
  `experiment_log_schema.yaml`.
- Data and branding: `data_sources.yaml` (sibling-repo data fetch map, read by
  `scripts/fetch_data.py`) and `nd_brand.yaml` (export styling).

## `config/` vs. `cohort_projections/config/` — which is authoritative

These are two different things and serve different roles:

- **`config/` (this directory)** is **runtime tunables as data** — YAML that the pipeline,
  benchmarking, and Observatory read at run time. To change projection behavior (rates,
  scope, method parameters, which profile is the champion), edit YAML here. This is the
  authoritative source for *configuration values*.
- **`cohort_projections/config/`** is a **Python package of fixed domain constants and
  mappings** — chiefly `race_mappings.py` (the canonical race/ethnicity categories and the
  source-specific crosswalks for Census/SEER/IRS), imported as
  `from cohort_projections.config import ...`. These are code-level invariants, not
  operator-tunable settings, and they ship inside the installed package.

Rule of thumb: if a value is something an analyst would tune between runs, it belongs in
this `config/` directory's YAML; if it is a structural mapping the code depends on
(canonical categories, source crosswalks), it belongs in `cohort_projections/config/`.

## Conventions and cautions

Profile filenames embed a date-stamped `config_id`; the run-id ↔ config-sha mapping for
the locked public run is tracked in the project trackers and ADRs (the production lock is
`m2026r1` / `cfg-20260611-production-lock`). Per the repository hygiene proposal, verify
that no tooling globs `projection_config.yaml` by exact name before any rename — the
config loader resolves it by that literal path.
