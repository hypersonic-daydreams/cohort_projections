# examples/

Status: current · Last Updated: 2026-06-23

Orientation README for the `examples/` directory. These are **standalone demo scripts**
that illustrate how to use the `cohort_projections` library end to end. They are teaching
and smoke-test artifacts, not part of the production pipeline (the production pipeline
lives in `scripts/pipeline/` and `scripts/projections/`). Several examples generate
*synthetic* input data so they can run without the real, gitignored datasets.

## What each example illustrates

- `run_basic_projection.py` — the minimal path: load/create input data (population,
  fertility, survival, migration), initialize the projection engine, run a projection, and
  inspect results. Start here.
- `run_multi_geography_example.py` — multi-geography projections (counties and places) with
  parallel processing, plus hierarchical aggregation (places → counties → state) and its
  validation.
- `process_fertility_example.py` — the fertility pipeline from raw SEER data to
  projection-ready rates.
- `process_survival_example.py` — converting SEER/CDC life tables into survival rates, with
  life-expectancy validation.
- `process_migration_example.py` — the migration pipeline (synthetic IRS county-to-county +
  international data) through the distribution pipeline into projection-ready rates.
- `fetch_census_data.py` — using `CensusDataFetcher` to pull PEP/ACS data from the Census
  Bureau APIs (needs a Census API key / network).
- `generate_outputs_example.py` — the output layer: exporting a run to Excel/CSV/Parquet/JSON,
  generating summary statistics and reports, building the standard visualizations, comparing
  scenarios, and assembling a complete stakeholder package.

## How to run

Run from the project root with the virtual environment active:

```bash
python examples/run_basic_projection.py
python examples/process_fertility_example.py
python examples/generate_outputs_example.py
```

Most examples are self-contained (they synthesize their own inputs). The exceptions need
external resources: `fetch_census_data.py` requires Census API access, and any example that
exercises the output layer may require the optional `--extra dev`/`--extra dashboard`
dependencies (matplotlib, openpyxl). For real production runs use the pipeline scripts in
`scripts/`, not these demos.
