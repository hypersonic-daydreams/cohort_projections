# ADR-066: Vintage 2025 PEP Base Population Refresh

**Status:** Accepted
**Date:** 2026-05-27
**Implemented:** 2026-05-27

## Context

The PUB-2026 public release uses 2025 as the projection base year. The Census
Bureau Vintage 2025 Population Estimates Program (PEP) county totals are now
available, and the release materials already cite the Vintage 2025 North Dakota
state total of 799,358.

The production county base-population loader was still able to fall back
silently to `data/raw/population/nd_county_population.csv`, whose current
population column is `population_2024` and sums to 796,568. That behavior could
produce a 2025 baseline projection anchored to a 2024 county total while the
methodology text and group-quarters inputs referenced Vintage 2025.

The shared Census PEP archive already contains the relevant Vintage 2025 county
totals at:

- `parquet/2020-2025/county/co-est2025-alldata.parquet`
- `csv/2020-2025/county/co-est2025-alldata.csv`

The same source file is also staged in the release scratchpad under
`../demography_scratchpad/2025_City_PopEst_Release/source-data/co-est2025-alldata.csv`.

## Decision

The default production `load_county_populations()` path now loads North Dakota
county totals from the shared Census PEP Vintage 2025 `co-est2025-alldata`
archive and uses `POPESTIMATE2025` as the county base total.

The legacy repo CSV remains supported only as an explicit path or as a fallback
when the shared PEP archive is unavailable. When a legacy CSV lacks
`population_2025`, the loader chooses the latest available `population_YYYY`
column and logs a warning rather than silently taking the first population-like
column.

Detailed age-sex-race allocation remains based on the latest detailed
characteristics files currently available to the projection system:

- county-specific age-sex-race distributions from `cc-est2024` data
- state single-year age-sex-race distribution from `SC-EST2024`

Those distributions are scaled to the Vintage 2025 county totals. If Census
releases Vintage 2025 county or state detailed characteristics files for the
same schema, those distributions should be refreshed in a later ADR-backed data
processing update.

## Consequences

Positive:

- The baseline projection base total now matches Census PEP Vintage 2025:
  North Dakota total population 799,358.
- The CBO-adjusted public baseline and its exports no longer begin from stale
  2024 county totals.
- The loader fails loudly on malformed Vintage 2025 PEP files and warns on
  legacy fallback behavior.

Trade-offs:

- The county age-sex-race structure is still a Vintage 2024 characteristics
  distribution scaled to Vintage 2025 totals. This is preferable to publishing
  2024 totals as the 2025 base, but it should be revisited when matching
  Vintage 2025 detailed characteristics files are available.
- Default local runs now depend on the shared Census PEP archive documented in
  ADR-034. Environments without that archive still fall back to the legacy CSV,
  with a warning.

## Implementation Results

- Updated `cohort_projections/data/load/base_population_loader.py` so the
  default county-population source resolves
  `parquet/2020-2025/county/co-est2025-alldata.parquet` from the shared PEP
  archive.
- Added regression coverage for default Vintage 2025 PEP loading and legacy CSV
  population-column selection.
- Regenerated production baseline projections and exports after the loader
  correction.
