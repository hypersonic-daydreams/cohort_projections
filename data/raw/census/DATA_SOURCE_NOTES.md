# Census Raw Data Source Notes

## Overview

This directory stores raw U.S. Census source files used by the cohort projection
pipeline, including ACS, decennial, and PEP extracts staged for reproducible
processing.

## Source

- Primary publisher: U.S. Census Bureau
- Primary portal: https://www.census.gov/
- Program families in this directory: ACS, Decennial Census, PEP

## Typical file provenance fields to capture per file

When adding files under this directory, document:
- file name and short description
- source URL
- download date (UTC)
- expected schema/columns
- related processing script(s)

## Processing linkage

Common downstream consumers include scripts under `scripts/data/` and
`scripts/data_processing/`, and loaders under
`cohort_projections/data/load/`.

## Historical notes

- Added 2026-02-26 to satisfy repository requirement that each
  `data/raw/{category}` directory has `DATA_SOURCE_NOTES.md`.
