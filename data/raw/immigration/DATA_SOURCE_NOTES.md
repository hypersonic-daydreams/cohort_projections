# Immigration Raw Data Source Notes

## Overview

This directory stores raw immigration-related source files used for
state/county immigration estimation and scenario construction.

## Source families represented

- Census-derived foreign-born and population-estimate inputs
- DHS series (LPR, naturalizations, nonimmigrants, refugees/asylees)
- State and partner profile files staged for harmonization

## Typical file provenance fields to capture per file

When adding files under this directory, document:
- file name and source agency/program
- source URL
- download date (UTC)
- key columns and coding conventions
- related ingestion/processing script(s)

## Processing linkage

Used by immigration pipeline components under `scripts/data_processing/` and
corresponding loaders/processors under `cohort_projections/data/`.

## Historical notes

- Added 2026-02-26 to satisfy repository requirement that each
  `data/raw/{category}` directory has `DATA_SOURCE_NOTES.md`.
