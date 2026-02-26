# Geographic Raw Data Source Notes

## Overview

This directory stores raw geographic reference files used by the projection
system (county/place identifiers, boundaries, and crosswalk-style reference
inputs).

## Source

- Primary sources: U.S. Census Bureau geographic products and project partner
  reference extracts used for North Dakota geography alignment.

## Typical file provenance fields to capture per file

When adding files under this directory, document:
- file name and geographic scope
- source URL
- download date (UTC)
- key identifier columns (e.g., FIPS/GEOID)
- processing script(s) that consume the file

## Processing linkage

Inputs are used by geographic loaders and projection scripts to enforce
state->county->place consistency.

## Historical notes

- Added 2026-02-26 to satisfy repository requirement that each
  `data/raw/{category}` directory has `DATA_SOURCE_NOTES.md`.
