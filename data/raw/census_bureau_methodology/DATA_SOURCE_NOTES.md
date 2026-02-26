# Census Bureau Methodology Source Notes

## Overview

This directory stores raw methodology references from the Census Bureau
(file-layout PDFs, methodology memos, and related technical references)
used to interpret PEP and related source datasets.

## Source

- Publisher: U.S. Census Bureau
- Reference landing page: https://www.census.gov/programs-surveys/popest/technical-documentation.html

## Typical file provenance fields to capture per file

When adding files under this directory, document:
- file name and what methodological question it supports
- source URL
- download date (UTC)
- relevant sections/pages used by project scripts or ADRs

## Processing linkage

These files support assumptions and parsing logic in data ingestion/extraction
scripts and ADR documentation.

## Historical notes

- Added 2026-02-26 to satisfy repository requirement that each
  `data/raw/{category}` directory has `DATA_SOURCE_NOTES.md`.
