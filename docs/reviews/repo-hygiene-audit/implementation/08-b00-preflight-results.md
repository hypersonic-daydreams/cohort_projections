# B00 Preflight Results

**Batch:** `B00` (`baseline_metrics_and_safety_harness`)  
**Date:** 2026-02-26  
**Operator:** codex

## Dry-Run Profiles

1. `DRY-BASELINE`  
Status: **pass**  
Window: `2026-02-26T17:50:28Z -> 2026-02-26T17:50:29Z`  
Evidence: `./dry-baseline-latest.log`

2. `DRY-CHECK-REPLAY`  
Status: **pass**  
Window: `2026-02-26T17:55:48Z -> 2026-02-26T17:55:51Z`  
Evidence: `./dry-check-replay-latest.log`  
Result summary: 27/27 adjudicated claim checks passed in replay.

## Quality Gates Snapshot

1. `ruff check .`  
Status: **fail** (baseline debt)  
Summary: `124` issues reported.

2. `mypy cohort_projections`  
Status: **fail** (baseline debt)  
Summary: `3` errors in `cohort_projections/data/load/base_population_loader.py`.

3. `pytest tests/ -q`  
Status: **pass**  
Summary: `1258 passed, 5 skipped, 33 warnings in 269.46s (0:04:29)`.

## Decision Interpretation for B00

- B00 remains **GO** because it is a planning/baseline batch with no production behavior changes.
- `ruff`/`mypy` failures are recorded as baseline repo debt to be addressed in later implementation batches.
