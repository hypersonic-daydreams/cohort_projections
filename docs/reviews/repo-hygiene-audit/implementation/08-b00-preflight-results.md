# B00 Preflight Results

**Batch:** `B00` (`baseline_metrics_and_safety_harness`)  
**Date:** 2026-02-26  
**Operator:** codex

## Dry-Run Profiles

1. `DRY-BASELINE`  
Status: **pass**  
Window: `2026-02-26T18:16:15Z -> 2026-02-26T18:16:15Z`  
Evidence: `./dry-baseline-latest.log`

2. `DRY-CHECK-REPLAY`  
Status: **pass**  
Window: `2026-02-26T18:16:21Z -> 2026-02-26T18:16:25Z`  
Evidence: `./dry-check-replay-latest.log`  
Result summary: 27/27 adjudicated claim checks passed in replay.

## Post-Change Replay (Affected Claims)

1. `RHA-013`  
Status: **pass**  
Run: `2026-02-26T18:18:33Z`  
Evidence: `../verification/evidence/20260226T181833Z_RHA-013.json`

2. `RHA-023`  
Status: **pass**  
Run: `2026-02-26T18:18:34Z`  
Evidence: `../verification/evidence/20260226T181834Z_RHA-023.json`

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

- B00 remained **GO** because it is a planning/baseline batch with no production behavior changes.
- Post-change replay for the affected claims also passed, so B00 was closed without reopening NO-GO.
- `ruff`/`mypy` failures are recorded as baseline repo debt to be addressed in later implementation batches.
