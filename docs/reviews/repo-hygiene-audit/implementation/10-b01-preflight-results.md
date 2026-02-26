# B01 Preflight and Replay Results

**Batch:** `B01` (`documentation_harmonization`)  
**Date:** 2026-02-26  
**Operator:** codex

## Required Pre-Edit Dry-Run Profiles

1. `DRY-DOCS`  
Status: **pass**  
Window: `2026-02-26T18:24:13Z -> 2026-02-26T18:24:14Z`  
Evidence: `./dry-docs-b01-preflight.log`

2. `DRY-CHECK-REPLAY`  
Status: **pass**  
Window: `2026-02-26T18:24:19Z -> 2026-02-26T18:24:23Z`  
Evidence: `./dry-check-replay-b01-preflight.log`  
Result summary: 27/27 adjudicated claim checks passed before B01 edits.

## Post-Edit Replay (Affected B01 Claims)

Replay log: `./check-replay-b01-affected-postedit.log`

1. `RHA-009`  
Status: **fail (expected after remediation)**  
Run: `2026-02-26T18:32:31Z`  
Evidence: `../verification/evidence/20260226T183231Z_RHA-009.json`  
Key output: `MISMATCH_COUNT=0`, `MISSING_COUNT=0`

2. `RHA-019`  
Status: **fail (expected after remediation)**  
Run: `2026-02-26T18:32:31Z`  
Evidence: `../verification/evidence/20260226T183231Z_RHA-019.json`  
Key output: `INDEX_EXISTS=1`, `NAVIGATION_EXISTS=0`, `AGENTS_DOC_INDEX_REF=1`

3. `RHA-020`  
Status: **fail (expected after remediation)**  
Run: `2026-02-26T18:32:31Z`  
Evidence: `../verification/evidence/20260226T183231Z_RHA-020.json`  
Key output: `README_2025_2045_COUNT=0`, `AGENTS_2025_2055_COUNT=1`

4. `RHA-021`  
Status: **fail (expected after remediation)**  
Run: `2026-02-26T18:32:31Z`  
Evidence: `../verification/evidence/20260226T183231Z_RHA-021.json`  
Key output: `DEVELOPMENT_TRACKER_LINES=67`

5. `RHA-022`  
Status: **fail (expected after remediation)**  
Run: `2026-02-26T18:32:31Z`  
Evidence: `../verification/evidence/20260226T183231Z_RHA-022.json`  
Key output: `PROCESS_README_LINES=74`

## Decision Interpretation for B01

- B01 pre-edit gates were **GO** and implementation proceeded within docs-only scope.
- Post-edit affected claim replays now fail because current checks assert the pre-fix problematic state.
- These post-edit failures are retained as evidence of remediation, and claim-check definitions require a future refresh to validate resolved-state conditions directly.
