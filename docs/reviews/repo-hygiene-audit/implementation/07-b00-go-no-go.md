# Repo Hygiene Audit Go/No-Go Checklist (B00)

Use this checklist before implementing any planned batch from `02-action-batches.yaml`.

## Batch Metadata

- Batch ID: `B00`
- Batch Name: `baseline_metrics_and_safety_harness`
- Date: `2026-02-26`
- Owner: `nhaarstad` (approver) / `codex` (execution support)
- Included Claims: `RHA-013`, `RHA-023`
- Dry-Run Profiles Required: `DRY-BASELINE`, `DRY-CHECK-REPLAY`

## 1. Scope Control

- [x] Batch is defined in `02-action-batches.yaml`.
- [x] Included claim IDs are explicitly listed.
- [x] Out-of-scope files/areas are documented.
- [x] No audit report text edits (`00-07*.md`) are included.

Notes:
- Out-of-scope for `B00`: runtime logic changes, pipeline behavior changes, data moves/deletes.

## 2. Preconditions

- [x] `B00` baseline metrics are available.
- [x] Dependencies for this batch are complete.
- [x] Current branch is clean enough to isolate batch changes.

Evidence:
- Baseline dry-run log: `./dry-baseline-latest.log`
- Check replay log: `./dry-check-replay-latest.log`
- Dashboard snapshot: `./06-dashboard-current.md`

## 3. Dry-Run Gates

- [x] Required dry-run profiles for this batch are identified.
- [x] All required dry-run commands completed.
- [x] Dry-run outputs captured in execution notes.
- [x] Any failed dry-run has a documented mitigation plan.

Execution notes:
- Completed: `DRY-BASELINE` at `2026-02-26T17:50:28Z -> 2026-02-26T17:50:29Z` (pass).
- Completed: `DRY-CHECK-REPLAY` at `2026-02-26T17:55:48Z -> 2026-02-26T17:55:51Z` (pass, 27/27 claim checks passing).
- Failures: none observed in `DRY-BASELINE`; mitigation path is to rerun profile and halt `B00` closeout until pass.

## 4. Quality Gates

- [x] `ruff check .` result recorded.
- [x] `mypy cohort_projections` result recorded (if batch touches code).
- [x] `pytest tests/ -q` result recorded (if batch touches code/behavior).
- [x] Claim-check replay plan for affected claims is ready.

Notes:
- `B00` is planning/baseline oriented; quality-gate commands were executed to capture baseline health.
- `ruff check .`: fail (`124` issues; baseline repo debt).
- `mypy cohort_projections`: fail (`3` errors in `base_population_loader.py`; baseline repo debt).
- `pytest tests/ -q`: pass (`1258 passed, 5 skipped`).

## 5. Risk and Rollback

- [x] Blast radius assessed and accepted.
- [x] Rollback plan written (files, commands, decision trigger).
- [x] Data safety confirmed (`data/raw/` remains immutable).

Risk assessment:
- Expected blast radius: **none to low** (planning docs + baseline metadata only).

Rollback plan:
1. Revert `implementation/` planning-file edits for `B00` if scope drift occurs.
2. Regenerate dashboard from prior committed state.
3. Trigger rollback if any runtime source path outside planning/docs is touched.

## 6. Decision

- [ ] **GO**: batch approved for implementation.
- [x] **GO**: batch approved for implementation.
- [ ] **NO-GO**: batch blocked pending actions below.

### Blocking Actions (if NO-GO re-opened)

1. Baseline quality-gate failures (ruff/mypy) are tracked as known debt for later batches, not B00 blockers.
