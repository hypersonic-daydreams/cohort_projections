# Repo Hygiene Audit Go/No-Go Checklist (B01)

Use this checklist before implementing any planned batch from `02-action-batches.yaml`.

## Batch Metadata

- Batch ID: `B01`
- Batch Name: `documentation_harmonization`
- Date: `2026-02-26`
- Owner: `nhaarstad` (approver) / `codex` (execution support)
- Included Claims: `RHA-009`, `RHA-019`, `RHA-020`, `RHA-021`, `RHA-022`
- Dry-Run Profiles Required: `DRY-DOCS`, `DRY-CHECK-REPLAY`

## 1. Scope Control

- [x] Batch is defined in `02-action-batches.yaml`.
- [x] Included claim IDs are explicitly listed.
- [x] Out-of-scope files/areas are documented.
- [x] No audit report text edits (`00-07*.md`) are included.

Notes:
- Out-of-scope for `B01`: runtime logic changes, pipeline behavior changes, data moves/deletes.

## 2. Preconditions

- [x] `B00` baseline metrics are available.
- [x] Dependencies for this batch are complete.
- [x] Current branch is clean enough to isolate batch changes.

Evidence:
- `B00` closeout: `./07-b00-go-no-go.md`, `./08-b00-preflight-results.md`
- Dashboard baseline: `./06-dashboard-current.md`

## 3. Dry-Run Gates

- [x] Required dry-run profiles for this batch are identified.
- [x] All required dry-run commands completed.
- [x] Dry-run outputs captured in execution notes.
- [x] Any failed dry-run has a documented mitigation plan.

Execution notes:
- Completed: `DRY-DOCS` at `2026-02-26T18:24:13Z -> 2026-02-26T18:24:14Z` (pass).
- Completed: `DRY-CHECK-REPLAY` at `2026-02-26T18:24:19Z -> 2026-02-26T18:24:23Z` (pass, 27/27 claims passing before B01 edits).
- Failures: none in required pre-edit dry-run profiles.

## 4. Quality Gates

- [x] `ruff check .` result recorded.
- [x] `mypy cohort_projections` result recorded (if batch touches code).
- [x] `pytest tests/ -q` result recorded (if batch touches code/behavior).
- [x] Claim-check replay plan for affected claims is ready.

Notes:
- `B01` is docs-only; no runtime code paths were edited.
- Baseline quality snapshot from `B00` retained for reference:
  - `ruff check .`: fail (`124` issues; baseline debt)
  - `mypy cohort_projections`: fail (`3` errors; baseline debt)
  - `pytest tests/ -q`: pass (`1258 passed, 5 skipped`)
- Post-edit affected-claim replay completed (see `10-b01-preflight-results.md`).

## 5. Risk and Rollback

- [x] Blast radius assessed and accepted.
- [x] Rollback plan written (files, commands, decision trigger).
- [x] Data safety confirmed (`data/raw/` remains immutable).

Risk assessment:
- Expected blast radius: **low** (documentation and tracking artifacts only).

Rollback plan:
1. Revert B01 docs edits if claim replay indicates unintended scope drift.
2. Restore removed navigation file and pre-edit tracker/readme content from archive snapshots if needed.
3. Trigger rollback if runtime source files outside docs/planning were touched.

## 6. Decision

- [x] **GO**: batch approved for implementation and executed.
- [ ] **NO-GO**: batch blocked pending actions below.

Closeout interpretation:
- Pre-edit GO gates passed (`DRY-DOCS`, `DRY-CHECK-REPLAY`).
- Post-edit affected claim checks are **failing by design** because current checks encode pre-remediation conditions; failures are treated as remediation evidence and tracked for claim-check refresh in later harmonization.

### Blocking Actions (if NO-GO re-opened)

1. If docs-only scope is violated, revert and re-run dry-runs.
2. If claim replay artifacts are missing, regenerate affected claim evidence.
3. If remediation intent is disputed, adjudicate B01 claim acceptance criteria before progressing to B02/B03.
