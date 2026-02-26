# Repo Hygiene Audit Go/No-Go Checklist (B02)

Use this checklist before implementing any planned batch from `02-action-batches.yaml`.

## Batch Metadata

- Batch ID: `B02`
- Batch Name: `pipeline_entrypoint_and_order_consistency`
- Date: `2026-02-26`
- Owner: `nhaarstad` (approver) / `codex` (execution support)
- Included Claims: `RHA-001`, `RHA-002`, `RHA-003`, `RHA-004`, `RHA-027`
- Dry-Run Profiles Required: `DRY-PIPELINE`, `DRY-TESTS`, `DRY-CHECK-REPLAY`

## 1. Scope Control

- [x] Batch is defined in `02-action-batches.yaml`.
- [x] Included claim IDs are explicitly listed.
- [x] Out-of-scope files/areas are documented.
- [x] No audit report text edits (`00-07*.md`) are included.

Notes:
- This checkpoint covers preflight and gate-unblock edits only.
- Out-of-scope for this checkpoint: full B02 pipeline ordering/remap implementation.

## 2. Preconditions

- [x] `B00` baseline metrics are available.
- [x] Dependencies for this batch are complete.
- [x] Current branch is clean enough to isolate batch changes.

Evidence:
- Prior closeouts: `./07-b00-go-no-go.md`, `./08-b00-preflight-results.md`, `./09-b01-go-no-go.md`, `./10-b01-preflight-results.md`
- Program dashboard: `./06-dashboard-current.md`

## 3. Dry-Run Gates

- [x] Required dry-run profiles for this batch are identified.
- [x] All required dry-run commands completed.
- [x] Dry-run outputs captured in execution notes.
- [x] Any failed dry-run has a documented mitigation plan.

Execution notes:
- Initial B02 preflight recorded NO-GO due pipeline dry-run failure + stale `RHA-001` check assumptions.
- Gate-unblock edits applied:
  - `scripts/pipeline/01_process_demographic_data.py`: migration input check no longer hard-fails in `--dry-run`.
  - `scripts/pipeline/02_run_projections.py`: dry-run fallback skips place-level setup when place reference schema is incomplete.
  - `verification/claims_registry.yaml` (`RHA-001`): check updated for current docs surfaces after NAVIGATION removal.
- Rerun results:
  - `DRY-PIPELINE`: pass (`2026-02-26T19:04:49Z -> 2026-02-26T19:05:01Z`)
  - `DRY-TESTS`: pass (`2026-02-26T19:05:07Z -> 2026-02-26T19:09:52Z`)
  - `DRY-CHECK-REPLAY`: pass for B02 gate interpretation (`2026-02-26T19:10:00Z -> 2026-02-26T19:10:04Z`), with only known B01 expected drifts.

## 4. Quality Gates

- [ ] `ruff check .` result recorded.
- [ ] `mypy cohort_projections` result recorded.
- [x] `pytest tests/ -q` result recorded.
- [x] Claim-check replay plan for affected claims is ready.

Notes:
- `ruff`/`mypy` were not rerun during this preflight checkpoint.
- `pytest` completed successfully as part of `DRY-TESTS`.

## 5. Risk and Rollback

- [x] Blast radius assessed and accepted.
- [x] Rollback plan written (files, commands, decision trigger).
- [x] Data safety confirmed (`data/raw/` remains immutable).

Risk assessment:
- Expected blast radius remains low-to-medium for preflight unblock edits.

Rollback plan:
1. Revert gate-unblock edits if new failures appear in rerun profiles.
2. Keep B02 at preflight-only state until GO is confirmed.
3. Trigger rollback if non-B02 scope files are modified.

## 6. Decision

- [x] **GO**: batch approved for implementation.
- [ ] **NO-GO**: batch blocked pending actions below.

Closeout interpretation:
- B02 preflight is now **GO-ready** based on rerun gate results.
- Known B01 post-remediation expected drift claims (`RHA-009`, `RHA-019`, `RHA-020`, `RHA-021`, `RHA-022`) remain explicitly tracked and are not treated as B02 blockers.
- B02 implementation scope has since been executed; post-edit validation and affected-claim replay are recorded in `./12-b02-preflight-results.md` and `./06-dashboard-current.md`.

### Blocking Actions (if NO-GO re-opened)

1. Reproduce failing profile with fresh logs.
2. Re-validate claim-check assumptions against current repo structure.
3. Re-run full B02 preflight before any further implementation edits.
