# Repo Hygiene Audit Go/No-Go Checklist (B03)

Use this checklist before implementing any planned batch from `02-action-batches.yaml`.

## Batch Metadata

- Batch ID: `B03`
- Batch Name: `config_path_and_version_hygiene`
- Date: `2026-02-26`
- Owner: `codex`
- Included Claims: `RHA-005`, `RHA-006`, `RHA-007`, `RHA-016`
- Dry-Run Profiles Required: `DRY-CONFIG`, `DRY-TESTS`, `DRY-LINT-TYPE`, `DRY-CHECK-REPLAY`

## 1. Scope Control

- [x] Batch is defined in `02-action-batches.yaml`.
- [x] Included claim IDs are explicitly listed.
- [x] Out-of-scope files/areas are documented.
- [x] No audit report text edits (`00-07*.md`) are included.

Notes:
- This checkpoint is preflight-only.
- Out-of-scope for this checkpoint: any production/config/path/version code edits.

## 2. Preconditions

- [x] `B00` baseline metrics are available.
- [x] Dependencies for this batch are complete.
- [x] Current branch is clean enough to isolate batch changes.

Evidence:
- Prior closeouts: `./07-b00-go-no-go.md`, `./08-b00-preflight-results.md`, `./09-b01-go-no-go.md`, `./10-b01-preflight-results.md`, `./11-b02-go-no-go.md`, `./12-b02-preflight-results.md`
- Program dashboard: `./06-dashboard-current.md`

## 3. Dry-Run Gates

- [x] Required dry-run profiles for this batch are identified.
- [x] All required dry-run commands completed.
- [x] Dry-run outputs captured in execution notes.
- [x] Any failed dry-run has a documented mitigation plan.

Execution notes:
- `DRY-CONFIG`: pass (`./dry-config-b03-preflight.log`)
- `DRY-TESTS`: pass (`./dry-tests-b03-preflight.log`)
- `DRY-LINT-TYPE`: fail (`./dry-lint-type-b03-preflight-ruff.log`, `./dry-lint-type-b03-preflight-mypy.log`)
- `DRY-CHECK-REPLAY`: command-pass with known expected drift on previously remediated claims (`./dry-check-replay-b03-preflight.log`)

## 4. Quality Gates

- [x] `ruff check .` result recorded.
- [x] `mypy cohort_projections` result recorded.
- [x] `pytest tests/ -q` result recorded.
- [x] Claim-check replay plan for affected claims is ready.

Notes:
- `ruff check .` failed with `124` findings (many outside B03 scope).
- `mypy cohort_projections` failed with `3` errors in `cohort_projections/data/load/base_population_loader.py`.

## 5. Risk and Rollback

- [x] Blast radius assessed and accepted.
- [x] Rollback plan written (files, commands, decision trigger).
- [x] Data safety confirmed (`data/raw/` remains immutable).

Risk assessment:
- Implementing B03 while lint/type gate is red would violate batch safety policy and may mask regressions.

Rollback plan:
1. Keep B03 in preflight state with no implementation edits.
2. Resolve or explicitly baseline-accept lint/type failures in a dedicated gate-unblock change.
3. Re-run full B03 preflight before any B03 implementation edit.

## 6. Decision

- [ ] **GO**: batch approved for implementation.
- [x] **NO-GO**: batch blocked pending actions below.

Closeout interpretation:
- B03 is **NO-GO** due failed `DRY-LINT-TYPE` gate.
- Per execution constraints, no B03 implementation edits were applied.

### Blocking Actions

1. Address `ruff` baseline violations or scope lint to approved target paths for batch gating.
2. Fix `mypy` type errors in `cohort_projections/data/load/base_population_loader.py` or define accepted baseline policy for B03 gate.
3. Re-run all B03 required preflight profiles and require GO before implementation.
