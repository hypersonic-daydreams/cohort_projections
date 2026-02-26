# Repo Hygiene Audit Go/No-Go Checklist

Use this checklist before implementing any planned batch from `02-action-batches.yaml`.

## Batch Metadata

- Batch ID: B04
- Batch Name: code_structure_and_test_scope_alignment
- Date: 2026-02-26
- Owner: codex

## 1. Scope Control

- [x] Batch is defined in `02-action-batches.yaml`.
- [x] Included claim IDs are explicitly listed.
- [x] Out-of-scope files/areas are documented.
- [x] No audit report text edits (`00-07*.md`) are included.

## 2. Preconditions

- [x] `B00` baseline metrics are available.
- [x] Dependencies for this batch are complete.
- [x] Current branch is clean enough to isolate batch changes.

## 3. Dry-Run Gates

- [x] Required dry-run profiles for this batch are identified.
- [x] All required dry-run commands completed.
- [x] Dry-run outputs captured in execution notes.
- [x] Any failed dry-run has a documented mitigation plan.

## 4. Quality Gates

- [x] `ruff check .` result recorded.
- [x] `mypy cohort_projections` result recorded (if batch touches code).
- [x] `pytest tests/ -q` result recorded (if batch touches code/behavior).
- [x] Claim-check replay plan for affected claims is ready.

## 5. Risk and Rollback

- [x] Blast radius assessed and accepted.
- [x] Rollback plan written (files, commands, decision trigger).
- [x] Data safety confirmed (`data/raw/` remains immutable).

Rollback plan:
1. `git checkout -- cohort_projections/__init__.py cohort_projections/core/cohort_component.py cohort_projections/data/process/__init__.py cohort_projections/data/process/residual_migration.py cohort_projections/output/reports.py scripts/projections/run_pep_projections.py tests/test_integration/test_adr021_modules.py tests/unit/test_adr021_modules.py`
2. Remove B04-only logs/checklists if rollback is complete.
3. Re-run `pytest tests/ -q` and B04 claim checks to confirm pre-change state.

## 6. Decision

- [x] **GO**: batch approved for implementation.
- [ ] **NO-GO**: batch blocked pending actions below.

### Blocking Actions (if NO-GO)

1. N/A
2. N/A
3. N/A
