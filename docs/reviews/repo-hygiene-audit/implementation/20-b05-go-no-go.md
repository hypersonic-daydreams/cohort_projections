# Repo Hygiene Audit Go/No-Go Checklist

Use this checklist before implementing any planned batch from `02-action-batches.yaml`.

## Batch Metadata

- Batch ID: B05
- Batch Name: repository_footprint_and_data_hygiene
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
1. `git checkout -- docs/reviews/repo-hygiene-audit/verification/claims_registry.yaml docs/reviews/repo-hygiene-audit/verification/progress.md docs/reviews/repo-hygiene-audit/verification/evidence/`
2. Restore dashboard/action/risk artifacts for B05 preflight if reverting this stage.
3. Re-run `python scripts/reviews/run_claim_checks.py run --status adjudicated` and `python scripts/reviews/run_claim_checks.py progress`.

## 6. Decision

- [ ] **GO**: batch approved for implementation.
- [x] **NO-GO**: batch blocked pending actions below.

### Blocking Actions (if NO-GO)

1. Archive/move/delete decisions for root clutter and duplicate footprints require explicit owner decisions before destructive changes.
2. B05 includes high-blast-radius repository boundary actions (`sdc_2024_replication` footprint) that need approved destination strategy.
3. Data-footprint deduplication plan must specify provenance-preserving canonical location before file removals.
