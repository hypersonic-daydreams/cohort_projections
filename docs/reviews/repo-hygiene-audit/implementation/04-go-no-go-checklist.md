# Repo Hygiene Audit Go/No-Go Checklist

Use this checklist before implementing any planned batch from `02-action-batches.yaml`.

## Batch Metadata

- Batch ID:
- Batch Name:
- Date:
- Owner:

## 1. Scope Control

- [ ] Batch is defined in `02-action-batches.yaml`.
- [ ] Included claim IDs are explicitly listed.
- [ ] Out-of-scope files/areas are documented.
- [ ] No audit report text edits (`00-07*.md`) are included.

## 2. Preconditions

- [ ] `B00` baseline metrics are available.
- [ ] Dependencies for this batch are complete.
- [ ] Current branch is clean enough to isolate batch changes.

## 3. Dry-Run Gates

- [ ] Required dry-run profiles for this batch are identified.
- [ ] All required dry-run commands completed.
- [ ] Dry-run outputs captured in execution notes.
- [ ] Any failed dry-run has a documented mitigation plan.

## 4. Quality Gates

- [ ] `ruff check .` result recorded.
- [ ] `mypy cohort_projections` result recorded (if batch touches code).
- [ ] `pytest tests/ -q` result recorded (if batch touches code/behavior).
- [ ] Claim-check replay plan for affected claims is ready.

## 5. Risk and Rollback

- [ ] Blast radius assessed and accepted.
- [ ] Rollback plan written (files, commands, decision trigger).
- [ ] Data safety confirmed (`data/raw/` remains immutable).

## 6. Decision

- [ ] **GO**: batch approved for implementation.
- [ ] **NO-GO**: batch blocked pending actions below.

### Blocking Actions (if NO-GO)

1.
2.
3.
