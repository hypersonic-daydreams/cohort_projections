# Repo Hygiene Audit Open Risks / Blockers Register

**Last Updated (UTC):** 2026-02-27T18:58:26Z  
**Scope:** B00-B06 implementation and residual risk remediation complete

## Purpose

Centralize unresolved risks and blockers so post-implementation sequencing does not lose
operational context.

## Active Items

None.

## Closed Items

### RB-003: Pipeline Dry-Run Coverage Gap

- Status: `closed` (2026-02-27)
- Resolution:
  1. Added explicit `--dry-run` support to `01a_compute_residual_migration.py`, `01b_compute_convergence.py`, and `01c_compute_mortality_improvement.py`.
  2. Updated `run_complete_pipeline.sh` to execute stages `01a/01b/01c` in dry-run mode rather than skipping them.
  3. Verified end-to-end dry-run pass (`RC=0`) with full 7-stage runner flow.

### RB-004: Full-Repo Lint/Type Debt

- Status: `closed` (2026-02-27)
- Resolution:
  1. Completed dedicated lint/type cleanup wave.
  2. `ruff check .` passes (`RC=0`).
  3. `mypy .` passes (`RC=0`).

### RB-001: Resolved-State Claim Check Redesign

- Status: `closed` (2026-02-26)
- Resolution: updated claim checks to assert resolved states; full adjudicated replay passes (`27/27`).

### RB-002: Baseline Metric Drift (`RHA-013`)

- Status: `closed` (2026-02-26; refreshed baseline 2026-02-27)
- Resolution: baseline assertion updated to current canonical package metrics (`CORE_PY_FILES=40`, `CORE_PY_LINES=17719`).

### RB-005: B05 Archive/Delete Strategy Approval

- Status: `closed` (2026-02-27)
- Resolution:
  1. Strategy decisions documented and approved (`22-b05-strategy-decisions.md`).
  2. Wave 1 compatibility implementation completed (`25-b05-wave1-implementation-results.md`).
  3. Wave 2 Step 1 archive actions completed (`26-b05-wave2-step1-results.md`).
  4. Wave 2 Step 2 extraction/placement actions completed (`27-b05-wave2-step2-results.md`).
  5. B05 claim checks replay now fully passing with resolved-state predicates.

## Sequencing Guidance

1. Maintain non-regression by keeping full-repo `ruff` and `mypy` in routine validation.
2. Keep claim replay and dry-run evidence current as publication-focused work proceeds.
