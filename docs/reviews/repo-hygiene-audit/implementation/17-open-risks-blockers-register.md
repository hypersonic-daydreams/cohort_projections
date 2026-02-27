# Repo Hygiene Audit Open Risks / Blockers Register

**Last Updated (UTC):** 2026-02-27T18:21:14Z  
**Scope:** B05 and B06 implementation complete; residual policy/quality risks tracked

## Purpose

Centralize unresolved risks and blockers so post-implementation sequencing does not lose
operational context.

## Active Items

### RB-003: Pipeline Dry-Run Coverage Gap

- Status: `open`
- Severity: `medium`
- Affected stages: `scripts/pipeline/01a_*`, `01b_*`, `01c_*`
- Current signal: `run_complete_pipeline.sh --dry-run` still skips stages `01a/01b/01c`.
- Impact: dry-run does not exercise full stage graph.
- Required action:
  1. Add `--dry-run` handling to skipped stages, or
  2. Codify skip as accepted limitation with dedicated check and rationale.
- Exit criterion: deterministic dry-run policy for all seven pipeline stages.

### RB-004: Full-Repo Lint/Type Debt

- Status: `open`
- Severity: `medium`
- Current signal: full baseline still fails (`ruff`, `mypy`) outside current batch scopes.
- Impact: quality gates rely on explicit non-regression policy by batch scope.
- Required action:
  1. Create dedicated debt-reduction wave (or ADR-backed policy) for baseline cleanup.
  2. Track debt trend over batches to prevent growth.
- Exit criterion: either full-repo lint/type pass or formally accepted long-term baseline policy.

## Closed Items

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

1. Execute dedicated RB-003/RB-004 remediation or policy-closeout wave.
2. Preserve explicit non-regression evidence for lint/type until debt is resolved or policy-accepted.
