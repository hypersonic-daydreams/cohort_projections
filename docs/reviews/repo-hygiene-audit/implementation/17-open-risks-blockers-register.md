# Repo Hygiene Audit Open Risks / Blockers Register

**Last Updated (UTC):** 2026-02-26T20:42:11Z  
**Scope:** Post-B03 closeout

## Purpose

Centralize unresolved risks and blockers so batch sequencing (`B04`-`B06`) does not
lose operational context.

## Active Items

### RB-001: Resolved-State Claim Check Redesign

- Status: `open`
- Severity: `high`
- Affected claims: `RHA-001`, `RHA-002`, `RHA-003`, `RHA-004`, `RHA-005`, `RHA-006`, `RHA-007`, `RHA-009`, `RHA-016`, `RHA-019`, `RHA-020`, `RHA-021`, `RHA-022`, `RHA-027`
- Current signal: these claims replay as `0/1` because checks still assert pre-remediation states.
- Impact: noisy verification baseline and ambiguity for B06 final harmonization.
- Required action:
  1. Add replacement resolved-state checks in `verification/claims_registry.yaml`.
  2. Run claim replay and adjudication update.
  3. Regenerate `verification/progress.md`.
- Exit criterion: resolved-state claims replay `1/1` with explicit evidence.

### RB-002: Baseline Metric Drift (`RHA-013`)

- Status: `open`
- Severity: `medium`
- Affected claim: `RHA-013`
- Current signal: replay shows `0/1` after implementation drift from pinned baseline values.
- Impact: baseline guardrail no longer reflects current repository state.
- Required action:
  1. Recompute current baseline metric values.
  2. Update claim check assertion and notes to new canonical baseline.
  3. Replay `RHA-013` and refresh progress tracker.
- Exit criterion: `RHA-013` replays `1/1` with updated pinned metrics.

### RB-003: Pipeline Dry-Run Coverage Gap

- Status: `open`
- Severity: `medium`
- Affected stages: `scripts/pipeline/01a_*`, `01b_*`, `01c_*`
- Current signal: `run_complete_pipeline.sh --dry-run` skips stages `01a/01b/01c`.
- Impact: dry-run does not exercise full stage graph.
- Required action:
  1. Add `--dry-run` handling to skipped stages, or
  2. codify skip as accepted limitation with dedicated check and rationale.
- Exit criterion: deterministic dry-run policy for all seven pipeline stages.

### RB-004: Full-Repo Lint/Type Debt

- Status: `open`
- Severity: `medium`
- Current signal: full baseline still fails (`ruff`, `mypy`) outside B03 scope.
- Impact: quality gates must continue using explicit non-regression policy by batch scope.
- Required action:
  1. Create dedicated debt-reduction wave (or ADR-backed policy) for baseline cleanup.
  2. Track debt trend over batches to prevent growth.
- Exit criterion: either full-repo lint/type pass or formally accepted long-term baseline policy.

## Sequencing Guidance

1. Run RB-001 and RB-002 before B06 final harmonization.
2. Treat RB-003 as required before declaring pipeline dry-run fully representative.
3. Keep RB-004 policy explicit in every batch gate report until retired.
