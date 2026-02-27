# Repo Hygiene Audit Open Risks / Blockers Register

**Last Updated (UTC):** 2026-02-26T21:26:06Z  
**Scope:** Post-B05 preflight + RB-001/RB-002 remediation

## Purpose

Centralize unresolved risks and blockers so remaining batch sequencing does not lose
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

### RB-005: B05 Archive/Delete Strategy Approval

- Status: `open`
- Severity: `high`
- Affected claims/batch: `B05` (`RHA-011`, `RHA-012`, `RHA-015`, `RHA-017`, `RHA-018`, `RHA-026`)
- Current signal: preflight gates pass, but implementation requires destructive or boundary-moving actions.
- Impact: implementation cannot proceed safely without explicit destination/provenance strategy.
- Required action:
  1. **Documented owner decisions** (2026-02-27) for:
     - extracting `sdc_2024_replication/` to its own repo under the local `demography/` directory, and
     - keeping the canonical SDC rate CSVs in that extracted repository.
  2. Produce an implementation-ready **references inventory + migration plan** (paths, links, docs, code) for the extraction and rate-path updates.
  3. Produce a **delete/archive list** (root clutter, stale exports, empty placeholders) with proposed destinations and retention policy.
- Exit criterion: written approved strategy and rollback plan allowing B05 implementation GO.

Owner-approved strategy decisions: `docs/reviews/repo-hygiene-audit/implementation/22-b05-strategy-decisions.md`

## Closed Items

### RB-001: Resolved-State Claim Check Redesign

- Status: `closed` (2026-02-26)
- Resolution: updated claim checks to assert resolved states; full adjudicated replay now passes (`27/27`).

### RB-002: Baseline Metric Drift (`RHA-013`)

- Status: `closed` (2026-02-26)
- Resolution: baseline assertion reset to current canonical metrics (`CORE_PY_FILES=39`, `CORE_PY_LINES=17604`), replay passes.

## Sequencing Guidance

1. Resolve RB-005 before any B05 destructive cleanup actions.
2. Keep RB-003 and RB-004 explicitly documented through B06 closeout.
3. After B05 implementation, run final B06 harmonization and full claim revalidation.
