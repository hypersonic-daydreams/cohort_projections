# Claim Verification Framework

This directory contains the agent-driven verification tooling for
`docs/reviews/repo-hygiene-audit/`.

## Purpose

Create a repeatable process that:
1. Registers audit claims as atomic units.
2. Defines deterministic checks for each claim.
3. Runs checks and stores immutable evidence artifacts.
4. Tracks claim-by-claim progress across agent sessions.

No findings are adjudicated here by default; this is scaffolding only.

## Files

| File | Purpose |
|---|---|
| `claims_registry.yaml` | Source of truth for claims, checks, status, and verdict fields |
| `progress.md` | Generated progress dashboard for claim-level tracking |
| `evidence/` | JSON artifacts from deterministic check runs |

## Agent Roles

1. `Claim Extractor Agent`
   - Adds claim entries to `claims_registry.yaml`.
   - Keeps claims atomic and testable.
2. `Check Designer Agent`
   - Adds deterministic `checks` + `assertion` rules to each claim.
3. `Evidence Runner Agent`
   - Runs checks via `scripts/reviews/run_claim_checks.py run`.
   - Produces JSON artifacts in `evidence/`.
4. `Adjudicator Agent`
   - Reviews evidence and updates `verdict`, `confidence`, and notes.
5. `Tracker Agent`
   - Regenerates `progress.md` and advances status fields.

## Deterministic Check Model

Each check is a command + explicit assertion rule.

Example assertion fields:
- `expect_exit_code`: exact return code expected
- `stdout_contains`: required substrings in stdout
- `stdout_not_contains`: forbidden substrings in stdout
- `stderr_contains`: required substrings in stderr
- `stderr_not_contains`: forbidden substrings in stderr
- `regex_match`: required stdout regex pattern(s)

## CLI Usage

From repository root:

```bash
python scripts/reviews/run_claim_checks.py list
python scripts/reviews/run_claim_checks.py progress
python scripts/reviews/run_claim_checks.py run --status checks_defined,in_progress
python scripts/reviews/run_claim_checks.py run --claim-id RHA-001 --fail-on-check-failure
```

Dry-run (selection preview without executing commands):

```bash
python scripts/reviews/run_claim_checks.py run --status checks_defined --dry-run
```
