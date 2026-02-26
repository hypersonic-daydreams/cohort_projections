# Repo Hygiene Audit Claim Verification Progress

**Generated (UTC):** 2026-02-26T17:23:03Z
**Registry:** `docs/reviews/repo-hygiene-audit/verification/claims_registry.yaml`
**Evidence Directory:** `docs/reviews/repo-hygiene-audit/verification/evidence`

## Summary

- Total claims: **0**
- Status counts:
  - _(no claims registered)_
- Verdict counts:
  - _(no claims registered)_

## Claim Tracker

| Claim ID | Severity | Type | Status | Verdict | Checks | Last Run (UTC) | Evidence |
|---|---|---|---|---|---:|---|---|
| _(none)_ | - | - | - | - | 0 | - | - |

## Agent Workflow

1. Claim Extractor Agent registers an atomic claim in `claims_registry.yaml`.
2. Check Designer Agent adds deterministic checks and assertions.
3. Evidence Runner Agent executes checks via `run_claim_checks.py run`.
4. Adjudicator Agent updates `verdict`, `confidence`, and `notes` in registry.
5. Tracker Agent regenerates this file via `run_claim_checks.py progress`.

## Notes

- This tracker is generated from the registry and latest evidence artifacts.
- Edit claims in YAML, then regenerate progress to reflect updates.
