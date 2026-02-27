# Repo Hygiene Audit Claim Verification Progress

**Generated (UTC):** 2026-02-27T18:21:14Z
**Registry:** `docs/reviews/repo-hygiene-audit/verification/claims_registry.yaml`
**Evidence Directory:** `docs/reviews/repo-hygiene-audit/verification/evidence`

## Summary

- Total claims: **27**
- Status counts:
  - `adjudicated`: 27
- Verdict counts:
  - `confirmed`: 24
  - `partially_confirmed`: 3

## Claim Tracker

| Claim ID | Severity | Type | Status | Verdict | Checks | Last Run (UTC) | Evidence |
|---|---|---|---|---|---:|---|---|
| RHA-001 | critical | structural | adjudicated | confirmed | 1/1 | 2026-02-27T18:14:00Z | [artifact](evidence/20260227T181400Z_RHA-001.json) |
| RHA-002 | critical | behavioral | adjudicated | confirmed | 1/1 | 2026-02-27T18:14:00Z | [artifact](evidence/20260227T181400Z_RHA-002.json) |
| RHA-003 | critical | behavioral | adjudicated | confirmed | 1/1 | 2026-02-27T18:14:00Z | [artifact](evidence/20260227T181400Z_RHA-003.json) |
| RHA-004 | critical | structural | adjudicated | confirmed | 1/1 | 2026-02-27T18:14:00Z | [artifact](evidence/20260227T181400Z_RHA-004.json) |
| RHA-005 | high | structural | adjudicated | confirmed | 1/1 | 2026-02-27T18:14:01Z | [artifact](evidence/20260227T181401Z_RHA-005.json) |
| RHA-006 | high | structural | adjudicated | confirmed | 1/1 | 2026-02-27T18:14:01Z | [artifact](evidence/20260227T181401Z_RHA-006.json) |
| RHA-007 | high | structural | adjudicated | confirmed | 1/1 | 2026-02-27T18:14:01Z | [artifact](evidence/20260227T181401Z_RHA-007.json) |
| RHA-008 | high | behavioral | adjudicated | confirmed | 1/1 | 2026-02-27T18:14:01Z | [artifact](evidence/20260227T181401Z_RHA-008.json) |
| RHA-009 | critical | structural | adjudicated | confirmed | 1/1 | 2026-02-27T18:14:01Z | [artifact](evidence/20260227T181401Z_RHA-009.json) |
| RHA-010 | high | structural | adjudicated | partially_confirmed | 1/1 | 2026-02-27T18:14:01Z | [artifact](evidence/20260227T181401Z_RHA-010.json) |
| RHA-011 | high | structural | adjudicated | confirmed | 1/1 | 2026-02-27T18:14:02Z | [artifact](evidence/20260227T181402Z_RHA-011.json) |
| RHA-012 | high | structural | adjudicated | confirmed | 1/1 | 2026-02-27T18:14:02Z | [artifact](evidence/20260227T181402Z_RHA-012.json) |
| RHA-013 | medium | structural | adjudicated | confirmed | 1/1 | 2026-02-27T18:14:02Z | [artifact](evidence/20260227T181402Z_RHA-013.json) |
| RHA-014 | medium | structural | adjudicated | confirmed | 1/1 | 2026-02-27T18:14:02Z | [artifact](evidence/20260227T181402Z_RHA-014.json) |
| RHA-015 | medium | structural | adjudicated | confirmed | 1/1 | 2026-02-27T18:14:02Z | [artifact](evidence/20260227T181402Z_RHA-015.json) |
| RHA-016 | medium | structural | adjudicated | confirmed | 1/1 | 2026-02-27T18:14:02Z | [artifact](evidence/20260227T181402Z_RHA-016.json) |
| RHA-017 | low | structural | adjudicated | confirmed | 1/1 | 2026-02-27T18:14:02Z | [artifact](evidence/20260227T181402Z_RHA-017.json) |
| RHA-018 | medium | structural | adjudicated | confirmed | 1/1 | 2026-02-27T18:14:02Z | [artifact](evidence/20260227T181402Z_RHA-018.json) |
| RHA-019 | medium | structural | adjudicated | partially_confirmed | 1/1 | 2026-02-27T18:14:02Z | [artifact](evidence/20260227T181402Z_RHA-019.json) |
| RHA-020 | medium | structural | adjudicated | confirmed | 1/1 | 2026-02-27T18:14:03Z | [artifact](evidence/20260227T181403Z_RHA-020.json) |
| RHA-021 | low | structural | adjudicated | confirmed | 1/1 | 2026-02-27T18:14:03Z | [artifact](evidence/20260227T181403Z_RHA-021.json) |
| RHA-022 | low | structural | adjudicated | confirmed | 1/1 | 2026-02-27T18:14:03Z | [artifact](evidence/20260227T181403Z_RHA-022.json) |
| RHA-023 | medium | structural | adjudicated | confirmed | 1/1 | 2026-02-27T18:14:03Z | [artifact](evidence/20260227T181403Z_RHA-023.json) |
| RHA-024 | medium | structural | adjudicated | confirmed | 1/1 | 2026-02-27T18:14:03Z | [artifact](evidence/20260227T181403Z_RHA-024.json) |
| RHA-025 | medium | structural | adjudicated | confirmed | 1/1 | 2026-02-27T18:14:03Z | [artifact](evidence/20260227T181403Z_RHA-025.json) |
| RHA-026 | low | structural | adjudicated | confirmed | 1/1 | 2026-02-27T18:14:03Z | [artifact](evidence/20260227T181403Z_RHA-026.json) |
| RHA-027 | high | interpretive | adjudicated | partially_confirmed | 1/1 | 2026-02-27T18:14:03Z | [artifact](evidence/20260227T181403Z_RHA-027.json) |

## Agent Workflow

1. Claim Extractor Agent registers an atomic claim in `claims_registry.yaml`.
2. Check Designer Agent adds deterministic checks and assertions.
3. Evidence Runner Agent executes checks via `run_claim_checks.py run`.
4. Adjudicator Agent updates `verdict`, `confidence`, and `notes` in registry.
5. Tracker Agent regenerates this file via `run_claim_checks.py progress`.

## Notes

- This tracker is generated from the registry and latest evidence artifacts.
- Edit claims in YAML, then regenerate progress to reflect updates.
