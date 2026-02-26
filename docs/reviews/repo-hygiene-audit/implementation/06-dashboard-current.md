# Repo Hygiene Audit Execution Dashboard (Current)

**Snapshot Date (UTC):** 2026-02-26T17:50:59Z
**Program Phase:** B00 Go-Ready
**Source Claims Registry:** `../verification/claims_registry.yaml`

## 0. Recommended Start

- **Safest first execution candidate:** `B00` (`baseline_metrics_and_safety_harness`)
- **Why:** No production behavior changes, no prerequisites, and only baseline/guardrail setup claims (`RHA-013`, `RHA-023`).
- **Entry command profile:** `DRY-BASELINE` then `DRY-CHECK-REPLAY`.
- **Checklist instance:** [07-b00-go-no-go.md](./07-b00-go-no-go.md) (currently **GO**).
- **Preflight record:** [08-b00-preflight-results.md](./08-b00-preflight-results.md).

## 1. Program Summary

- Total verified claims: **27**
- Verified verdict split: **confirmed=24**, **partially_confirmed=3**
- Claims in planned batches: **27**
- Claims completed in implementation: **0**
- Claims pending implementation: **27**

## 2. Batch Status

| Batch | Name | Status | Owner | Started | Last Updated | Notes |
|---|---|---|---|---|---|---|
| B00 | baseline_metrics_and_safety_harness | go_ready | nhaarstad | 2026-02-26 | 2026-02-26 | Dry-runs complete; GO checklist approved. |
| B01 | documentation_harmonization | blocked_by_prerequisites |  |  | 2026-02-26 | Waiting on: B00 |
| B02 | pipeline_entrypoint_and_order_consistency | blocked_by_prerequisites |  |  | 2026-02-26 | Waiting on: B00, B01 |
| B03 | config_path_and_version_hygiene | blocked_by_prerequisites |  |  | 2026-02-26 | Waiting on: B00 |
| B04 | code_structure_and_test_scope_alignment | blocked_by_prerequisites |  |  | 2026-02-26 | Waiting on: B00, B02, B03 |
| B05 | repository_footprint_and_data_hygiene | blocked_by_prerequisites |  |  | 2026-02-26 | Waiting on: B00, B01 |
| B06 | final_harmonization_and_claim_revalidation | blocked_by_prerequisites |  |  | 2026-02-26 | Waiting on: B01, B02, B03, B04, B05 |

## 3. Dry-Run Gate Results

| Profile | Batch | Status | Timestamp | Operator | Evidence/Log |
|---|---|---|---|---|---|
| DRY-BASELINE | B00 | pass | 2026-02-26T17:50:28Z -> 2026-02-26T17:50:29Z | codex | [dry-baseline-latest.log](./dry-baseline-latest.log) |
| DRY-DOCS | B01 | not_run |  |  |  |
| DRY-PIPELINE | B02 | not_run |  |  |  |
| DRY-CONFIG | B03 | not_run |  |  |  |
| DRY-LINT-TYPE | B03/B04/B06 | not_run |  |  |  |
| DRY-TESTS | B02/B03/B04/B06 | not_run |  |  |  |
| DRY-REPO-CLEANUP | B05 | not_run |  |  |  |
| DRY-CHECK-REPLAY | All | pass | 2026-02-26T17:55:48Z -> 2026-02-26T17:55:51Z | codex | [dry-check-replay-latest.log](./dry-check-replay-latest.log) |

## 4. Claim-Level Implementation Tracker

| Claim ID | Severity | Verdict (Verified) | Planned Batch | Implementation Status | Re-verified? | Notes |
|---|---|---|---|---|---|---|
| RHA-001 | critical | confirmed | B02 | not_started | no | Ghost projection runner is referenced but missing |
| RHA-002 | critical | confirmed | B02 | not_started | no | Pipeline shell runner executes only three Python stages |
| RHA-003 | critical | confirmed | B02 | not_started | no | Additional numbered pipeline steps are not called by runner |
| RHA-004 | critical | confirmed | B02 | not_started | no | Pipeline numbering collision for 01-prefixed scripts |
| RHA-005 | high | confirmed | B03 | not_started | no | Version declarations are inconsistent across locations |
| RHA-006 | high | confirmed | B03 | not_started | no | Duplicate ConfigLoader implementations coexist |
| RHA-007 | high | confirmed | B03 | not_started | no | Hardcoded absolute paths exist in data scripts |
| RHA-008 | high | confirmed | B04 | not_started | no | run_pep_projections dynamically imports pipeline module |
| RHA-009 | critical | confirmed | B01 | not_started | no | ADR README index is stale for late-series ADRs |
| RHA-010 | high | partially_confirmed | B04 | not_started | no | Tests include substantial sibling-repo-focused suites |
| RHA-011 | high | confirmed | B05 | not_started | no | sdc_2024_replication dominates repository file count |
| RHA-012 | high | confirmed | B05 | not_started | no | Markdown documentation line count exceeds Python line count |
| RHA-013 | medium | confirmed | B00 | not_started | no | Core package size is 39 files and 17,759 lines |
| RHA-014 | medium | confirmed | B04 | not_started | no | Three oversized modules exceed 1,300 lines |
| RHA-015 | medium | confirmed | B05 | not_started | no | Root-level clutter includes tracked and stray artifacts |
| RHA-016 | medium | confirmed | B03 | not_started | no | Five raw-data subdirectories are missing DATA_SOURCE_NOTES.md |
| RHA-017 | low | confirmed | B05 | not_started | no | Stale low_growth exports remain without matching projections |
| RHA-018 | medium | confirmed | B05 | not_started | no | Five SDC rate files are triplicated across three directories |
| RHA-019 | medium | partially_confirmed | B01 | not_started | no | Multiple navigation documents coexist |
| RHA-020 | medium | confirmed | B01 | not_started | no | README projection horizon conflicts with AGENTS horizon |
| RHA-021 | low | confirmed | B01 | not_started | no | DEVELOPMENT_TRACKER is lengthy |
| RHA-022 | low | confirmed | B01 | not_started | no | data/process module README is very large |
| RHA-023 | medium | confirmed | B00 | not_started | no | cohort_projections package has no circular imports |
| RHA-024 | medium | confirmed | B04 | not_started | no | Median age and dependency ratio functions are duplicated |
| RHA-025 | medium | confirmed | B04 | not_started | no | example_usage.py and version.py are orphaned from imports |
| RHA-026 | low | confirmed | B05 | not_started | no | Placeholder directories remain empty |
| RHA-027 | high | partially_confirmed | B02 | not_started | no | Pipeline fragmentation claim is materially supported by repository state |

## 5. Open Risks / Blocks

1. `B01` onward remain blocked until `B00` is formally marked complete (implementation actions executed and logged).
2. High-blast-radius work (`B04`, `B05`) requires explicit rollback design before execution.
3. Partially confirmed claims (`RHA-010`, `RHA-019`, `RHA-027`) need tighter acceptance criteria before closeout.

## 6. Decision Log

| Date | Decision | Impacted Batches/Claims | Rationale | Owner |
|---|---|---|---|---|
| 2026-02-26 | Set initial safest execution candidate to `B00` | B00 (`RHA-013`, `RHA-023`) | Lowest risk, no behavior changes, establishes baseline guardrails. | codex |
| 2026-02-26 | Completed preflight and set `B00` to GO | B00 | `DRY-BASELINE` and `DRY-CHECK-REPLAY` passed; quality-gate baseline recorded. | codex |
