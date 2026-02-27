# Development Tracker

Canonical, current-state tracker for the North Dakota cohort projections repository.

**Last Updated:** 2026-02-27  
**Projection Horizon:** 2025-2055  
**Status:** Publication preparation and repo-hygiene execution

## Purpose

Use this file for active status only. Historical session detail is archived to:

- `docs/archive/DEVELOPMENT_TRACKER_2026-02-26.md`

## Current Snapshot

| Area | Status | Notes |
|------|--------|-------|
| Core projection engine | complete | County/state production path is operational for all scenarios. |
| Data processing pipeline | complete | Inputs and transforms are in place; no active blocker. |
| Documentation alignment | complete | B01 documentation harmonization complete. |
| Repo-hygiene program | in progress | B00-B04 complete; B05 Wave 1 complete; B05 Wave 2 Step 1 complete (archive dir + low_growth + Ward file); remaining Wave 2 extraction/placement actions pending. |
| Test health baseline | stable | Latest recorded baseline: `1258 passed, 5 skipped`. |
| Claim replay health | stable | `27/27` adjudicated claims passing after resolved-state check redesign. |

## Active Priorities

1. Execute remaining B05 Wave 2 extraction/cleanup actions (SDC extraction + remaining root-clutter placements).
2. Maintain reproducible execution records for each B05 execution wave and replay affected claims.
3. Keep RB-003 and RB-004 tracked until dedicated remediation wave or policy closeout.

## Repo-Hygiene Batch State

| Batch | Name | Status | Notes |
|------|------|--------|-------|
| B00 | baseline_metrics_and_safety_harness | complete | Baseline metrics and guardrails pinned. |
| B01 | documentation_harmonization | complete | ADR index, navigation, horizon, and oversized docs harmonized. |
| B02 | pipeline_entrypoint_and_order_consistency | complete | Canonical runner and stage wiring remediated. |
| B03 | config_path_and_version_hygiene | complete | Version/config/path hygiene remediated. |
| B04 | code_structure_and_test_scope_alignment | complete | Import boundary, helper dedupe, test scope realignment, orphan wiring complete. |
| B05 | repository_footprint_and_data_hygiene | in progress | Wave 1 complete; Wave 2 Step 1 complete; remaining Wave 2 actions pending. |
| B06 | final_harmonization_and_claim_revalidation | blocked | Depends on B05 implementation completion. |

## Current Working Agreements

- Run commands in project venv: `source .venv/bin/activate`.
- Do not modify raw inputs in `data/raw/`.
- Use dry-run and claim-replay gates before and after batch edits.
- Keep audit narrative files (`docs/reviews/repo-hygiene-audit/00-07*.md`) unchanged unless explicitly requested.

## Near-Term Next Actions

1. Execute remaining Wave 2 archive placements and SDC extraction from `24-b05-delete-archive-proposal.md`.
2. Reconcile B05 claim checks for post-remediation state (`RHA-017` currently expected drift after archival).
3. Run B06 final harmonization once B05 closes.

## Deferred / Later Work

- City/place projection expansion under ADR-033.
- Optional TIGER integration and geospatial export enhancements.
- Additional publication-facing formatting and package cleanup.

## References

- `AGENTS.md`
- `docs/reviews/repo-hygiene-audit/implementation/06-dashboard-current.md`
- `docs/reviews/repo-hygiene-audit/implementation/17-open-risks-blockers-register.md`
- `docs/reviews/repo-hygiene-audit/implementation/02-action-batches.yaml`
- `docs/reviews/repo-hygiene-audit/verification/progress.md`
