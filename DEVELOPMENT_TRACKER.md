# Development Tracker

Canonical, current-state tracker for the North Dakota cohort projections repository.

**Last Updated:** 2026-02-26  
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
| Documentation alignment | complete | B01 documentation harmonization complete with expected drift documented. |
| Repo-hygiene program | in progress | B00/B01/B02 complete; next executable batch is B03. |
| Test health baseline | stable | Latest recorded baseline: `1258 passed, 5 skipped`. |

## Active Priorities

1. Execute repo-hygiene `B03` config/path/version hygiene after B02 closeout.
2. Keep methodology and governance docs synchronized with implemented ADR statuses.
3. Maintain reproducible execution records for every implementation batch.

## Repo-Hygiene Batch State

| Batch | Name | Status | Notes |
|------|------|--------|-------|
| B00 | baseline_metrics_and_safety_harness | complete | Baseline metrics and guardrails pinned. |
| B01 | documentation_harmonization | complete | ADR index, navigation, horizon, and oversized docs were harmonized and replayed. |
| B02 | pipeline_entrypoint_and_order_consistency | complete | Canonical runner added, full stage order wired, and post-edit claim replay captured expected remediation drift. |
| B03 | config_path_and_version_hygiene | ready | Can run after B01 if sequencing keeps context clean. |
| B04 | code_structure_and_test_scope_alignment | blocked | Depends on B03. |
| B05 | repository_footprint_and_data_hygiene | ready | Prerequisites from B00/B01 are satisfied. |
| B06 | final_harmonization_and_claim_revalidation | blocked | Depends on B03-B05. |

## Current Working Agreements

- Run commands in project venv: `source .venv/bin/activate`.
- Do not modify raw inputs in `data/raw/`.
- Use dry-run and claim-replay gates before and after batch edits.
- Keep audit narrative files (`docs/reviews/repo-hygiene-audit/00-07*.md`) unchanged unless explicitly requested.

## Near-Term Next Actions

1. Run B03 pre-edit dry-run profiles before touching config/path/version surfaces.
2. Define resolved-state claim checks for B01 and B02 findings before final harmonization closeout.
3. Keep claim-replay evidence refreshed after each batch closeout.

## Deferred / Later Work

- City/place projection expansion under ADR-033.
- Optional TIGER integration and geospatial export enhancements.
- Additional publication-facing formatting and package cleanup.

## References

- `AGENTS.md`
- `docs/reviews/repo-hygiene-audit/implementation/06-dashboard-current.md`
- `docs/reviews/repo-hygiene-audit/implementation/02-action-batches.yaml`
- `docs/reviews/repo-hygiene-audit/verification/progress.md`
