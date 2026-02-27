# Development Tracker

Canonical, current-state tracker for the North Dakota cohort projections repository.

**Last Updated:** 2026-02-27  
**Projection Horizon:** 2025-2055  
**Status:** Publication preparation and post-implementation repo-hygiene follow-through

## Purpose

Use this file for active status only. Historical session detail is archived to:

- `docs/archive/DEVELOPMENT_TRACKER_2026-02-26.md`

## Current Snapshot

| Area | Status | Notes |
|------|--------|-------|
| Core projection engine | complete | County/state production path is operational for all scenarios. |
| Data processing pipeline | complete | Inputs and transforms are in place; no active blocker. |
| Documentation alignment | complete | B01 documentation harmonization complete. |
| Repo-hygiene program | complete_with_open_risk_tracking | B00-B06 implemented; full adjudicated replay now `27/27` passing; RB-003 and RB-004 remain tracked for dedicated remediation/policy closeout. |
| Test health baseline | stable | Latest recorded baseline: `1258 passed, 5 skipped`. |
| Claim replay health | stable | `27/27` adjudicated claims passing (latest full replay: 2026-02-27T18:14:03Z). |

## Active Priorities

1. Plan and execute RB-003 remediation or policy-closeout for full pipeline dry-run coverage.
2. Plan and execute RB-004 remediation or policy-closeout for full-repo lint/type debt.
3. Maintain claim replay stability while transitioning from implementation to publication-focused work.

## Repo-Hygiene Batch State

| Batch | Name | Status | Notes |
|------|------|--------|-------|
| B00 | baseline_metrics_and_safety_harness | complete | Baseline metrics and guardrails pinned. |
| B01 | documentation_harmonization | complete | ADR index, navigation, horizon, and oversized docs harmonized. |
| B02 | pipeline_entrypoint_and_order_consistency | complete | Canonical runner and stage wiring remediated. |
| B03 | config_path_and_version_hygiene | complete | Version/config/path hygiene remediated. |
| B04 | code_structure_and_test_scope_alignment | complete | Import boundary, helper dedupe, test scope realignment, orphan wiring complete. |
| B05 | repository_footprint_and_data_hygiene | complete | Wave 1 and Wave 2 (Step 1 + Step 2) complete; extraction and cleanup actions executed. |
| B06 | final_harmonization_and_claim_revalidation | complete_with_open_risk_tracking | Full claim replay passing; RB-003/RB-004 remain open. |

## Current Working Agreements

- Run commands in project venv: `source .venv/bin/activate`.
- Do not modify raw inputs in `data/raw/`.
- Use dry-run and claim-replay gates before and after batch edits.
- Keep audit narrative files (`docs/reviews/repo-hygiene-audit/00-07*.md`) unchanged unless explicitly requested.

## Near-Term Next Actions

1. Define and schedule RB-003 dry-run stage coverage remediation (or accepted limitation policy).
2. Define and schedule RB-004 lint/type debt remediation (or accepted baseline policy).
3. Keep repo-hygiene artifacts and claim evidence current as publication tasks proceed.

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
