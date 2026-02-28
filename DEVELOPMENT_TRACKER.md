# Development Tracker

Canonical, current-state tracker for the North Dakota cohort projections repository.

**Last Updated:** 2026-02-28  
**Projection Horizon:** 2025-2055  
**Status:** Publication preparation with repo-hygiene closeout complete; this file is the single source of truth for remaining projection-development work.

## Purpose

Use this file for active status only. Historical session detail is archived to:

- `docs/archive/DEVELOPMENT_TRACKER_2026-02-26.md`

## Source Of Truth Rule

1. When asked "what work remains for population projections," answer from this file first.
2. Treat these sections as canonical: `Projection Development Backlog (Canonical)`, `Documentation Consistency Queue`, and `Deferred / Later Work`.
3. Do not treat archived docs/reviews as open work unless the item is explicitly listed here.
4. If new open work is discovered, add it here with a source link before ending the session.

## Current Snapshot

| Area | Status | Notes |
|------|--------|-------|
| Core projection engine | complete | County/state production path is operational for all scenarios. |
| Data processing pipeline | complete | Inputs and transforms are in place; no active blocker. |
| Documentation alignment | complete | B01 documentation harmonization complete. |
| Repo-hygiene program | complete | B00-B06 implemented; full adjudicated replay `27/27` passing; RB-003 and RB-004 remediated and closed. |
| Test health baseline | stable | Latest recorded baseline: `1258 passed, 5 skipped`. |
| Claim replay health | stable | `27/27` adjudicated claims passing (latest full replay: 2026-02-27T18:14:03Z). |

## Projection Development Backlog (Canonical)

| ID | Work Item | Status | Next Milestone | Source |
|------|------|------|------|------|
| PP-001 | Publication-facing output QA and dissemination packaging | active | Re-run publication QA checks on current exports and finalize release packaging checklist | `DEVELOPMENT_TRACKER.md`, `docs/methodology.md` |
| PP-002 | Non-regression validation cadence during publication work | active | Run and record next full validation cycle (`run_complete_pipeline.sh --dry-run`, `ruff`, `mypy`, `pytest`) | `docs/reviews/repo-hygiene-audit/implementation/06-dashboard-current.md` |
| PP-003 | City/place projection workstream reactivation (ADR-033) | scoping_checklist_defined | Execute Phase 1 scoping checklist and issue explicit go/no-go decision before implementation | `docs/governance/adrs/033-city-level-projection-methodology.md`, `docs/governance/adrs/054-state-county-aggregation-reconciliation.md` |

## PP-003 Phase 1 Scoping Checklist (Canonical)

| Step ID | Task | Status | Definition of Done | Evidence Artifact |
|------|------|------|------|------|
| PP3-S01 | Scope envelope | pending | Confirm place universe, projection horizon, and output granularity tiers (HIGH/MODERATE/LOWER/EXCLUDED) for the first release | Add scoped statement to `DEVELOPMENT_TRACKER.md` and cross-reference ADR-033 |
| PP3-S02 | Historical place data readiness | pending | Verify place-level annual history coverage needed for backtesting (target window: 2000-2024) and identify any gaps requiring imputation or exclusion rules | Add short data readiness note under `docs/reviews/` and link it from tracker |
| PP3-S03 | Place-to-county boundary mapping strategy | pending | Define authoritative mapping and handling rules for boundary/vintage changes so place shares can be compared consistently over time | Add mapping rules note under `docs/reviews/` and link it from tracker |
| PP3-S04 | Modeling spec for share-trending | pending | Select candidate trend specifications and constraint approach (share sum <= 100%, balance-of-county handling) for Phase 1 testing | Add Phase 1 model spec note under `docs/reviews/` and link it from tracker |
| PP3-S05 | Backtesting design and metrics | pending | Lock backtest design (train/test windows) and acceptance metrics by confidence tier | Add validation design note under `docs/reviews/` and link it from tracker |
| PP3-S06 | Output contract | pending | Define Phase 1 deliverables: projection files, metadata fields, QA summary tables, and workbook impacts | Add output contract note under `docs/reviews/` and link it from tracker |
| PP3-S07 | Approval gate (required) | pending | Human approval recorded to proceed from scoping to implementation (Tier 3 methodology-change control) | Approval note in tracker + ADR/review cross-reference |
| PP3-S08 | Implementation kickoff packet | pending | Publish execution-ready task list (files, tests, validation gates, and ADR touchpoints) | Add implementation plan doc under `docs/plans/` and link it from tracker |

**Go/No-Go Rule:** PP-003 implementation starts only after `PP3-S01` through `PP3-S07` are complete and explicitly marked `go`.

## Documentation Consistency Queue

| ID | Item | Status | Action Needed | Source |
|------|------|------|------|------|
| DOC-001 | ADR-033 deferral rationale still cites unresolved state-county discrepancy as a blocker | closed_2026-02-28 | Completed: ADR-033 deferral section and revision history now reflect ADR-054 accepted/implemented state and sequencing-based deferral rationale | `docs/governance/adrs/033-city-level-projection-methodology.md`, `docs/governance/adrs/054-state-county-aggregation-reconciliation.md` |
| DOC-002 | 2026-02-23 projection output review "minor issues" may be stale | closed_2026-02-28 | Completed: revalidation recorded in-place; both minor issues are now explicitly marked resolved/stale | `docs/reviews/2026-02-23-projection-output-review.md` |

## Repo-Hygiene Batch State

| Batch | Name | Status | Notes |
|------|------|--------|-------|
| B00 | baseline_metrics_and_safety_harness | complete | Baseline metrics and guardrails pinned. |
| B01 | documentation_harmonization | complete | ADR index, navigation, horizon, and oversized docs harmonized. |
| B02 | pipeline_entrypoint_and_order_consistency | complete | Canonical runner and stage wiring remediated. |
| B03 | config_path_and_version_hygiene | complete | Version/config/path hygiene remediated. |
| B04 | code_structure_and_test_scope_alignment | complete | Import boundary, helper dedupe, test scope realignment, orphan wiring complete. |
| B05 | repository_footprint_and_data_hygiene | complete | Wave 1 and Wave 2 (Step 1 + Step 2) complete; extraction and cleanup actions executed. |
| B06 | final_harmonization_and_claim_revalidation | complete | Full claim replay passing; residual risks closed in follow-up remediation wave. |

## Current Working Agreements

- Run commands in project venv: `source .venv/bin/activate`.
- Do not modify raw inputs in `data/raw/`.
- Use dry-run and claim-replay gates before and after batch edits.
- Keep audit narrative files (`docs/reviews/repo-hygiene-audit/00-07*.md`) unchanged unless explicitly requested.

## Near-Term Next Actions

1. Keep the documentation consistency queue current; add and resolve new cross-document drift items as they appear.
2. Keep repo-hygiene evidence current while executing publication tasks (`PP-001`, `PP-002`).
3. Execute `PP3-S01` through `PP3-S03` to establish the initial city/place scoping baseline.

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
