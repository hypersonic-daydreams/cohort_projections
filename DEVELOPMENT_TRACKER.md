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
| Test health baseline | stable | Latest recorded baseline: `1247 passed, 5 skipped` (excl. integration). ADR-056 accepted; PP4-01 through PP4-05 closed — 127 new tests added covering GQ separation, pipeline orchestrators, and base population loaders. |
| Claim replay health | stable | `27/27` adjudicated claims passing (latest full replay: 2026-02-27T18:14:03Z). |

## Projection Development Backlog (Canonical)

| ID | Work Item | Status | Next Milestone | Source |
|------|------|------|------|------|
| PP-001 | Publication-facing output QA and dissemination packaging | active | Collect stakeholder sign-off on 2026-02-28 package set and keep release checklist current for final publication handoff | `docs/reviews/2026-02-28-publication-output-qa-packaging-checklist.md`, `docs/reviews/repo-hygiene-audit/implementation/30-pp001-pp002-publication-followthrough-results.md` |
| PP-002 | Non-regression validation cadence during publication work | active | Run and record the next cadence cycle after the next material projection/config change (same gate set) | `docs/reviews/repo-hygiene-audit/implementation/06-dashboard-current.md`, `docs/reviews/repo-hygiene-audit/implementation/30-pp001-pp002-publication-followthrough-results.md` |
| PP-003 | City/place projection workstream reactivation (ADR-033) | active | IMP-08 backtest runner implementation completed (module + runner + tests + targeted lint/type checks recorded 2026-02-28); proceed to IMP-09 execution/winner selection on ND production artifacts | `docs/plans/pp3-s08-implementation-kickoff.md`, `docs/reviews/2026-02-28-pp3-s07-approval-gate.md` |
| PP-004 | Test coverage gap closure (ADR-056) | active | Close priority coverage gaps identified in ADR-056 Decision 6; align with PP-002 periodic review cadence | `docs/governance/adrs/056-testing-strategy-maturation.md`, `docs/guides/test-maintenance-practices.md` |

## PP-003 Phase 1 Scoping Checklist (Canonical)

| Step ID | Task | Status | Definition of Done | Evidence Artifact |
|------|------|------|------|------|
| PP3-S01 | Scope envelope | completed_2026-02-28 | Confirm place universe, projection horizon, and output granularity tiers (HIGH/MODERATE/LOWER/EXCLUDED) for the first release | Scoped statement below (this file) + ADR-033 cross-reference |
| PP3-S02 | Historical place data readiness | completed_2026-02-28 | Verify place-level annual history coverage needed for backtesting (target window: 2000-2024) and identify any gaps requiring imputation or exclusion rules | `docs/reviews/2026-02-28-place-data-readiness-note.md` |
| PP3-S03 | Place-to-county boundary mapping strategy | completed_2026-02-28 | Define authoritative mapping and handling rules for boundary/vintage changes so place shares can be compared consistently over time | `docs/reviews/2026-02-28-place-county-mapping-strategy-note.md` |
| PP3-S04 | Modeling spec for share-trending | completed_2026-02-28 | Select candidate trend specifications and constraint approach (share sum <= 100%, balance-of-county handling) for Phase 1 testing | `docs/reviews/2026-02-28-pp3-s04-modeling-spec.md` |
| PP3-S05 | Backtesting design and metrics | completed_2026-02-28 | Lock backtest design (train/test windows) and acceptance metrics by confidence tier | `docs/reviews/2026-02-28-pp3-s05-backtesting-design.md` |
| PP3-S06 | Output contract | completed_2026-02-28 | Define Phase 1 deliverables: projection files, metadata fields, QA summary tables, and workbook impacts | `docs/reviews/2026-02-28-pp3-s06-output-contract.md` |
| PP3-S07 | Approval gate (required) | approved_2026-02-28 | Human approval recorded to proceed from scoping to implementation (Tier 3 methodology-change control) | `docs/reviews/2026-02-28-pp3-s07-approval-gate.md` |
| PP3-S08 | Implementation kickoff packet | completed_2026-02-28 | Publish execution-ready task list (files, tests, validation gates, and ADR touchpoints) | `docs/plans/pp3-s08-implementation-kickoff.md` |

**Go/No-Go Rule:** PP-003 implementation starts only after `PP3-S01` through `PP3-S07` are complete and explicitly marked `go`.

## PP-003 Phase 1 Implementation Checklist (IMP-01 to IMP-04)

| Step ID | Task | Status | Definition of Done | Evidence Artifact |
|------|------|------|------|------|
| IMP-01 | Build place-county crosswalk | completed_2026-02-28 | Production run complete: primary crosswalk `357` rows (`355` active + `2` historical_only), multicounty detail `14` rows (`7` places), major-place spot checks PASS | `scripts/data/build_place_county_crosswalk.py`, `data/processed/geographic/place_county_crosswalk_2020.csv`, `data/processed/geographic/place_county_crosswalk_2020_multicounty_detail.csv`, `tests/test_data/test_place_county_crosswalk.py` |
| IMP-02 | Assemble place population history (2000-2024) | completed_2026-02-28 | Production run complete: `8,915` rows, year coverage `2000-2024` contiguous, no null populations, county join complete | `scripts/data/assemble_place_population_history.py`, `data/processed/place_population_history_2000_2024.parquet`, `tests/test_data/test_place_population_history.py` |
| IMP-03 | Compute historical place shares | completed_2026-02-28 | Production run complete: `10,240` rows (`8,915` place + `1,325` balance), county-year share identity holds (`max abs error = 0.0`), epsilon clamping bounds enforced | `cohort_projections/data/process/place_shares.py`, `data/processed/place_shares_2000_2024.parquet`, `tests/test_data/test_place_shares.py` |
| IMP-04 | Assign confidence tiers | completed_2026-02-28 | Tier enrichment written to crosswalk; active-place counts exactly `HIGH=9`, `MODERATE=9`, `LOWER=72`, `EXCLUDED=265` | `scripts/data/build_place_county_crosswalk.py`, `scripts/data/assemble_place_population_history.py`, `data/processed/geographic/place_county_crosswalk_2020.csv`, `tests/test_data/test_place_county_crosswalk.py`, `tests/test_data/test_place_population_history.py` |

**Execution note:** Runtime artifact generation is complete (2026-02-28). Human plausibility review remains recommended for multicounty primary assignments before publication.

## PP-003 Phase 2 Implementation Checklist (IMP-05 onward)

| Step ID | Task | Status | Definition of Done | Evidence Artifact |
|------|------|------|------|------|
| IMP-05 | Core share-trending module | completed_2026-02-28 | Implemented S04 logit-linear engine (`OLS`/`WLS`, midpoint centering, proportional + cap-and-redistribute constraints, balance reconciliation, county orchestration) with required edge-case handling and full kickoff test matrix coverage | `cohort_projections/data/process/place_share_trending.py`, `cohort_projections/data/process/__init__.py`, `tests/test_data/test_place_share_trending.py` |
| IMP-06 | Place projection orchestrator | completed_2026-02-28 | Implemented county-driven place projection orchestration (`run_place_projections`), tier-specific age/sex allocation (`HIGH` 18x2, `MODERATE` 6x2, `LOWER` total-only), and S06 output writing contract (per-place parquet/metadata/summary + `places_summary.csv` + `places_metadata.json` with required parquet footer metadata keys) with synthetic end-to-end contract tests | `cohort_projections/data/process/place_projection_orchestrator.py`, `cohort_projections/data/process/__init__.py`, `tests/test_data/test_place_projection_orchestrator.py` |
| IMP-07 | Configuration additions | completed_2026-02-28 | Added `place_projections` block to canonical config (paths, model, tiers, backtest windows, output years/key years) and added tests validating defaults plus configuration consumption by both IMP-05 (`trend_all_places_in_county`) and IMP-06 (`run_place_projections`) | `config/projection_config.yaml`, `tests/test_config/test_place_projection_config.py` |
| IMP-08 | Backtest runner script + module | completed_2026-02-28 | Implemented backtest computation module (`run_single_variant`, per-place metrics, tier aggregates, weighted scoring, tie-break winner selection) plus standalone runner executing A-I/A-II/B-I/B-II across windows with output artifacts and synthetic matrix coverage tests including EXCLUDED informational-tier handling | `cohort_projections/data/process/place_backtest.py`, `scripts/backtesting/run_place_backtest.py`, `tests/test_data/test_place_backtest.py`, `cohort_projections/data/process/__init__.py` |

### IMP-05 Verification Evidence (2026-02-28)

- `source .venv/bin/activate && pytest tests/test_data/test_place_share_trending.py` -> `14 passed`.
- `source .venv/bin/activate && ruff check cohort_projections/data/process/place_share_trending.py tests/test_data/test_place_share_trending.py cohort_projections/data/process/__init__.py` -> `All checks passed!`.
- `source .venv/bin/activate && mypy cohort_projections/data/process/place_share_trending.py` -> `Success: no issues found in 1 source file`.

### IMP-06 Verification Evidence (2026-02-28)

- `source .venv/bin/activate && pytest tests/test_data/test_place_projection_orchestrator.py` -> `4 passed`.
- `source .venv/bin/activate && ruff check cohort_projections/data/process/place_projection_orchestrator.py tests/test_data/test_place_projection_orchestrator.py cohort_projections/data/process/__init__.py` -> `All checks passed!`.
- `source .venv/bin/activate && mypy cohort_projections/data/process/place_projection_orchestrator.py` -> `Success: no issues found in 1 source file`.

### IMP-07 Verification Evidence (2026-02-28)

- `source .venv/bin/activate && pytest tests/test_config/test_place_projection_config.py tests/test_data/test_place_projection_orchestrator.py` -> `7 passed`.
- `source .venv/bin/activate && ruff check tests/test_config/test_place_projection_config.py` -> `All checks passed!`.
- `source .venv/bin/activate && mypy cohort_projections/data/process/place_projection_orchestrator.py cohort_projections/data/process/place_share_trending.py` -> `Success: no issues found in 2 source files`.

### IMP-08 Verification Evidence (2026-02-28)

- `source .venv/bin/activate && pytest tests/test_data/test_place_backtest.py tests/test_config/test_place_projection_config.py tests/test_data/test_place_projection_orchestrator.py` -> `15 passed`.
- `source .venv/bin/activate && ruff check cohort_projections/data/process/place_backtest.py scripts/backtesting/run_place_backtest.py tests/test_data/test_place_backtest.py cohort_projections/data/process/__init__.py` -> `All checks passed!`.
- `source .venv/bin/activate && mypy cohort_projections/data/process/place_backtest.py` -> `Success: no issues found in 1 source file`.

## PP-003 Scope Envelope (S01 Result)

Scope statement confirmed on 2026-02-28 (ADR-033 alignment):

- **Projection universe (first release):** 355 active ND incorporated places (2024 PEP place file in `data/raw/geographic/nd_places.csv`), with historical-only handling for dissolved places discovered in S02.
- **Projection horizon:** 2025-2055 annual, aligned to county/state production horizon.
- **Output granularity tiers (ADR-033 thresholds, based on 2024 population):**
  - `HIGH` (>10,000): 9 places
  - `MODERATE` (2,500-10,000): 9 places
  - `LOWER` (500-2,500): 72 places
  - `EXCLUDED` (<500): 265 places
- **Constraint frame:** place outputs remain county-constrained with explicit balance-of-county remainder (full constraint specification continues in `PP3-S04`).

## PP-004 Test Coverage Gap Closure Checklist (Canonical)

Priority coverage gaps from ADR-056 Decision 6 and `docs/guides/test-maintenance-practices.md`. Items are ranked by risk to projection accuracy.

| Step ID | Task | Status | Definition of Done | Reference |
|------|------|------|------|------|
| PP4-01 | GQ separation function tests (ADR-055) | completed_2026-02-28 | 42 tests in `test_gq_separation.py` covering all 5 GQ functions; `base_population_loader.py` 36.4%→68.2%, `residual_migration.py` 54.0%→85.4% | `tests/test_data/test_gq_separation.py` |
| PP4-02 | Pipeline orchestrator tests | completed_2026-02-28 | 38 tests in `test_pipeline_orchestrators.py` covering `run_residual_migration_pipeline` and `run_convergence_pipeline`; `convergence_interpolation.py` 44.3%→88.1% | `tests/test_data/test_pipeline_orchestrators.py` |
| PP4-03 | Base population loader tests | completed_2026-02-28 | 47 tests in `test_base_population_loader.py` covering all 3 loader entry points plus GQ separation; ADR-054 invariant verified | `tests/test_data/test_base_population_loader.py` |
| PP4-04 | Pre-commit hook blind spot fix | completed_2026-02-28 | `files` pattern updated to `^(cohort_projections|tests)/.*\.py$` so test-only changes trigger pytest | `.pre-commit-config.yaml` |
| PP4-05 | Silent skip audit | completed_2026-02-28 | Skip count: 5 (1 PyMC + 4 upstream bug); 222 silent-skip-capable tests all running; all IMPORTS_AVAILABLE gates resolve; baseline established | See PP4-05 audit results below |
| PP4-06 | First PP-002-aligned periodic coverage review | completed_2026-02-28 | All 5 checklist items PASS: 1,247 tests passed (5 skipped, unchanged); coverage stable at baselines; 194 IMPORTS_AVAILABLE gates unchanged; 137 integration tests pass; skip count steady at 5 | `docs/reviews/2026-02-28-pp4-06-periodic-coverage-review.md` |

**Approach:** Gaps are closed incrementally as modules are touched (ADR-056 Decision 6), not as a standalone sprint. PP4-01 through PP4-06 are complete. The PP-004 workstream is closed.

### PP4-05 Audit Results (2026-02-28)

- **Skip count baseline: 5** (1 PyMC slow test + 4 upstream `sigma_u.tolist()` bug in `test_bayesian_var.py`)
- **Alert threshold:** If skip count rises above 5, investigate immediately
- **Silent-skip-capable tests: 222** (194 IMPORTS_AVAILABLE-gated + 28 data-file-gated) — all currently running, none skipping
- **All IMPORTS_AVAILABLE gates resolve to True** in dev environment (matplotlib, seaborn, openpyxl, geopandas, pymc all installed)
- **Coverage deltas from PP4-01/02/03:** `base_population_loader.py` 36.4%→68.2%, `residual_migration.py` 54.0%→85.4%, `convergence_interpolation.py` 44.3%→88.1%

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
2. Keep repo-hygiene evidence current while executing publication tasks (`PP-001`, `PP-002`) and refresh package QA records as new export vintages are generated.
3. Continue PP-003 Phase 2 implementation (`IMP-09` onward): run backtest matrix on ND artifacts, select winner, and publish S05 evidence tables.
4. ~~Execute `PP4-06`~~ Completed 2026-02-28. PP-004 workstream closed. Future periodic reviews follow the PP-002 cadence.

## Deferred / Later Work

- City/place projection Phase 2+ expansion under ADR-033 (rolling-origin backtests, multi-county place splitting, housing-unit method).
- Optional TIGER integration and geospatial export enhancements.
- Additional publication-facing formatting and package cleanup.

## References

- `AGENTS.md`
- `docs/reviews/repo-hygiene-audit/implementation/06-dashboard-current.md`
- `docs/reviews/repo-hygiene-audit/implementation/30-pp001-pp002-publication-followthrough-results.md`
- `docs/reviews/2026-02-28-publication-output-qa-packaging-checklist.md`
- `docs/reviews/2026-02-28-place-data-readiness-note.md`
- `docs/reviews/2026-02-28-place-county-mapping-strategy-note.md`
- `docs/reviews/2026-02-28-pp3-s04-modeling-spec.md`
- `docs/reviews/2026-02-28-pp3-s05-backtesting-design.md`
- `docs/reviews/2026-02-28-pp3-s06-output-contract.md`
- `docs/reviews/2026-02-28-pp3-s07-approval-gate.md`
- `docs/plans/pp3-s08-implementation-kickoff.md`
- `docs/reviews/repo-hygiene-audit/implementation/17-open-risks-blockers-register.md`
- `docs/reviews/repo-hygiene-audit/implementation/02-action-batches.yaml`
- `docs/reviews/repo-hygiene-audit/verification/progress.md`
- `docs/governance/adrs/056-testing-strategy-maturation.md`
- `docs/guides/test-maintenance-practices.md`
- `docs/guides/test-suite-reference.md`
- `docs/guides/testing-workflow.md`
