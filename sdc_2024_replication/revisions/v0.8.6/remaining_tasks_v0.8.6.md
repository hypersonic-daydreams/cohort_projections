---
title: "v0.8.6 Remaining Tasks Tracker"
date_created: 2026-01-04
status: "complete"
archived_tracker: "sdc_2024_replication/revisions/v0.8.6/progress_tracker_v0.8.6_critique_v0.8.5.md"
status_update_memo: "sdc_2024_replication/revisions/v0.8.6/status_update_2026-01-06.md"
---

# v0.8.6 Remaining Tasks Tracker

## Context
- Archived full tracker: `sdc_2024_replication/revisions/v0.8.6/progress_tracker_v0.8.6_critique_v0.8.5.md`
- Latest status memo: `sdc_2024_replication/revisions/v0.8.6/status_update_2026-01-06.md`
- ADR-025 documents post-2020 refugee coverage and missing-state handling.

## Remaining Data Tasks
- [x] OCR RPC refugee PDFs for FY2022 and FY2024; attempt extraction; update processed outputs + manifests.
  - FY2022 OCR yields a 49-state panel with Hawaii/Wyoming omitted in the source (left missing); FY2021/FY2023/FY2024 partial panels improved (46/48/50 states); ND-only placeholder removed.
- [x] LPR FY2007-FY2013 yearbook ZIPs (manual download) and ingest via `sdc_2024_replication/data_immigration_policy/scripts/process_dhs_lpr_data.py`.
  - FY2012 state totals extracted from `data/raw/immigration/dhs_lpr/yearbook_pdfs/Yearbook_Immigration_Statistics_2012.pdf`.
- [x] Decide whether additional 1980s/1990s Census components are needed (scope/approval).
  - Decision: **defer** pre-2000 components for v0.8.6 (not required for the regime/uncertainty objectives; comparability costs likely exceed benefits).

## Integration and Pipeline Checks
- [x] Decide whether to model Amerasian/SIV as separate covariate or merged humanitarian series (approval required).
  - Decision: keep USRAP refugee treatment/exposure strict; model Amerasian/SIV as a separate durable series linked to capacity scenarios with a default post-2024 sunset (ADR-026).
- [x] Confirm downstream scripts reference updated outputs (no hard-coded paths).
  - Centralized `sdc_2024_replication/scripts/statistical_analysis/data_loader.py` now resolves the analysis directory via `config/projection_config.yaml` and supports `SDC_ANALYSIS_DATA_SOURCE={auto,db,files}` to avoid brittle hard-coded paths.
- [x] Update database/loader plumbing if using PostgreSQL tables (e.g., `rpc.refugee_arrivals`).
  - `sdc_2024_replication/scripts/database/db_config.py` now uses environment variables (no user-specific credentials in code); `data_loader.py` falls back to processed parquet/CSV when DB is unavailable or incomplete.
- [x] Reconcile partial FY2021-FY2024 refugee data and prevent ND-only placeholders from being treated as full panels.
  - Missing states are left missing (no zero-fill); state-panel analyses drop incomplete states post-2020; national totals use official FY2021-2024 values (ADR-025).
- [x] Build LPR multi-year panel variants needed for modeling (state totals + any aggregation variants).
  - New script: `sdc_2024_replication/data_immigration_policy/scripts/build_dhs_lpr_panel_variants.py` (writes states-only panel, balanced panel, US totals, ND shares).

## Modeling Updates (Approval Required if Results Change)
- [x] Update Travel Ban DiD/event-study with extended refugee data and refined timing.
  - Added ADR-027 to preserve pre-COVID causal estimand and add an explicitly-labeled FY2024 regime-dynamics supplement; excluded pseudo-nationalities (e.g., `Total`) from DiD units; added appendix figure `fig_app_event_study_extended`.
- [x] Increase Monte Carlo scenario simulation counts and parallelize for reproducible multi-core execution (ADR-028).
- [x] Refit wave duration metrics; reassess right-censoring and hazard ratios.
  - Senior-scholar guidance (verbatim memo): `sdc_2024_replication/revisions/v0.8.6/senior_scholar_memo_wave_duration.md`
  - Pre-declared spec grid + runbook: `sdc_2024_replication/revisions/v0.8.6/wave_duration_refit_spec_grid.md`
  - Implementation complete: Module 8 supports fixed baseline / year completion / gap tolerance / min-peak + tagged outputs; Module 9 supports `--duration-tag`; unit tests added.
  - Runs complete: P0/S1/S2/S3 executed; delta table + right-censoring/pause audit + Module 9 downstream-impact comparison recorded in the spec-grid doc. Decision: P0 primary, S1 sensitivity; manuscript duration/scenario text and captions updated (ADR-029).
- [x] Add regime-aware modeling for long-run PEP series (state-space or regime dummies).
  - [x] Decision documented (ADR-030): `docs/governance/adrs/030-pep-regime-aware-modeling-long-run-series.md`
  - [x] Implementation spec grid + definition-of-done runbook: `sdc_2024_replication/revisions/v0.8.6/pep_regime_modeling_spec_grid.md`
  - [x] Execute spec grid (P0 + sensitivities) and save reproducible artifacts under `sdc_2024_replication/scripts/statistical_analysis/`.
    - Results JSON: `sdc_2024_replication/scripts/statistical_analysis/results/module_B1_pep_regime_modeling.json`
    - Slopes table: `sdc_2024_replication/scripts/statistical_analysis/results/module_B1_pep_regime_modeling_slopes.csv`
    - Figure: `sdc_2024_replication/scripts/statistical_analysis/figures/module_B1_pep_regime_modeling__P0.png`
  - [x] Update manuscript text/appendix to reference the regime-aware diagnostic outputs (label as measurement-regime diagnostic, not ND-specific causal estimate).
    - Methods pointer: `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/02_data_methods.tex`
    - Appendix figure/table: `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/06_appendix.tex`
- [x] Integrate LPR + refugee + ACS covariates into forecasting (appendix near-term anchor; keep Moderate baseline).
  - [x] Decision + scope control documented (ADR-031): `docs/governance/adrs/031-covariate-conditioned-near-term-forecast-anchor.md`
  - [x] Appendix runbook/spec created: `sdc_2024_replication/revisions/v0.8.6/covariate_forecasting_appendix_spec.md`
  - [x] Implement covariate-conditioned 2025--2029 appendix anchor outputs (P0 + at least one sensitivity) under `sdc_2024_replication/scripts/statistical_analysis/`.
    - Script: `sdc_2024_replication/scripts/statistical_analysis/module_app_covariate_anchor.py`
    - Results JSON: `sdc_2024_replication/scripts/statistical_analysis/results/module_app_covariate_anchor.json`
    - Forecast table: `sdc_2024_replication/scripts/statistical_analysis/results/module_app_covariate_anchor_2025_2029.csv`
    - Figure: `sdc_2024_replication/scripts/statistical_analysis/figures/fig_app_covariate_anchor_2025_2029.png`
  - [x] Add appendix figure/table + short narrative in `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/06_appendix.tex`.
- [x] Reassess uncertainty envelopes after data fusion (avoid double-counting variance).
  - Decision record: ADR-032 `docs/governance/adrs/032-uncertainty-envelopes-two-band-approach.md`
  - Implementation spec/runbook: `sdc_2024_replication/revisions/v0.8.6/uncertainty_envelope_two_band_spec.md`
  - Implementation: Module 9 now reports nested bands (baseline-only PI + wave-adjusted uncertainty envelope).
  - Key artifacts:
    - Results JSON: `sdc_2024_replication/scripts/statistical_analysis/results/module_9_scenario_modeling.json`
    - Publication Figure 8: `sdc_2024_replication/scripts/statistical_analysis/journal_article/figures/fig_08_scenarios.pdf`
    - Updated caption: `sdc_2024_replication/scripts/statistical_analysis/journal_article/figures/figure_captions.tex`

## Documentation and Article Updates
- [x] Draft v0.8.6 response-to-critique document in `sdc_2024_replication/revisions/v0.8.6/`.
- [x] Update data/methods narrative in `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/02_data_methods.tex`.
- [x] Update results/figures in `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/03_results.tex` and `.../figures/`.
  - Scenario table/text and Figure 8 now align with `module_9_scenario_modeling` outputs (adds reproducible Immigration Policy scenario at 0.65× Moderate).
- [x] Update discussion/limitations in `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/04_discussion.tex`.
- [x] Add/refresh citations in `sdc_2024_replication/scripts/statistical_analysis/journal_article/references.bib`.
- [x] Update `sdc_2024_replication/METHODOLOGY_SPEC.md` if methodology changes are adopted.
  - Added a v0.8.6 addendum pointing to the adopted ADRs/runbooks (replication pipeline enhancements vs. the SDC 2024 spreadsheet method).
- [x] Update `data/raw/immigration/census_population_estimates/README_census_vintages.md` if new vintages are added.
  - Added `data/raw/immigration/census_population_estimates/README_census_vintages.md` documenting included vintages and interpretation as measurement regimes.

## Methodological Pitfalls to Document
- [x] Time-base mismatch: FY vs PEP-year vs calendar year (document mapping logic).
- [x] Net vs gross flows: ensure consistent interpretation across sources.
- [x] Vintage revisions: treat Census vintage choice as sensitivity dimension.
- [x] Measurement error (ACS MOE; LPR reporting lags).
- [x] Double counting: avoid stacking variance from multiple correlated signals.
- [x] Partial series risk: prevent ND-only data from being misinterpreted as national panel data.

## Validation and Testing
- [x] Add/extend tests for new data ingestion and model components.
  - Added unit tests for the data loader file-fallback mode and LPR panel variant builder.
- [x] Add unit tests for post-2020 missing-state filtering in Module 8.
- [x] Run targeted test suite for updated modules.
