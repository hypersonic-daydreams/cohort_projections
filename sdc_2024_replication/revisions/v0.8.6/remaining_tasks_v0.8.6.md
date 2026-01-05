---
title: "v0.8.6 Remaining Tasks Tracker"
date_created: 2026-01-04
status: "active"
archived_tracker: "sdc_2024_replication/revisions/v0.8.6/progress_tracker_v0.8.6_critique_v0.8.5.md"
status_update_memo: "sdc_2024_replication/revisions/v0.8.6/status_update_2026-01-04.md"
---

# v0.8.6 Remaining Tasks Tracker

## Context
- Archived full tracker: `sdc_2024_replication/revisions/v0.8.6/progress_tracker_v0.8.6_critique_v0.8.5.md`
- Latest status memo: `sdc_2024_replication/revisions/v0.8.6/status_update_2026-01-04.md`
- ADR-025 documents post-2020 refugee coverage and missing-state handling.

## Remaining Data Tasks
- [x] OCR RPC refugee PDFs for FY2022 and FY2024; attempt extraction; update processed outputs + manifests.
  - FY2022 OCR yields a 49-state panel with Hawaii/Wyoming omitted in the source (left missing); FY2021/FY2023/FY2024 partial panels improved (46/48/50 states); ND-only placeholder removed.
- [x] LPR FY2007-FY2013 yearbook ZIPs (manual download) and ingest via `sdc_2024_replication/data_immigration_policy/scripts/process_dhs_lpr_data.py`.
  - FY2012 state totals extracted from `data/raw/immigration/dhs_lpr/yearbook_pdfs/Yearbook_Immigration_Statistics_2012.pdf`.
- [ ] Decide whether additional 1980s/1990s Census components are needed (scope/approval).

## Integration and Pipeline Checks
- [ ] Decide whether to model Amerasian/SIV as separate covariate or merged humanitarian series (approval required).
- [ ] Confirm downstream scripts reference updated outputs (no hard-coded paths).
- [ ] Update database/loader plumbing if using PostgreSQL tables (e.g., `rpc.refugee_arrivals`).
- [x] Reconcile partial FY2021-FY2024 refugee data and prevent ND-only placeholders from being treated as full panels.
  - Missing states are left missing (no zero-fill); state-panel analyses drop incomplete states post-2020; national totals use official FY2021-2024 values (ADR-025).
- [ ] Build LPR multi-year panel variants needed for modeling (state totals + any aggregation variants).

## Modeling Updates (Approval Required if Results Change)
- [ ] Update Travel Ban DiD/event-study with extended refugee data and refined timing.
- [ ] Refit wave duration metrics; reassess right-censoring and hazard ratios.
- [ ] Add regime-aware modeling for long-run PEP series (state-space or regime dummies).
- [ ] Integrate LPR + refugee + ACS covariates into forecasting (ARIMAX or state-space).
- [ ] Reassess uncertainty envelopes after data fusion (avoid double-counting variance).

## Documentation and Article Updates
- [ ] Draft v0.8.6 response-to-critique document in `sdc_2024_replication/revisions/v0.8.6/`.
- [ ] Update data/methods narrative in `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/02_data_methods.tex`.
- [ ] Update results/figures in `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/03_results.tex` and `.../figures/`.
- [ ] Update discussion/limitations in `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/04_discussion.tex`.
- [ ] Add/refresh citations in `sdc_2024_replication/scripts/statistical_analysis/journal_article/references.bib`.
- [ ] Update `sdc_2024_replication/METHODOLOGY_SPEC.md` if methodology changes are adopted.
- [ ] Update `data/raw/immigration/census_population_estimates/README_census_vintages.md` if new vintages are added.

## Methodological Pitfalls to Document
- [ ] Time-base mismatch: FY vs PEP-year vs calendar year (document mapping logic).
- [ ] Net vs gross flows: ensure consistent interpretation across sources.
- [ ] Vintage revisions: treat Census vintage choice as sensitivity dimension.
- [ ] Measurement error (ACS MOE; LPR reporting lags).
- [ ] Double counting: avoid stacking variance from multiple correlated signals.
- [ ] Partial series risk: prevent ND-only data from being misinterpreted as national panel data.

## Validation and Testing
- [ ] Add/extend tests for new data ingestion and model components.
- [x] Add unit tests for post-2020 missing-state filtering in Module 8.
- [ ] Run targeted test suite for updated modules.
