---
title: "v0.8.6 Critique Implementation Tracker (ChatGPT 5.2 Pro on v0.8.5)"
date_created: 2026-01-04
source_critique: "sdc_2024_replication/revisions/v0.8.6/critique_chatgpt_5_2_pro_v0.8.5.md"
target_revision: "v0.8.6"
status: "archived"
description: "Progress tracker for implementing the v0.8.5 critique; structured for future AI agents with discrete, auditable tasks."
archived_date: 2026-01-04
superseded_by: "sdc_2024_replication/revisions/v0.8.6/remaining_tasks_v0.8.6.md"
---

# v0.8.6 Critique Implementation Tracker

## Archive Notice (2026-01-04)
This tracker is archived to preserve the full v0.8.6 record. Active work continues in:
- `sdc_2024_replication/revisions/v0.8.6/remaining_tasks_v0.8.6.md`
Recent updates (OCR extraction and FY2024 partial coverage) are tracked in the remaining-tasks tracker.

## Usage
- Mark tasks as complete with `[x]`.
- Add short notes directly under each task when decisions are made.
- Keep additions concise; link to detailed docs/scripts instead of pasting large content.

## Status Update Memo
- 2026-01-04: [v0.8.6 status update memo](./status_update_2026-01-04.md)

## Governance and Scope (Do First)
- [x] Confirm which critique items are in-scope for v0.8.6 vs deferred; list them in the scope notes below.
  - Scope confirmed (2026-01-04): in-scope items listed below; approval required before any result-changing implementation.
- [x] Flag methodology changes that affect results and confirm human approval before implementation (Tier 3).
  - Approval received 2026-01-04 for items #1-#5; proceed with documented changes only.
- [x] Decide whether a new ADR is required for data-fusion/regime-model changes; draft stub if needed.
  - Drafted: `docs/governance/adrs/024-immigration-data-extension-fusion.md`.
- [x] Review existing inventories and data manifests to avoid duplication:
  - `REPOSITORY_INVENTORY.md`
  - `data/DATA_MANIFEST.md`
  - `sdc_2024_replication/DATA_UPDATE_PLAN.md`
  - `sdc_2024_replication/data_immigration_policy/MANIFEST.md`
  - Reviewed for coverage, temporal alignment notes, and raw/processed paths.
- [x] Identify existing scripts and loaders before creating new ones (`scripts/`, `sdc_2024_replication/scripts/`).
  - Existing: `sdc_2024_replication/data_immigration_policy/scripts/process_refugee_data.py`, `process_dhs_lpr_data.py`, `combine_census_vintages.py`, `process_b05006.py`, `download_b05006.py`; `sdc_2024_replication/scripts/prepare_immigration_policy_data.py`, `verify_pipeline.py`; `scripts/` pipeline/validation utilities.

## Scope Notes (Fill In)
Scope confirmed (2026-01-04): items checked below are in-scope for v0.8.6; implementation awaits explicit approval where noted.
Approval granted (2026-01-04): items #1-#5 approved; SIV/Amerasian sourcing should prioritize RPC archives, with alternate sources allowed for coverage/trust (document decisions).

In-scope items for v0.8.6:
- [x] Extend refugee arrivals beyond FY2020 using existing RPC archives and reconcile partial 2021-2024 ND-only data (requires approval if results change).
- [x] Replace FY-to-PEP-year approximation with month-aware alignment when monthly data are available (requires approval).
- [x] Incorporate or explicitly evaluate SIV/Amerasian flows as parallel humanitarian signals (requires approval).
- [x] Audit existing DHS LPR multi-year data and integrate into modeling where appropriate (requires approval).
- [x] Leverage long-run Census components (2000-2024 available) for regime-aware modeling and sensitivity checks (requires approval).
- [x] Evaluate adding ACS moved-from-abroad inflow proxy and document measurement error (requires approval).

Deferred items:
- [ ] 1980s-era Census components integration (pending data availability and approval).
- [ ] Full state-space fusion model (scope/complexity decision pending).

## Existing Data Inventory (Fill In)
| Dataset | Current Coverage | Current Location | Notes |
|---|---|---|---|
| Refugee arrivals (RPC, state×nationality) | FY2002–FY2020 full; FY2021–FY2024 ND-only totals | `data/processed/immigration/analysis/refugee_arrivals_by_state_nationality.parquet` | 2021–2024 entries are ND-only totals from manual PDF/news extraction; not a full panel. |
| Refugee arrivals raw archives | FY2012–FY2024 (xls/xlsx + pdf) | `data/raw/immigration/refugee_arrivals/` | PDFs for FY2021–FY2024 not fully parsed; `orr_prm_1975_2018_v1.dta` and `PRM_Refugee_Admissions_Report_Nov_2025.xlsx` also present; duplicate PDFs in `data/raw/immigration/rpc_archives/`. |
| ND refugee arrivals FY2020–FY2024 (manual) | FY2020–FY2024 ND totals; partial breakdowns | `data/raw/immigration/nd_refugee_arrivals_fy2020_2024.csv` | Mix of refugees/SIV/parole; useful for ND-only checks. |
| ND refugee arrivals by nationality (manual) | FY2021 + FY2023 only | `data/raw/immigration/nd_refugee_arrivals_by_nationality.csv` | FY2022/2024 extraction issues noted. |
| LPR by state (DHS/OIS) | FY2000–FY2006, FY2014–FY2023 | `data/processed/immigration/analysis/dhs_lpr_by_state_time.parquet` | Raw XLSX in `data/raw/immigration/dhs_lpr/` plus `yearbook_lpr_2023_all_tables.xlsx`; FY2007–FY2013 gap remains. |
| LPR by country/region (DHS/OIS) | FY2014–FY2023 | `data/processed/immigration/analysis/dhs_lpr_by_country_time.parquet` | Country/region labels in `region_country_of_birth`. |
| LPR by state×country | FY2023 only | `data/processed/immigration/analysis/dhs_lpr_by_state_country.parquet` | Single-year cross-section. |
| PEP components (combined vintages) | 2000–2024 | `data/processed/immigration/state_migration_components_2000_2024.csv` | Raw vintages in `data/raw/immigration/census_population_estimates/`. |
| PEP components (analysis subset) | 2010–2024 | `data/processed/immigration/analysis/combined_components_of_change.csv` | 53 rows per year; used in recent analyses. |
| ACS foreign-born stock by origin | 2009–2023 | `data/processed/immigration/analysis/acs_foreign_born_by_state_origin.parquet` | Stock, not flow; MOE included. |
| ACS foreign-born ND share | 2009–2023 | `data/processed/immigration/analysis/acs_foreign_born_nd_share.parquet` | Stock-based share. |
| ACS moved-from-abroad inflow | 2010–2023 | `data/processed/immigration/analysis/acs_moved_from_abroad_by_state.parquet` | Flow proxy from ACS B07007; raw inputs in `data/raw/immigration/acs_migration/` with MOE preserved. |

## Data Acquisition (New or Extended Sources)
- [x] RPC monthly arrivals FY2021–FY2025 (refugees by state/nationality).
  - Check existing RPC PDFs in `data/raw/immigration/refugee_arrivals/` and `data/raw/immigration/rpc_archives/` before downloading.
  - Decide raw storage location (`data/raw/immigration/`) and document rationale.
  - Record “data as of” download date and source URL in the manifest.
  - Downloaded RPC archive PDFs for FY2021–FY2024 “as of” releases from rpc.state.gov; FY2022 PDF is image-only, FY2024 encoded text (see manifest).
  - PRM report is national-level monthly refugee admissions (no state breakdown; no SIV series noted).
- [x] RPC Amerasian/SIV arrivals by state/nationality (parallel humanitarian series).
  - Ensure consistent identifiers with refugee series.
  - Downloaded FY2021–FY2024 Amerasian/SIV PDFs from RPC archives (processing pending).
- [x] DHS/OIS multi-year LPR tables (state/year + class or origin).
  - Verify licensing/terms and data availability window.
  - Raw `lpr_state_county_2007-2023.xlsx` + `yearbook_lpr_2023_all_tables.xlsx` already in `data/raw/immigration/dhs_lpr/`.
  - Added FY2000-FY2006 yearbook ZIPs + supplemental tables (downloaded 2026-01-04 via browser; programmatic access blocked).
  - Organized extra DHS downloads into `data/raw/immigration/dhs_refugees_asylees/` and `data/raw/immigration/dhs_nonimmigrants/`; removed duplicate `LPR2006_0 (1).zip`.
- [ ] Census PEP components long-run series (1980+ where available).
  - Pull by decade series and archive vintage notes.
  - Current raw dir only has NST-EST2009/2020/2024 state totals; no 1980s/1990s components present.
- [x] ACS “moved from abroad” annual series (ND and state panel).
  - Capture MOE/SE metadata if available.
  - Downloaded ACS B07007 group data for 2010–2023; stored in `data/raw/immigration/acs_migration/` with year-specific label JSON files.

## Task Breakdown by Critique Theme
### Refugee Arrivals Extension (FY2021+)
- [x] Audit current processed refugee file for 2021–2024 partial coverage; document in manifest.
  - Confirmed FY2021–FY2024 are ND-only rows (state_count=1); documented in `data/DATA_MANIFEST.md`.
- [x] Extract FY2021–FY2024 state×nationality from RPC PDFs or obtain alternative source.
  - FY2021/FY2023 full panels extracted; FY2022 image-only and FY2024 encoded, leaving ND-only placeholders (documented in manifest).
- [x] Create/extend monthly-to-PEP-year crosswalk and store in processed metadata.
  - Crosswalk saved as `data/processed/immigration/analysis/refugee_fy_month_to_pep_year_crosswalk.csv`.
- [x] Regenerate refugee arrivals parquet and log differences vs prior series.
  - New monthly and PEP-year outputs saved; FY2022/FY2024 remain ND-only due to unreadable PDFs.

### SIV/Amerasian Parallel Series
- [x] Identify usable SIV/Amerasian series (state- or national-level) and document source limits.
  - RPC archive PDFs (FY2021–FY2024) identified and processed into state×nationality panels.
- [ ] Decide whether to model as separate covariate or merged humanitarian series (approval required).
- [x] Update processing outputs and data manifests accordingly.
  - Outputs saved under `data/processed/immigration/analysis/amerasian_siv_arrivals_*` and documented in `data/DATA_MANIFEST.md`.

### LPR Multi-Year Panel
- [x] Review raw DHS LPR files (2007–2023) and current processed coverage (2014–2023).
  - Raw files confirmed in `data/raw/immigration/dhs_lpr/`; processed `dhs_lpr_by_state_time.parquet` covers FY2014–FY2023 (54 states/territories per year).
  - `process_dhs_lpr_data.py` now reads config-driven `data/raw/immigration/dhs_lpr/`.
- [x] Expand processing to earlier years if feasible; normalize state identifiers.
  - Parsed Table 4 from LPR2006_0.zip to extend state totals to FY2000-FY2006; FY2007-FY2013 remain a gap pending additional yearbooks.
- [x] Produce state×year and origin×year panels for modeling inputs.
  - Re-ran `process_dhs_lpr_data.py` after fixing raw path; outputs refreshed for FY2014–FY2023.

### Long-Run PEP Components + Regime Handling
- [x] Confirm combined components series (2000–2024) aligns with current modeling inputs.
  - `state_migration_components_2000_2024.csv` spans 2000–2024; `analysis/combined_components_of_change.csv` spans 2010–2024 with 53 states per year.
- [x] Add explicit regime markers (pre/post-2010, pre/post-2020) for variance shifts.
  - Derived file: `data/processed/immigration/state_migration_components_2000_2024_with_regime.csv`.
- [ ] Decide whether additional 1980s/1990s series are needed (scope/approval).

### ACS “Moved From Abroad” Proxy
- [x] Identify ACS table/source with annual moved-from-abroad counts for states.
  - ADR-021 ACS migration report points to B07007/B07407 (mobility by citizenship) for aggregate abroad counts; PUMS needed for origin-state flows.
- [x] Build processing script and store outputs in `data/processed/immigration/analysis/`.
  - Outputs: `acs_moved_from_abroad_by_state.parquet`, `acs_moved_from_abroad_by_state.csv`.
- [x] Document MOE handling and any smoothing decisions.
  - MOE preserved from ACS B07007; no smoothing applied.

### Model and Output Updates
- [ ] Update Travel Ban DiD/event-study with extended refugee data and refined timing.
- [ ] Refit wave duration metrics; reassess right-censoring and hazard ratios.
- [ ] Integrate new covariates into forecasting models (ARIMAX/state-space) after approval.

## Data Handling Checklist (Apply to Each New Dataset)
- [x] Verify raw file location under `data/raw/immigration/` and record source URL + download date.
  - RPC refugee + Amerasian/SIV archive URLs recorded in `data/DATA_MANIFEST.md` (2026-01-04).
  - DHS yearbook extras filed into `dhs_refugees_asylees/` and `dhs_nonimmigrants/`; duplicate LPR ZIP removed; NATZ 2004 supplemental kept under `dhs_naturalizations/historical_downloads/`.
- [x] Add or update manifest entry describing coverage, caveats, and parsing notes.
- [x] Ensure processed outputs land in `data/processed/immigration/` with clear naming.
- [x] Add validation summary (basic counts, min/max year, missingness) in a short log.
  - `data/processed/immigration/analysis/immigration_v0.8.6_validation.md` (plus dataset-specific ACS validation log).
- [ ] Confirm downstream scripts reference the updated outputs (no hard-coded paths).

## Data Processing and Integration
- [x] Build/extend refugee + SIV ingestion pipeline (monthly → PEP-year).
  - Review `sdc_2024_replication/data_immigration_policy/scripts/process_refugee_data.py` before adding new scripts.
  - Create explicit FY-to-PEP-year crosswalk (Jul–Jun mapping).
  - Store derived annual series in `data/processed/immigration/` with provenance notes.
  - Refugee monthly + PEP-year outputs created; Amerasian/SIV outputs created via `process_siv_amerasian_data.py`.
- [ ] Update database/loader plumbing if using PostgreSQL tables (e.g., `rpc.refugee_arrivals`).
  - Document schema changes and refresh steps.
- [ ] Build LPR multi-year panel and normalize identifiers.
  - Review `sdc_2024_replication/data_immigration_policy/scripts/process_dhs_lpr_data.py`.
  - State totals extracted for FY2000–FY2006 and FY2014–FY2023; FY2007–FY2013 gap remains.
  - Add aggregation variants needed for gravity/forecasting modules.
- [x] Build ACS inflow proxy series (document smoothing or error model).
  - Processed from ACS B07007; MOE retained; no smoothing.
- [x] Construct long-run PEP components panel with regime markers (pre/post-2020, decennial boundaries).
  - `state_migration_components_2000_2024_with_regime.csv` includes regime flags and `regime_period`.
- [ ] Reconcile partial FY2021–FY2024 refugee data already in processed files and prevent accidental overwrites.
- [x] Add validation checks for new series (no negative flows; timing alignment sanity checks).
  - Validation log includes year coverage and negative-value checks.
- [x] Document all new processed outputs in `data/DATA_MANIFEST.md` (and any sdc_2024_replication manifests used).

## Methodology and Modeling Updates (Requires Approval if Results Change)
- [ ] Update Travel Ban DiD/event-study to use extended refugee data and post-2021 period.
- [ ] Refit wave duration models with extended arrivals; reassess right-censoring.
- [ ] Add regime-aware modeling for long-run PEP series (state-space or regime dummies).
- [ ] Integrate LPR + refugee + ACS covariates into forecasting (ARIMAX or state-space).
- [ ] Reassess uncertainty envelopes after data fusion (avoid double-counting variance).

## Journal Article and Documentation Updates
- [ ] Draft v0.8.6 response-to-critique document in `sdc_2024_replication/revisions/v0.8.6/`.
- [ ] Update data/methods narrative for new sources and timing alignment:
  - `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/02_data_methods.tex`
- [ ] Update results and figures impacted by new data:
  - `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/03_results.tex`
  - `sdc_2024_replication/scripts/statistical_analysis/journal_article/figures/`
- [ ] Update discussion/limitations to reflect resolved constraints and remaining caveats:
  - `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/04_discussion.tex`
- [ ] Add/refresh citations in `sdc_2024_replication/scripts/statistical_analysis/journal_article/references.bib`.
- [ ] Update `sdc_2024_replication/METHODOLOGY_SPEC.md` if methodology changes are adopted.
 - [x] Update `data/raw/immigration/refugee_arrivals/MANIFEST.md` to reflect any FY2021–FY2024 extraction changes.
 - [ ] Update `data/raw/immigration/census_population_estimates/README_census_vintages.md` if new vintages are added.

## Methodological Pitfalls to Evaluate (Document Outcomes)
- [ ] Time-base mismatch: FY vs PEP-year vs calendar year (document mapping logic).
- [ ] Net vs gross flows: ensure consistent interpretation across sources.
- [ ] Vintage revisions: treat Census vintage choice as sensitivity dimension.
- [ ] Measurement error (ACS MOE; LPR reporting lags).
- [ ] Double counting: avoid stacking variance from multiple correlated signals.
 - [ ] Partial series risk: prevent ND-only data from being misinterpreted as national panel data.

## Validation and Reproducibility
- [x] Wrap new analysis scripts in `log_execution` for reproducibility.
  - `download_b07007.py` and `process_b07007.py` wrapped.
- [ ] Add/extend tests for new data ingestion and model components.
- [ ] Run targeted test suite for updated modules.
- [x] Record validation summaries in a concise report (link, do not paste).
  - `data/processed/immigration/analysis/immigration_v0.8.6_validation.md`.

## Progress Log (Fill In)
- 2026-01-04: Tracker created.
- 2026-01-04: Added ACS B07007 moved-from-abroad inflow proxy; extended LPR state totals to FY2000–FY2006; added regime markers to PEP components; organized DHS yearbook downloads into product-specific raw folders and removed duplicate LPR ZIP; refreshed manifests and validation log.
- 2026-01-04: Filled initial scope notes and existing data inventory; added response stub.
- 2026-01-04: Added ACS B07007 moved-from-abroad series (raw + processed), updated manifests, and wrote validation summaries.
- 2026-01-04: Added regime-marked PEP components output and updated config/manifest entries.
- 2026-01-04: Reorganized DHS yearbook extras into `dhs_refugees_asylees/` and `dhs_nonimmigrants/` with new manifests.
- 2026-01-04: Extended LPR state time series to FY2000-FY2006 using LPR2006 yearbook Table 4 and refreshed outputs.
- 2026-01-04: Updated `.gitignore` to allow tracking raw data `MANIFEST.md` files.
- 2026-01-04: Added status update memo with data reattempt outcomes and next actions.
