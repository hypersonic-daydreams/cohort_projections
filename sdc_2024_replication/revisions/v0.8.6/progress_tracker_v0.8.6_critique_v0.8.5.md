---
title: "v0.8.6 Critique Implementation Tracker (ChatGPT 5.2 Pro on v0.8.5)"
date_created: 2026-01-04
source_critique: "sdc_2024_replication/revisions/v0.8.6/critique_chatgpt_5_2_pro_v0.8.5.md"
target_revision: "v0.8.6"
status: "planning"
description: "Progress tracker for implementing the v0.8.5 critique; structured for future AI agents with discrete, auditable tasks."
---

# v0.8.6 Critique Implementation Tracker

## Usage
- Mark tasks as complete with `[x]`.
- Add short notes directly under each task when decisions are made.
- Keep additions concise; link to detailed docs/scripts instead of pasting large content.

## Governance and Scope (Do First)
- [ ] Confirm which critique items are in-scope for v0.8.6 vs deferred; list them in the scope notes below.
- [ ] Flag methodology changes that affect results and confirm human approval before implementation (Tier 3).
- [x] Decide whether a new ADR is required for data-fusion/regime-model changes; draft stub if needed.
  - Drafted: `docs/governance/adrs/024-immigration-data-extension-fusion.md`.
- [ ] Review existing inventories and data manifests to avoid duplication:
  - `REPOSITORY_INVENTORY.md`
  - `data/DATA_MANIFEST.md`
  - `sdc_2024_replication/DATA_UPDATE_PLAN.md`
  - `sdc_2024_replication/data_immigration_policy/MANIFEST.md`
- [ ] Identify existing scripts and loaders before creating new ones (`scripts/`, `sdc_2024_replication/scripts/`).

## Scope Notes (Fill In)
In-scope items for v0.8.6:
- [ ] Extend refugee arrivals beyond FY2020 using existing RPC archives and reconcile partial 2021-2024 ND-only data (requires approval if results change).
- [ ] Replace FY-to-PEP-year approximation with month-aware alignment when monthly data are available (requires approval).
- [ ] Incorporate or explicitly evaluate SIV/Amerasian flows as parallel humanitarian signals (requires approval).
- [ ] Audit existing DHS LPR multi-year data and integrate into modeling where appropriate (requires approval).
- [ ] Leverage long-run Census components (2000-2024 available) for regime-aware modeling and sensitivity checks (requires approval).
- [ ] Evaluate adding ACS moved-from-abroad inflow proxy and document measurement error (requires approval).

Deferred items:
- [ ] 1980s-era Census components integration (pending data availability and approval).
- [ ] Full state-space fusion model (scope/complexity decision pending).

## Existing Data Inventory (Fill In)
| Dataset | Current Coverage | Current Location | Notes |
|---|---|---|---|
| Refugee arrivals (RPC, state×nationality) | FY2002–FY2020 full; FY2021–FY2024 ND-only totals | `data/processed/immigration/analysis/refugee_arrivals_by_state_nationality.parquet` | 2021–2024 entries are ND-only totals from manual PDF/news extraction; not a full panel. |
| Refugee arrivals raw archives | FY2012–FY2024 (xls/xlsx + pdf) | `data/raw/immigration/refugee_arrivals/` | PDFs for FY2021–FY2024 not fully parsed; manifest documents gaps. |
| ND refugee arrivals FY2020–FY2024 (manual) | FY2020–FY2024 ND totals; partial breakdowns | `data/raw/immigration/nd_refugee_arrivals_fy2020_2024.csv` | Mix of refugees/SIV/parole; useful for ND-only checks. |
| ND refugee arrivals by nationality (manual) | FY2021 + FY2023 only | `data/raw/immigration/nd_refugee_arrivals_by_nationality.csv` | FY2022/2024 extraction issues noted. |
| LPR by state (DHS/OIS) | FY2014–FY2023 | `data/processed/immigration/analysis/dhs_lpr_by_state_time.parquet` | Raw XLSX in `data/raw/immigration/dhs_lpr/` (2007–2023). |
| LPR by country/region (DHS/OIS) | FY2014–FY2023 | `data/processed/immigration/analysis/dhs_lpr_by_country_time.parquet` | Country/region labels in `region_country_of_birth`. |
| LPR by state×country | FY2023 only | `data/processed/immigration/analysis/dhs_lpr_by_state_country.parquet` | Single-year cross-section. |
| PEP components (combined vintages) | 2000–2024 | `data/processed/immigration/state_migration_components_2000_2024.csv` | Raw vintages in `data/raw/immigration/census_population_estimates/`. |
| PEP components (analysis subset) | 2010–2024 | `data/processed/immigration/analysis/combined_components_of_change.csv` | 53 rows per year; used in recent analyses. |
| ACS foreign-born stock by origin | 2009–2023 | `data/processed/immigration/analysis/acs_foreign_born_by_state_origin.parquet` | Stock, not flow; MOE included. |
| ACS foreign-born ND share | 2009–2023 | `data/processed/immigration/analysis/acs_foreign_born_nd_share.parquet` | Stock-based share. |
| ACS moved-from-abroad inflow | Not yet present |  | Needs acquisition/processing. |

## Data Acquisition (New or Extended Sources)
- [ ] RPC monthly arrivals FY2021–FY2025 (refugees by state/nationality).
  - Check existing RPC PDFs in `data/raw/immigration/refugee_arrivals/` and `data/raw/immigration/rpc_archives/` before downloading.
  - Decide raw storage location (`data/raw/immigration/`) and document rationale.
  - Record “data as of” download date and source URL in the manifest.
- [ ] RPC Amerasian/SIV arrivals by state/nationality (parallel humanitarian series).
  - Ensure consistent identifiers with refugee series.
- [ ] DHS/OIS multi-year LPR tables (state/year + class or origin).
  - Verify licensing/terms and data availability window.
- [ ] Census PEP components long-run series (1980+ where available).
  - Pull by decade series and archive vintage notes.
- [ ] ACS “moved from abroad” annual series (ND and state panel).
  - Capture MOE/SE metadata if available.

## Task Breakdown by Critique Theme
### Refugee Arrivals Extension (FY2021+)
- [ ] Audit current processed refugee file for 2021–2024 partial coverage; document in manifest.
- [ ] Extract FY2021–FY2024 state×nationality from RPC PDFs or obtain alternative source.
- [ ] Create/extend monthly-to-PEP-year crosswalk and store in processed metadata.
- [ ] Regenerate refugee arrivals parquet and log differences vs prior series.

### SIV/Amerasian Parallel Series
- [ ] Identify usable SIV/Amerasian series (state- or national-level) and document source limits.
- [ ] Decide whether to model as separate covariate or merged humanitarian series (approval required).
- [ ] Update processing outputs and data manifests accordingly.

### LPR Multi-Year Panel
- [ ] Review raw DHS LPR files (2007–2023) and current processed coverage (2014–2023).
- [ ] Expand processing to earlier years if feasible; normalize state identifiers.
- [ ] Produce state×year and origin×year panels for modeling inputs.

### Long-Run PEP Components + Regime Handling
- [ ] Confirm combined components series (2000–2024) aligns with current modeling inputs.
- [ ] Add explicit regime markers (pre/post-2010, pre/post-2020) for variance shifts.
- [ ] Decide whether additional 1980s/1990s series are needed (scope/approval).

### ACS “Moved From Abroad” Proxy
- [ ] Identify ACS table/source with annual moved-from-abroad counts for states.
- [ ] Build processing script and store outputs in `data/processed/immigration/analysis/`.
- [ ] Document MOE handling and any smoothing decisions.

### Model and Output Updates
- [ ] Update Travel Ban DiD/event-study with extended refugee data and refined timing.
- [ ] Refit wave duration metrics; reassess right-censoring and hazard ratios.
- [ ] Integrate new covariates into forecasting models (ARIMAX/state-space) after approval.

## Data Handling Checklist (Apply to Each New Dataset)
- [ ] Verify raw file location under `data/raw/immigration/` and record source URL + download date.
- [ ] Add or update manifest entry describing coverage, caveats, and parsing notes.
- [ ] Ensure processed outputs land in `data/processed/immigration/` with clear naming.
- [ ] Add validation summary (basic counts, min/max year, missingness) in a short log.
- [ ] Confirm downstream scripts reference the updated outputs (no hard-coded paths).

## Data Processing and Integration
- [ ] Build/extend refugee + SIV ingestion pipeline (monthly → PEP-year).
  - Review `sdc_2024_replication/data_immigration_policy/scripts/process_refugee_data.py` before adding new scripts.
  - Create explicit FY-to-PEP-year crosswalk (Jul–Jun mapping).
  - Store derived annual series in `data/processed/immigration/` with provenance notes.
- [ ] Update database/loader plumbing if using PostgreSQL tables (e.g., `rpc.refugee_arrivals`).
  - Document schema changes and refresh steps.
- [ ] Build LPR multi-year panel and normalize identifiers.
  - Review `sdc_2024_replication/data_immigration_policy/scripts/process_dhs_lpr_data.py`.
  - Add aggregation variants needed for gravity/forecasting modules.
- [ ] Build ACS inflow proxy series (document smoothing or error model).
- [ ] Construct long-run PEP components panel with regime markers (pre/post-2020, decennial boundaries).
- [ ] Reconcile partial FY2021–FY2024 refugee data already in processed files and prevent accidental overwrites.
- [ ] Add validation checks for new series (no negative flows; timing alignment sanity checks).
- [ ] Document all new processed outputs in `data/DATA_MANIFEST.md` (and any sdc_2024_replication manifests used).

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
 - [ ] Update `data/raw/immigration/refugee_arrivals/MANIFEST.md` to reflect any FY2021–FY2024 extraction changes.
 - [ ] Update `data/raw/immigration/census_population_estimates/README_census_vintages.md` if new vintages are added.

## Methodological Pitfalls to Evaluate (Document Outcomes)
- [ ] Time-base mismatch: FY vs PEP-year vs calendar year (document mapping logic).
- [ ] Net vs gross flows: ensure consistent interpretation across sources.
- [ ] Vintage revisions: treat Census vintage choice as sensitivity dimension.
- [ ] Measurement error (ACS MOE; LPR reporting lags).
- [ ] Double counting: avoid stacking variance from multiple correlated signals.
 - [ ] Partial series risk: prevent ND-only data from being misinterpreted as national panel data.

## Validation and Reproducibility
- [ ] Wrap new analysis scripts in `log_execution` for reproducibility.
- [ ] Add/extend tests for new data ingestion and model components.
- [ ] Run targeted test suite for updated modules.
- [ ] Record validation summaries in a concise report (link, do not paste).

## Progress Log (Fill In)
- 2026-01-04: Tracker created.
- 2026-01-04: Filled initial scope notes and existing data inventory; added response stub.
