---
title: "Long-Run PEP Regime-Aware Modeling: Spec Grid + Definition of Done (v0.8.6)"
date_created: 2026-01-07
status: "active"
related_tracker: "sdc_2024_replication/revisions/v0.8.6/remaining_tasks_v0.8.6.md"
related_response: "sdc_2024_replication/revisions/v0.8.6/response_to_critique.md"
related_adr: "docs/governance/adrs/030-pep-regime-aware-modeling-long-run-series.md"
---

# Long-Run PEP Regime-Aware Modeling: Spec Grid + Definition of Done (v0.8.6)

This document operationalizes the v0.8.6 remaining task:
- “Add regime-aware modeling for long-run PEP series (state-space or regime dummies).”

It is designed to be auditable and reproducible (spec-first), following the same pattern as the wave-duration refit runbook.

## 1. Scope and Interpretation (What this task is / is not)

### In-scope
- Add an explicit, regime-aware statistical treatment of the **long-run (2000–2024) spliced PEP components series** that:
  - Treats decennial-vintage boundaries as measurement regimes.
  - Separates the one-year COVID shock (2020) from post-2020 dynamics.
  - Provides a small set of pre-declared specifications (primary + sensitivities).

### Out-of-scope (v0.8.6)
- Full **state-space fusion** of PEP with LPR/refugee/ACS signals (tracked separately; Tier 3 implications).
- Redefining the paper’s primary estimand away from **PEP net international migration**.

## 2. Background Findings (v0.8.5 → v0.8.6)

This runbook reflects an explicit investigation of what was implemented in v0.8.5 and what v0.8.6 intends:

- **Geography used for the “PEP series” in the paper**: state-level PEP net international migration, with North Dakota as the focal series and an all-states panel for some diagnostics/ITS.
- **Primary time-series window (v0.8.5 and v0.8.6)**: 2010–2024 (n=15), stored as `data/processed/immigration/analysis/nd_migration_summary.csv`.
- **Long-run regime context series (v0.8.6)**: 2000–2024 (n=25) spliced across three PEP vintages and exported with regime markers as `data/processed/immigration/state_migration_components_2000_2024_with_regime.csv`.
- **Vintages**:
  - v0.8.5 already treats the post-2020 period as a distinct measurement regime and documents the splicing logic in narrative terms.
  - v0.8.6 operationalizes that into a canonical long-run export with explicit regime markers.
- **Estimand note (important)**: this is not “residual net migration” constructed from births/deaths; it is the PEP **international migration component** used as the dependent series. Births/deaths are available in the broader panel (`data/processed/immigration/analysis/combined_components_of_change.csv`) but are not used to redefine the dependent variable for this task.
- **Terminology guardrail**: decennial-vintage “measurement regimes” should not be conflated with the separate “policy regime” variable `R_t` from ADR-021.

## 2. Key Questions This Task Must Answer

1. Do slope/variance properties differ across the PEP vintage regimes (2000s vs 2010s vs 2020s)?
2. How sensitive are long-run conclusions to the COVID outlier year (2020)?
3. What is the minimum “regime-aware” structure that is stable under n≈25 and interpretable in the manuscript?

## 3. Canonical Inputs (Do not improvise paths)

### Primary (short) series (n=15)
- `data/processed/immigration/analysis/nd_migration_summary.csv`
  - Coverage: 2010–2024
  - Usage: the paper’s primary ND migration time series for forecasting/scenario modules.

### Long-run (spliced) series (n=25)
- `data/processed/immigration/state_migration_components_2000_2024_with_regime.csv`
  - Coverage: 2000–2024 across Vintage 2009, 2020, 2024
  - Includes: `vintage` and `regime_*` markers.
  - Usage: robustness/diagnostic regime-aware modeling.

## 4. Pre-Declared Spec Grid

This grid is intentionally small: one primary spec plus 2–3 sensitivities that map to identifiable threats (COVID outlier, heteroskedasticity, intervention shape).

### P0 (Primary): Piecewise vintage regimes + COVID pulse + HAC inference
- Data: long-run series (2000–2024), ND only.
- Regimes: 2000s / 2010s / 2020s (vintage-based).
- Model: piecewise slopes with level shifts at 2010 and 2020, plus a **COVID pulse** (2020 only).
- Inference: HAC (Newey–West) with 2 lags.
- Deliverable: one table (slopes by regime) + one figure (series with regime shading + fitted piecewise trends).

### S1 (Sensitivity): Exclude 2020 entirely (no intervention term)
- Same as P0 but drop the 2020 observation (tests whether COVID dominates slope estimates).

### S2 (Sensitivity): WLS by regime variance (downweight high-variance regime)
- Same as P0 but use regime-variance weights to reduce heteroskedastic leverage.
- Interpretation: robustness check for inference stability, not a “better” estimator.

### S3 (Sensitivity): Alternative COVID specification (step or recovery ramp)
- Replace COVID pulse with either a step (2020+) or ramp (years since 2020) intervention.
- Only keep if it adds interpretive clarity in the manuscript; otherwise report briefly and discard.

## 5. Implementation Plan (Concrete steps)

### Step A — Data extraction and validation
1. Extract ND series from `state_migration_components_2000_2024_with_regime.csv` (state identifier = “North Dakota” or FIPS 38 depending on column).
2. Confirm:
   - Year coverage is exactly 2000–2024 (25 observations for ND).
   - Vintage regime counts are 10/10/5 for (2000s/2010s/2020s).
   - 2020 is flagged for intervention logic.

### Step B — Model execution
Use the existing regime-aware utilities in:
- `sdc_2024_replication/scripts/statistical_analysis/module_regime_aware/`

Preferred options for the execution wrapper:
1. **Option 1 (recommended):** add a small runner script dedicated to this task that:
   - Loads the canonical processed CSV.
   - Runs P0/S1/S2/S3.
   - Writes a single JSON bundle + figures to stable locations.
2. **Option 2 (acceptable):** adapt `sdc_2024_replication/scripts/statistical_analysis/module_B1_regime_aware_models.py` to load from canonical inputs (not ADR report artifacts), then run.

### Step C — Artifact generation for the manuscript
- Table: regime-specific slope estimates (+ robust SE), plus an equality-of-slopes test if feasible.
- Figure: ND international migration time series with (a) vintage coloring, (b) regime boundaries, and (c) P0 fitted trends.
- Short narrative: 1–2 paragraphs in the data/methods or robustness section explaining that this is a measurement-regime diagnostic supporting cautious interpretation.

## 6. Implementation Definition of Done (DOD)

This task is “done” when all items below are satisfied and linked in the tracker.

### A. Decision + discoverability
- [ ] ADR accepted or explicitly retained as proposed: `docs/governance/adrs/030-pep-regime-aware-modeling-long-run-series.md`.
- [ ] This runbook is linked from the v0.8.6 tracker: `sdc_2024_replication/revisions/v0.8.6/remaining_tasks_v0.8.6.md`.

### B. Reproducible artifacts
- [ ] Outputs exist on disk for P0 and at least S1/S2 (S3 optional):
  - Results JSON bundle (single file) saved under `sdc_2024_replication/scripts/statistical_analysis/results/`.
  - At least one figure saved under `sdc_2024_replication/scripts/statistical_analysis/figures/`.
- [ ] Outputs record:
  - Input file paths and row counts.
  - Spec tag (P0/S1/S2/S3) and key parameters (lags, intervention type).

### C. Manuscript integration (v0.8.6 draft)
- [ ] `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/02_data_methods.tex` references the regime-aware diagnostic and properly labels “measurement regime” vs “policy regime”.
- [ ] Any new figure/table is either (a) included in the main text or (b) placed in appendix with a clear pointer.

### D. Quality gates
- [ ] No raw data files modified under `data/raw/`.
- [ ] `pytest tests/ -q` passes after any code changes (if code is modified for execution).

## 7. Suggested Commands (Runbook)

Assuming environment activation via `direnv` and dependency sync via `uv`:

1. Run the modeling script:
   - `uv run python sdc_2024_replication/scripts/statistical_analysis/run_pep_regime_modeling.py`
2. Validate tests:
   - `uv run pytest tests/ -q`

## 8. Tracker Update Checklist (What to paste into the tracker when complete)

When closing the tracker item, add links to:
- ADR: `docs/governance/adrs/030-pep-regime-aware-modeling-long-run-series.md`
- Runbook: `sdc_2024_replication/revisions/v0.8.6/pep_regime_modeling_spec_grid.md`
- Results JSON: `sdc_2024_replication/scripts/statistical_analysis/results/module_B1_pep_regime_modeling.json`
- Slopes table: `sdc_2024_replication/scripts/statistical_analysis/results/module_B1_pep_regime_modeling_slopes.csv`
- Figure(s): `sdc_2024_replication/scripts/statistical_analysis/figures/module_B1_pep_regime_modeling__P0.png`
