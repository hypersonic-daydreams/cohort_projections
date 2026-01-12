---
title: "Covariate-Conditioned Near-Term Forecast Anchor (2025–2029): Appendix Spec (v0.8.6)"
date_created: 2026-01-07
status: "active"
related_tracker: "sdc_2024_replication/revisions/v0.8.6/remaining_tasks_v0.8.6.md"
related_response: "sdc_2024_replication/revisions/v0.8.6/response_to_critique.md"
related_adr: "docs/governance/adrs/031-covariate-conditioned-near-term-forecast-anchor.md"
---

# Covariate-Conditioned Near-Term Forecast Anchor (2025–2029): Appendix Spec (v0.8.6)

This document operationalizes the v0.8.6 remaining task:
- “Integrate LPR + refugee + ACS covariates into forecasting (ARIMAX or state-space).”

It is written spec-first (auditable/runbook style) and is intentionally scoped to **appendix-only robustness**, consistent with ADR-031.

## 1. Scope and Interpretation (What this task is / is not)

### In-scope (v0.8.6)
- Implement a **covariate-conditioned near-term forecast anchor** for **2025–2029** and present it in the appendix as a robustness overlay:
  - Compare covariate-conditioned forecasts to the existing univariate baseline.
  - Emphasize interpretability and stability over model complexity.
  - Document time-base alignment rules explicitly.

### Out-of-scope (v0.8.6)
- Replacing the **Moderate** baseline scenario with a covariate-conditioned baseline through 2045.
- Full state-space “data fusion” where \PEP, refugees, LPR, and ACS are treated as multiple noisy measurements of a single latent immigration factor.
- Feeding the covariate-conditioned model into the main scenario engine or Monte Carlo wave simulation (avoid double counting).

## 2. Decision Summary (Why this approach is the rigorous choice)

The appendix anchor is adopted because:
- A covariate-conditioned model is only a coherent “baseline” if future covariate paths are specified and defensible for 20 years.
- With \(T \approx 15\), high-parameter ARIMAX/state-space models are fragile; the paper’s thesis is deep uncertainty, not “best model.”
- A near-term (2025–2029) robustness check is feasible, useful, and reviewer-proof while keeping the manuscript’s spine intact.

Canonical decision record:
- ADR-031: `docs/governance/adrs/031-covariate-conditioned-near-term-forecast-anchor.md`

## 3. Canonical Inputs (Do not improvise paths)

### Forecast target (dependent series)
- `data/processed/immigration/analysis/nd_migration_summary.csv`
  - Columns: `year`, `nd_intl_migration` (and U.S. comparators)
  - Coverage: 2010–2024 (n=15)
  - Interpretation: ND \PEP net international migration component (persons; net)

### Covariate candidates

#### A. Refugee arrivals (administrative flow)
- Long-run FY state×nationality panel:
  - `data/processed/immigration/analysis/refugee_arrivals_by_state_nationality.parquet`
  - Coverage: FY2002–FY2024
- Month-aware FY→\PEP-year mapping (FY2021+ only; used for exact alignment checks):
  - `data/processed/immigration/analysis/refugee_arrivals_by_state_nationality_pep_year.parquet`
  - `data/processed/immigration/analysis/refugee_fy_month_to_pep_year_crosswalk.csv`

#### B. LPR admissions (administrative durable-admissions flow)
- ND and U.S. totals + ND share:
  - `data/processed/immigration/analysis/dhs_lpr_nd_share_time.parquet`
  - Coverage: FY2000–FY2023

#### C. ACS covariate (survey-based flow proxy; optional sensitivity)
- ACS 1-year estimate table B07007:
  - `data/processed/immigration/analysis/acs_moved_from_abroad_by_state.parquet`
  - Coverage: 2010–2023
  - Recommended series: `variable == "B07007_028E"` (“Moved from abroad” × “Foreign born”), using `estimate` as the covariate and `margin_of_error` as documentation (not mechanically modeled in v0.8.6).

## 4. Covariate Construction (Definitions + Alignment Rules)

The appendix anchor must be explicit about time bases. The goal is not “perfect alignment” (often impossible historically), but a **documented rule** with a sensitivity check.

### 4.1. Define \(y_t\)
- \(y_t\) := `nd_intl_migration` for \PEP year \(t\) (2010–2024).

### 4.2. Define refugee covariate \(x^{ref}_t\)

**Primary definition (P0)**:
- \(x^{ref}_t\) := total refugee arrivals to North Dakota in fiscal year \(t\) (FY basis), aggregated across nationalities from `refugee_arrivals_by_state_nationality.parquet`.

**Justification**:
- FY and \PEP-year are not identical, but the appendix anchor is explicitly a robustness check. FY totals provide a durable administrative signal with long coverage.
- Exact month-level FY→\PEP-year alignment is available only for FY2021+; we use it as a validation/sensitivity tool rather than forcing a discontinuous construction across the full 2010–2024 window.

**Sensitivity (S1; alignment)**:
- For 2021–2024, build an alternative \(x^{ref,pep}_t\) using `refugee_arrivals_by_state_nationality_pep_year.parquet` (aggregated to ND total for \PEP-year \(t\)) and assess whether results materially differ from FY-based \(x^{ref}_t\).

### 4.3. Define LPR covariate \(x^{lpr}_t\)

**Primary definition (P0)**:
- \(x^{lpr}_t\) := `nd_lpr_count` in fiscal year \(t\) from `dhs_lpr_nd_share_time.parquet`.

**Optional sensitivity (S2; scale)**:
- Use `nd_share_pct` (ND share of U.S. LPR) instead of the count to reduce sensitivity to national swings.

### 4.4. Define ACS covariate \(x^{acs}_t\) (optional)

**Definition (S3; ACS included)**:
- \(x^{acs}_t\) := ACS moved-from-abroad, foreign-born estimate for North Dakota in calendar year \(t\), using:
  - `state_name == "North Dakota"`
  - `variable == "B07007_028E"`
  - `estimate` as the series value

**Interpretation and guardrail**:
- This is a noisy proxy for international inflow and is included only as a sensitivity due to MOE and small-state sampling error.

## 5. Model Specification (Appendix Anchor)

### 5.1. Primary model (P0): local-level state-space regression

Use a parsimonious “local level + regression” model (dynamic regression / structural time series):

- Observation equation:
  - \(y_t = \mu_t + \beta_{ref}\,x^{ref}_{t-1} + \beta_{lpr}\,x^{lpr}_{t-1} + \varepsilon_t\)
- State (level) equation:
  - \(\mu_t = \mu_{t-1} + \eta_t\)
- Errors:
  - \(\varepsilon_t \sim \mathcal{N}(0, \sigma_\varepsilon^2)\)
  - \(\eta_t \sim \mathcal{N}(0, \sigma_\eta^2)\)

**Notes**:
- The one-year lag structure is intentional: it (i) avoids requiring contemporaneous covariates through 2024 for fitting, and (ii) makes the forecast exercise feasible with covariate publication lags.
- Covariates should be standardized (z-scored) on the training window to improve numerical stability; store the means/SDs in the results JSON for reproducibility.

### 5.2. Sensitivity specs (small, threat-mapped)

- **S1 (Refugee alignment)**: substitute \(x^{ref,pep}\) where available (2021–2024) and evaluate whether coefficients/forecasts are qualitatively stable.
- **S2 (LPR scaling)**: replace \(x^{lpr}\) count with `nd_share_pct`.
- **S3 (ACS included)**: add \(x^{acs}_{t-1}\) as an additional regressor, explicitly labeled “noisy survey proxy.”

Keep the grid intentionally small to avoid “methods festival” optics.

## 6. Forecast Construction (2025–2029) and Output Interpretation

### 6.1. Forecast horizon
- Produce conditional forecasts for \PEP years 2025–2029.

### 6.2. Covariate paths for 2024–2028 (needed due to lags)

Because the model uses \(x_{t-1}\), forecasting \(y_{2025}\) requires \(x_{2024}\), forecasting \(y_{2026}\) requires \(x_{2025}\), etc.

**Default rule (P0; minimal assumptions)**:
- **LPR**: hold `nd_lpr_count` constant at FY2023 for FY2024–FY2028 (LOCF) unless FY2024 becomes available.
- **ACS** (if used): hold at 2023 for 2024–2028 (LOCF).
- **Refugees**: hold at FY2024 for FY2025–FY2028 (LOCF); do not attempt to layer wave simulation here.

This keeps the appendix anchor focused on “does conditioning matter?” rather than “can we forecast each covariate?”

### 6.3. Reporting and language in the appendix

The appendix results should be labeled explicitly as:
- “Covariate-conditioned near-term diagnostic (appendix robustness)” and
- “Not used to parameterize the main scenario engine.”

Interpretation target:
- If forecasts are similar to the baseline: this supports the paper’s thesis that uncertainty is high and covariates do not reliably “solve” near-term forecasting in ND.
- If forecasts differ: report the direction and magnitude, but emphasize that differences arise under conditional assumptions and do not replace the main Moderate baseline.

## 7. Validation (Rolling-Origin Backtesting)

At minimum, report a short-horizon forecast comparison using rolling-origin evaluation.

### Recommended evaluation
- Expanding window, 1-step ahead forecasts over 2017–2024 (or the largest feasible window given covariate coverage).
- Metrics:
  - MASE (consistent with current reporting)
  - Interval coverage (optional; interpret cautiously)
- Compare:
  - Univariate baseline (current ARIMA/random-walk anchor)
  - Appendix covariate-conditioned anchor (P0)

Success criteria:
- Appendix anchor should not catastrophically underperform the baseline (robustness check).
- Report results descriptively; do not oversell significance given small samples.

## 8. Manuscript Integration (Appendix deliverables)

### Artifacts to produce
- One figure:
  - Baseline vs covariate-conditioned point forecasts for 2025–2029 with 80%/95% intervals (overlay).
- One table:
  - Year-by-year forecast summary (baseline vs covariate-conditioned; point, 80% PI, 95% PI).

### Target manuscript locations (planned)
- Appendix LaTeX section:
  - `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/06_appendix.tex`
- Results should be described in a short paragraph emphasizing scope and limitations.

## 9. Implementation Definition of Done (DOD)

This task is “done” when all items below are satisfied and linked in the tracker.

### A. Decision + discoverability
- [ ] ADR accepted: `docs/governance/adrs/031-covariate-conditioned-near-term-forecast-anchor.md`
- [ ] This runbook linked from the tracker item in:
  - `sdc_2024_replication/revisions/v0.8.6/remaining_tasks_v0.8.6.md`

### B. Reproducible artifacts
- [ ] Outputs exist on disk for P0 and at least one sensitivity:
  - Results JSON saved under `sdc_2024_replication/scripts/statistical_analysis/results/`
  - Figure saved under `sdc_2024_replication/scripts/statistical_analysis/figures/`
- [ ] Outputs record:
  - Exact input paths
  - Year coverage and row counts after lagging
  - Spec tags (P0/S1/S2/S3) and key parameters (lags, covariates included)

### C. Manuscript integration
- [ ] Appendix figure/table included or referenced in `06_appendix.tex` with correct labeling.

### D. Quality gates
- [ ] No raw data files modified under `data/raw/`.
- [ ] If code changes are made: `pytest tests/ -q` passes.

## 10. Tracker Update Checklist (What to paste into the tracker when complete)

When closing the tracker item, add links to:
- ADR: `docs/governance/adrs/031-covariate-conditioned-near-term-forecast-anchor.md`
- Runbook: `sdc_2024_replication/revisions/v0.8.6/covariate_forecasting_appendix_spec.md`
- Results JSON: (to be created) `sdc_2024_replication/scripts/statistical_analysis/results/module_app_covariate_anchor.json`
- Figure: (to be created) `sdc_2024_replication/scripts/statistical_analysis/figures/fig_app_covariate_anchor_2025_2029.png`
- Manuscript pointer: `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/06_appendix.tex`
