# Article Revision Coordination Plan

## Response to ChatGPT 5.2-Pro Critique (December 29, 2025)

**Source Critique:** `output/ChatGPT_5-2-Pro_article_draft_critique.md`

**Non-Destructive Principle:** All revision work creates NEW scripts in `revision_scripts/` and outputs in `revision_outputs/`. Original `module_*.py` scripts and `results/` outputs remain untouched for auditability.

---

## Human Decisions (Confirmed December 29, 2025)

| Decision | Resolution | Rationale |
|----------|------------|-----------|
| **Paper positioning** | Methods for understanding immigration dynamics to improve forecasting | Forecasting is the goal; understanding dynamics is the means |
| **Synthetic Control** | Agent R06b-SCM will evaluate and decide | Sub-agent assesses value-add before keep/drop decision |
| **ML Section** | Include with backtesting | Assess predictive value empirically |
| **Duration Analysis** | Attempt connection to wave forecasting | Keep and develop forecasting application |
| **Ensemble demonstration** | Operational demonstration | Show working combination, not just conceptual |

---

## Mandatory Documentation Standard

**CRITICAL REQUIREMENT:** Every agent must produce a `METHODOLOGY_NOTES.md` file documenting:

1. **Methods Implemented:** Name, mathematical specification, key assumptions
2. **Implementation Details:** Library/function used, parameter choices, any deviations from standard approach
3. **Data Transformations:** Any preprocessing, filtering, alignment applied
4. **Interpretation Guidance:** How to read the outputs, what the numbers mean
5. **Limitations & Caveats:** Known issues, edge cases, assumptions that may not hold
6. **References:** Key citations for methods used

This documentation serves two purposes:
- **Verification:** Allows later agents and humans to audit the analysis
- **Report Construction:** Provides source material for writing revised paper sections

### Template for METHODOLOGY_NOTES.md

```markdown
# Methodology Notes: Agent [ID]

## Overview
[Brief description of what this agent does]

## Methods Implemented

### [Method Name 1]
**Mathematical Specification:**
[LaTeX-style equations or clear mathematical description]

**Implementation:**
- Library: [e.g., statsmodels, scipy, sklearn]
- Function: [e.g., `sm.tsa.stattools.adfuller()`]
- Parameters: [key parameter choices and why]

**Assumptions:**
- [Assumption 1]
- [Assumption 2]

**Interpretation:**
[How to read the output]

### [Method Name 2]
[...]

## Data Transformations Applied
| Input | Transformation | Output | Rationale |
|-------|---------------|--------|-----------|
| [col] | [transform]   | [col]  | [why]     |

## Key Findings
[Summary of main results]

## Limitations & Caveats
- [Limitation 1]
- [Limitation 2]

## References
- [Citation 1]
- [Citation 2]
```

---

## Directory Structure for Revision Work

```
journal_article/
├── revision_scripts/           # NEW: All revision analysis scripts
│   ├── R01_estimand_mapping/   # Critique #1: Estimand clarity
│   ├── R02_module_integration/ # Critique #2: Tighter narrative logic
│   ├── R03_inference_reframe/  # Critique #3: Small-sample inference
│   ├── R04_backtesting/        # Critique #4: Forecast validation
│   ├── R05_gravity_repair/     # Critique #5: Gravity model fixes
│   ├── R06_causal_robustness/  # Critique #6: DiD/SCM/Bartik repair
│   ├── R07_scenario_arithmetic/# Critique #7: Scenario consistency
│   ├── R08_duration_forecast/  # Critique #8 (revised): Duration → Forecasting
│   └── R10_acs_uncertainty/    # Critique #10: ACS MOE integration
├── revision_outputs/           # NEW: All revision analysis outputs
│   ├── R01_estimand/
│   │   ├── METHODOLOGY_NOTES.md   # REQUIRED
│   │   ├── STATUS.md              # REQUIRED
│   │   └── [analysis outputs]
│   ├── R02_module_integration/
│   ├── R03_inference/
│   ├── R04_backtesting/
│   ├── R05_gravity/
│   ├── R06_causal/
│   ├── R07_scenario/
│   ├── R08_duration/
│   ├── R09_completion/
│   ├── R10_acs/
│   ├── R11_tone/
│   └── R12_temporal/
├── revision_sections/          # NEW: Revised LaTeX sections
│   ├── partial/                # Section fragments for assembly
│   ├── 02_data_methods_v2.tex
│   ├── 03_results_v2.tex
│   ├── 04_discussion_v2.tex
│   ├── 05_conclusion_v2.tex
│   └── 06_appendix_v2.tex
├── revision_figures/           # NEW: Fixed/new figures
└── REVISION_COORDINATION_PLAN.md  # This file
```

---

## Critique Summary and Agent Assignments

### High Priority (Publishability Blockers)

| # | Critique | Agent ID | Complexity | Dependencies |
|---|----------|----------|------------|--------------|
| 1 | Clarify forecasting target/estimand | R01 | Medium | None |
| 3 | Small-sample inference reframing | R03 | Medium | R01 |
| 4 | Forecast backtesting validation | R04 | High | R01, R03 |
| 6 | Causal inference repair (DiD/SCM/Bartik) | R06 | High | None |
| 7 | Scenario arithmetic consistency | R07 | Medium | R04 |
| 9 | Missing references and figures | R09 | Medium | All others |

### Medium Priority (Strengthening)

| # | Critique | Agent ID | Complexity | Dependencies |
|---|----------|----------|------------|--------------|
| 2 | Module narrative integration | R02 | Medium | R01, R04, R08 |
| 5 | Gravity model specification | R05 | Medium | None |
| 8 | Duration analysis → forecasting | R08 | Medium | R01 |
| 10 | ACS measurement error integration | R10 | Medium | R05 |

### Lower Priority (Polish)

| # | Critique | Agent ID | Complexity | Dependencies |
|---|----------|----------|------------|--------------|
| 11 | Tone adjustment | R11 | Low | All others |
| 12 | Calendar/Fiscal year harmonization | R12 | Low | R01 |

---

## Detailed Agent Specifications

### Agent R01: Estimand & Measurement Mapping

**Goal:** Create a definitive mapping between data sources and the forecast target, resolving the ambiguity the reviewer identified.

**Paper Positioning Context:** The paper is about methods for understanding immigration dynamics in order to improve forecasting. The estimand definition must support both understanding (decomposition, drivers) AND forecasting (predictable target).

**Scope:**
- Define PEP net international migration as the primary forecast target
- Map relationships between PEP, RPC, LPR, and ACS foreign-born stocks
- Calculate correlations and coverage ratios for overlapping years
- Create decomposition showing how substreams relate to aggregate
- Produce a clear "Estimand & Measurement" section for the paper

**Input Files:**
- `data/processed/immigration/analysis/nd_migration_summary.csv` (PEP data)
- `data/processed/immigration/analysis/refugee_arrivals_by_state_nationality.parquet` (RPC)
- `data/processed/immigration/analysis/dhs_lpr_by_state_time.parquet` (LPR)
- `data/processed/immigration/analysis/acs_foreign_born_nd_share.parquet` (ACS stocks)

**Output Files:**
- `revision_outputs/R01_estimand/METHODOLOGY_NOTES.md` (REQUIRED)
- `revision_outputs/R01_estimand/STATUS.md` (REQUIRED)
- `revision_outputs/R01_estimand/estimand_mapping.json`
- `revision_outputs/R01_estimand/source_correlations.csv`
- `revision_outputs/R01_estimand/decomposition_table.csv`
- `revision_outputs/R01_estimand/fy_vs_cy_alignment.csv`
- `revision_scripts/R01_estimand_mapping/estimand_analysis.py`
- `revision_sections/partial/estimand_measurement_subsection.tex`

**Key Questions to Answer:**
1. What fraction of PEP net migration is explained by refugee arrivals in overlapping years?
2. How do LPR flows compare to PEP in magnitude and timing?
3. What is the correlation between ACS stock changes and PEP flows?
4. How should FY data be aligned with CY data (lag structure)?

**METHODOLOGY_NOTES.md Must Include:**
- Definition of PEP net international migration (what it includes/excludes)
- Data source alignment methodology (FY to CY conversion approach)
- Correlation calculation approach
- Decomposition methodology

**Success Criteria:**
- Clear statement: "The forecast target is PEP net international migration (CY)"
- Quantified decomposition showing refugee vs. non-refugee components
- Explicit FY→CY alignment strategy documented
- METHODOLOGY_NOTES.md complete

---

### Agent R02: Module Integration & Forecasting Pipeline

**Goal:** Establish a coherent forecasting pipeline narrative connecting all analysis modules to the estimand.

**Paper Positioning Context:** Each module should contribute to either (a) understanding dynamics OR (b) improving forecasts. Modules that do neither should be flagged for removal or repositioning.

**Scope:**
- Review each module's contribution to understanding OR forecasting
- Map module outputs to the estimand defined by R01
- Design a clear pipeline diagram showing module relationships
- Position paper as "dynamics understanding → forecasting improvement"

**Input Files:**
- All `module_*_*.json` results in `results/`
- `revision_outputs/R01_estimand/` (depends on R01)
- `revision_outputs/R04_backtesting/` (depends on R04)
- `revision_outputs/R08_duration/` (depends on R08)

**Output Files:**
- `revision_outputs/R02_module_integration/METHODOLOGY_NOTES.md` (REQUIRED)
- `revision_outputs/R02_module_integration/STATUS.md` (REQUIRED)
- `revision_outputs/R02_module_integration/pipeline_specification.json`
- `revision_outputs/R02_module_integration/module_roles.md`
- `revision_outputs/R02_module_integration/orphan_assessment.md`
- `revision_figures/fig_forecasting_pipeline.pdf`
- `revision_sections/partial/framework_overview.tex`

**Module Classification Task:**
For each of the 9 original modules, classify as:
| Role | Description | Example |
|------|-------------|---------|
| Decomposition | Breaks down aggregate into interpretable parts | Refugee vs. non-refugee split |
| Policy Sensitivity | Estimates response to policy changes | Travel ban DiD |
| Predictor | Provides covariates or forecasts | ARIMA, ML models |
| Validation | Tests model assumptions or performance | Backtesting, residual diagnostics |
| Descriptive | Characterizes patterns without forecasting use | (flag for removal or repositioning) |

**METHODOLOGY_NOTES.md Must Include:**
- Classification criteria for each role
- Mapping of modules to roles with justification
- Pipeline diagram specification (nodes, edges, data flows)
- Decision rationale for any modules marked "descriptive only"

**Success Criteria:**
- Each module clearly labeled with role(s)
- Pipeline diagram showing data flow from inputs → modules → forecast
- Clear statement: "Understanding dynamics improves forecasting through [specific mechanisms]"
- Orphan modules identified with recommendation (keep descriptive, reposition, or remove)

---

### Agent R03: Small-Sample Inference Reframing

**Goal:** Correct the KPSS contradiction, reframe test-centric language, and adopt small-sample-appropriate methods.

**Scope:**
- Audit all p-value interpretations for n=15 appropriateness
- Fix KPSS contradiction (text says "fails to reject" but table shows rejection)
- Reframe claims as "consistent with" rather than "establishes"
- Implement break-robust unit root tests (Zivot-Andrews, Lee-Strazicich)
- Rename "confidence intervals" to "prediction intervals" where appropriate

**Input Files:**
- `results/module_2_1_1_unit_root_tests.json`
- `results/module_2_1_2_structural_breaks.json`
- `sections/03_results.tex` (current Results section)

**Output Files:**
- `revision_outputs/R03_inference/METHODOLOGY_NOTES.md` (REQUIRED)
- `revision_outputs/R03_inference/STATUS.md` (REQUIRED)
- `revision_scripts/R03_inference_reframe/break_robust_unit_roots.py`
- `revision_outputs/R03_inference/break_robust_tests.json`
- `revision_outputs/R03_inference/kpss_resolution.md`
- `revision_outputs/R03_inference/inference_audit.md`
- `revision_outputs/R03_inference/terminology_changes.csv`

**Specific Fixes:**
1. KPSS narrative: Examine original results, determine true result, document resolution
2. Unit root with break: Run Zivot-Andrews test allowing for structural break at 2020
3. Language audit: Systematic list of all overconfident claims with replacements
4. Interval terminology: Map all "confidence interval" → "prediction interval" for forecasts

**METHODOLOGY_NOTES.md Must Include:**
- Zivot-Andrews test specification (null hypothesis, break location, critical values)
- Lee-Strazicich test if implemented
- KPSS test interpretation (what rejection means vs. non-rejection)
- Small-sample inference guidelines followed

**Success Criteria:**
- KPSS contradiction resolved with documented explanation
- Break-robust unit root test results available
- Complete terminology change list
- All claims appropriately hedged for n=15

---

### Agent R04: Forecast Backtesting and Validation

**Goal:** Implement proper rolling-origin backtesting to validate forecasting claims and demonstrate operational ensemble.

**Scope:**
- Implement rolling-origin evaluation (train 2010-t, predict t+1)
- Compare against three benchmarks: naive (last value), mean, trend
- Calculate point metrics (MAE, RMSE) and interval calibration (coverage)
- **Demonstrate operational ensemble combination** (per human decision)
- Include ML models in backtesting if data permits

**Input Files:**
- `data/processed/immigration/analysis/nd_migration_summary.csv`
- `results/module_2_1_arima_model.json`
- `results/module_6_machine_learning.json`
- `results/module_9_scenario_modeling.json`

**Output Files:**
- `revision_outputs/R04_backtesting/METHODOLOGY_NOTES.md` (REQUIRED)
- `revision_outputs/R04_backtesting/STATUS.md` (REQUIRED)
- `revision_scripts/R04_backtesting/rolling_origin_evaluation.py`
- `revision_scripts/R04_backtesting/ensemble_combination.py`
- `revision_outputs/R04_backtesting/backtest_results.json`
- `revision_outputs/R04_backtesting/point_accuracy_metrics.csv`
- `revision_outputs/R04_backtesting/interval_calibration.csv`
- `revision_outputs/R04_backtesting/benchmark_comparison.csv`
- `revision_outputs/R04_backtesting/ensemble_weights.json`
- `revision_outputs/R04_backtesting/ml_backtest_results.json` (if ML included)
- `revision_figures/fig_backtest_accuracy.pdf`
- `revision_figures/fig_interval_calibration.pdf`
- `revision_figures/fig_ensemble_performance.pdf`
- `revision_sections/partial/forecasting_validation_subsection.tex`

**Methodology:**

1. **Rolling Origin Evaluation:**
   For t in [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]:
   - Fit model(s) on 2010 to t-1
   - Forecast t
   - Compare to actual
   - Record point forecast and prediction intervals

2. **Benchmarks:**
   - Naive: ŷ_t = y_{t-1}
   - Mean: ŷ_t = mean(y_{2010:t-1})
   - Drift: ŷ_t = y_{t-1} + (y_{t-1} - y_{2010}) / (t-1-2010)

3. **Metrics:**
   - Point: MAE, RMSE, MAPE
   - Interval: 80% and 95% empirical coverage
   - Relative: RMSE ratio vs. naive (skill score)

4. **Ensemble Combination (Operational):**
   - Simple average of available forecasts
   - Inverse-MSE weighted average
   - Compare ensemble vs. best single model

**METHODOLOGY_NOTES.md Must Include:**
- Rolling origin procedure (exact training/test splits)
- Metric definitions (MAE, RMSE, coverage calculation)
- Ensemble weighting formula
- Handling of 2020 COVID outlier (include/exclude sensitivity)

**Success Criteria:**
- 9 backtest observations (2016-2024)
- Clear benchmark comparison table
- Interval calibration plot showing empirical vs. nominal coverage
- Operational ensemble demonstrated with weights and performance

---

### Agent R05: Gravity Model Specification Repair

**Goal:** Fix the gravity model's distance handling, causal language, and standard error concerns.

**Scope:**
- Clarify distance treatment in full specification
- Reframe "causal network effect" as "predictive association"
- Investigate standard error computation (clustering, overdispersion)
- Document cross-section limitation (FY2023 only)

**Input Files:**
- `results/module_5_gravity_model.json`
- `results/module_5_gravity_network.json`
- `data/processed/immigration/analysis/dhs_lpr_by_state_country.parquet`
- `data/processed/immigration/analysis/dhs_lpr_by_state_time.parquet`

**Output Files:**
- `revision_outputs/R05_gravity/METHODOLOGY_NOTES.md` (REQUIRED)
- `revision_outputs/R05_gravity/STATUS.md` (REQUIRED)
- `revision_scripts/R05_gravity_repair/gravity_with_distance.py`
- `revision_scripts/R05_gravity_repair/gravity_robust_se.py`
- `revision_outputs/R05_gravity/gravity_full_spec.json`
- `revision_outputs/R05_gravity/robust_se_comparison.csv`
- `revision_outputs/R05_gravity/distance_analysis.md`
- `revision_outputs/R05_gravity/language_corrections.md`

**Specific Fixes:**
1. **Distance:** Either include log(distance) in specification OR provide explicit justification for exclusion (e.g., destination-state fixed effects absorb it)
2. **Language:** Replace all "causal" with "associational" in gravity section
3. **Standard errors:**
   - Compute vanilla, robust, clustered-by-origin, clustered-by-destination
   - Report comparison table
   - Note measurement error in diaspora stock (ACS)
4. **Panel limitation:** Explain why cross-section only (FY2023 detail limitation)

**METHODOLOGY_NOTES.md Must Include:**
- Full PPML gravity specification with all terms defined
- Distance variable source and calculation (if included)
- Clustering rationale (why cluster by origin vs. destination)
- PPML vs. OLS comparison and interpretation differences

**Success Criteria:**
- Full specification includes or explicitly discusses distance
- No unjustified causal language
- SE comparison table showing vanilla vs. clustered vs. robust
- Clear limitation statement about cross-sectional identification

---

### Agent R06: Causal Inference Robustness

**Goal:** Strengthen or appropriately limit DiD, SCM, and Bartik claims.

**Scope:**
This is the most complex agent, handling three distinct causal methods. The SCM decision is delegated to a sub-evaluation within this agent.

#### 6a: DiD on Travel Ban
- Re-estimate with clustering at nationality level
- Consider PPML instead of log(y+1)
- Generate missing event study figure
- Add sensitivity checks (alternative post periods, placebo)

#### 6b: Synthetic Control Evaluation (Decision Agent)
- **EVALUATE BEFORE DECIDING:** Assess what value SCM adds
- If valuable: Reframe as exposure-weighted design (differential exposure to ban)
- If not valuable: Recommend dropping with documented rationale
- Implement generalized SCM or interactive fixed effects if keeping

#### 6c: Bartik Shift-Share
- Document base period and share construction completely
- Implement appropriate shift-share inference (Borusyak et al. 2022 or Goldsmith-Pinkham et al. 2020)
- Clarify units in Table 7

**Input Files:**
- `results/module_7_causal_inference.json`
- `results/module_7_did_estimates.json`
- `results/module_7_synthetic_control.json`
- `results/module_7_event_study.parquet`
- `data/processed/immigration/analysis/refugee_arrivals_by_state_nationality.parquet`
- `data/processed/immigration/analysis/combined_components_of_change.csv`

**Output Files:**
- `revision_outputs/R06_causal/METHODOLOGY_NOTES.md` (REQUIRED)
- `revision_outputs/R06_causal/STATUS.md` (REQUIRED)
- `revision_scripts/R06_causal_robustness/did_clustered.py`
- `revision_scripts/R06_causal_robustness/did_ppml.py`
- `revision_scripts/R06_causal_robustness/event_study_figure.py`
- `revision_scripts/R06_causal_robustness/scm_evaluation.py`
- `revision_scripts/R06_causal_robustness/bartik_proper_inference.py`
- `revision_outputs/R06_causal/did_robust.json`
- `revision_outputs/R06_causal/did_sensitivity.json`
- `revision_outputs/R06_causal/event_study_data.csv`
- `revision_outputs/R06_causal/scm_evaluation_decision.md` (THE DECISION DOCUMENT)
- `revision_outputs/R06_causal/scm_revised.json` (if kept)
- `revision_outputs/R06_causal/bartik_revised.json`
- `revision_outputs/R06_causal/bartik_specification.md`
- `revision_figures/fig_event_study.pdf` (THE MISSING FIGURE)
- `revision_figures/fig_did_sensitivity.pdf`
- `revision_sections/partial/causal_inference_revised.tex`

**SCM Evaluation Criteria (for Decision):**
The SCM should be KEPT if:
1. It provides insight not available from DiD (e.g., counterfactual trajectory visualization)
2. The exposure-weighted reframing is defensible (ND high exposure vs. donors low exposure)
3. Pre-treatment fit is acceptable (RMSPE < threshold)
4. Placebo tests show ND effect is unusual relative to donor distribution

The SCM should be DROPPED if:
1. National policy shock makes all states "treated" with no clean counterfactual
2. Pre-treatment fit is poor
3. Exposure reframing is not credible for this application
4. Results contradict DiD without clear explanation

**METHODOLOGY_NOTES.md Must Include:**
- DiD specification: outcome, treatment indicator, fixed effects, standard error clustering
- Event study specification: leads/lags, reference period, inference
- SCM evaluation criteria and decision rationale
- Bartik specification: base period shares, national shocks, instrument construction
- Shift-share inference approach (which method and why)

**Success Criteria:**
- Event study figure exists and shows pre-trend assessment
- DiD SEs are clustered appropriately with comparison table
- SCM decision is documented with clear rationale
- Bartik specification fully documented with proper inference

---

### Agent R07: Scenario Arithmetic Consistency

**Goal:** Fix the arithmetic inconsistencies in Table 9 scenario projections.

**Scope:**
- Verify 8% annual growth compounds correctly
- Resolve 2010-2019 slope (+72/year) vs 2024 baseline contradiction
- Reconcile CV discrepancy (82.5% vs 0.39)
- Rename "credible intervals" if using frequentist bootstrap
- Create appendix with exact equations

**Input Files:**
- `results/module_9_scenario_modeling.json`
- `results/module_9_scenario_projections.parquet`
- `results/module_9_monte_carlo.parquet`
- `results/module_1_1_summary_statistics.json` (for CV)

**Output Files:**
- `revision_outputs/R07_scenario/METHODOLOGY_NOTES.md` (REQUIRED)
- `revision_outputs/R07_scenario/STATUS.md` (REQUIRED)
- `revision_scripts/R07_scenario_arithmetic/scenario_verification.py`
- `revision_outputs/R07_scenario/scenario_equations.md`
- `revision_outputs/R07_scenario/verified_projections.csv`
- `revision_outputs/R07_scenario/arithmetic_audit.md`
- `revision_outputs/R07_scenario/cv_reconciliation.md`
- `revision_outputs/R07_scenario/monte_carlo_specification.md`
- `revision_sections/partial/scenario_appendix.tex`

**Specific Fixes:**
1. **8% growth:** Verify 5,126 × 1.08^21 = ? Compare to reported 2045 value
2. **2010-2019 slope:** Clarify: Is anchor point 2019 level OR 2024 level? Document clearly
3. **CV discrepancy:** Explain difference between historical CV (82.5%) and Monte Carlo CV (0.39)
   - Possible explanation: Monte Carlo uses forecast-period stabilized CV, not historical
4. **Terminology:** "credible intervals" → "prediction intervals" (unless Bayesian posterior)

**METHODOLOGY_NOTES.md Must Include:**
- Exact equation for each scenario projection
- Monte Carlo procedure: distribution assumed, parameters, number of draws
- CV calculation: sample vs. population, period used
- Interval terminology: frequentist prediction interval vs. Bayesian credible interval

**Success Criteria:**
- All scenario numbers reproducible from documented equations
- CV discrepancy explained with clear rationale
- Appendix with complete Monte Carlo specification
- Correct interval terminology

---

### Agent R08: Duration Analysis → Forecasting Connection

**Goal:** Connect duration analysis to operational wave forecasting (per human decision to attempt this).

**Scope:**
- Review existing duration analysis results
- Develop connection between wave duration estimates and forecast scenarios
- Create "expected remaining duration" forecast based on wave lifecycle
- Integrate with ensemble framework from R04

**Input Files:**
- `results/module_8_duration_analysis.json`
- `results/module_8_hazard_model.json`
- `results/module_8_wave_durations.json`
- `revision_outputs/R01_estimand/` (for target alignment)
- `revision_outputs/R04_backtesting/` (for ensemble integration)

**Output Files:**
- `revision_outputs/R08_duration/METHODOLOGY_NOTES.md` (REQUIRED)
- `revision_outputs/R08_duration/STATUS.md` (REQUIRED)
- `revision_scripts/R08_duration_forecast/wave_forecast.py`
- `revision_outputs/R08_duration/wave_forecast_scenarios.json`
- `revision_outputs/R08_duration/duration_to_forecast_mapping.md`
- `revision_outputs/R08_duration/hazard_interpretation.md`
- `revision_figures/fig_wave_lifecycle.pdf`
- `revision_sections/partial/duration_forecasting_subsection.tex`

**Forecasting Connection Approach:**
1. **Wave State Classification:** Where are we in current wave? (early, peak, declining)
2. **Expected Duration:** Given historical wave durations, expected time to next transition
3. **Scenario Conditioning:** Different forecast paths conditioned on wave state
4. **Hazard-Based Forecast:** Use hazard model to estimate probability of wave transition in each future period

**METHODOLOGY_NOTES.md Must Include:**
- Cox proportional hazards model specification
- Wave definition criteria (how waves are identified)
- Forecast construction from hazard estimates
- Limitations of duration-based forecasting

**Success Criteria:**
- Clear mapping from duration analysis to forecast scenarios
- Wave state classification for current period
- Duration-based forecast with uncertainty
- Integration point with ensemble identified

---

### Agent R09: References and Figures Completion

**Goal:** Eliminate all placeholder citations and missing figures.

**Scope:**
- Find and fix all "(??)" and "(?)" placeholders in LaTeX
- Generate all missing figures (ACF/PACF, event study from R06)
- Verify figure/table cross-references
- Complete references.bib with any missing entries

**Input Files:**
- `sections/*.tex` (all current sections)
- `references.bib`
- All `revision_figures/` from other agents
- All `revision_outputs/*/` for data needed for figures

**Output Files:**
- `revision_outputs/R09_completion/METHODOLOGY_NOTES.md` (REQUIRED - for figure generation methods)
- `revision_outputs/R09_completion/STATUS.md` (REQUIRED)
- `revision_outputs/R09_completion/placeholder_audit.md`
- `revision_outputs/R09_completion/missing_figures_list.md`
- `revision_outputs/R09_completion/reference_additions.md`
- `revision_scripts/R09_completion/generate_acf_pacf.py`
- `revision_figures/fig_acf_pacf.pdf`
- `references_additions.bib` (new entries to merge)

**METHODOLOGY_NOTES.md Must Include:**
- ACF/PACF calculation parameters (lags, confidence bands)
- Figure generation settings (size, font, color scheme)
- Reference search methodology

**Success Criteria:**
- Zero "(??)" or "(?)" placeholders
- All referenced figures exist
- All referenced tables exist
- References section complete

---

### Agent R10: ACS Measurement Error Integration

**Goal:** Properly acknowledge and address ACS uncertainty in derived quantities.

**Scope:**
- Report MOE for key ACS-derived quantities (LQs, diaspora stocks)
- Consider measurement-error attenuation in gravity models
- Aggregate origins where MOE is high
- Document which ACS file (1-year vs 5-year) is used

**Input Files:**
- `data/processed/immigration/analysis/acs_foreign_born_by_state_origin.parquet`
- `data/processed/immigration/analysis/acs_foreign_born_nd_share.parquet`
- `results/module_1_2_location_quotients.json`
- `results/module_5_gravity_model.json`
- `revision_outputs/R05_gravity/` (for gravity model context)

**Output Files:**
- `revision_outputs/R10_acs/METHODOLOGY_NOTES.md` (REQUIRED)
- `revision_outputs/R10_acs/STATUS.md` (REQUIRED)
- `revision_scripts/R10_acs_uncertainty/moe_analysis.py`
- `revision_outputs/R10_acs/key_estimates_with_moe.csv`
- `revision_outputs/R10_acs/high_moe_origins.csv`
- `revision_outputs/R10_acs/aggregation_recommendations.md`
- `revision_outputs/R10_acs/attenuation_assessment.md`
- `revision_outputs/R10_acs/acs_file_documentation.md`

**METHODOLOGY_NOTES.md Must Include:**
- ACS MOE interpretation (90% confidence interval)
- MOE propagation for derived quantities (ratios, LQs)
- Aggregation rules for high-MOE origins
- Measurement error attenuation bias in regression

**Success Criteria:**
- Key LQ and diaspora estimates reported with MOE
- High-MOE origins identified with aggregation recommendations
- Gravity model results include measurement error caveat
- ACS file source (1-year vs 5-year) documented

---

### Agent R11: Tone Adjustment

**Goal:** Soften overconfident language while maintaining clarity and assertiveness.

**Scope:**
- Audit for "honestly characterizing," "this demonstrates," "proves," "establishes"
- Replace with "consistent with," "suggests," "within this short series"
- Maintain assertive but appropriately hedged style
- Ensure claims align with identification strength

**Input Files:**
- All `sections/*.tex`
- `revision_outputs/R03_inference/terminology_changes.csv` (from R03)
- All other agents' output sections

**Output Files:**
- `revision_outputs/R11_tone/METHODOLOGY_NOTES.md` (REQUIRED - document tone guidelines)
- `revision_outputs/R11_tone/STATUS.md` (REQUIRED)
- `revision_outputs/R11_tone/language_audit.md`
- `revision_outputs/R11_tone/replacements.csv`
- `revision_outputs/R11_tone/tone_guidelines.md`

**METHODOLOGY_NOTES.md Must Include:**
- Tone guidelines for different claim types
- Mapping of identification strength to appropriate language
- Examples of before/after replacements

**Success Criteria:**
- No remaining overconfident declarations
- Claims aligned with identification strength
- Consistent tone throughout paper

---

### Agent R12: Calendar/Fiscal Year Harmonization

**Goal:** Establish consistent CY/FY handling throughout the paper.

**Scope:**
- Document all data sources with their temporal basis
- Create explicit harmonization strategy
- Ensure consistent labeling ("FY arrivals" vs "CY net migration")
- Check for any temporal misalignment in analysis

**Input Files:**
- Data dictionary from `SUBAGENT_COORDINATION.md`
- `revision_outputs/R01_estimand/fy_vs_cy_alignment.csv` (from R01)
- All `sections/*.tex`

**Output Files:**
- `revision_outputs/R12_temporal/METHODOLOGY_NOTES.md` (REQUIRED)
- `revision_outputs/R12_temporal/STATUS.md` (REQUIRED)
- `revision_outputs/R12_temporal/temporal_alignment_table.md`
- `revision_outputs/R12_temporal/harmonization_strategy.md`
- `revision_outputs/R12_temporal/terminology_consistency_audit.md`
- `revision_sections/partial/temporal_alignment_note.tex`

**METHODOLOGY_NOTES.md Must Include:**
- Each data source's temporal basis (CY, FY, multi-year)
- Alignment approach for cross-source analysis
- Terminology conventions adopted

**Success Criteria:**
- Every data source labeled with temporal basis
- Explicit CY/FY conversion documented
- Consistent terminology throughout paper

---

## Execution Phases

### Phase 1: Foundation (Parallel)
**Agents:** R01, R05, R06, R10

- R01 establishes the estimand (foundational for all)
- R05 repairs gravity model (independent)
- R06 handles causal inference including SCM decision (independent)
- R10 analyzes ACS uncertainty (independent, informs R05)

**Deliverables:** Estimand definition, gravity repairs, causal robustness, ACS characterization

### Phase 2: Core Analysis (Parallel after Phase 1)
**Agents:** R03, R04, R08

- R03 reframes inference (needs R01 for context)
- R04 backtesting (needs R01 for target definition)
- R08 duration→forecasting (needs R01 for target alignment)

**Deliverables:** Inference audit, backtest results, duration-forecast connection

### Phase 3: Consistency & Integration (Parallel after Phase 2)
**Agents:** R02, R07

- R02 integrates modules into pipeline (needs R01, R04, R08)
- R07 fixes scenario arithmetic (needs R04 for backtest context)

**Deliverables:** Integrated pipeline, verified scenarios

### Phase 4: Completion (Sequential after Phase 3)
**Agents:** R09, R11, R12

- R09 fills references and figures (needs all analysis complete)
- R11 adjusts tone (needs section drafts)
- R12 harmonizes temporal (needs R01, R09)

**Deliverables:** Complete references/figures, consistent tone, temporal harmony

### Phase 5: Integration
- Compile all `revision_sections/` into new draft
- Merge all `METHODOLOGY_NOTES.md` into technical appendix
- Generate `article_draft_v2.pdf`
- Create diff document comparing v1 to v2

---

## Agent Communication Protocol

Each agent MUST:

1. **Read this coordination plan** before starting
2. **Check `revision_outputs/`** for outputs from dependency agents
3. **Write all outputs** to designated directories
4. **Create METHODOLOGY_NOTES.md** documenting all methods (MANDATORY)
5. **Create STATUS.md** in their output directory with:
   - Completion status (Complete/Partial/Blocked)
   - Key findings/decisions made
   - Any blockers or cross-agent dependencies discovered
   - Files created
6. **Never modify files** in `results/` or original `sections/`

### STATUS.md Template

```markdown
# Agent [ID] Status

## Completion Status
[Complete | Partial | Blocked]

## Key Findings
- [Finding 1]
- [Finding 2]

## Decisions Made
- [Decision 1 with rationale]
- [Decision 2 with rationale]

## Files Created
- [file1.py]
- [file2.json]
- [METHODOLOGY_NOTES.md]

## Blockers/Dependencies
- [None | Description of blocker]

## Notes for Downstream Agents
- [Any information other agents should know]
```

---

## Quality Criteria for Revision

### Minimum Bar for Each Agent
- [ ] All output files created per specification
- [ ] METHODOLOGY_NOTES.md complete and detailed
- [ ] STATUS.md documents completion
- [ ] No placeholder text in outputs
- [ ] JSON/CSV outputs are valid and parseable
- [ ] LaTeX compiles without errors (if applicable)

### Overall Revision Success
- [ ] All 11 critiques addressed
- [ ] No internal inconsistencies
- [ ] Event study figure exists
- [ ] Backtesting results demonstrate forecast skill (or honestly document lack thereof)
- [ ] Scenario arithmetic verified and reproducible
- [ ] Causal claims appropriately limited or strengthened
- [ ] SCM decision documented with clear rationale
- [ ] Duration analysis connected to forecasting
- [ ] Operational ensemble demonstrated
- [ ] All METHODOLOGY_NOTES.md files complete
- [ ] Paper compiles to PDF without warnings

---

## Appendix A: File Inventory

### Original Analysis (DO NOT MODIFY)
```
sdc_2024_replication/scripts/statistical_analysis/
├── module_1_1_descriptive_statistics.py
├── module_1_2_geographic_concentration.py
├── module_2_1_1_unit_root_tests.py
├── module_2_1_2_structural_breaks.py
├── module_2_1_arima.py
├── module_2_2_var_cointegration.py
├── module_3_1_panel_data.py
├── module_3_2_network_effects.py
├── module_4_regression_extensions.py
├── module_5_gravity_network.py
├── module_6_machine_learning.py
├── module_7_causal_inference.py
├── module_8_duration_analysis.py
├── module_9_scenario_modeling.py
└── results/
    └── [all .json and .parquet files]
```

### Original Article (DO NOT MODIFY)
```
journal_article/
├── main.tex
├── preamble.tex
├── sections/
│   ├── 01_introduction.tex
│   ├── 02_data_methods.tex
│   ├── 03_results.tex
│   ├── 04_discussion.tex
│   ├── 05_conclusion.tex
│   └── 06_appendix.tex
├── figures/
├── references.bib
└── output/
    └── article_draft.pdf
```

---

## Appendix B: Methodology Documentation Index

After all agents complete, the following METHODOLOGY_NOTES.md files will exist:

| Agent | Location | Key Methods Documented |
|-------|----------|----------------------|
| R01 | `revision_outputs/R01_estimand/` | Estimand definition, source alignment |
| R02 | `revision_outputs/R02_module_integration/` | Module classification, pipeline design |
| R03 | `revision_outputs/R03_inference/` | Break-robust tests, inference guidelines |
| R04 | `revision_outputs/R04_backtesting/` | Rolling origin, ensemble combination |
| R05 | `revision_outputs/R05_gravity/` | PPML gravity, SE clustering |
| R06 | `revision_outputs/R06_causal/` | DiD, SCM evaluation, Bartik |
| R07 | `revision_outputs/R07_scenario/` | Scenario equations, Monte Carlo |
| R08 | `revision_outputs/R08_duration/` | Hazard models, wave forecasting |
| R09 | `revision_outputs/R09_completion/` | Figure generation |
| R10 | `revision_outputs/R10_acs/` | MOE propagation, aggregation |
| R11 | `revision_outputs/R11_tone/` | Tone guidelines |
| R12 | `revision_outputs/R12_temporal/` | Temporal harmonization |

These will be compiled into a technical appendix for the revised paper.

---

*Last Updated: December 29, 2025*
*Status: APPROVED - Ready for Agent Execution*
*Human Decisions: Confirmed*
