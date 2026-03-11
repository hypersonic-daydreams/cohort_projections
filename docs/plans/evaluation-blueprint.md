---
title: "Population Projection Evaluation Blueprint"
created: 2026-03-11
status: accepted
author: ChatGPT
purpose: >
  Formal evaluation framework for population projection models covering
  accuracy, bias, structural realism, sensitivity, and benchmarking.
related_docs:
  - config/evaluation_config.yaml
  - cohort_projections/analysis/evaluation/
related_adrs:
  - ADR-057 (Rolling-Origin Backtests)
  - ADR-061 (College Fix Model Revision)
---

# Population Projection Evaluation Blueprint

## Purpose

This blueprint defines a formal evaluation framework for population projection models for North Dakota and its counties. It is designed to support repeated, structured testing of projection methods across multiple horizons, geographies, demographic components, and parameter settings. The goal is to make it easy to identify where a model performs well, where it fails, and why.

This framework is intended for operational use with AI agents and scripted workflows inside a project folder. It is not just a checklist of tests. It is a full evaluation system with standardized inputs, outputs, diagnostics, and reporting.

## Primary objectives

The evaluation system should answer four core questions for every model version or experimental run:

1. **Accuracy**: How close are projections to realized values?
2. **Bias**: Are errors systematically high or low?
3. **Realism**: Do projections behave demographically and structurally as expected?
4. **Stability**: Does the model remain well-behaved under alternative assumptions, windows, shocks, and small perturbations?

A good model for official use should perform credibly across all four dimensions, not just optimize a single aggregate error metric.

## Context and scope

This blueprint assumes:

* Geographic scope includes **North Dakota state and counties**.
* Historical data span approximately **2000 through 2024**, with some preliminary **2025** data available.
* Projection methods may include:

  * a baseline cohort-component model,
  * more advanced cohort-component variants,
  * tuned or hybrid models,
  * benchmark and blended alternatives.
* Evaluation includes both **backtesting** and **walkforward / rolling-origin testing**.

## Design principles

1. **Standardize all runs** so outputs can be compared across methods.
2. **Evaluate by horizon**, not just overall average performance.
3. **Separate total-population performance from age-structure performance**.
4. **Decompose errors by component** whenever possible.
5. **Benchmark everything** against simpler alternatives.
6. **Record every experimental run** with metadata and parameters.
7. **Visualize failure patterns** so issues can be seen quickly.
8. **Use multiple metrics** to avoid being fooled by one scoring rule.

## Evaluation architecture

The evaluation system should be built around five modules:

1. **Forecast Accuracy Module**
2. **Structural Realism Module**
3. **Sensitivity and Robustness Module**
4. **Benchmark and Comparison Module**
5. **Reporting and Diagnostic Visualization Module**

Each module should consume standardized projection outputs and produce standardized diagnostic tables.

## Standard evaluation dimensions

Every run should be evaluated across these dimensions.

### Geography

* State total
* County total
* County-type groups if defined

### Horizon

Recommended forecast horizons:

* 1 year
* 2 years
* 3 years
* 5 years
* 10 years
* 15 years
* 20 years

If data limits require shorter horizons for some backtests, preserve the same horizon labels where possible.

### Target variables

* Total population
* Age-group population
* Sex-specific population if used
* Births
* Deaths
* Net migration
* In-migration and out-migration if separately modeled

### Model / run identity

Each run should include:

* `run_id`
* `model_name`
* `model_family`
* `projection_origin_year`
* `training_window`
* `data_vintage`
* `parameter_set`
* `notes`

## Required data structures

The evaluation system should use a consistent set of tables or files.

### 1. Experiment registry

Stores metadata for each run.

Suggested fields:

* `run_id`
* `timestamp`
* `model_name`
* `model_family`
* `baseline_flag`
* `projection_origin_year`
* `training_start_year`
* `training_end_year`
* `data_vintage`
* `parameter_json`
* `scenario_name`
* `notes`

### 2. Projection results table

Stores projected and realized values in tidy form.

Suggested fields:

* `run_id`
* `geography`
* `geography_type`
* `year`
* `horizon`
* `sex`
* `age_group`
* `target`
* `projected_value`
* `actual_value`
* `base_value`

### 3. Component table

Stores births, deaths, migration, and related components.

Suggested fields:

* `run_id`
* `geography`
* `year`
* `horizon`
* `component`
* `projected_component_value`
* `actual_component_value`

### 4. Diagnostics table

Stores calculated metrics.

Suggested fields:

* `run_id`
* `metric_name`
* `metric_group`
* `geography`
* `geography_group`
* `target`
* `horizon`
* `value`
* `comparison_run_id`
* `notes`

### 5. Visualization-ready summary tables

Create pre-aggregated tables for:

* horizon curves
* county-by-horizon heatmaps
* county bias maps
* component decomposition plots
* parameter sweep plots

## Module 1: Forecast Accuracy

This module measures predictive performance.

### Core metrics

At minimum, compute:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* Mean Absolute Percentage Error (MAPE)
* Median Absolute Percentage Error
* Weighted Absolute Percentage Error (WAPE or similar)
* Mean Signed Error
* Mean Signed Percentage Error

### Required slices

Each metric should be summarized by:

* model / run
* projection origin
* horizon
* geography
* county-type group if available
* target variable
* age group if applicable

### Required tests

1. **Total population accuracy by county and horizon**
2. **Total population accuracy at state level by horizon**
3. **Age-group accuracy by county and horizon**
4. **Age-group accuracy at state level**
5. **Signed bias by county and horizon**
6. **Weighted vs unweighted percentage-error comparisons**

### Rank and direction tests

To assess pattern recognition rather than only magnitude:

* Spearman rank correlation of projected vs actual county growth
* Directional accuracy for classifying growth vs decline
* Top-decile and bottom-decile capture for county growth rates

## Module 2: Structural Realism

This module evaluates whether projections look demographically sensible.

### Component realism

Evaluate births, deaths, and migration separately.

Required tests:

1. **Birth projection error by horizon**
2. **Death projection error by horizon**
3. **Net migration projection error by horizon**
4. **In-migration and out-migration error**, if separately modeled

### Age-structure realism

Required tests:

1. **Age-distribution divergence over time**
2. **Five-year age-band accuracy**
3. **Age-schedule smoothness diagnostics**
4. **Plausibility of fertility, mortality, and migration age profiles**

Recommended divergence measures:

* Jensen-Shannon divergence
* Kullback-Leibler divergence where appropriate
* absolute difference summaries across normalized age distributions

### Cohort continuity

Required tests:

1. **Cohort survival continuity checks**
2. **Expected cohort transition residuals**
3. **Identification of abrupt age-pattern breaks**

### Accounting and coherence checks

Required tests:

1. **Population accounting identity check**

   * next year population = prior population + births - deaths + net migration
2. **Age-sex totals sum correctly to county total**
3. **County totals sum to state total**, if state consistency is enforced
4. **Rates remain within plausible bounds**

### Distributional realism

Evaluate whether the county system behaves plausibly as a whole.

Required tests:

1. **Distribution of county population sizes vs actual**
2. **Distribution of county growth rates vs actual**
3. **Variance and skewness comparison across counties**
4. **Over-dispersion or under-dispersion diagnostics**

## Module 3: Sensitivity and Robustness

This module identifies which assumptions matter and where fragility appears.

### Parameter sensitivity

For every important tuning parameter, run low, medium, and high settings, or a larger grid where needed.

Parameters may include:

* migration smoothing strength
* recency weighting strength
* fertility trend damping
* mortality improvement assumptions
* shrinkage toward state or regional patterns
* mean-reversion strength
* caps and floors on rates
* age-profile smoothing parameters
* county-specific correction multipliers

Required outputs:

* parameter value vs near-term error
* parameter value vs long-term error
* parameter value vs bias
* parameter value vs realism score
* parameter value vs stability score

### Interaction sensitivity

Test selected parameter pairs that are likely to interact.

Priority pairs include:

* recency weighting × migration smoothing
* mean reversion × migration caps
* shrinkage × county-specific adjustment
* fertility trend × age-structure smoothing

Use factorial or fractional-factorial designs where computationally useful.

### Base-year sensitivity

Required tests:

* re-run models using different projection origins
* compare long-run trajectories for nearby base years
* flag cases where one unusual year strongly changes long-horizon results

### History-window sensitivity

Required tests:

* full-history training window
* shorter recent-history windows
* rolling fixed-width windows
* exclusion of early years
* exclusion of recent years where appropriate

### Shock-year sensitivity

Required tests:

* all years included
* pandemic-distorted years excluded
* boom/bust years excluded
* extreme years winsorized or smoothed
* selected anomalous migration years damped or replaced

### Data-vintage sensitivity

Required tests:

* with and without preliminary 2025 data
* with alternate historical estimate vintages, if available
* base-population revision sensitivity

### Small perturbation stability tests

Introduce small changes to key inputs and observe response.

Examples:

* base population ±0.5% and ±1%
* births ±1%
* deaths ±1%
* migration ±5%
* small age-specific migration perturbations

Required outputs:

* sensitivity index by county and horizon
* stability ranking of runs
* list of counties or age groups with disproportionate response

### Monte Carlo uncertainty propagation

Where feasible, simulate uncertainty in:

* starting population
* births
* deaths
* migration
* age-specific schedules

Required outputs:

* forecast interval width by county and horizon
* component contribution to uncertainty
* counties with highest uncertainty amplification

## Module 4: Benchmark and Comparison Framework

This module ensures that all advanced methods are evaluated relative to credible alternatives.

### Required benchmark families

Every candidate method should be compared with:

1. **2024 baseline / official cohort-component method**
2. **Simple carry-forward method**
3. **Moving-average rates method**
4. **Simple mean-reverting cohort-component method**
5. **Trend extrapolation benchmark**, if relevant
6. **State-share or ratio-share benchmark**, if relevant

### Component-swapping tests

Required tests:

* advanced migration + simple fertility + simple mortality
* simple migration + advanced fertility + advanced mortality
* advanced fertility only
* advanced mortality only
* advanced migration only

These runs are essential for identifying the component that drives long-run underperformance.

### Horizon-blended tests

Required tests:

* advanced method in near-term and simple method in long-term
* weighted blend by horizon
* gradual damping of migration assumptions after selected years

### Ensemble tests

Recommended tests:

* simple average ensemble
* weighted ensemble by horizon performance
* county-type-specific ensemble
* county-specific model-selection or ensemble scheme

## Module 5: Backtesting Design

This module defines how historical testing should be conducted.

### Rolling-origin walkforward testing

Required design:

* choose multiple projection origins
* train through origin year
* project forward to all available realized years
* evaluate by horizon and origin

This should be the default framework for model comparison.

### Fixed-origin long-horizon tests

Required design:

* select origins that allow 5-, 10-, 15-, and 20-year evaluation where possible
* compare decay in performance over increasing horizon lengths

### Regime-specific backtesting

Define and test across historical regimes relevant to North Dakota. Potential regimes may include:

* stable periods
* energy boom periods
* energy bust or slowdown periods
* pandemic disruption period
* post-pandemic normalization period

Required outputs:

* performance by regime
* horizon-specific performance within regime
* benchmark comparison within regime

### Leave-one-period-out testing

Recommended design:

* omit one major historical period from training
* test on that omitted period
* rotate across major periods

This helps detect overdependence on particular eras.

## Error decomposition framework

This framework is used when a model misses and the cause needs to be identified.

### Component contribution decomposition

Required approach:

For each county and horizon, estimate contribution of:

* base population error
* births error
* deaths error
* migration error
* interaction / compounding effects

A practical approximation is to replace one projected component at a time with realized values and measure reduction in total error.

### Temporal accumulation decomposition

Required outputs:

* error accumulation curves by year since projection origin
* identification of whether long-run miss emerged early or late
* cumulative component error paths

### Spatial decomposition

Required outputs:

* county-level maps of signed error
* county-level maps of improvement vs baseline
* county-level maps of instability under perturbation

### Cohort decomposition

Required outputs:

* contribution of age groups to later total-population error
* identification of age bands that drive downstream problems in births, deaths, or migration

## Diagnostic visualization requirements

The evaluation system should generate a compact but comprehensive dashboard or report for each major run set.

### Required visuals

1. **Horizon profile plot**

   * error metric vs horizon
   * one line per model
2. **County-by-horizon heatmap**

   * cell color represents error or improvement over benchmark
3. **Bias map by county**

   * signed overprojection or underprojection
4. **Component blame chart**

   * births, deaths, migration, and base-population contributions
5. **Age-distribution divergence plot**

   * divergence over horizon
6. **Parameter-response plot**

   * metric vs parameter value
7. **Stability scatterplot**

   * near-term accuracy vs long-term accuracy, with marker size/color reflecting robustness or realism

### Optional but useful visuals

* county rank-accuracy plots
* county distribution comparison plots
* fan charts from Monte Carlo runs
* scenario comparison plots

## Model scorecard

Each run should be summarized using a multi-axis scorecard.

### Required top-level scores

1. **Near-term accuracy score**
2. **Long-term accuracy score**
3. **Bias / calibration score**
4. **Age-structure realism score**
5. **Robustness / stability score**
6. **Interpretability / defensibility score**

Interpretability may require a partially qualitative assessment, but should still be recorded in a structured way.

### Composite scoring

A composite score may be computed for model ranking, but all sub-scores must remain visible. No run should be selected based only on a single aggregate number.

## Recommended starter battery

The following starter battery should be implemented first if a phased rollout is needed.

1. Total-population MAE by county and horizon
2. Total-population MAPE by county and horizon
3. Weighted APE by county and horizon
4. Signed percentage error by county and horizon
5. State-level accuracy by horizon
6. Age-group accuracy by horizon
7. County rank-order growth accuracy
8. Growth/decline directional accuracy
9. Benchmark comparison vs 2024 method
10. Error heatmap by county × horizon
11. Bias map by county
12. Births projection error
13. Deaths projection error
14. Migration projection error
15. Age-distribution divergence score
16. Cohort continuity residuals
17. County-to-state coherence check
18. Recency weighting sensitivity sweep
19. Smoothing parameter sensitivity sweep
20. Mean-reversion sensitivity sweep
21. Base-year sensitivity test
22. Shock-year exclusion test
23. Component-swapping benchmark runs
24. Small perturbation stability test
25. Horizon-blended hybrid benchmark

## Recommended implementation phases

### Phase 1: Foundation

Build:

* experiment registry
* standardized run outputs
* projection results table
* diagnostics table
* benchmark comparison framework

Deliverables:

* one tidy results pipeline
* basic horizon plots
* county × horizon heatmaps
* benchmark score tables

### Phase 2: Core diagnostics

Add:

* component scoring
* age-structure diagnostics
* cohort continuity checks
* county-type grouping analysis
* bias maps

Deliverables:

* structural realism report
* component decomposition outputs
* county-level failure summaries

### Phase 3: Sensitivity framework

Add:

* parameter sweeps
* base-year sensitivity
* history-window sensitivity
* shock-year tests
* perturbation stability tests

Deliverables:

* parameter response tables
* stability rankings
* root-cause flags for fragile assumptions

### Phase 4: Robustness and uncertainty

Add:

* Monte Carlo uncertainty propagation
* ensemble tests
* horizon blending
* regime-specific backtests

Deliverables:

* uncertainty summaries
* robustness scorecards
* model frontier comparison

## Suggested project-folder structure

```text
project/
  data/
    raw/
    processed/
    vintages/
  configs/
    models/
    scenarios/
    parameter_sweeps/
  runs/
    <run_id>/
      metadata.json
      projections.parquet
      components.parquet
      diagnostics.parquet
      plots/
      summary.md
  benchmarks/
  evaluation/
    metrics/
    diagnostics/
    decomposition/
    sensitivity/
    reporting/
  dashboards/
  notebooks/
  docs/
    evaluation_blueprint.md
```

## AI-agent operating guidance

An AI agent working inside the project folder should follow these rules.

### Required agent workflow

1. Read experiment registry and identify candidate runs.
2. Generate or update missing diagnostics in standardized format.
3. Compare all serious runs against baseline benchmarks.
4. Flag counties, horizons, and components with repeated failure.
5. Summarize likely root causes using structured evidence.
6. Recommend next experiments based on observed weaknesses.

### Agent output expectations

For every major comparison cycle, the agent should produce:

* top-performing runs by scorecard dimension
* clear near-term vs long-term tradeoff summary
* list of counties with recurring issues
* list of parameters most associated with instability
* recommendation for next sensitivity tests
* recommendation on whether hybrid or simpler methods should be retained for certain horizons

## Decision rules for model refinement

When deciding whether to keep, modify, or reject a model version, apply these rules.

### Keep or promote a model when

* it outperforms benchmarks on both weighted and unweighted metrics in relevant horizons,
* it does not show strong systematic bias,
* age structure remains realistic,
* sensitivity tests show acceptable stability,
* performance is not concentrated only in one historical regime.

### Modify a model when

* near-term accuracy is good but long-term accuracy degrades,
* migration assumptions dominate long-run error,
* county results are too volatile under small perturbations,
* the model benefits clearly from stronger smoothing, shrinkage, or mean reversion.

### Reject or constrain a model when

* it underperforms simple benchmarks across multiple horizons,
* errors are unstable under modest perturbations,
* age structure becomes implausible,
* county-to-state coherence breaks down,
* apparent gains depend too heavily on one unusual historical period.

## Recommended immediate priorities for the current project

Given the current observation that the advanced approach performs better in the near term but worse in the long term, the highest-priority analyses are:

1. **Horizon-specific error curves against the 2024 baseline**
2. **County × horizon heatmaps of improvement or deterioration**
3. **Signed bias summaries by horizon**
4. **Migration-focused component decomposition**
5. **Recency weighting and smoothing sweeps**
6. **Mean-reversion sensitivity runs**
7. **Shock-year exclusion or dampening tests**
8. **Component-swapping runs to isolate the source of long-run weakness**
9. **Horizon-blended hybrid model tests**
10. **Age-distribution divergence diagnostics**

## End state

The final evaluation environment should function like a projection-model observatory rather than a single notebook or one-off test script. It should allow fast comparison of many model variants, surface problems automatically, preserve full run history, and generate evidence-based recommendations for refinement.

In practical terms, the end state is a reproducible system where any new projection method can be dropped in, scored, diagnosed, compared to benchmarks, stress-tested, and summarized for decision-making with minimal manual cleanup.

That is the standard needed for a high-trust official-use projection workflow.
