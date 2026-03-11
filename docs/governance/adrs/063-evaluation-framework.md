# ADR-063: Evaluation Framework Architecture

## Status
Accepted

## Date
2026-03-11

## Context

The cohort projection system produces population projections for North Dakota and its 53 counties across multiple scenarios (baseline, high growth, restricted) and method variants (m2026, m2026r1). As the number of model variants, parameter configurations, and experiment sweeps grew, the project lacked a systematic, reproducible way to assess projection accuracy across multiple dimensions, compare models against naive benchmarks, test sensitivity to parameter changes, and produce standardized scorecards for model selection.

Evaluation had been handled through ad-hoc scripts and manual inspection of walk-forward validation results (ADR-057). This approach was not reproducible across experiments, used inconsistent metrics depending on who wrote the script, and could not produce composite rankings across the multiple dimensions that matter for an official projection (accuracy, bias, structural realism, stability).

### Requirements
- Evaluate projection accuracy stratified by geography (state, county groups, individual counties), horizon (5, 10, 15, 20 years), age structure, and historical regime (pre-boom, boom, post-boom, recent).
- Compare every model variant against naive benchmarks (carry-forward, linear trend, average growth) to ensure the cohort-component model adds value.
- Test sensitivity to parameter perturbations, base-year changes, and history-window variations.
- Produce a composite scorecard that ranks model variants across multiple dimensions with configurable weights.
- Standardize all evaluation inputs and outputs so results are comparable across experiments.

### Challenges
- County-group stratification (Bakken, college, reservation, rural, urban) is ND-specific and not supported by generic forecast evaluation packages.
- Evaluation must integrate with the existing walk-forward validation infrastructure (ADR-057) without duplicating metric computation.
- The framework must be configuration-driven to support experiment sweeps with varying thresholds and weights.

## Decision

### Decision 1: Five-Module Evaluation Architecture

**Decision**: Implement the evaluation framework as five cohesive modules, each consuming standardized projection outputs and producing standardized diagnostic tables.

**Rationale**:
- Separating concerns (accuracy, realism, sensitivity, benchmarking, reporting) allows each module to evolve independently.
- Standardized input/output contracts enable composition: the reporting module can consume outputs from any combination of the other four.
- Matches the structure defined in the Evaluation Blueprint (`docs/plans/evaluation-blueprint.md`).

The five modules are:

1. **Forecast Accuracy** (`forecast_accuracy.py`): MAE, RMSE, MAPE, WAPE, bias metrics stratified by geography, horizon, age group, and historical regime.
2. **Structural Realism** (`structural_realism.py`): Age distribution divergence (Jensen-Shannon / Kullback-Leibler), cohort continuity checks, accounting identity verification, rate bounds validation.
3. **Sensitivity & Robustness** (`sensitivity.py`): Parameter perturbation analysis, Monte Carlo error propagation, base-year and history-window sensitivity testing.
4. **Benchmark Comparison** (`benchmark_comparison.py`, `benchmark_runners.py`): Pairwise comparison against naive methods (carry-forward, linear trend, average growth), component-swap analysis, horizon blending evaluation.
5. **Reporting & Visualization** (`visualization.py`): Standardized diagnostic plots and report generation.

**Alternatives Considered**:
- **Single monolithic evaluation script**: Rejected because it would mix concerns, making it difficult to run partial evaluations or extend individual dimensions.
- **External evaluation package** (e.g., `sktime`, `darts`): Rejected because these are designed for univariate time series, not multi-geography cohort-component models with ND-specific county-group stratification and regime awareness.

### Decision 2: Canonical Data Structures

**Decision**: Define canonical dataclasses for all evaluation inputs and outputs in a shared `data_structures.py` module.

**Rationale**:
- Enforces consistent structure across all modules (DRY principle).
- Makes the evaluation API self-documenting: consumers know exactly what fields are available.
- Enables type checking and IDE support.

The canonical structures are:
- `RunIdentity`: Identifies a single projection run (method, scenario, parameters, timestamp).
- `ExperimentRegistryEntry`: Metadata for an experiment sweep entry.
- `ProjectionResultRecord`: Standardized projection output (county, year, age, sex, population).
- `ComponentRecord`: Decomposed projection components (births, deaths, migration).
- `DiagnosticRecord`: Per-run diagnostic metrics and flags.
- `ScorecardEntry`: Per-model composite score with axis-level detail.

**Alternatives Considered**:
- **Plain dictionaries**: Rejected because they lack type safety, are error-prone, and don't self-document the expected schema.
- **Pydantic models**: Considered but rejected as over-engineered for internal-only data structures; dataclasses with `__post_init__` validation are sufficient.

### Decision 3: Six-Axis Weighted Scorecard

**Decision**: Implement a `ModelScorecard` class that computes a composite score across six weighted axes for model ranking.

**Rationale**:
- A single aggregate metric (e.g., MAPE alone) can be gamed or misleading. A multi-axis scorecard forces models to perform credibly across accuracy, bias, realism, and stability simultaneously.
- Configurable weights allow the scorecard to reflect stakeholder priorities (e.g., weighting accuracy higher for official projections, or stability higher for long-horizon scenarios).
- Axis weights and thresholds are defined in `config/evaluation_config.yaml`, not hard-coded.

**Alternatives Considered**:
- **Single-metric ranking** (MAPE only): Rejected because it ignores bias, structural realism, and sensitivity, which are critical for official projections.
- **Unweighted average**: Rejected because not all axes are equally important; accuracy and bias should typically dominate.

### Decision 4: Configuration-Driven Thresholds

**Decision**: All evaluation thresholds, axis weights, county-group definitions, and metric parameters are defined in `config/evaluation_config.yaml`.

**Rationale**:
- Avoids hard-coded assumptions that would require code changes to adjust.
- Enables experiment sweeps to use different threshold profiles without forking evaluation code.
- Consistent with the project's configuration management strategy (ADR-005).

### Decision 5: Orchestrator Pattern

**Decision**: Implement an `EvaluationRunner` class that orchestrates the full evaluation pipeline, tying all five modules together with a single entry point.

**Rationale**:
- Provides a simple API for running complete evaluations: load config, run all modules, produce scorecard and reports.
- Individual modules remain independently callable for targeted analysis.
- The orchestrator handles module sequencing, shared state, and output aggregation.

## Consequences

### Positive
1. **Reproducible evaluation**: Every model variant is assessed with identical metrics, stratifications, and thresholds, enabling apples-to-apples comparison.
2. **Clear decision criteria**: The composite scorecard provides an objective basis for model selection, reducing reliance on subjective judgment.
3. **Integration with walk-forward validation**: The forecast accuracy module consumes walk-forward results (ADR-057) directly, avoiding metric duplication.
4. **Configuration-driven**: Thresholds and weights can be adjusted for different evaluation contexts without code changes.
5. **Extensible**: New metrics, stratifications, or evaluation dimensions can be added to individual modules without restructuring the framework.

### Negative
1. **Upfront complexity**: Five modules with canonical data structures and configuration is more infrastructure than ad-hoc scripts, requiring more initial development effort.
2. **Configuration surface area**: The evaluation config file adds another configuration artifact to maintain.
3. **Scorecard weight subjectivity**: While the scorecard is systematic, the choice of axis weights still involves judgment.

### Risks and Mitigations

**Risk**: Scorecard weights are chosen poorly, causing misleading model rankings.
- **Mitigation**: Weights are configurable and documented. Sensitivity analysis of the scorecard itself (varying weights) can reveal whether rankings are robust to weight choices.

**Risk**: Evaluation framework diverges from the metrics used in walk-forward validation.
- **Mitigation**: Shared `metrics.py` and `utils.py` modules provide a single source of truth for metric computation. Both walk-forward and evaluation modules import from these shared modules.

**Risk**: Structural realism checks are too strict, rejecting models that are accurate but violate a minor demographic constraint.
- **Mitigation**: Realism checks produce diagnostic flags, not hard rejections. The scorecard weights realism as one axis among six, so a minor violation does not dominate the composite score.

## Implementation Notes

### Key Functions/Classes
- `EvaluationRunner`: Orchestrator that runs the full evaluation pipeline.
- `ModelScorecard`: Computes composite weighted scores across six axes.
- `RunIdentity`: Identifies a projection run with method, scenario, and parameter metadata.
- `ProjectionResultRecord`: Standardized projection output record.
- `DiagnosticRecord`: Per-run diagnostic metrics and flags.
- `ScorecardEntry`: Per-model axis-level scores and composite ranking.

### Configuration Integration
Evaluation parameters are defined in `config/evaluation_config.yaml`, including:
- Metric selection and stratification dimensions
- County-group definitions (Bakken, college, reservation, rural, urban)
- Scorecard axis weights and pass/fail thresholds
- Benchmark method configurations
- Sensitivity perturbation ranges

### Testing Strategy
- 154 tests across 8 test files in `tests/test_analysis/test_evaluation/`.
- Unit tests for each module: metric computation, structural checks, sensitivity analysis, benchmark comparison, scorecard weighting, visualization output.
- Integration tests via `test_runner.py` verifying end-to-end orchestration.
- All tests use synthetic data to avoid coupling to specific data vintages.

### Module Inventory (13 files, ~4,800 LOC)

| Module | Purpose |
|--------|---------|
| `data_structures.py` | Canonical dataclasses for runs, results, diagnostics, scorecards |
| `schemas.py` | Shared validation schemas |
| `utils.py` | Common utilities (stratification, grouping, filtering) |
| `metrics.py` | Metric computation (MAE, RMSE, MAPE, WAPE, bias) |
| `forecast_accuracy.py` | Stratified forecast accuracy evaluation |
| `structural_realism.py` | Age distribution, cohort continuity, identity checks |
| `sensitivity.py` | Parameter perturbation and robustness analysis |
| `benchmark_comparison.py` | Pairwise model-vs-benchmark comparison |
| `benchmark_runners.py` | Naive benchmark implementations (carry-forward, linear, average) |
| `scorecard.py` | Composite weighted scorecard computation |
| `visualization.py` | Diagnostic plots and report generation |
| `runner.py` | EvaluationRunner orchestrator |
| `__init__.py` | Package exports |

### Documentation
- [ ] Update `docs/methodology.md` if this ADR changes formulas, rates, data sources, or projection logic

## References

1. **Evaluation Blueprint**: `docs/plans/evaluation-blueprint.md` -- full specification of the evaluation framework design.
2. **Hyndman & Koehler (2006)**: "Another look at measures of forecast accuracy" -- foundational reference for MAE, MAPE, MASE metric selection.
3. **Swanson et al. (2011)**: "Subnational population forecasts: do users think they are accurate enough?" -- context for multi-dimensional evaluation of demographic projections.

## Revision History

- **2026-03-11**: Initial version (ADR-063) - Evaluation framework architecture based on Evaluation Blueprint.

## Related ADRs

- [ADR-057](057-rolling-origin-backtests.md): Rolling-Origin Backtests (walk-forward validation integration)
- [ADR-061](061-college-fix-model-revision.md): College Fix Model Revision (evaluation framework applied to experiment sweeps)
- [ADR-056](056-testing-strategy-maturation.md): Testing Strategy Maturation (testing patterns for evaluation tests)
- [ADR-005](005-configuration-management-strategy.md): Configuration Management Strategy (config-driven design)
