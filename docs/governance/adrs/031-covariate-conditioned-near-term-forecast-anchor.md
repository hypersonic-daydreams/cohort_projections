# ADR-031: Covariate-Conditioned Near-Term Forecast Anchor (Appendix-Only)

## Status
Accepted

## Date
2026-01-07

## Context

The v0.8.6 revision tracker includes a remaining task: **“Integrate LPR + refugee + ACS covariates into forecasting (ARIMAX or state-space).”** This task sits at the intersection of two constraints that are central to the paper’s goals (as clarified in v0.8.5):

1. **Estimand discipline**: the forecast target is North Dakota \PEP net international migration (\(Y_t\)), with administrative series (refugees, LPR) and survey series (ACS) treated as auxiliary signals rather than alternative outcomes.
2. **Deep uncertainty in a short series**: the primary forecast window is 2010–2024 (\(T \approx 15\)), featuring a major structural break/outlier (2020) and post-2020 volatility that limits the credibility of high-parameter forecasting models.

Integrating covariates can strengthen the paper by:
- providing an explicit “predictability stress test” (do policy-linked signals materially improve near-term forecasts?),
- improving scenario narrative discipline (showing what a covariate-conditioned mapping would imply),
- addressing reviewer/referee expectations that covariates described in methods should appear in results (or be explicitly scoped as sensitivity work).

However, **promoting a covariate-conditioned model to replace the Moderate baseline** would require the paper to also specify (and defend) **future paths of the covariates** (refugees, LPR, ACS) through 2045. That turns the “baseline forecast” into a family of conditional scenarios, increases modeling surface area, and risks target drift away from the \PEP estimand.

### Requirements
- Preserve the manuscript’s core narrative: a transparent baseline forecast + explicit scenarios under policy sensitivity.
- Avoid overfitting and fragile inference under \(T \approx 15\).
- Keep the Moderate baseline interpretable and reproducible without needing additional upstream forecasting models for covariates.
- Make time-base alignment explicit (FY vs \PEP-year vs ACS calendar year) and avoid “quiet” mismatches.
- Avoid double counting (e.g., refugee covariates and refugee-wave adjustments both injecting the same signal into uncertainty).

### Challenges
- **Forecast-feasibility**: a long-horizon (2025–2045) covariate-conditioned baseline is only coherent if \(X_{t+h}\) is specified.
- **Time-base mismatch**: refugee and LPR series are fiscal-year based; ACS is calendar-year based; \PEP uses demographic years.
- **Measurement error**: ACS “moved from abroad” and foreign-born stocks have substantial sampling error (MOE), especially for small states.
- **Partial coverage**: some refugee month-level crosswalk coverage begins in FY2021; earlier years require approximation if aligned to \PEP-year.

## Decision

### Decision 1: Keep the Moderate baseline unchanged (univariate anchor)

**Decision**: Retain the existing Moderate scenario baseline (anchored to the univariate ARIMA/random-walk logic for 2025–2029 with the existing dampened-trend extension thereafter), rather than replacing it with a covariate-conditioned baseline.

**Rationale**:
- The Moderate baseline is the paper’s “reference trajectory” for stakeholders; redefining it as conditional on future covariate paths would require additional assumptions that are not currently defendable for 20-year horizons.
- With \(T \approx 15\), covariate-conditioned models can be unstable; the paper’s main claim is not “we beat the random walk,” but “uncertainty is large and policy-sensitive.”
- Preserves estimand discipline: \(Y_t\) remains the forecast object; covariates remain auxiliary.

**Alternatives Considered**:
- Replace Moderate with a full ARIMAX/state-space forecast through 2045: rejected due to the need to forecast covariates (or assume strong policy paths) and the resulting increase in complexity and interpretive burden.

### Decision 2: Add a covariate-conditioned *near-term* anchor (2025–2029) as appendix robustness

**Decision**: Implement a covariate-conditioned near-term (2025–2029) forecast as an **appendix-only robustness check**, using a parsimonious state-space “local level + regression” model (dynamic regression), and report it as a sensitivity overlay relative to the univariate baseline.

**Rationale**:
- The near-term horizon is where auxiliary signals are most plausibly informative and where assumptions about covariate paths can be kept minimal and transparent.
- A local-level state-space regression nests the random-walk baseline but allows covariates to explain deviations, reducing spurious regression risk without introducing high-order ARMA structure.
- Presenting this in an appendix strengthens the paper’s credibility (the covariates were considered) without making the main results depend on fragile coefficient estimates.

**Implementation (model class)**:
- Observation equation: \(y_t = \mu_t + \boldsymbol{\beta}'\mathbf{x}_t + \varepsilon_t\)
- State equation (local level): \(\mu_t = \mu_{t-1} + \eta_t\)
- Errors: \(\varepsilon_t \sim \mathcal{N}(0, \sigma_\varepsilon^2)\), \(\eta_t \sim \mathcal{N}(0, \sigma_\eta^2)\)

**Covariates (parsimonious default)**:
- \(x_t^{\text{ref}}\): ND refugee arrivals aligned to \PEP-year (aggregation + documented mapping; see appendix spec).
- \(x_t^{\text{lpr}}\): ND LPR admissions (or ND share of U.S. LPR) aligned to \PEP-year via a documented lag/mapping rule.
- Optional \(x_t^{\text{acs}}\): ACS “moved from abroad, foreign born” (B07007) as a noisy flow proxy; include only as sensitivity.

**Forecasting of covariates**:
- For the appendix anchor, covariate paths for 2025–2029 must be specified using **minimal, transparent rules** (e.g., last-observation carried forward or short-window mean). The goal is robustness, not a new covariate-forecasting sub-pipeline.

**Alternatives Considered**:
- Omit covariates entirely: rejected because it weakens the response to the “covariates” remaining-task item and misses an opportunity for a credible robustness check.
- Joint multivariate state-space fusion (forecasting \(Y_t\) with a latent immigration factor driven by multiple observed series): deferred as a separate workstream due to higher complexity and greater risk of results changes.

### Decision 3: Treat the covariate-conditioned forecast as *diagnostic*, not a scenario-engine input

**Decision**: The appendix anchor is reported as a diagnostic overlay and does **not** replace the Moderate baseline or mechanically re-parameterize scenario trajectories in Module 9.

**Rationale**:
- Prevents double counting with existing wave-duration adjustments and Monte Carlo uncertainty logic.
- Keeps the manuscript’s spine intact: scenarios remain policy-indexed narratives anchored to an interpretable baseline.

## Consequences

### Positive
1. Preserves the paper’s estimand discipline and “deep uncertainty” narrative.
2. Adds a rigorous, reviewer-friendly robustness check demonstrating that covariates were integrated in a forecast-feasible way.
3. Avoids a large increase in modeling surface area (no need to build and defend 20-year covariate forecasts).
4. Creates a clear bridge between policy-linked signals and near-term forecasts without over-claiming.

### Negative
1. The paper does not claim a single covariate-conditioned “best” baseline forecast.
2. Appendix work adds implementation and documentation overhead (alignment rules, sensitivity notes).

### Risks and Mitigations

**Risk**: Spurious regression / unstable coefficients under \(T \approx 15\).
- **Mitigation**: Use a local-level state-space model; restrict covariates to 1–2 primary signals; treat ACS as sensitivity; report rolling-origin near-term performance.

**Risk**: Time-base mismatch (FY vs \PEP-year vs calendar-year) undermines credibility.
- **Mitigation**: Use explicit mapping/lag rules and report them in the appendix; include a sensitivity that varies the mapping (e.g., one-year lag).

**Risk**: Double counting refugee signal (covariate + wave adjustment).
- **Mitigation**: Keep the appendix anchor separate from Module 8 wave adjustments; do not feed the appendix model into the main Monte Carlo scenario engine.

## Implementation Notes

### Inputs (canonical processed files)
- \PEP target series:
  - `data/processed/immigration/analysis/nd_migration_summary.csv`
  - (equivalent source) `data/processed/immigration/analysis/combined_components_of_change.csv` (state panel)
- Refugee series:
  - `data/processed/immigration/analysis/refugee_arrivals_by_state_nationality.parquet` (FY; long run)
  - `data/processed/immigration/analysis/refugee_arrivals_by_state_nationality_pep_year.parquet` (month-aware mapping for FY2021+)
  - `data/processed/immigration/analysis/refugee_fy_month_to_pep_year_crosswalk.csv`
- LPR series:
  - `data/processed/immigration/analysis/dhs_lpr_by_state_time.parquet` (FY totals)
  - `data/processed/immigration/analysis/dhs_lpr_nd_share_time.parquet` (ND vs U.S. totals/shares)
- ACS covariate (optional):
  - `data/processed/immigration/analysis/acs_moved_from_abroad_by_state.parquet` (ACS table B07007)

### Code integration (planned)
- Implement the appendix anchor as an additional output under:
  - `sdc_2024_replication/scripts/statistical_analysis/module_9_scenario_modeling.py`
  - or a small helper under `sdc_2024_replication/scripts/statistical_analysis/module_09_scenario/`

### Testing Strategy
- Add unit tests for covariate construction and alignment helpers (year coverage, missingness, lag rules).
- Add a lightweight regression test asserting the appendix output artifacts are generated and schema-stable.

## References
1. v0.8.6 remaining tasks tracker: `sdc_2024_replication/revisions/v0.8.6/remaining_tasks_v0.8.6.md`
2. v0.8.6 appendix spec memo: `sdc_2024_replication/revisions/v0.8.6/covariate_forecasting_appendix_spec.md`
3. ADR-024: `docs/governance/adrs/024-immigration-data-extension-fusion.md` (data sources and time-base alignment)
4. ADR-025: `docs/governance/adrs/025-refugee-coverage-missing-state-handling.md` (post-2020 refugee panel handling)
5. ADR-030: `docs/governance/adrs/030-pep-regime-aware-modeling-long-run-series.md` (scope boundary with regime-aware work)

## Revision History
- 2026-01-07: Initial version (ADR-031) - Keep Moderate baseline; add appendix-only covariate-conditioned near-term anchor (2025–2029).

## Related ADRs
- ADR-024: Immigration Data Extension and Fusion Strategy (inputs)
- ADR-025: Refugee Coverage and Missing-State Handling (post-2020 handling)
- ADR-028: Monte Carlo Simulation Rigor and Parallelization (uncertainty philosophy)
- ADR-030: Regime-Aware Modeling for Long-Run PEP Series (separating regime diagnostics from fusion work)
