# ADR-030: Regime-Aware Modeling for Long-Run PEP Net International Migration Series

## Status
Accepted

## Date
2026-01-07

## Context

The v0.8.6 revision cycle includes a remaining task: **“Add regime-aware modeling for long-run PEP series (state-space or regime dummies).”** The core analytical tension is that the project uses two related, but not identical, PEP-derived series:

1. **Primary forecasting/inference window (n=15):** North Dakota PEP net international migration, 2010–2024 (`data/processed/immigration/analysis/nd_migration_summary.csv`).
2. **Long-run context series (n=25):** Census PEP components spliced across vintages, 2000–2024 (`data/processed/immigration/state_migration_components_2000_2024_with_regime.csv`).

Across 2000–2024, there are known **measurement-regime boundaries** aligned with decennial-vintage transitions (2009→2010; 2019/2020→2020+) and a substantive shock/outlier year (COVID-19 in 2020). This creates a high risk of conflating genuine structural change with vintage-methodology shifts if a single homogeneous time-series model is fit to the full long-run series without explicit regime structure.

This ADR formalizes a regime-aware modeling choice that is:
- Consistent with ADR-020’s **hybrid approach** (primary claims on 2010–2024; long-run as robustness/diagnostic).
- Stable under small samples and high volatility.
- Transparent and easy to communicate in the manuscript.

### Requirements
- Treat decennial-vintage boundaries as explicit regimes (not implicit “noise”).
- Maintain interpretability: regime effects should map to identifiable time periods.
- Avoid overfitting: models must remain stable under n=15 and n=25 constraints.
- Separate *measurement-regime artifacts* from *real shocks* where possible (especially 2020).
- Provide a spec grid (primary + limited sensitivities) with reproducible outputs.

### Challenges
- **COVID-19 outlier (2020):** a large discontinuity that confounds “regime change” with a one-year shock.
- **Variance heterogeneity:** post-2020 volatility is materially higher than prior regimes; naive OLS inference can be misleading.
- **Vintages are not fully comparable in levels:** the long-run series is explicitly spliced; cross-regime level comparisons are hazardous.
- **Scope control:** full state-space fusion (PEP + LPR + refugee + ACS) is deferred (Tier 3 implications) and should not be smuggled into this task.

## Decision

### Decision 1: Use regime-dummy / intervention models as the primary “regime-aware” implementation for v0.8.6

**Decision**: Implement regime awareness via **regime indicators + piecewise trends + COVID intervention terms**, estimated with robust inference, rather than a latent regime-switching or full state-space model.

**Rationale**:
- This matches the project’s current evidence base: explicit vintage boundaries and known shocks are already documented in ADR-020 and the data/methods narrative.
- Regime dummies/interventions are transparent and stable in small samples, while latent switching can overfit and be fragile for n≈25.
- Keeps this task separable from the broader (and deferred) “fusion model” workstream.

**Implementation (model class)**:
- **Piecewise (vintage-regime) slopes** across {2000s, 2010s, 2020s} with level shifts at 2010 and 2020.
- **COVID intervention** as a pulse (2020 only) in the long-run diagnostic model.
- **Robust inference** via HAC standard errors; optional WLS sensitivity using regime-specific variance weights.

**Existing code assets**:
- Piecewise trends and regime dummies: `sdc_2024_replication/scripts/statistical_analysis/module_regime_aware/piecewise_trends.py`
- COVID intervention construction: `sdc_2024_replication/scripts/statistical_analysis/module_regime_aware/covid_intervention.py`
- Robust/heteroskedastic inference utilities: `sdc_2024_replication/scripts/statistical_analysis/module_regime_aware/robust_inference.py`

**Alternatives Considered**:
- **Markov-switching / latent regimes**: rejected for v0.8.6 due to instability/overfit risk under n≈25 and weaker interpretability.
- **Full state-space (Kalman) model**: deferred because it is best treated as part of the data-fusion task (LPR/refugee/ACS covariates), which is explicitly not started and carries Tier 3 implications.

### Decision 2: Treat the long-run model as diagnostic/robustness; keep primary forecasting anchored to 2010–2024

**Decision**: Use long-run (2000–2024) regime-aware modeling for **diagnostic and robustness** purposes, while keeping the main forecasting target and scenario engine anchored to the 2010–2024 series.

**Rationale**:
- Preserves the “hybrid” identification logic: causal claims and core forecast calibration remain on the most comparable window.
- Uses the long-run series to quantify variance shifts and regime-dependent slopes without overstating cross-vintage comparability.

**Alternatives Considered**:
- Promote 2000–2024 to the primary forecasting series: rejected due to measurement comparability risks and interpretation complexity.

## Consequences

### Positive
1. Regime handling becomes explicit, auditable, and easy to explain to reviewers.
2. Avoids fragile latent-regime estimation on short series.
3. Clean separation between (a) regime-aware diagnostics and (b) later fusion/state-space work.
4. Provides a clear spec grid structure (primary + 2–3 sensitivities) suitable for the v0.8.6 tracker.

### Negative
1. Regime dummies do not “solve” vintage comparability; they only structure it transparently.
2. Interpretation still requires careful language: cross-regime level comparisons remain discouraged.

### Risks and Mitigations

**Risk**: Readers interpret regime dummies as causal “policy regimes,” not measurement regimes.
- **Mitigation**: Use “vintage regime / measurement regime” terminology for decennial boundaries; reserve “policy regime” for the separate `R_t` framework (ADR-021).

**Risk**: COVID year dominates inference and drives spurious slope estimates.
- **Mitigation**: Include an explicit COVID pulse and report a sensitivity excluding 2020.

**Risk**: Heteroskedasticity leads to misleading standard errors.
- **Mitigation**: Default to HAC SE; include WLS-by-regime-variance sensitivity where appropriate.

## Implementation Notes

### Inputs (canonical)
- `data/processed/immigration/analysis/nd_migration_summary.csv` (2010–2024; ND and U.S. totals/shares)
- `data/processed/immigration/state_migration_components_2000_2024_with_regime.csv` (2000–2024; spliced vintages; includes regime markers)

### Outputs (expected)
- JSON results + figures generated by the regime-aware modeling run (see v0.8.6 runbook in `sdc_2024_replication/revisions/v0.8.6/`).
- A short manuscript paragraph describing the regime-aware diagnostic and the role of measurement regimes.

### Testing Strategy
- Run `pytest tests/ -q` after implementing any code changes associated with this ADR.

## References
1. ADR-020: `docs/governance/adrs/020-extended-time-series-methodology-analysis.md` (hybrid approach and vintage risks)
2. ADR-024: `docs/governance/adrs/024-immigration-data-extension-fusion.md` (long-run components + regime markers; fusion deferred)
3. v0.8.6 tracker: `sdc_2024_replication/revisions/v0.8.6/remaining_tasks_v0.8.6.md`

## Revision History
- 2026-01-07: Initial version (ADR-030) - Proposed regime-dummy/intervention approach for long-run PEP regime-aware modeling.

## Related ADRs
- ADR-020: Extended Time Series Methodology Analysis (foundational rationale)
- ADR-021: Immigration Status Durability Methodology (separate policy-regime variable `R_t`)
- ADR-024: Immigration Data Extension and Fusion Strategy (scope boundary with fusion/state-space work)
