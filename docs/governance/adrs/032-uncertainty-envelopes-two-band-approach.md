# ADR-032: Two-Band Uncertainty After Data Fusion (Avoid Double-Counting Variance)

## Status
Accepted

## Date
2026-01-07

## Context

The v0.8.6 revision tracker includes a remaining task:

> “Reassess uncertainty envelopes after data fusion (avoid double-counting variance).”

In the current forecasting pipeline, **Module 9** generates a Monte Carlo distribution for North Dakota \PEP net international migration using:

1. A baseline stochastic projection process (trend + innovations, with ARIMA-based near-term standard errors when available), and
2. An optional stochastic **wave persistence** adjustment derived from **Module 8** (hazard-based duration model applied to detected active refugee-wave structure).

This design raises a legitimate inference/communication hazard:

- If the baseline forecast uncertainty is estimated from the historical ND series, that series already contains wave-driven volatility. Adding a stochastic wave process on top of the baseline can therefore **inflate uncertainty** if interpreted as a calibrated prediction interval (i.e., a form of “variance stacking”).

The v0.8.5 critique explicitly highlighted this risk and suggested bracketing uncertainty with two bands (baseline-only vs baseline+wave), paired with language that these intervals are **conservative uncertainty envelopes**, not strictly calibrated predictive probabilities.

The paper’s intent is exploratory and decision-supportive (small-state volatility + policy sensitivity) rather than a claim of precisely calibrated probabilistic forecasting. The appropriate response is therefore to make the uncertainty decomposition explicit and auditable.

## Decision

### Decision 1: Report two uncertainty bands (nested) rather than a single “all-in” interval

**Decision**: Implement and report **two** Monte Carlo-based uncertainty summaries for the scenario projections:

1. **Baseline-only prediction interval (PI)**: Monte Carlo uncertainty from the baseline stochastic projection process *without* wave persistence adjustments.
2. **Wave-adjusted uncertainty envelope**: Monte Carlo uncertainty from the same baseline process *plus* hazard-based wave persistence draws.

**Rationale**:
- Makes the potential double-counting issue transparent.
- Aligns with the paper’s “deep uncertainty” thesis while avoiding over-claiming calibration.
- Gives readers and planners a choice: “model-based PI” vs “conservative envelope.”

### Decision 2: Use shared baseline draws to isolate the incremental impact of wave persistence

**Decision**: Generate baseline-only and wave-adjusted distributions from the **same baseline simulation draws** (same seed/chunking), then add wave adjustments to obtain the envelope.

**Rationale**:
- Ensures differences between bands are attributable to the wave process, not to different Monte Carlo noise realizations.
- Preserves reproducibility (seeded, chunk-deterministic execution; ADR-028).

## Consequences

### Positive
1. Removes the appearance of “quiet variance stacking” by explicitly bracketing uncertainty.
2. Improves interpretability: baseline uncertainty vs wave persistence contribution are separable.
3. Supports reviewer expectations for honest interval interpretation in short, regime-shifting series.

### Negative
1. Adds reporting surface area (two sets of bands/tables).
2. Readers may ask why not fully re-estimate baseline variance conditional on wave state; we treat that as a potential follow-on (see Alternatives).

## Risks and Mitigations

**Risk**: Readers interpret the wave-adjusted envelope as a calibrated 95% PI.
- **Mitigation**: Label the wave-adjusted band as an **uncertainty envelope** and explicitly state it may be conservative due to overlapping variance sources.

**Risk**: The two-band figure becomes visually busy.
- **Mitigation**: Use nested, clearly labeled shading (inner PI; outer envelope), and limit to 50%/95% only.

**Risk**: Results change from v0.8.5/v0.8.6 interim outputs.
- **Mitigation**: Treat changes as a reporting/interpretation improvement; regenerate figures and update manuscript language accordingly.

## Alternatives Considered

1. **Full data-fusion state-space model** (joint latent process measured by PEP, refugees, LPR, ACS):
   - Deferred due to complexity and higher risk of methodology-driven results changes.
2. **Re-estimate baseline innovations after de-waving** (fit baseline to a “de-waved” series, then add waves back stochastically):
   - Conceptually clean, but requires additional modeling assumptions (what is “wave” vs baseline) and introduces additional specification choices.
3. **Drop wave adjustments entirely**:
   - Rejected because wave persistence is substantively relevant for policy-sensitive planning and is already supported by Module 8 evidence.

## Implementation Notes

### Code Changes (planned/implemented)
- `sdc_2024_replication/scripts/statistical_analysis/module_9_scenario_modeling.py`
  - Produce baseline-only and wave-adjusted Monte Carlo summaries from shared draws.
  - Persist both interval endpoints in `module_9_scenario_modeling.json`.
  - Update fan chart output to display nested bands.
- `sdc_2024_replication/scripts/statistical_analysis/journal_article/create_publication_figures.py`
  - Update Figure 8 to display the two-band uncertainty.
- `sdc_2024_replication/scripts/statistical_analysis/journal_article/figures/figure_captions.tex`
  - Update caption language to distinguish PI vs envelope.

### Testing Strategy
- Run `pytest tests/ -q` after code changes.
- Smoke-run Module 9 with a small draw count (e.g., `--n-draws 2000`) to confirm both bands are generated and serialized.

## Related ADRs
- ADR-024: Immigration Data Extension and Fusion Strategy (double-counting risk)
- ADR-028: Monte Carlo Simulation Rigor and Parallelization (reproducible chunking)
- ADR-029: Wave Duration Refit (hazard-based wave persistence feeding Module 9)
- ADR-031: Covariate-Conditioned Near-Term Forecast Anchor (explicitly avoids feeding appendix model into the scenario engine to prevent double counting)

## Revision History
- 2026-01-07: Initial version (ADR-032) - Adopt two-band uncertainty reporting to avoid double-counting variance after fusion.
