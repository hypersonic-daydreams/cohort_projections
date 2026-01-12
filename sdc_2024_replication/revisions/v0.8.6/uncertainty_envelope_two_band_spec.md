---
title: "Two-Band Uncertainty Envelopes After Fusion: Implementation Spec (v0.8.6)"
date_created: 2026-01-07
status: "active"
related_tracker: "sdc_2024_replication/revisions/v0.8.6/remaining_tasks_v0.8.6.md"
related_adr: "docs/governance/adrs/032-uncertainty-envelopes-two-band-approach.md"
---

# Two-Band Uncertainty Envelopes After Fusion: Implementation Spec (v0.8.6)

This runbook operationalizes the v0.8.6 remaining task:

- “Reassess uncertainty envelopes after data fusion (avoid double-counting variance).”

Decision record:
- ADR-032: `docs/governance/adrs/032-uncertainty-envelopes-two-band-approach.md`

## 1. Goal (What “done” looks like)

Produce **two nested uncertainty summaries** for the scenario fan chart:

1. **Baseline-only PI**: Monte Carlo prediction intervals from the baseline stochastic projection process (no wave persistence adjustment).
2. **Wave-adjusted envelope**: The same baseline draws plus hazard-based wave persistence draws from Module 8.

Outputs must be reproducible (seeded) and clearly labeled “PI” vs “envelope” in the figure caption and manuscript text.

## 2. Scope and Interpretation

### In-scope (v0.8.6)
- Two-band reporting and figure updates (PI vs envelope).
- Shared-draw implementation so the incremental contribution of wave persistence is isolated.
- Minimal manuscript edits to clarify interpretation.

### Out-of-scope (v0.8.6)
- Re-fitting baseline innovation variance conditional on “de-waved” series.
- Full data-fusion state-space model (latent factor measured by multiple sources).

## 3. Implementation Tasks

### 3.1. Module 9: generate both bands from shared draws

File:
- `sdc_2024_replication/scripts/statistical_analysis/module_9_scenario_modeling.py`

Requirements:
- Monte Carlo simulation returns:
  - baseline simulations (no wave adjustment)
  - wave-adjusted simulations (baseline + wave draws)
  - wave adjustment summaries (already present)
- JSON output includes both interval endpoints at minimum for 2030 and 2045 (50% and 95%).
- Preserve existing CLI flags and chunk-deterministic reproducibility (ADR-028).

Suggested output keys:
- `results.monte_carlo.baseline_only`
- `results.monte_carlo.wave_adjusted`
- `results.confidence_intervals.baseline_only`
- `results.confidence_intervals.wave_adjusted`

### 3.2. Figure 8: display nested bands

File:
- `sdc_2024_replication/scripts/statistical_analysis/journal_article/create_publication_figures.py`

Update:
- Use baseline-only PI as the **inner** band.
- Use wave-adjusted envelope as the **outer** band.
- Keep shading visually distinct and avoid clutter (recommend: show 50% + 95% for PI, and 95% for envelope).

### 3.3. Figure caption: clarify PI vs envelope

File:
- `sdc_2024_replication/scripts/statistical_analysis/journal_article/figures/figure_captions.tex`

Update caption text to:
- Define baseline-only band as prediction interval.
- Define wave-adjusted band as conservative uncertainty envelope.
- Explicitly note potential overlap (double-counting risk) motivating the two-band presentation.

### 3.4. Manuscript text: align terminology

Files (minimum touch):
- `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/02_data_methods.tex`
- `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/03_results.tex`

Update:
- State that Monte Carlo uncertainty is reported as nested PI (baseline) and envelope (baseline + wave persistence).
- Avoid calling the envelope a calibrated 95% PI; call it a conservative envelope.

## 4. Execution Commands (Reproducible)

Run Module 9 (rigorous, with wave inputs if available):

```bash
uv run python sdc_2024_replication/scripts/statistical_analysis/module_9_scenario_modeling.py --rigorous --duration-tag P0
```

Generate publication figures (including Figure 8):

```bash
uv run python sdc_2024_replication/scripts/statistical_analysis/journal_article/create_publication_figures.py
```

## 5. Validation Checklist

- [ ] Module 9 JSON contains both `baseline_only` and `wave_adjusted` uncertainty summaries.
- [ ] Figure 8 shows two bands with correct labels (PI vs envelope).
- [ ] Manuscript text and caption do not over-claim calibrated predictive probability for the envelope.
- [ ] `pytest tests/ -q` passes.

## 6. Definition of Done (DOD)

This tracker item can be marked complete when:
- ADR-032 is merged/accepted (this doc references it).
- Module 9 produces two-band uncertainty outputs.
- Figure 8 and its caption are updated and regenerate without errors.
- Manuscript text references the two-band interpretation.
