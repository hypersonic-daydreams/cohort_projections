# ADR-029: Wave Duration Refit for Right-Censoring and Hazard Stability

## Status
Accepted

## Date
2026-01-07

## Context

The v0.8.6 critique highlighted a limitation in the duration (wave persistence) module: wave duration and Cox hazard estimates were tied to an FY2020 endpoint and therefore subject to right-censoring. Extending the refugee arrivals series through FY2024+ should (a) reduce right-censoring for the FY2020-censored cohort, (b) reveal whether apparent “terminations” were actually “pauses,” and (c) potentially change survival curves and hazard ratios.

Two code-path issues must be controlled to interpret FY2024 extensions as a censoring update (rather than an unintended definition drift):

1. **Baseline window drift**: the legacy baseline for “above 150% of baseline” could shift when the sample endpoint is extended (because the baseline was implicitly defined relative to the full sample window).
2. **Sparse year coverage**: the state×nationality refugee panel can omit explicit zero years; treating missing years as adjacent can spuriously create “consecutive-year” runs.

### Requirements
- Extend duration estimation to FY2024 while keeping baseline definitions stable.
- Ensure the “≥2 consecutive years above threshold” rule reflects consecutive fiscal years, not consecutive non-missing observations.
- Support reproducible multi-spec execution (primary + sensitivities) without overwriting artifacts.
- Allow downstream Module 9 Monte Carlo wave adjustments to load the chosen duration spec explicitly.

### Challenges
- Post-2020 RPC archive reports omit some states in some years (ADR-025); state-panel duration estimation must avoid implicit zero-fill of omitted states while still avoiding spurious gaps inside retained state×nationality series.
- The duration module outputs feed the Module 9 wave-adjustment mechanism; changes must be explicit and auditable (Tier 3).

## Decision

### Decision 1: Adopt a fixed pre–Travel Ban baseline window and complete missing years (P0 primary)

**Decision**: Use a fixed baseline window ending FY2016 and fill missing years with `arrivals = 0` inside each state×nationality series prior to wave detection.

**Rationale**:
- Fixes baseline-window drift when extending the endpoint beyond FY2020.
- Ensures that “two consecutive years” corresponds to consecutive fiscal years.
- Prevents sparse panels from producing spurious wave runs.

**Implementation**:
- Module 8 parameterization: `sdc_2024_replication/scripts/statistical_analysis/module_8_duration_analysis.py`
  - `--end-year 2024`
  - `--baseline-end-year 2016`
  - `--fill-missing-years`
  - `--gap-tolerance-years 0`
  - `--min-peak-arrivals 0`
  - Tagged outputs via `--tag P0` to avoid overwriting artifacts.

**Alternatives Considered**:
- Keep the legacy “first half of sample” baseline rule: rejected because it confounds endpoint extension with definition drift.
- Restrict duration analysis to FY2002–FY2020 only: rejected because it fails to address the critique about right-censoring and pause behavior.

### Decision 2: Treat pause behavior as an explicit sensitivity (S1) and plumb tagged outputs into Module 9

**Decision**: Define a single sensitivity that allows a 1-year below-threshold gap inside a wave (`gap_tolerance_years=1`) and add explicit loading of tagged Module 8 outputs in Module 9 via `--duration-tag`.

**Rationale**:
- Allows an interpretable “pause/resume” taxonomy without forcing it into the primary definition.
- Keeps downstream scenario modeling reproducible: the scenario engine can be rerun under different duration specs without manual file copying.

**Implementation**:
- Module 8 sensitivity: `--gap-tolerance-years 1 --tag S1`
- Module 9 integration: `sdc_2024_replication/scripts/statistical_analysis/module_9_scenario_modeling.py --duration-tag {P0,S1}`
- Tagged-load helper: `sdc_2024_replication/scripts/statistical_analysis/module_08_duration/wave_registry.py::load_duration_model_bundle(..., tag=...)`

**Alternatives Considered**:
- Promote gap tolerance to the primary spec: rejected for v0.8.6; retained as a sensitivity because the downstream effects are small and the baseline-definition fixes address the core measurement risks.
- Manual copying/overwriting of default result filenames: rejected as error-prone and non-reproducible.

## Consequences

### Positive
1. FY2024 extensions can be interpreted as a censoring/pause reassessment rather than a moving baseline definition.
2. Wave detection is robust to sparse year coverage in the arrivals panel.
3. Downstream scenario modeling can explicitly tie outputs to duration-spec choices.

### Negative
1. Filling missing years with zeros can change wave segmentation relative to the legacy implementation for sparse series.
2. Post-2020 incomplete state reporting still requires explicit state filtering (ADR-025) to avoid implicit missing-as-zero for omitted states.

### Risks and Mitigations

**Risk**: Duration and hazard estimates change materially relative to FY2020 results.
- **Mitigation**: Treat as an approved Tier 3 methodology update; record a pre-declared spec grid and sensitivity results in `sdc_2024_replication/revisions/v0.8.6/wave_duration_refit_spec_grid.md`, and rerun Module 9 under P0 and S1 to quantify downstream impact.

## Implementation Notes

### Key Files
- Module 8: `sdc_2024_replication/scripts/statistical_analysis/module_8_duration_analysis.py`
- Duration utilities: `sdc_2024_replication/scripts/statistical_analysis/module_08_duration/wave_registry.py`
- Module 9: `sdc_2024_replication/scripts/statistical_analysis/module_9_scenario_modeling.py`

### Testing Strategy
- Unit tests for wave detection edge cases live in `tests/unit/test_module_8_duration_analysis.py`.
- Run `pytest tests/ -q` after changes (v0.8.6 status: passing).

## References
1. `sdc_2024_replication/revisions/v0.8.6/senior_scholar_memo_wave_duration.md` (verbatim guidance)
2. `sdc_2024_replication/revisions/v0.8.6/wave_duration_refit_spec_grid.md` (spec grid, runbook, delta tables, Module 9 impact)
3. `sdc_2024_replication/revisions/v0.8.6/critique_chatgpt_5_2_pro_v0.8.5.md` (critique item “Wave duration right-censoring and wave taxonomy”)
4. ADR-025: Refugee coverage and missing-state handling

## Revision History
- 2026-01-07: Initial version (ADR-029) - Fixed-baseline + year-completion wave duration refit with explicit sensitivity and Module 9 integration.
