---
title: "Wave Duration Refit (Module 8): Spec Grid + Implementation Runbook (v0.8.6)"
date_created: 2026-01-07
status: "complete"
related_tracker: "sdc_2024_replication/revisions/v0.8.6/remaining_tasks_v0.8.6.md"
related_response: "sdc_2024_replication/revisions/v0.8.6/response_to_critique.md"
related_memo: "sdc_2024_replication/revisions/v0.8.6/senior_scholar_memo_wave_duration.md"
scope: "Refit Module 8 wave duration metrics; reassess right-censoring and hazard ratios; document primary vs sensitivity specifications; define implementation checkpoints."
notes: "P0 adopted as primary (approved) and S1 retained as sensitivity; Module 8 grid executed; Module 9 rerun to quantify downstream impact."
---

# Wave Duration Refit (Module 8): Spec Grid + Implementation Runbook (v0.8.6)

## Purpose
This document operationalizes the v0.8.6 remaining task:

- `Refit wave duration metrics; reassess right-censoring and hazard ratios.`

It defines (1) a **pre-declared primary specification** and a small set of **sensitivities**, (2) the comparison outputs required to claim we “reassessed right-censoring,” and (3) the implementation steps needed to run multiple specifications without overwriting results.

## Scope and Touchpoints
- **Module 8 (duration analysis)**: `sdc_2024_replication/scripts/statistical_analysis/module_8_duration_analysis.py`
  - Outputs used by later steps:
    - `sdc_2024_replication/scripts/statistical_analysis/results/module_8_duration_analysis.json`
    - `sdc_2024_replication/scripts/statistical_analysis/results/module_8_hazard_model.json`
    - `sdc_2024_replication/scripts/statistical_analysis/results/module_8_wave_durations.json`
- **Module 9 (scenario engine)** consumes Module 8 outputs via:
  - `sdc_2024_replication/scripts/statistical_analysis/module_08_duration/wave_registry.py`
  - `sdc_2024_replication/scripts/statistical_analysis/module_9_scenario_modeling.py`

## Key Risk: “Endpoint extension” can change more than censoring
Before interpreting differences as “reduced right-censoring,” we must control two code-path issues:

1. **Baseline window drift**: `module_8_duration_analysis.py` currently computes the baseline window as “first half of sample,” which changes when FY2024 is added. This can change wave identification even if censoring were unchanged.
2. **Sparse year coverage**: the refugee arrivals panel is not a complete state×nationality×year grid; many state-nationality pairs omit years with zero arrivals. The current wave-run detection treats missing years as adjacent, which can spuriously merge non-consecutive years into a “consecutive” wave (a direct confound for “paused vs ended”).

These two issues motivate explicit specification dimensions (baseline window and year completion).

## Pre-Declared Specifications (“Spec Grid”)
Definitions below assume the same wave rule used in the manuscript/code: arrivals exceed **150% of baseline** for **≥2 consecutive years** (unless a gap-tolerance sensitivity is applied).

### Parameters and Notation
- **End year**: last fiscal year included in the refugee arrivals series (FY2024 in the refit).
- **Baseline window**: fiscal years used to compute the baseline median for each state×nationality series.
- **Year completion**: whether missing years are imputed as `arrivals = 0` before wave detection.
- **Gap tolerance**: maximum number of consecutive below-threshold years allowed *within* a wave before declaring termination (captures “pause/resumption”).
- **Min peak arrivals**: absolute minimum `peak_arrivals` required for a wave to be included (filters tiny/noisy waves).

### Spec Table
| Spec ID | Role | End Year | Baseline Window | Year Completion | Gap Tolerance | Min Peak Arrivals | Primary Question Answered |
|---|---|---:|---|---|---:|---:|---|
| **P0** | Primary (recommended) | 2024 | FY2002–FY2016 (fixed, pre–Travel Ban) | Yes (fill missing years with 0) | 0 | 0 | “With extended data and a stable baseline definition, what are the updated durations and hazard ratios?” |
| **S1** | Sensitivity (pause taxonomy) | 2024 | FY2002–FY2016 | Yes | 1 | 0 | “Do short pauses materially change durations/hazards?” |
| **S2** | Sensitivity (comparability baseline) | 2024 | FY2002–FY2011 (matches pre-extension implied baseline) | Yes | 0 | 0 | “Are changes primarily censoring/data-extension vs baseline choice?” |
| **S3** | Sensitivity (small-wave filter) | 2024 | FY2002–FY2016 | Yes | 0 | 10 | “Are hazards driven by many tiny waves?” |

### Notes on Specification Status (Tier 3)
- The “primary” designation above is a **recommendation** for scientific defensibility (fixed baseline + year completion). Adopting it as the paper’s main result is a **Tier 3 methodology change** because it can change wave definitions and hazard ratios.
- If approval is not granted for adopting P0 as the main spec, we can still run the same grid as a **diagnostic appendix** while preserving the legacy method as primary.
- Approval granted (2026-01-07): **P0 is primary**, **S1 is sensitivity**.

## Required Outputs and Comparison Artifacts
For each spec run, capture:
- `n_waves_identified`, `n_censored`, censoring rate
- KM: median duration + survival at 2/3/5 years
- Cox PH: concordance index, hazard ratios + p-values for the main covariates (`log_intensity`, `peak_arrivals`, timing/region terms)
- A short “pause audit” summary: among waves censored under FY2020, what fraction end/continue/pause by FY2024 (when applicable)

### Delta Table (filled; 2026-01-07)
| Metric | FY2020 (reference) | FY2024 P0 | Δ (P0–ref) | FY2024 S1 | FY2024 S2 | FY2024 S3 |
|---|---:|---:|---:|---:|---:|---:|
| Waves identified | 940 | 2,057 | +1,117 | 1,954 | 2,087 | 1,631 |
| Censoring rate (%) | 10.0 | 25.5 | +15.5 | 28.1 | 27.0 | 28.1 |
| KM median duration (years) | 3.0 | 3.0 | +0.0 | 4.0 | 3.0 | 4.0 |
| KM S(3) | 0.318 | 0.481 | +0.163 | 0.622 | 0.496 | 0.555 |
| Cox concordance | 0.769 | 0.777 | +0.008 | 0.740 | 0.783 | 0.765 |
| HR(`log_intensity`) | 0.412 | 0.591 | +0.179 | 0.619 | 0.626 | 0.623 |
| HR(`peak_arrivals`) | 0.656 | 0.565 | -0.090 | 0.689 | 0.473 | 0.597 |

**Notes**
- FY2020 reference comes from legacy outputs: `sdc_2024_replication/scripts/statistical_analysis/results/module_8_duration_analysis.json`.
- FY2024 specs come from tagged outputs:
  - `.../results/module_8_duration_analysis__P0.json`
  - `.../results/module_8_duration_analysis__S1.json`
  - `.../results/module_8_duration_analysis__S2.json`
  - `.../results/module_8_duration_analysis__S3.json`
- `HR(peak_arrivals)` corresponds to the model’s `peak_arrivals` covariate after the script’s stability scaling (`peak_arrivals / 1000`).

### Right-Censoring and “Pause” Audit (S2-style; 2026-01-07)
To assess *right-censoring* (not just overall censoring rates), we looked at what happened to waves that were **censored at FY2020** once FY2021–FY2024 became available.

Audit setup:
- Spec: baseline window ends FY2011 (`baseline_end_year=2011`), `fill_missing_years=True`, `gap_tolerance_years=0` (S2-style).
- States restricted to those with complete FY2021–FY2024 “Total” coverage (same inclusion logic as the Module 8 pipeline when FY2024 is present).

Results:
- Waves through FY2020: 1,573 total; 218 censored at FY2020 (13.9%).
- Among the 218 FY2020-censored waves:
  - 82 (37.6%) continue through FY2024 (still active/censored at FY2024).
  - 136 (62.4%) are now observed to end by FY2024.
    - End-year distribution: FY2020 (82), FY2021 (51), FY2022 (1), FY2023 (2).
  - 106 of the 136 ended waves (77.9%) have a later wave restart (“pause/resume” under `gap_tolerance_years=0`).
    - Gap-length distribution (years below threshold between wave episodes): 1-year gap (83), 2-year gap (23).

Interpretation:
- Extending to FY2024 **resolves a majority of FY2020 right-censoring** for the pre-2021 wave cohort (censoring falls from 13.9% to 5.2% for that cohort).
- The *overall* FY2024 censoring rate is high because many waves start in FY2021+ and are mechanically right-censored at FY2024:
  - New waves starting FY2021+: 499; censored among new: 474 (95.0%).

### Module 9 Downstream Impact (P0 vs S1; 2026-01-07)
To quantify downstream effects on the scenario engine, we reran Module 9 under both duration specs:
- P0: `module_9_scenario_modeling.py --duration-tag P0 --rigorous`
- S1: `module_9_scenario_modeling.py --duration-tag S1 --rigorous`

Notes:
- Deterministic scenario endpoints (CBO/Moderate/Policy/Zero/Pre-2020 trend) are unchanged because scenario construction does not depend on the duration tag.
- The duration tag affects only the **hazard-based wave persistence adjustment** inside the Monte Carlo simulation.

Key Monte Carlo deltas (units: persons, net international migration to ND):
| Metric | P0 | S1 | Δ (P0–S1) |
|---|---:|---:|---:|
| MC p50 (2030) | 7,409 | 7,415 | -6 |
| MC p5 (2030) | 3,128 | 2,989 | +139 |
| MC p95 (2030) | 11,638 | 11,713 | -75 |
| MC p50 (2045) | 9,119 | 9,064 | +55 |
| MC p5 (2045) | 3,407 | 3,345 | +61 |
| MC p95 (2045) | 14,806 | 14,755 | +52 |
| Mean wave adjustment (2030) | 2,042.2 | 2,049.5 | -7.3 |
| Mean wave adjustment (2045) | 399.6 | 334.5 | +65.0 |

Summary:
- Largest year-specific difference over 2025–2045 in the MC median is ~586 persons (year 2037).
- P0 vs S1 differences are small relative to the overall uncertainty envelope, supporting S1 as a sensitivity rather than an alternative primary spec.

## Implementation Runbook (What We Need to Build/Run)

### A. Code changes required (so we can actually run the grid)
To run multiple specs without overwriting, we need to add a small amount of plumbing to Module 8:
1. **Parameterize baseline window** (fixed end-year, not “first half of sample”).
2. **Optionally fill missing years** in each state×nationality series with `arrivals = 0` before wave detection.
3. **Implement gap tolerance** (e.g., allow a 1-year below-threshold gap inside a wave).
4. **Add a min-peak filter** on candidate waves.
5. **Add output tagging** (e.g., `--tag P0`) so outputs become:
   - `module_8_duration_analysis__P0.json`
   - `module_8_hazard_model__P0.json`
   - `module_8_wave_durations__P0.json`

Recommended approach: keep the existing default outputs unchanged (backward-compatible), and only write tagged outputs when a tag is provided.

### B. Execution workflow (once parameterization exists)
Run inside the project virtual environment. Example invocation pattern:
- `.venv/bin/python sdc_2024_replication/scripts/statistical_analysis/module_8_duration_analysis.py --end-year 2024 --baseline-end-year 2016 --fill-missing-years --gap-tolerance-years 0 --min-peak-arrivals 0 --tag P0`
- Repeat for `S1`, `S2`, `S3`.

### C. Integration workflow (Module 9)
Only after selecting the final primary spec:
- Module 9 supports loading tagged Module 8 outputs via `--duration-tag` (so you can avoid manual copying).
- Rerun Module 9 and record whether scenario paths materially change. If they do, document deltas as attributable to the wave-duration update.

Example:
- `.venv/bin/python sdc_2024_replication/scripts/statistical_analysis/module_9_scenario_modeling.py --duration-tag P0 --n-draws 25000 --n-jobs 0`

## Decision Criteria (When a Sensitivity Becomes the Primary)
A spec change (gap tolerance, baseline window, min-peak) should be promoted from sensitivity to primary only if:
1. It resolves a clearly demonstrated measurement problem (e.g., “pause” behavior is common and the current definition misclassifies it), **and**
2. The qualitative conclusions of the paper remain coherent (no whiplash), **and**
3. The revised definition can be explained transparently in the Methods section with a clear substantive rationale (not “it fits better”).

## Approval Checkpoint (Tier 3)
Before adopting P0 (or any sensitivity) as the manuscript’s primary duration model:
- Record the spec choice and rationale in the v0.8.6 response matrix (`sdc_2024_replication/revisions/v0.8.6/response_to_critique.md`).
- Confirm explicit approval for the methodology change, since it can affect reported hazard ratios and downstream scenario simulation.
