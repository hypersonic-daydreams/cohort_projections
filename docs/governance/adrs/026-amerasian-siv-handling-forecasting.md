# ADR-026: Amerasian/SIV Handling in Status Decomposition and Scenario Forecasts

## Status
Accepted

## Date
2026-01-06

## Context

The RPC archives publish Amerasian and Special Immigrant Visa (SIV) arrivals separately from
USRAP refugee arrivals. For North Dakota in FY2021--FY2024, the extracted Amerasian/SIV series
is non-trivial relative to USRAP refugees and is overwhelmingly Afghanistan-linked.

Two places in the analysis pipeline require an explicit decision:

1. **Causal identification (DiD/event study)**: where the estimand and exposure should remain
   narrowly defined for interpretability.
2. **Forecasting / decomposition**: where the objective is to model future international
   in-migration behaviorally, but without silently misclassifying durable arrivals as
   temporary/parole by construction.

The short Amerasian/SIV history (FY2021--FY2024) makes independent time-series modeling
highly unstable and difficult to defend.

## Decision

### Decision 1: Keep USRAP refugees as the exposure for DiD/event study

- Treat USRAP refugee arrivals as the strict treatment/exposure series for causal analyses.
- Do not merge Amerasian/SIV into the USRAP refugee series.

### Decision 2: Model Amerasian/SIV as a separate durable series in forecasting/decomposition

- Keep Amerasian/SIV as a separate series, but treat it as **durable humanitarian** for
  behavioral purposes (retention/durability similar to refugees).
- In the ND status accounting (two-component decomposition), classify:
  - `durable_humanitarian = refugees(USRAP) + amerasian_siv`
  - `parole_proxy = PEP - durable_humanitarian - other_share`
  - where `other_share` is a fixed non-humanitarian share of PEP (per ADR-021 residual method).

### Decision 3: Scenario linkage and default sunset

- In scenario projections, link Amerasian/SIV to the same **local capacity trajectory**
  used for refugee allocation (Rec #3 capacity multiplier), rather than tying it directly
  to refugee ceilings or an assumed fixed refugee:SIV ratio.
- Apply a conservative **default sunset** after the base year (2024) via an exponential
  half-life with a non-zero floor to avoid extrapolating Afghanistan-era SIV volumes
  indefinitely while still allowing residual SIV-like flows.

## Implementation

- Data extraction remains separate:
  - `data/processed/immigration/analysis/amerasian_siv_arrivals_by_state_nationality.parquet`
- Two-component decomposition uses SIV explicitly:
  - `sdc_2024_replication/scripts/statistical_analysis/module_10_two_component_estimand.py`
- Policy scenario projections include SIV as durable, capacity-linked, and sunset:
  - `sdc_2024_replication/scripts/statistical_analysis/module_9b_policy_scenarios.py`

## Consequences

### Positive

- Causal estimates remain interpretable (USRAP-only exposure).
- Forecasting/decomposition avoids misclassifying durable humanitarian arrivals as parole.
- Separation improves transparency and makes assumptions auditable.

### Negative

- The SIV series is short; parameters (baseline, sunset) are assumption-driven and must be
  described as such.
- Scenario outputs change when SIV is incorporated into durable totals.

## Alternatives Considered

1. **Merge Amerasian/SIV into USRAP refugees everywhere**: rejected (definition drift harms
   interpretability for causal work).
2. **Independent SIV time-series forecasting**: rejected (FY2021--FY2024 too short; high risk
   of overfitting and unstable extrapolation).
3. **Flat SIV with no sunset**: rejected (implicitly assumes Afghanistan-era SIV persists).
