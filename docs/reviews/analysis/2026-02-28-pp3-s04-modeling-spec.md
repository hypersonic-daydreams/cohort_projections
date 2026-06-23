# 2026-02-28 Modeling Specification for Share-Trending (PP3-S04)

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-02-28 |
| **Reviewer** | Claude (AI Agent) -- requires human review before acceptance |
| **Scope** | PP3-S04 trend model specification, constraint mechanism, and backtest variant matrix for Phase 1 city/place projections |
| **Status** | Revised -- human decisions incorporated 2026-02-28; final approval at PP3-S07 gate |
| **Related ADR** | ADR-033 |

---

## 1. Purpose

Define the mathematical specification for the Phase 1 place-level share-trending model. This document specifies:

- The trend model form (logit-linear).
- Two fitting-window variants to be tested via backtest (equal-weight vs. recency-weighted).
- Two constraint mechanisms to be tested via backtest (proportional rescaling vs. cap-and-redistribute).
- The balance-of-county treatment (independently trended and reconciled).
- Edge case handling rules.

All four items above were resolved by human review. No open questions remain.

## 2. Trend Model Form (Decision 1)

### 2.1 Logit-Linear Specification

For each place *p* in county *c*, let `s_p(t)` denote the place's share of county population in year *t*:

```
s_p(t) = population_p(t) / population_c(t)
```

The share is transformed via the logit function:

```
logit(s) = log(s / (1 - s))
```

A linear model is fit on the logit scale:

```
logit(s_p(t)) = a_p + b_p * t
```

where `a_p` (intercept) and `b_p` (slope) are estimated from historical data (2000-2024). Projected shares are obtained by back-transforming:

```
s_p(t_future) = 1 / (1 + exp(-(a_p + b_p * t_future)))
```

### 2.2 Properties

- **Bounded output**: The logistic back-transform guarantees `s_p(t) in (0, 1)` for all finite values of `a + b*t`, eliminating the need for post-hoc clamping of individual shares.
- **Asymptotic behavior**: As `b*t` grows large, `s_p(t)` approaches 1; as `b*t` grows large and negative, `s_p(t)` approaches 0. Extreme trend extrapolation produces asymptotic convergence rather than unbounded growth or negative values.
- **Linearizable**: OLS/WLS on the logit scale is computationally trivial (no iterative optimization).

### 2.3 Epsilon Clamping for Boundary Shares

The logit function is undefined at `s = 0` and `s = 1`. Before applying the logit transform, clamp observed shares:

```
s_clamped = max(epsilon, min(1 - epsilon, s_observed))
```

**Epsilon value: 0.001** (one-tenth of one percent of county population).

Rationale: An epsilon of 0.001 corresponds to a place holding 0.1% of its county. For the smallest ND county (Slope, ~700 population), this floor is less than 1 person -- small enough to avoid materially distorting the trend, but large enough to keep the logit finite.

Implementation note: Apply clamping to the observed share series before fitting. If a projected share falls below epsilon after back-transform, set the projected population to zero for that place-year (the place has effectively vanished). This can only happen with a strongly negative slope over a very long horizon.

## 3. Fitting Window Variants (Decision 2)

Both variants use the full 2000-2024 history (25 annual observations per place). They differ only in how observations are weighted during regression.

### 3.1 Variant A: Equal-Weight OLS

Ordinary least squares on the logit-transformed shares:

```
minimize sum_{t=2000}^{2024} [logit(s_p(t)) - (a_p + b_p * t)]^2
```

All 25 observations receive equal weight. This variant treats the full 2000-2024 history as equally informative.

### 3.2 Variant B: Recency-Weighted WLS

Weighted least squares with exponentially declining weights:

```
w(t) = lambda^(2024 - t)       where lambda = 0.9
```

```
minimize sum_{t=2000}^{2024} w(t) * [logit(s_p(t)) - (a_p + b_p * t)]^2
```

Weight table (selected years):

| Year | `2024 - t` | `w(t) = 0.9^(2024-t)` |
|------|-----------|----------------------|
| 2024 | 0 | 1.000 |
| 2020 | 4 | 0.656 |
| 2015 | 9 | 0.387 |
| 2010 | 14 | 0.229 |
| 2005 | 19 | 0.135 |
| 2000 | 24 | 0.080 |

Half-life: approximately 7 years (`0.9^7 = 0.478`). Observations older than ~14 years carry less than 25% of the weight of the most recent observation.

This variant prioritizes recent share dynamics while retaining the full history for slope estimation stability.

### 3.3 Variant Selection Protocol

The winning variant is selected by the S05 backtesting protocol (primary window: train 2000-2014, test 2015-2024). Selection criterion: **population-weighted MedAPE** across all three projected tiers (HIGH, MODERATE, LOWER).

**One model is selected for all tiers** -- no per-tier model selection. See Section 5 for the full variant matrix and selection algorithm.

## 4. Share Constraint Mechanism (Decision 3)

After projecting individual place shares within a county, the sum of shares may exceed 1.0 in some projection years. Two constraint mechanisms are tested.

### 4.1 Option I: Proportional Rescaling

If, for county *c* in projection year *t*:

```
S_c(t) = sum_{p in c} s_p(t) > 1.0
```

then rescale every place share proportionally:

```
s_p_adj(t) = s_p(t) / S_c(t)       for all p in c
```

After rescaling, `sum s_p_adj(t) = 1.0` and the balance-of-county share is zero for that year.

Properties:
- Simple and deterministic.
- All places are adjusted equally in proportional terms.
- A place with a declining share is pulled down further, which may over-penalize declining places.

### 4.2 Option II: Cap-and-Redistribute

If `S_c(t) > 1.0`, identify the set of *growing* places (places whose projected share in year *t* exceeds their base-year share in 2025):

```
G_c(t) = {p in c : s_p(t) > s_p(2025)}
D_c(t) = {p in c : s_p(t) <= s_p(2025)}
```

Declining/stable places keep their projected shares. The excess is redistributed only among growing places:

```
excess = S_c(t) - 1.0

For p in D_c(t):
    s_p_adj(t) = s_p(t)                    # unchanged

For p in G_c(t):
    s_p_adj(t) = s_p(t) - excess * [s_p(t) / sum_{q in G_c(t)} s_q(t)]
```

**Edge case**: If all places grew (`D_c(t)` is empty), every place is in `G_c(t)` and the formula reduces to proportional rescaling (Option I).

**Edge case**: If after redistribution any growing place's adjusted share falls below its base-year share (over-correction), clamp it at the base-year share and re-distribute the remaining excess among the other growing places iteratively.

Properties:
- Preserves the direction of declining trends.
- Places losing share are not penalized for other places' growth.
- Slightly more complex to implement due to the iterative edge case.

### 4.3 Constraint Selection Protocol

The winning constraint mechanism is selected by the same S05 backtest, crossed with the fitting-window variants (see Section 5). The selection criterion is the same population-weighted MedAPE.

## 5. Backtest Variant Matrix

### 5.1 The 2x2 Matrix

| Variant ID | Fitting Window | Constraint Mechanism |
|------------|---------------|---------------------|
| **A-I** | Equal-weight OLS | Proportional rescaling |
| **A-II** | Equal-weight OLS | Cap-and-redistribute |
| **B-I** | Recency-weighted WLS (lambda=0.9) | Proportional rescaling |
| **B-II** | Recency-weighted WLS (lambda=0.9) | Cap-and-redistribute |

### 5.2 Backtest Execution

Each variant is run through the S05 primary backtest window:

- **Training period**: 2000-2014 (15 years of share observations).
- **Test period**: 2015-2024 (10 years of out-of-sample evaluation).
- For each variant, fit the logit-linear model on training data, project shares for 2015-2024, apply the variant's constraint mechanism, multiply by actual county population to get projected place populations, and compute error metrics against actual place populations.

### 5.3 Selection Criterion: Population-Weighted MedAPE

For each variant *v*, compute a single scalar score:

1. For each place *p*, compute `MAPE_p` = mean of `APE(p, t)` across all test years (per S05 Section 3.1).
2. For each tier *k* (HIGH, MODERATE, LOWER), compute `MedAPE_k` = median of `MAPE_p` across places in the tier.
3. Compute the population-weighted aggregate:

```
Score(v) = sum_{k in {HIGH, MOD, LOWER}} [W_k * MedAPE_k(v)]
            / sum_{k in {HIGH, MOD, LOWER}} W_k
```

where `W_k` is the total 2024 population of all places in tier *k*.

The variant with the lowest `Score(v)` wins. Population weighting ensures that accuracy in large places (which represent the bulk of projected population) is prioritized, while still incorporating signal from smaller tiers.

### 5.4 Tie-Breaking

If two variants produce identical `Score(v)` to three decimal places, prefer the simpler specification:
1. Prefer equal-weight (A) over recency-weighted (B).
2. Prefer proportional rescaling (I) over cap-and-redistribute (II).

### 5.5 One Winner for All Tiers

The winning variant is applied uniformly to all tiers and all places. There is no per-tier or per-place model selection. This avoids overfitting to tier-specific noise and keeps the model specification simple and auditable.

## 6. Balance-of-County Treatment (Decision 4)

### 6.1 Independent Balance Trend

For each county *c*, compute the historical balance-of-county share:

```
s_bal(t) = 1 - sum_{p in c} s_p(t)
```

where the sum runs over all places in the projection universe for that county (population >= 500 per ADR-033).

Fit an independent logit-linear trend to `s_bal(t)` using the same fitting variant that wins the backtest (equal-weight or recency-weighted). The independently trended balance share provides a cross-check on the place-level trends.

### 6.2 Reconciliation Procedure

In each projection year *t*, for each county *c*:

1. Compute the sum of independently projected place shares: `S_places(t) = sum s_p(t)`.
2. Retrieve the independently trended balance share: `s_bal_trend(t)`.
3. Compute the total: `T(t) = S_places(t) + s_bal_trend(t)`.

**If `T(t) = 1.0`** (within floating-point tolerance of 1e-9): no adjustment needed.

**If `T(t) != 1.0`**: apply the winning constraint mechanism (from Section 4) to reconcile. Specifically:

- Treat the balance-of-county as an additional "place" in the constraint algorithm.
- Under proportional rescaling (Option I): scale all shares (places + balance) proportionally so they sum to 1.0.
- Under cap-and-redistribute (Option II): identify which components (places and/or balance) grew vs. declined, and redistribute the discrepancy among growing components only.

### 6.3 QA Signal

After reconciliation, compute the adjustment magnitude for each county-year:

```
adjustment(c, t) = |T(t) - 1.0|
```

If `adjustment(c, t) > 0.05` (more than 5 percentage points of reconciliation) in any projection year, flag the county in the QA outlier table. Large adjustments indicate that place trends and balance trends are internally inconsistent for that county, warranting manual review.

## 7. Edge Cases

### 7.1 Places with Zero Share in Historical Years

A place with zero population in some historical years will have `s_p(t) = 0` for those years.

**Rule**: Apply epsilon clamping (Section 2.3) before logit transform. The clamped share of 0.001 ensures a finite logit value. If the place has zero population for the majority of the historical window, the fitted trend will have a strongly negative intercept, producing near-zero projected shares -- which is the correct behavior for a place that was essentially unpopulated.

### 7.2 Places Near Tier Boundaries

A place whose 2024 population is within 5% of a tier threshold (9,500-10,000 for HIGH/MODERATE; 2,375-2,500 for MODERATE/LOWER; 475-500 for LOWER/EXCLUDED) could shift tiers under minor population revision.

**Rule**: Tier assignment uses 2024 PEP population as published (per S05 Section 6.2). Boundary places are flagged as `tier_boundary` in backtest and production output for reviewer awareness. No special modeling treatment -- the place is modeled identically regardless of tier; only the output detail level differs by tier.

### 7.3 Counties with Only One Projected Place

If a county has exactly one place in the projection universe, the share constraint is trivially satisfied (one share <= 1.0 by the logit bound). The balance-of-county is `1 - s_p(t)`.

**Rule**: No special handling. The single-place trend and independent balance trend are both fit and reconciled normally. The reconciliation adjustment should be near zero since there are no competing place trends.

### 7.4 Counties with No Projected Places

If all places in a county fall below the 500-population EXCLUDED threshold, no place-level projections are produced for that county.

**Rule**: No balance-of-county output is generated. The county projection stands alone. These counties are listed in the run-level metadata (`places_metadata.json`) under a `counties_without_places` field for completeness.

### 7.5 Dissolved Places (Bantry, Churchs Ferry)

Bantry city (04740, dissolved by 2020) and Churchs Ferry city (14140, dissolved by 2020) are present in 2000-2019 data but absent from 2020-2024.

**Rule**: Per S02/S03 rules, these places are flagged `historical_only` in the crosswalk. They are included in historical county-share sums (so that county shares remain correct in training years) but excluded from the projection universe, backtest evaluation, and production output. Their historical share is absorbed into the balance-of-county for their respective counties in years after dissolution.

### 7.6 Multi-County Places

Places assigned via `multi_county_primary` in the S03 crosswalk have their entire population attributed to a single primary county. If growth is actually occurring in the secondary county, the primary-county share trend may be noisy.

**Rule**: Model these places identically to single-county places using the primary county assignment. Flag all `multi_county_primary` places in the backtest per-place detail table (per S05 Section 6.4). Systematic accuracy differences between `single_county` and `multi_county_primary` places are noted for potential Phase 2 refinement (population splitting).

## 8. Implementation Notes

### 8.1 Epsilon Value

Use `epsilon = 0.001` for logit clamping. This value is defined as a named constant in the model configuration, not hard-coded in the fitting function, so it can be adjusted without code changes if backtesting reveals sensitivity.

### 8.2 Population-Weighted MedAPE Computation

Step-by-step algorithm for variant selection:

```python
for variant in [A_I, A_II, B_I, B_II]:
    for place in projection_universe:
        # Compute APE for each test year
        ape_values = [abs(proj[t] - actual[t]) / actual[t] * 100
                      for t in test_years]
        place.mape = mean(ape_values)

    for tier in [HIGH, MODERATE, LOWER]:
        tier_places = [p for p in projection_universe if p.tier == tier]
        tier.medape = median([p.mape for p in tier_places])
        tier.weight = sum([p.population_2024 for p in tier_places])

    variant.score = (
        sum(tier.weight * tier.medape for tier in tiers)
        / sum(tier.weight for tier in tiers)
    )

winner = min(variants, key=lambda v: v.score)
```

### 8.3 Reconciliation Algorithm

For a given county *c* and projection year *t*:

```python
# Collect projected shares (places + balance)
shares = {p: s_p(t) for p in county_places}
shares['balance'] = s_bal_trend(t)
total = sum(shares.values())

if abs(total - 1.0) < 1e-9:
    return shares  # no adjustment needed

if constraint_winner == 'proportional':
    return {k: v / total for k, v in shares.items()}

elif constraint_winner == 'cap_and_redistribute':
    base_shares = {p: s_p(2025) for p in county_places}
    base_shares['balance'] = s_bal_trend(2025)

    growing = {k for k, v in shares.items() if v > base_shares[k]}
    stable  = {k for k, v in shares.items() if v <= base_shares[k]}

    if not growing:
        # All declined -- proportional fallback
        return {k: v / total for k, v in shares.items()}

    excess = total - 1.0
    growing_total = sum(shares[k] for k in growing)

    adjusted = dict(shares)
    for k in growing:
        adjusted[k] -= excess * (shares[k] / growing_total)

    # Iterative clamping if over-corrected
    while any(adjusted[k] < base_shares[k] for k in growing):
        clamped = {k for k in growing if adjusted[k] < base_shares[k]}
        for k in clamped:
            adjusted[k] = base_shares[k]
        growing -= clamped
        if not growing:
            break
        remaining_excess = sum(adjusted.values()) - 1.0
        growing_total = sum(adjusted[k] for k in growing)
        for k in growing:
            adjusted[k] -= remaining_excess * (adjusted[k] / growing_total)

    return adjusted
```

### 8.4 Time Variable Centering

When fitting `logit(s) = a + b*t`, center the time variable at the midpoint of the fitting window to reduce collinearity between intercept and slope:

```
t_centered = t - mean(t_fitting_window)
```

For the full 2000-2024 window, `mean = 2012`. For the backtest training window 2000-2014, `mean = 2007`. Centering does not affect the projected shares (the back-transform is invariant to linear reparameterization), but it improves numerical stability of the regression coefficient estimates.

### 8.5 Configuration Parameters

All model parameters should be defined in `config/projection_config.yaml` (or a dedicated place-projection config section), not hard-coded:

| Parameter | Value | Section |
|-----------|-------|---------|
| `epsilon` | 0.001 | Logit clamping (2.3) |
| `lambda` | 0.9 | WLS decay rate (3.2) |
| `history_start` | 2000 | Fitting window start (3) |
| `history_end` | 2024 | Fitting window end (3) |
| `reconciliation_flag_threshold` | 0.05 | Balance QA flag (6.3) |

## 9. Dependencies

| Dependency | Source Step | Required Artifact | Status |
|------------|-----------|-------------------|--------|
| Place-to-county crosswalk | PP3-S03 | `data/processed/geographic/place_county_crosswalk_2020.csv` | Rules defined, artifact not yet built |
| Historical place population series | PP3-S02 | Assembled long-format place population file (2000-2024) | Data sources verified, assembly script not yet built |
| County population actuals (for shares) | Existing | `data/processed/pep_county_components_2000_2025.parquet` or equivalent county totals | Available |
| Backtest protocol | PP3-S05 | Backtest design and acceptance thresholds | Defined (this document's variant matrix is consumed by S05) |
| County projections (for production forward runs) | Existing | `data/projections/{scenario}/county/*.parquet` | Available |

**Sequencing**: S04 (this document) defines what the backtest tests; S05 defines how the backtest is evaluated. Both must be finalized before the backtest can be executed. S06 (output contract) defines how the winning model's results are packaged. All three feed into the S07 approval gate.

## 10. Human Decisions Record

All modeling decisions in this document were made by human review on 2026-02-28. No open questions remain.

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **D1: Trend model form** | Logit-linear | Bounded shares by construction; linearizable; standard in demographic share-trending |
| **D2: Fitting window** | Backtest decides between equal-weight OLS (Variant A) and recency-weighted WLS with lambda=0.9 (Variant B) | Both use full 2000-2024 history; backtest selects winner by population-weighted MedAPE |
| **D3: Constraint mechanism** | Backtest decides between proportional rescaling (Option I) and cap-and-redistribute (Option II) | Crossed with D2 to form 2x2 matrix; same selection criterion |
| **D4: Balance-of-county** | Independently trended + reconciled | Independent trend provides QA cross-check; reconciliation uses winning constraint mechanism |

---

| Attribute | Value |
|-----------|-------|
| **Last Updated** | 2026-02-28 |
| **Version** | 1.0 |
