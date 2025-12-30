# Bartik/Shift-Share Specification

## Identification Strategy Assessment
Your current “Bartik” construction is a classic **shift-share exposure index**:
- Shares: each state’s baseline (2010) share of each nationality.
- Shifts: national changes in arrivals by nationality over time.
- Index: B_st = Σ_n share_sn,2010 × ΔNatArrivals_n,t

As coded, you then run a **first-stage regression** of state-year international migration on this index with state and year FE, producing a strong first stage (F≈22). This shows relevance, but **it is not yet a full IV design** (no second stage, and no defended exclusion restriction).

The key modern concern: in shift-share settings, residuals can be correlated across states with similar shares, so “vanilla” robust or clustered-by-state SE can badly over-reject. Reviewers will expect you to follow modern inference.

## Modern Inference Approach (what to cite / what to do)
You should anchor the discussion in three core references:
- Goldsmith-Pinkham, Sorkin & Swift (2020, AER) on how Bartik identification works and when it fails.
- Adão, Kolesár & Morales (2019, QJE) on shift-share inference and why naive SE over-reject.
- Borusyak, Hull & Jaravel (2022, ReStud) on SSIV / exposure-robust inference; plus their practical guide (JEP 2025) for implementation.

Implementation recommendations:
1. **Use exposure-robust (“shock-level”) inference**:
   - Re-express the design at the shock level (nationality-year shocks) and compute standard errors that account for the shift-share structure (AKM / BHJ).
   - In R, consider packages like `ShiftShareSE` (AKM-style) or implement BHJ’s recommended procedures.
2. **Leave-one-out shift**:
   - Construct national shocks excluding each state’s own contribution to national totals to avoid mechanical correlation (especially relevant if some states are large in specific nationalities).
3. **Base period sensitivity**:
   - Keep 2010 as baseline (plausibly predetermined for a 2017 shock), but show robustness to:
     - 2008 or 2005 baseline,
     - or a multi-year average baseline (e.g., 2008–2010).
4. **Explicit exclusion restriction**:
   - State clearly what must be true for the instrument to be valid (e.g., conditional on FE, national nationality-level shocks affect a state only through predicted refugee/migrant inflows).

## Interpretation Guidance (the “4.36” coefficient)
Right now, **4.36 is a first-stage slope**, not a causal effect:
- It means: a 1-unit increase in the Bartik index is associated with +4.36 units of the dependent variable (international migration), conditional on state and year FE.
- The unit of the Bartik index is “predicted arrivals” (a weighted sum of national nationality changes), so you must clarify whether it is in persons, per-capita, or some scaled unit.

What the paper should do:
- Label the estimate as **First-stage relevance** and report it as such.
- Add an interpretation example using the index’s SD:
  - “A 1 SD increase in the Bartik index predicts X additional international migrants (or Y per 1,000).”
- Do **not** call it an “IV estimate” unless you add a second stage.

## Required Additions (to survive reviewer scrutiny)
1. A formal equation for the index, and a short paragraph explaining shares vs shocks.
2. A table describing:
   - baseline year,
   - number of shocks (nationalities),
   - overlap years,
   - and distribution of the index.
3. Correct inference:
   - AKM/BHJ SE (or an equivalent exposure-robust method), plus a robustness check using alternative clustering.
4. Clear scope:
   - If your goal is corroboration of the Travel Ban effect, shift-share needs to be aligned to that shock (e.g., build a “banned-nationality shock index” post-2017), not a broad refugee-flow predictor.
