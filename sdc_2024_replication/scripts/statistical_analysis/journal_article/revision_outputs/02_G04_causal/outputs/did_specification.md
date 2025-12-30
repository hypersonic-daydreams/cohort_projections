# DiD Specification Recommendations

## Current Specification Assessment
Your Travel Ban design is conceptually strong because it uses **nationality-level variation** within the same destination context, so “national policy shock” is not automatically fatal: the policy targets specific nationalities, giving you a plausible treated vs control contrast. The current implementation is:

- Outcome: **log(arrivals + 1)**
- Units: nationality-year
- Treatment: the original 7 EO-13769 nationalities
- Post: 2018–2019 (first full post year is 2018; 2020 excluded for COVID)
- Model: TWFE (nationality FE + year FE) estimated via OLS with heteroskedastic-robust SE

Main findings (as currently estimated):
- ATT ≈ -1.38 in log points, implying ~75% reduction in treated-country arrivals post-ban.
- Pre-trend test p≈0.18; event-study joint pretrend p≈0.15.

What’s good:
- Clear estimand (short-run policy effect on arrivals from targeted nationalities).
- Lots of control units (many non-treated nationalities) and many pre years.

What’s fragile:
- Inference: HC3/HC1 is not the DiD default in panels with repeated observations by unit.
- Functional form: log(y+1) is workable but not the current best practice for flow counts with many zeros.
- Parallel trends: long pre window (2002–2017) mixes very different global refugee regimes; early “lead” coefficients can be noisy and distract from the key pre-period close to treatment.
- Only **2 post years** (2018–2019) limits dynamic interpretation.

## Recommended Changes
### 1) Standard errors
**Recommendation:** Cluster at the nationality level (one-way).
Rationale: serial correlation and within-nationality shocks across years make heteroskedastic-only SE optimistic.

Add two robustness layers:
- **Wild cluster bootstrap** p-values (clustered by nationality), to handle small effective treated-group size (7 treated nationalities) and potential non-normality.
- (Optional) Randomization inference / placebo treated sets: repeatedly draw 7 “fake treated” countries matched on pre-period mean and variance of arrivals.

### 2) Functional form (log(y+1) vs PPML)
**Recommendation:** Re-estimate with **PPML (Poisson pseudo-maximum likelihood) with nationality and year fixed effects**, and cluster by nationality.

Why:
- PPML naturally handles zeros without arbitrary +1.
- PPML is robust to many forms of heteroskedasticity and gives semi-elasticity-style interpretation in multiplicative models.

Report both:
- Primary: PPML FE estimate of the ATT (incidence rate ratio or % change).
- Secondary: log(y+1) OLS for transparency/comparability.

### 3) Sample / timing
Keep 2018 as first full post year, but add explicit sensitivity:
- Include 2017 as a “partial” year with a separate indicator (or exclude 2017 entirely and use 2002–2016 as pre) to show robustness to the transition year.
- Restrict pre-period to a more comparable window (e.g., 2010–2017) and show that results are not driven by early years with very different global patterns.
- Consider a “stacked” design around 2017/2018 if you add additional policy events; otherwise, keep the estimand explicitly short-run.

## Sensitivity Checks Required
1. **Alternative treated sets**:
   - Original 7 (baseline) vs updated lists from later proclamations (as a robustness exercise).
2. **Placebo policy dates**:
   - Pretend treatment occurs in 2014 or 2015 and verify no “effect.”
3. **Placebo treated groups**:
   - Assign treatment to a set of non-banned countries matched on pre-trends / region / baseline volume.
4. **Exclude high-leverage controls**:
   - Drop the top-k highest-volume control nationalities to ensure results aren’t driven by a few giant senders.
5. **Trend specifications**:
   - Add treated-group-specific linear trends; check whether ATT remains negative and sizable.
6. **Aggregation robustness**:
   - Run the same design on (i) totals, (ii) per-capita rates, and (iii) inverse-hyperbolic-sine transform as an alternative to log+1.

## Event Study Improvements
1. **Plot is non-optional**: include the event-study figure in the main text, with:
   - clearly labeled reference year (2017),
   - 95% CI,
   - vertical line at the policy period,
   - and a joint test of all pre-coefficients.
2. **Bin extreme leads**: combine very early pre-years into bins (e.g., ≤ -8) to reduce clutter and avoid over-reading noisy early coefficients.
3. **Focus attention**: include an inset or second panel zoomed to -5…+2 years, where the identifying variation is most credible.
4. **Match the estimator**: if PPML is the primary estimator, produce a PPML-based event study as well (not only OLS log+1).
