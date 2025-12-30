# Recommendations for Revising the Small-Sample Forecasting Methodology (n=15)

This note prioritizes changes needed for a top-tier demography journal revision focused on *credible inference under very small samples*.

---

## Critical changes (must fix before re-review)

### 1) Reconcile KPSS narrative vs robustness table (internal consistency)
- Ensure the **same series and specification** are referenced in text and in tables.
- Remember KPSS **H0 is stationarity**. Stars in a KPSS column mean “evidence *against* stationarity,” not “stationary.”
- Add a footnote that **statsmodels KPSS p-values are approximate and often capped at 0.10**.

**Deliverable for the revision:** one paragraph that states:
(i) ADF null, (ii) KPSS null, (iii) what each test did/did not reject, and (iv) a cautious interpretation (“diagnostic only”).

### 2) Fix scenario labeling so arithmetic is reproducible from stated assumptions
The scenario numbers are reproducible *given the implemented rules*, but the labels can mislead.

- **CBO Full:** It is not “8% growth from 2024 baseline.” It is:
  - 2025–2029: **1.1 × ARIMA level**
  - 2030–2045: **8% compound growth**
- **Pre-2020 Trend:** It is not anchored to 2024. It is a **counterfactual anchored at 2019** (pre-COVID endpoint).

**Deliverable:** an appendix with explicit formulas for each scenario and a one-line code snippet showing the update rule.

### 3) Rename “credible intervals” and clean up interval terminology
Unless you are fitting an explicit Bayesian model with priors/posterior:
- Replace **credible interval** → **prediction interval** (or “simulation-based prediction interval”).
- Distinguish:
  - **confidence interval** (uncertainty about a parameter),
  - **prediction interval** (uncertainty about a future observation).

### 4) Add out-of-sample backtesting with naive benchmarks
For a forecasting paper, validation is the spine.

- Implement **rolling-origin evaluation** (expanding window).
- Compare at least:
  1) naive random-walk (last observation),
  2) expanding mean/median,
  3) national-driver regression.
- Report **MAE/RMSE** and **interval coverage**.

Even with few folds, this will satisfy the “does it forecast better than nothing?” reviewer question.

---

## Important changes (strongly recommended)

### 5) Break-robust stationarity framing (don’t overclaim I(1))
With n=15 and a 2020/2021 disruption, unit-root tests are too brittle to “declare” the DGP.

**Recommended approach:**
- Treat unit-root tests as **descriptive diagnostics**.
- Add **break-aware alternatives** (even if low power):
  - Perron-style tests with an exogenous break date (2020),
  - Zivot–Andrews (endogenous single break),
  - Lee–Strazicich LM tests (breaks under both null and alternative),
  - Clemente–Montañés–Reyes (additive outlier vs innovational outlier).
- For forecasting, fit a **local-level state-space model** or **ARIMA with an intervention dummy** for 2020 and compare via backtesting.

### 6) COVID handling: treat 2020 as an intervention, not “just another year”
You don’t have to delete 2020, but you must model it honestly.

Options to report:
- **Indicator/intervention dummy** (one-year shock),
- **Robust error model** (heavy tails / t innovations),
- **Winsorization sensitivity** (show that results are not driven by the single dip).

State explicitly whether 2020 is treated as:
- measurement anomaly,
- temporary shock, or
- regime change.

---

## Minor / editorial (still worth fixing)

- Replace “optimal” with “selected” everywhere.
- Replace “establishes/proves” with “suggests/consistent with.”
- Round and simplify overly precise p-values given n=15.
- Ensure calendar-year vs fiscal-year terminology is consistent throughout.

---

## Suggested appendix contents (for transparency and reviewer confidence)

1. **Full backtesting tables** (rolling-origin log + summary metrics).
2. **Scenario equations and step-by-step arithmetic** (show 2045 endpoints).
3. **Sensitivity to COVID handling** (with/without dummy; robust vs not).
4. **Stationarity diagnostics** (plots + tests), labeled explicitly as low-power.

---

## Talking points for responding to reviewers

- “We do not claim unit-root tests identify the true integration order in n=15; we use them as diagnostics and prioritize predictive validation.”
- “We add rolling-origin backtests against naive benchmarks; our contribution is disciplined uncertainty quantification and transparent scenario design, not point-precise prediction.”
- “We correct terminology: simulation percentile bands are prediction intervals, not Bayesian credible intervals.”
- “We make scenario arithmetic reproducible with explicit update rules and an appendix.”

This is how you turn “small sample” from a liability into the core methodological contribution.
