# KPSS Resolution for n=15 Annual ND International Migration (2010–2024)

## What KPSS actually tests (and why the draft’s wording got into trouble)

The **KPSS test** (Kwiatkowski–Phillips–Schmidt–Shin) flips the usual unit-root logic:

- **H0 (null):** the series is **stationary** (either **level-stationary** or **trend-stationary**, depending on the regression setting).
- **H1 (alternative):** the series is **non-stationary** (contains a unit root / stochastic trend).

In the provided scripts, KPSS is run with `regression="c"` (constant only), which corresponds to testing **level-stationarity** (no deterministic trend). That is *not* the same as a “trend-stationarity” test (which would use `regression="ct"`).

**Important implementation detail (Statsmodels):** KPSS p-values are based on a lookup table and are often **reported as 0.10 when p ≥ 0.10**. So a printed `p = 0.10` should be described as **p ≥ 0.10** (not as “exactly 0.10”).

---

## The KPSS results in the provided unit-root output

Below are the KPSS outcomes from the unit-root results file (Module 2.1.1 output).

### A) ND international migration (`nd_intl_migration`)

- **KPSS (level):** statistic = **0.323**, p-value reported = **0.10** → **fail to reject stationarity at 5%** (interpret as p ≥ 0.10).
- **KPSS (first difference):** statistic = **0.189**, p-value reported = **0.10** → **fail to reject stationarity at 5%**.

**Interpretation:** Under KPSS, both the level series and the first-difference series look “not inconsistent with stationarity.” This does *not* prove stationarity; it means KPSS has not found strong evidence *against* stationarity.

### B) ND share of US international migration (`nd_share_of_us_intl_pct`) — where the KPSS rejection occurs

- **KPSS (level):** statistic = **0.070**, p-value reported = **0.10** → **fail to reject level-stationarity**.
- **KPSS (first difference):** statistic = **0.500**, p-value reported ≈ **0.0417** → **reject level-stationarity at 5%**.

**Interpretation:** This is the only KPSS rejection at 5% in the provided output. If the draft’s robustness table shows KPSS “**” for a *level* test, it is likely either:

1) a transcription/star-coding error,
2) the table is referring to a different series than the narrative,
3) the table used a different KPSS specification (e.g., trend-stationary `regression="ct"`), or
4) the author misread the KPSS null (common: treating KPSS like ADF).

### C) A note on “auto” lag selection in tiny samples

In the output, KPSS for the **differenced share series** uses **13 lags** on a 14-point differenced series. That is a red flag: in very small samples, “auto” bandwidth/lag rules can behave erratically, and the KPSS statistic can become highly sensitive to that choice.

For a revision, it is reasonable to:
- report a small set of sensitivity results with **fixed small lags** (e.g., 0, 1, 2), and
- treat any single KPSS rejection as suggestive rather than dispositive.

---

## Is there a contradiction, or just imprecise language?

### What’s definitely wrong
If the narrative says **“KPSS fails to reject stationarity in levels”** but the robustness table marks **KPSS rejecting stationarity in levels**, both cannot be true **for the same series/specification**.

Given the provided results file:

- For **`nd_intl_migration`**, KPSS **does not reject** stationarity in levels (p ≥ 0.10).
- The **rejection** (p ≈ 0.042) occurs for the **differenced `nd_share_of_us_intl_pct`** series, not `nd_intl_migration` levels.

### The most likely cause
The mismatch is most consistent with **variable/specification mixing** (level vs differenced; migration vs share), plus **KPSS-null confusion**.

---

## How to describe these results correctly in a journal article (recommended language)

Use phrasing that:
1) states the null explicitly,
2) acknowledges low power in n=15, and
3) avoids “accepting” a null.

**Suggested text (for `nd_intl_migration`):**

> “Stationarity diagnostics were inconclusive in this short annual series (n=15), particularly given the COVID-era disruption. The ADF test on levels does not reject a unit root, while the KPSS test (null: level-stationarity) does not reject stationarity (p ≥ 0.10). Taken together, these results are consistent with either (i) an approximately integrated process or (ii) a stationary process with structural breaks/outliers. We therefore treat integration order as a modeling choice rather than a settled fact, and prioritize out-of-sample forecast validation and robustness checks (including a 2020 intervention/outlier sensitivity).”

**Suggested text (for the share series, where KPSS rejects after differencing):**

> “For the ND share series, KPSS rejects level-stationarity of the first-difference (p ≈ 0.04), while ADF strongly rejects a unit root in the first-difference. Given the very short sample and sensitivity to lag/bandwidth selection, we treat these tests as descriptive diagnostics rather than definitive evidence about the true data-generating process.”

---

## What the combined ADF + KPSS evidence suggests about integration order

### Key point for n=15:
With **structural breaks (2020/2021) and a one-year extreme outlier (2020)**, standard unit-root/stationarity tests can easily disagree. In this setting, the strongest defensible claim is not “the series is I(1), random walk,” but:

- “Tests are **consistent with** I(1) *or* broken-trend stationarity; we therefore use models that are robust to both.”

For **`nd_intl_migration`**, the pattern is:

- **ADF level:** fails to reject unit root
- **KPSS level:** fails to reject stationarity
- **ADF difference:** rejects unit root
- **KPSS difference:** fails to reject stationarity

This is compatible with I(1), but the level KPSS non-rejection prevents a confident claim that the level is non-stationary. A clean revision is to label the integration order as **“ambiguous; treated as I(1) for conservative differencing, with sensitivity checks.”**

---

## Summary table of unit-root/stationarity diagnostics (from the provided output)

| Series | ADF level p | KPSS level p (H0: stationary) | PP level p | ADF diff p | KPSS diff p | Safe conclusion to write |
|---|---:|---:|---:|---:|---:|---|
| nd_intl_migration | 0.556 | ≥0.10 | 0.867 | 0.0025 | ≥0.10 | “Ambiguous; differencing is defensible, but breaks/outliers likely dominate test behavior.” |
| nd_share_of_us_intl_pct | 0.285 | ≥0.10 | 0.00051 | ~0.00000005 | 0.0417 | “Tests conflict; treat as diagnostic only; prioritize break-robust modeling and forecasting validation.” |
| us_intl_migration | 0.0020 | ≥0.10 | 0.952 | 0.157 | ≥0.10 | “Mixed evidence (ADF vs PP); do not overinterpret; treat as broadly stable for use as a driver.” |

---

## Action items to resolve the draft’s internal inconsistency

1. **Fix the robustness table** so that KPSS “stars” reflect *rejection of stationarity* (KPSS null) and match the p-values for the same series/specification.
2. **Add a footnote:** “KPSS p-values are approximate; statsmodels reports 0.10 when p ≥ 0.10.”
3. **Add a ‘diagnostic, not definitive’ sentence** wherever stationarity language appears.
4. **Shift evidentiary weight to forecasting**: add rolling-origin accuracy and interval calibration (backtesting), rather than leaning on unit-root test declarations.
