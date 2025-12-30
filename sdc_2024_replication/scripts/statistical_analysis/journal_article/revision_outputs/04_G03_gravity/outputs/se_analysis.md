# Standard Error Analysis for the FY2023 Gravity/Network PPML

This note diagnoses why the reported PPML standard errors look implausibly small and lays out a defensible inference strategy consistent with modern gravity practice.

---

## 1. What the current script is doing (and not doing)

### 1.1 Variance estimator used now

The current implementation fits Poisson GLMs via statsmodels and uses the **default covariance** from maximum-likelihood Poisson (i.e., *model-based* SE assuming Var\((Y|X)=\mu\) and independent observations).

It does **not**:

- request heteroskedasticity-robust (“sandwich”) SE,
- cluster SE by destination state or origin country,
- apply any correction for overdispersion,
- propagate ACS measurement error from margins of error (MOE).

### 1.2 A second, separate issue: dropping zeros

The script filters the estimation sample to **flow > 0** before estimation. That is:

- selection on the dependent variable, and
- inconsistent with the stated motivation for PPML (“handles zeros naturally”).

Even if inference were corrected, dropping zeros changes the estimating sample and can mechanically reduce uncertainty.

---

## 2. Evidence of severe overdispersion (why Poisson MLE SE are too small)

From your saved model diagnostics:

- **Full gravity PPML**: Pearson \(\chi^2 \approx 2{,}630{,}523\) on \(df \approx 2{,}676\)
- **Simple PPML**: Pearson \(\chi^2 \approx 8{,}262{,}772\) on \(df \approx 2{,}678\)

That implies a dispersion factor:

\[
\hat{\phi} \;=\; \frac{\chi^2_{\text{Pearson}}}{df_{\text{resid}}}
\]

- Full model: \(\hat{\phi} \approx 983\)
- Simple model: \(\hat{\phi} \approx 3{,}085\)

If you were doing a quasi-Poisson adjustment, SE would inflate roughly by \(\sqrt{\hat{\phi}}\):

- Full model: \(\sqrt{983} \approx 31\)
- Simple model: \(\sqrt{3085} \approx 56\)

This is not “a small technicality”: it explains why you see diaspora SEs on the order of \(0.001\) even though the data are extremely noisy.

**Bottom line:** you cannot report Poisson MLE SE in this setting and expect reviewers to accept them.

---

## 3. Recommended inference strategy (what to report)

### 3.1 Minimum acceptable: heteroskedasticity-robust SE (sandwich)

Report at least HC1/HC3 robust SE for PPML:

- This treats Poisson as a quasi-MLE and remains valid even when Var\((Y|X)\neq\mu\).
- In gravity applications with heteroskedasticity and many zeros, robust SE are standard.

### 3.2 Better (and usually expected): clustered SE

The OD cells \( (o,s) \) are not independent. Two obvious correlation structures:

- **Destination-state clustering:** unobserved state-level shocks, processing differences, and “gatewayness” induce correlation across origins within a state.
- **Origin-country clustering:** origin-specific shocks/policies induce correlation across states within an origin.

**Recommended:** report **two-way clustered** SE by destination state and origin country.

If two-way clustering is difficult in your current Python workflow, a common compromise is:

- report state-clustered and origin-clustered SE as separate robustness columns, and
- interpret inference conservatively (use the larger SE as a “worst-case” bound).

### 3.3 Bootstrap (very defensible for a forecasting paper)

Because the paper’s goal is forecasting (not hypothesis testing), bootstrap is attractive:

- resample origins and/or states (block bootstrap),
- optionally draw ACS diaspora stocks from their MOE-implied distribution inside each replication,
- refit PPML,
- summarize the distribution of \(\hat{\beta}\) and predicted ND flows.

This produces uncertainty intervals that align with “planning under uncertainty” rather than fragile p-values.

---

## 4. How to implement robust / clustered inference (practical notes)

### 4.1 Robust SE in statsmodels (conceptual)

In statsmodels you can obtain sandwich SE via robust covariance results after fitting the GLM.

Pseudo-code:

```python
model = sm.GLM(y, X, family=sm.families.Poisson())
res = model.fit()

# Heteroskedasticity-robust (sandwich) SE
res_hc1 = res.get_robustcov_results(cov_type="HC1")

# Clustered SE by state (destination)
res_cl_state = res.get_robustcov_results(cov_type="cluster", groups=df["state"])

# Clustered SE by origin
res_cl_origin = res.get_robustcov_results(cov_type="cluster", groups=df["origin"])
```

For **two-way clustering**, use a multi-way cluster covariance routine (e.g., Cameron–Gelbach–Miller). In statsmodels this is typically done via sandwich covariance utilities (implementation detail depends on your installed version).



#### Two-way clustered covariance (state and origin) in statsmodels

Statsmodels provides a helper for two-way clustering:

```python
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups
import numpy as np

model = sm.GLM(y, X, family=sm.families.Poisson())
res = model.fit()

cov_tw, cov_state, cov_origin = cov_cluster_2groups(res, df["state"], df["origin"])
se_tw = np.sqrt(np.diag(cov_tw))

# diaspora SE under two-way clustering (example)
i = list(res.params.index).index("log_diaspora")
diaspora_se_tw = se_tw[i]
```

This gives you the two-way clustered variance estimator that reviewers expect for state–origin gravity cells.

### 4.2 Keep zeros (critical)

Before any inference fix matters, correct the data step:

- build the full \(o \times s\) grid for the set of origins you keep,
- set missing DHS flows to **0**,
- keep diaspora stock as 0 where appropriate (or keep as missing only if ACS does not report the category).

---

## 5. Handling ACS measurement error (MOE)

Diaspora stocks from ACS are **estimates** with sampling uncertainty. Treating them as fixed regressors:

- overstates precision,
- can attenuate \(\hat{\beta}\) (classical measurement error),
- and can mislead forecasting intervals.

### 5.1 Minimal acknowledgement (must be in text)

State explicitly that diaspora stocks are ACS estimates with MOE and that regression uncertainty does not fully incorporate that sampling error.

### 5.2 Preferred: simulation / multiple imputation

A practical approach:

1. Convert MOE to an approximate standard error:
   - ACS MOE is often reported at 90% confidence; if so, \(SE \approx \text{MOE}/1.645\).
2. For \(b=1,\dots,B\):
   - draw \(\tilde{Diaspora}_{os}^{(b)} \sim \mathcal{N}(Diaspora_{os}, SE_{os}^2)\), truncated at 0,
   - re-estimate PPML,
   - store \(\hat{\beta}^{(b)}\) and predicted ND flows.
3. Report simulation-based uncertainty (quantiles) for coefficients and forecasts.

This “propagates” ACS sampling error into your forecasting uncertainty.

---

## 6. What to change in the paper’s reporting

1. Replace “SE” and p-values from default Poisson MLE with:
   - robust SE (HC1/HC3) and
   - clustered SE (state, origin, and preferably two-way).
2. Add an overdispersion diagnostic line:
   - report \(\chi^2_{\text{Pearson}}/df\) to justify robust inference.
3. For the forecasting audience:
   - emphasize effect sizes and predictive performance,
   - present prediction intervals from bootstrap/simulation rather than “highly significant” p-values.
