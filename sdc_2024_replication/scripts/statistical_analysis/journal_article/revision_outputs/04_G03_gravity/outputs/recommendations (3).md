# Revision Recommendations: Gravity Model + Network Language

This note gives concrete, publication-facing revisions to make the gravity/network sections (i) technically defensible and (ii) aligned with the paper’s forecasting goal.

---

## 1. Reframe the gravity model as a predictive allocation model (not a causal design)

### What you can credibly claim from FY2023 only

With a single cross-section:

- You **can** estimate and use cross-sectional associations (predictive relationships) between diaspora stock and FY2023 LPR admissions across state–origin cells.
- You **cannot** identify a causal “network effect” (i.e., that increasing diaspora stock would *cause* higher admissions) because:
  - diaspora stock is endogenous to historical flows and state attractiveness,
  - contemporaneous stock and flow are mechanically related,
  - unobserved factors jointly affect both.

**Operational takeaway:** treat diaspora as a *predictor* for ND planning, not as a mechanism test.

---

## 2. Specification changes reviewers will expect

### 2.1 Stop dropping zeros (highest priority)

**Change:** estimate PPML on a dataset that includes \(Y_{os}=0\).

**Why:** PPML’s main advantage over log-OLS is exactly that it handles zeros. Dropping them is a self-inflicted wound.

**Paper language to add:** “We include zero admissions cells and estimate PPML on the full origin–state grid for the origin categories available in ACS.”

### 2.2 Add distance—or explicitly justify why it is not included

**Preferred:** include \(\ln(\text{Distance}_{os})\) as a bilateral cost term.

If you cannot include it, add a transparent justification specific to DHS LPR admissions:

- many LPRs are adjustments of status (distance at admission is not the migration cost),
- settlement may reflect within-U.S. mobility more than origin-to-destination travel distance.

Even then, distance should appear as a robustness check if feasible because it is standard in gravity and relatively easy to compute.

### 2.3 Use origin and destination fixed effects (multilateral resistance)

Replace “origin mass” and “destination mass” regressors with:

- origin-country FE and
- destination-state FE

This is the gravity-correct way to absorb mass and unobserved attractiveness. It also reduces omitted-variable bias in your diaspora coefficient.

---

## 3. Inference fixes (why your SE are being rejected on sight)

### 3.1 Use robust + clustered standard errors

Current SE are from default Poisson MLE and are inconsistent with the overdispersion in the data.

**Minimum columns for the main regression table:**

1. PPML + HC1 robust SE
2. PPML + cluster-by-state SE
3. PPML + cluster-by-origin SE
4. PPML + two-way cluster (state & origin) SE (preferred)

### 3.2 Add an overdispersion diagnostic in the text/table notes

Report Pearson \(\chi^2/df\) for each model. If it’s \(\gg 1\), say: “We therefore use sandwich/clustered SE throughout.”

### 3.3 Propagate ACS measurement error (forecasting-oriented)

Because diaspora stocks come with MOE, treat them as noisy:

- include a short paragraph noting measurement error / attenuation,
- add a simulation-based robustness check drawing diaspora from the MOE distribution and re-estimating PPML,
- and propagate that uncertainty into forecast intervals.

---

## 4. Recommended “repaired” model lineup for the paper

### Model A (baseline gravity-correct predictor)

PPML with origin FE + destination FE + diaspora + distance:

\[
\mu_{os} = \exp\Big(\alpha_s + \gamma_o + \beta \ln(1+\text{Diaspora}_{os}) - \delta \ln(\text{Distance}_{os})\Big)
\]

Report two-way clustered SE.

### Model B (add feasible bilateral or policy controls)

Add \(X_{os}\) only if you can define and defend it cleanly (examples):

- refugee-origin indicator / recent conflict indicator,
- English language indicator,
- visa/adjustment composition if available,
- origin-region categories.

### Model C (forecasting robustness)

- negative binomial as a check,
- or PPML with alternative transforms (e.g., \(\ln(\text{Diaspora})\) for diaspora>0 + a zero-stock dummy).

---

## 5. Suggested language revisions (before/after)

These are “drop-in” edits. Replace causal/mechanism wording with predictive wording.

### Example 1: Causal → predictive

**Before:** “Diaspora networks have a causal effect on new immigrant admissions.”
**After:** “In FY2023 cross-sectional data, larger state–origin diaspora stocks are **associated with** higher LPR admissions, conditional on origin and destination controls. We interpret this as a **predictive diaspora–flow relationship** consistent with network mechanisms, without claiming causality.”

### Example 2: Elasticity wording

**Before:** “A 1% increase in diaspora stock increases admissions by 0.10%.”
**After:** “The estimated diaspora coefficient implies that, **within the FY2023 cross-section**, a 1% higher diaspora stock is **associated with** roughly 0.10% higher LPR admissions (robust/clustered inference reported).”

### Example 3: “Network effects” section header

**Before:** “Causal Network Effects”
**After:** “Diaspora Stocks as Predictors of Destination Choice”

### Example 4: ND-specific interpretation

**Before:** “ND has weak network effects.”
**After:** “North Dakota’s estimated diaspora association is smaller than in gateway settings, consistent with ND’s smaller and more recent immigrant communities and the larger role of institutionally mediated channels (e.g., refugee placement).”

### Example 5: Cross-section limitation statement (add verbatim)

**Add:** “Because FY2023 is a single cross-section, we do not identify dynamic network effects or causal impacts of diaspora size. The gravity estimates are used as **predictive weights** for forecasting and scenario analysis.”

---

## 6. What would be needed for causal identification (optional, but be explicit)

If the paper wants to talk about causality, reviewers will expect at least one of:

1. **Panel flows** \(Y_{os,t}\) over multiple years + lagged diaspora \(Diaspora_{os,t-1}\) with origin and destination FE and clustered SE.
2. **IV / shift-share instrument** for diaspora stock (e.g., historical settlement shares interacted with national-level shocks).
3. **Natural experiments** (e.g., quasi-random refugee assignment, sudden policy changes affecting certain origins differentially) with credible identification.

If you cannot do this, the clean move is to *not* claim it.

---

## 7. Integration with the paper’s forecasting narrative

A clean way to connect Module 5 to the projection goal:

- Module 2/4 (time series) forecasts **total international migration to ND** (your estimand).
- Module 5 (gravity) provides **origin composition / allocation weights**:
  - given a forecast of national-origin totals (or scenarios), gravity allocates likely shares to ND based on diaspora and distance.
- Uncertainty from gravity (robust SE + ACS MOE simulation) feeds into scenario ranges for planning.

This turns gravity into a useful forecasting component and avoids the “identification limbo” trap.

---

## 8. Acknowledgment-of-limitations paragraph (drop-in text)

“Gravity estimates are obtained from a single FY2023 cross-section of DHS LPR admissions matched to ACS diaspora stocks. As such, the results should be interpreted as cross-sectional associations useful for prediction and allocation, not as causal network effects or dynamic multipliers. Inference is reported using heteroskedasticity-robust and origin/destination clustered standard errors because admissions counts are highly overdispersed and correlated within origins and destinations. Diaspora stocks are ACS estimates with margins of error; we propagate this sampling uncertainty using a Monte Carlo sensitivity analysis that draws diaspora stocks from their MOE-implied sampling distribution and recomputes coefficients and forecast intervals. Finally, DHS LPR admissions mix admission pathways (new arrivals and adjustments of status) and may not map cleanly to origin-to-destination migration costs; distance results are therefore interpreted as reduced-form spatial gradients rather than structural migration-cost elasticities.”
