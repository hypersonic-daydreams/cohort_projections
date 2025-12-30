# Revised Gravity Model Specification (FY2023 cross-section)

This note provides a revised, defensible gravity-model specification for **state-by-origin LPR admissions** in FY2023, designed primarily as a **forecasting / allocation tool** (not a causal design).

---

## 1. What “gravity” means here

Let:

- \(o\) index **origin countries**
- \(s\) index **U.S. destination states** (including DC if present)
- \(Y_{os}\) be **FY2023 DHS LPR admissions** with intended residence in state \(s\) and country of birth \(o\)

A canonical gravity model says flows are increasing in “mass” at origin and destination and decreasing in migration costs:

\[
\mathbb{E}[Y_{os}] \;=\; G \cdot O_o^{\alpha} \cdot D_s^{\gamma} \cdot C_{os}^{-\delta} \cdot N_{os}^{\theta},
\]

where:

- \(O_o\): origin “mass” (push/supply of migrants from \(o\))
- \(D_s\): destination “mass” (pull/attractiveness of state \(s\))
- \(C_{os}\): bilateral migration cost (distance, frictions)
- \(N_{os}\): **diaspora network stock** (a “network friction-reducer”)

For empirical work, gravity is almost always implemented in **multiplicative form** with a log link.

---



## 1.5 Review of the current three specifications (what they are and what they miss)

### Model 1 (Simple): \( \text{Flow}_{os} \sim \ln(1+\text{Diaspora}_{os}) \)

- **What it is:** a bivariate reduced-form relationship between admissions and diaspora stock.
- **What it is not:** a gravity model in the usual sense (no origin/destination mass terms, no bilateral cost terms, no multilateral resistance controls).
- **Main omissions:** origin heterogeneity (push), destination heterogeneity (pull), and bilateral migration frictions (distance/cost). Any “network elasticity” here will generally be upward biased because diaspora stock is correlated with both origin intensity and state attractiveness.

### Model 2 (Full): \( \text{Flow}_{os} \sim \ln(1+\text{Diaspora}_{os}) + \ln(\text{OriginMass}_o) + \ln(\text{DestMass}_s) \)

- **What it is:** a more gravity-like reduced-form regression that controls for broad origin and destination scale.
- **Key concern:** in your data, \(\text{OriginMass}_o\) and \(\text{DestMass}_s\) are **U.S. foreign-born stocks**, not origin-country population or destination population. That is fine for prediction, but it should be described precisely.
- **Main omissions:** (i) **distance/cost**; (ii) multilateral resistance (origin and destination fixed effects are usually preferred to “mass” proxies); (iii) correlated shocks across origins within a state and across states within an origin (inference).

### Model 3 (State FE): \( \text{Flow}_{os} \sim \ln(1+\text{Diaspora}_{os}) + \alpha_s \)

- **What it improves:** \(\alpha_s\) absorbs destination attractiveness and “destination mass” (including your \(\ln(\text{DestMass}_s)\)), which is closer to modern gravity practice than Model 2’s size proxy.
- **Main omissions:** (i) origin fixed effects \(\gamma_o\) (without them, the diaspora coefficient still partly reflects origin intensity); (ii) distance/cost; (iii) robust/clustered inference.
- **If you keep a three-model progression in the paper:** Model 3 should ideally become “State FE + Origin FE + distance” and be your baseline.


## 2. Recommended empirical specification (preferred)

### 2.1 PPML with origin and destination fixed effects

Use Poisson Pseudo-Maximum Likelihood (PPML) with high-dimensional fixed effects:

\[
Y_{os} \sim \text{Poisson}(\mu_{os}),
\]

\[
\mu_{os} \;=\; \exp\Big(
\underbrace{\alpha_s}_{\text{destination FE}}
\;+\;
\underbrace{\gamma_o}_{\text{origin FE}}
\;+\;
\beta \ln(1 + \text{Diaspora}_{os})
\;-\;
\delta \ln(\text{Distance}_{os})
\;+\;
X_{os}'\lambda
\Big).
\]

**Interpretation (forecasting):** \(\beta\) is a *cross-sectional diaspora elasticity* (approximately) among the included origin–state cells **in FY2023**, conditional on origin and destination fixed effects and the included bilateral controls.

### Why this is “more gravity-correct” than the current setup

- \(\alpha_s\) and \(\gamma_o\) absorb *destination and origin mass* plus all other time-invariant origin or state factors in this cross-section (economic scale, processing intensity, “gatewayness,” etc.).
- Including FE is the standard way to address **multilateral resistance** (the fact that destinations compete with each other and origins allocate across destinations).
- With FE, your bilateral coefficients (diaspora, distance) are interpreted off **within-origin and within-destination variation**, which is much harder to fake via omitted variables.

### 2.2 What to do with “mass” variables under FE

If you include \(\alpha_s\) and \(\gamma_o\), then:

- \(\log(\text{DestMass}_s)\) is collinear with \(\alpha_s\)
- \(\log(\text{OriginMass}_o)\) is collinear with \(\gamma_o\)

So: **drop the mass regressors** when you include the corresponding FE. In gravity, that’s a feature, not a bug.

---

## 3. Acceptable alternative specifications (when you want masses explicitly)

If (for expository reasons) you want to show “mass” elasticities rather than FE, you can estimate:

\[
\mu_{os} = \exp\Big(
\kappa
\;+\;
\beta \ln(1 + \text{Diaspora}_{os})
\;+\;
\phi \ln(\text{OriginMass}_{o})
\;+\;
\psi \ln(\text{DestMass}_{s})
\;-\;
\delta \ln(\text{Distance}_{os})
\Big),
\]

but you should treat this as a **reduced-form predictive regression**, not a structural gravity model with multilateral resistance fully handled.

---

## 4. Variable definitions (aligned to your data)

### Flow (dependent variable)

- \(Y_{os}\): DHS **LPR admissions** in FY2023, by intended state of residence \(s\) and country of birth \(o\).

### Diaspora stock (key predictor)

- \(\text{Diaspora}_{os}\): ACS 2023 estimate of foreign-born residents in state \(s\) born in country \(o\).
- Recommended transform: \(\ln(1+\text{Diaspora}_{os})\) to keep zero stocks.

**Note on interpretation:** With \(\ln(1+x)\), the coefficient is approximately an elasticity for large \(x\), but not exactly for small stocks. In text, call it a “semi-elasticity in \(\ln(1+\text{stock})\)” or report an elasticity evaluated at meaningful stock sizes.

### Origin “mass” and destination “mass”

If you must include them explicitly:

- \(\text{OriginMass}_o\): national total foreign-born in the U.S. from origin \(o\) (ACS 2023).
- \(\text{DestMass}_s\): state total foreign-born (ACS 2023).

**Caution:** These are **U.S.-resident stocks**, not origin-country populations or destination total populations. That’s okay for prediction, but it should be described precisely.

### Distance (recommended)

- \(\text{Distance}_{os}\): great-circle distance (km) between an origin-country centroid/capital and a destination-state population centroid (or a major metro).

**If you cannot include distance:** state clearly why. A defensible justification in this setting is that DHS LPR admissions include many **adjustments of status** where “distance from origin at time of admission” is not an actual migration cost. Even then, distance should be included as a robustness check if feasible.

---

## 5. Estimator choice (PPML vs OLS vs negative binomial vs zero-inflated)

### PPML (recommended default)

PPML is the standard in modern gravity work because:

- It models \(\mathbb{E}[Y_{os}|X]\) in multiplicative form (log link).
- It naturally accommodates **zero flows** (no need for \(\log(Y+1)\)).
- With **robust (sandwich) variance**, it remains valid under heteroskedasticity and overdispersion as a quasi-MLE.

### Log-linear OLS (not recommended)

OLS on \(\log(Y+1)\):

- changes the estimand and interpretation,
- is sensitive to heteroskedasticity (classic gravity bias),
- and forces arbitrary handling of zeros.

### Negative binomial (optional robustness)

Negative binomial can fit overdispersed counts, but it imposes a distributional form. It is useful as a robustness check; it is not “more correct” than PPML unless you are willing to defend that distribution.

### Zero-inflated models (generally unnecessary here)

Before moving to zero-inflation, first:

- keep the zeros in PPML,
- add appropriate FE,
- and check whether the remaining zeros look “structural” (true impossibility) versus just low intensity.

Zero-inflated models can be fragile and hard to interpret in gravity settings.

---

## 6. Practical implementation requirements (critical)

### 6.1 Do **not** drop zeros

A PPML gravity model should be estimated on the full set of relevant origin–state pairs:

- If DHS reports only positive flows, explicitly construct the full \(o \times s\) grid and set missing flows to **0**.
- Dropping \(Y_{os}=0\) changes the estimand (selection on the dependent variable) and undermines the “PPML handles zeros” rationale.

### 6.2 Use lagged stocks for forecasting

For an operational forecast, use \(\text{Diaspora}_{os,t-1}\) (or earlier) when possible to avoid contemporaneous mechanical correlation between stock and flow.

---

## 7. Recommended reporting and use in the paper

- Call \(\beta\) a **“cross-sectional diaspora–flow elasticity (association)”**.
- Report results with:
  - heteroskedasticity-robust SE,
  - clustered SE (destination and origin),
  - and (ideally) sensitivity to ACS measurement error via simulation.
- In the forecasting narrative, treat gravity as an **allocation / scaling tool** that helps map expected national-origin inflows into a plausible ND-origin composition, not as a causal mechanism test.
