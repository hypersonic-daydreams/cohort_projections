# G01 Specifications: Estimand, Identification Assumptions, and Module Modifications

## 1. Formal estimand definition

### 1.1 Primary forecasting estimand

Let \(Y_t\) denote **North Dakota net international migration** in year \(t\), measured using the **Census Bureau Population Estimates Program (PEP)** “international migration” component of change for North Dakota (persons; net). This is the object the paper projects.

A forecasting estimand should be stated as a *predictive* quantity. A clean definition is:

\[
\mathcal{P}_{t,H}(\cdot) \equiv \Pr\left( Y_{t+1},\ldots,Y_{t+H} \in \cdot \mid \mathcal{F}_t \right),
\]

where \(\mathcal{F}_t\) is the information set available through year \(t\) (past PEP values and any covariates used), and \(H\) is the projection horizon (e.g., \(H=21\) for 2025–2045 when \(t=2024\)).

You may also report a point functional (median/mean forecast path):

\[
\hat{y}_{t+h} \equiv \mathbb{E}[Y_{t+h} \mid \mathcal{F}_t] \quad \text{or} \quad \operatorname{median}(Y_{t+h}\mid \mathcal{F}_t),
\]

but in a demographic methods venue the *distribution* (intervals, calibration) is often the real contribution.

### 1.2 Secondary objects and their roles (not competing estimands)

Define these explicitly to prevent “target drift”:

- \(R^{\text{RPC}}_{s,t}\): Refugee arrivals to state \(s\) (or ND) in fiscal year \(t\) from RPC. **Gross inflow**, fiscal-year timing, initial placement.

- \(L^{\text{DHS}}_{s,o,t}\): DHS LPR admissions to state \(s\) from origin \(o\) (FY; often only one cross-section in your paper). **Gross inflow**, intended residence.

- \(B^{\text{ACS}}_{s,o,t}\): ACS foreign-born **stock** in state \(s\) from origin \(o\) (survey estimate with MOE). **Stock**, not a flow.

Clarify: these are used to build predictors \(X_t\), to motivate decomposition, and to stress-test scenario assumptions for \(Y_t\).

---

## 2. Identification assessment by causal claim (Module 7 and related)

### 2.1 Travel Ban DiD (refugee arrivals by nationality)

**What is the estimand here?**
This design targets an average post-policy change for treated nationalities, within ND:

\[
\tau \equiv \mathbb{E}\left[ Y_{c,t}(1) - Y_{c,t}(0) \mid c \in \mathcal{T},\ t \ge t_0 \right],
\]
where \(c\) indexes nationality, \(\mathcal{T}\) is the set of Travel-Ban nationalities, and \(t_0\) is the chosen “post” onset.

**Core assumptions needed for causal interpretation:**

1. **Parallel trends (nationality-level):** In the absence of the policy, treated and control nationalities’ arrivals to ND would have evolved similarly over time (in levels or in the chosen link scale).
2. **No anticipatory behavior / correct timing:** No treatment effect before \(t_0\) (or explicitly model anticipation/transition).
3. **Stable composition and measurement:** Nationality definitions and measurement do not change differentially at \(t_0\).
4. **SUTVA / no interference across nationalities:** Outcomes for one nationality are not affected by treatment status of another (hard in migration: substitution is plausible).
5. **Correct outcome model and inference:** Count outcomes with zeros require an estimator robust to heteroskedasticity and zero mass (PPML is a strong default).

**Assessment of credibility (given your current implementation):**

- Your current approach uses log(arrivals+1) with two-way FE and HC3 SE. With many zeros and over-dispersion, this is fragile.
- “Parallel trends supported” based on a single pre-trend test is **not enough** for top-tier review; you need an event-study figure and robustness.

**Treatment timing (2017 vs 2018):**
The Travel Ban begins in Jan 2017 (calendar), but your analysis sets \(t_0=2018\) (first “fully treated” year). This can be defensible *if* you (i) justify the mapping given FY timing, and (ii) show sensitivity:
- treat 2017 as post,
- treat 2017 as transition and drop it,
- keep 2018 as post (your current choice).

**Recommended modifications:**

- Use **PPML DiD** with nationality and year FE and **cluster SE by nationality**.
- Include an **event study** and report: (i) pre-period coefficients; (ii) joint test; (iii) robustness to alternative \(t_0\).
- Add **placebo treated groups** (pseudo-treated nationalities) and a “negative control” policy period.

---

### 2.2 Synthetic control for ND (state-level PEP international migration rate)

**What is the “treatment”?**
As currently written, the implicit treatment is “post-2017 policy regime shift.” But this is a national regime shift that affects all states, so there are no truly untreated donors.

**Why standard synthetic control is not identified here (as causal):**

- Synthetic control requires that the donor pool represents the counterfactual evolution of the treated unit *in the absence of treatment*.
- When the shock is national, donors are also “treated,” so the synthetic series cannot be interpreted as “ND without the policy.”

**What it *can* be used for:**

- A descriptive benchmark: “ND vs a peer-weighted synthetic that matches pre-2017.”
- A forecasting device: borrowing strength from similar states to produce a baseline trajectory.
- A diagnostic for **structural divergence** (not causal attribution).

**Keep / modify / drop recommendation:**

- **Modify (preferred):** Keep it, but reframe as descriptive benchmarking and remove causal language (“treatment effect”).
- **Drop (acceptable):** If space is tight, drop rather than defend an indefensible causal interpretation.
- **Full redesign (only if you must claim causality):** Move to an **exposure-based** design (states stratified by pre-period exposure to the policy) and use generalized synthetic control / synthetic DiD logic with explicit identifying assumptions.

---

### 2.3 Gravity model network effects (Module 5)

**What is estimated?**
A reduced-form relationship between flow outcomes and diaspora stock proxies (plus other gravity terms).

**Causal status:**
As written, these coefficients should be treated as **associational/predictive**. Diaspora stock is:
- a function of prior flows (mechanical endogeneity),
- correlated with unobserved state and origin traits (policies, labor demand, selection),
- measured with error (ACS MOE), which can bias estimates.

**Recommended language and presentation changes:**

- Replace “network effect” with “diaspora association” or “diaspora predictor.”
- Interpret coefficients as *predictive elasticities* (“associated with”) and explicitly list confounding channels.
- If you want a causal claim, you need a credible instrument or quasi-experiment for diaspora stock (usually outside the scope of this paper).

---

### 2.4 Bartik shift-share (Module 7)

**Construction (your current summary):**

- Baseline shares: \(s_{i,o,2010}\) = state \(i\)’s share of origin \(o\) at baseline (2010).
- National shocks: \(g_{o,t}\) = national change for origin \(o\) at time \(t\).
- Instrument: \(Z_{i,t} = \sum_o s_{i,o,2010} g_{o,t}\).

**Identification assumptions:**

1. **Predetermined shares:** \(s_{i,o,2010}\) is fixed before the shocks and not chosen in response to future outcomes.
2. **Exogenous shocks:** national-origin shocks \(g_{o,t}\) are uncorrelated with state-level unobservables affecting the outcome (conditional on controls).
3. **No “reflection” / leave-one-out:** if national shocks include the state itself, they can mechanically correlate with the error; leave-one-out shocks mitigate this.
4. **Many small shocks / appropriate aggregation:** inference should account for common shocks inducing cross-state correlation.

**Instrument strength:**
A first-stage F ≈ 22.5 is a good sign, but with shift-share the relevant issue is not only weak instruments—it's **mis-sized standard errors** if you treat observations as independent.

**Recommended inference:**

- Use a modern shift-share inference approach (AKM/BHJ-style).
- Report robustness with two-way clustering (state and year) as a *secondary* check, not the main inference.

---

## 3. Recommended modifications by module (to align with the estimand)

### Module 1 (Descriptive + concentration)

- State explicitly that these modules describe \(Y_t\) and its uncertainty/volatility.
- When using ACS concentration and LQs: label them as composition diagnostics and feature construction for forecasting, not separate targets.

### Module 2 (Time series + breaks)

- Frame ARIMA/structural breaks as baseline forecast models for \(Y_t\).
- Avoid causal language (“policy-driven breaks”) unless tied to a causal design; use “breaks coincide with…”

### Module 3 (Panel + network)

- Clarify that panel models borrow strength across states to improve predictive performance for ND.
- Network/stock persistence should be positioned as a predictor dynamic (useful for scenario timing), not a causal mechanism estimate.

### Module 4 (Regression extensions)

- Treat robust/quantile/beta regressions as sensitivity checks and distributional modeling tools for \(Y_t\) or its share/rate transformations.
- Avoid “significance hunting” language; emphasize effect sizes and predictive relevance.

### Module 5 (Gravity + networks)

- Treat gravity models as a way to generate origin weights / predictors (e.g., expected composition), not causal network elasticities.
- Explicitly acknowledge diaspora endogeneity and ACS measurement error.

### Module 6 (Machine learning)

- Ensure evaluation respects time ordering (rolling-origin or blocked CV).
- Present ML as predictive benchmarking for \(Y_t\) (accuracy + calibration), not as “discovering causes.”

### Module 7 (Causal inference)

- Clearly label the causal outcomes (refugee arrivals; state PEP rates) as **auxiliary** to the forecasting estimand.
- Implement the DiD/shift-share fixes above; reframe or drop synthetic control causal claims.

### Module 8 (Duration / wave analysis)

- Tie wave duration results directly to scenario generation for the refugee component (expected persistence of refugee inflows).

### Module 9 (Scenario modeling)

- Define each scenario as an assumption set about \(Y_t\) (or its components) and ensure arithmetic matches stated assumptions.
- Make explicit how causal-module estimates enter scenarios (if they do); otherwise, keep causal results as contextual only.

---

## 4. Structural Causal Model (SCM/DAG) framework decision

### Recommendation: **Include a minimal DAG (appendix), not a full SCM reframe**

A small DAG is useful here because it forces you to be honest about what is identified and what is merely predictive. But a full SCM layer across the whole nine-module pipeline will likely distract from the forecasting contribution and invite reviewers to demand more causal machinery than you intend to supply.

**How to implement:**

- Include **one DAG per causal module** (Travel Ban DiD; Bartik IV).
- Keep nodes high-level: Policy regime, national admissions, state resettlement capacity, diaspora, local economy, and the measured outcome.
- Use the DAG caption to list the specific identifying assumptions (parallel trends; exogenous shocks; predetermined shares).

**Key nodes and edges (suggested):**

- Travel Ban: Policy \(\rightarrow\) national refugee admissions/processing \(\rightarrow\) ND refugee arrivals; global conflict \(\rightarrow\) admissions and arrivals; state capacity \(\rightarrow\) arrivals.
- Bartik: baseline shares + national origin shocks \(\rightarrow\) predicted inflow \(\rightarrow\) state inflow; unobserved state shocks \(\rightarrow\) state inflow (assume orthogonality to predicted inflow given controls).
