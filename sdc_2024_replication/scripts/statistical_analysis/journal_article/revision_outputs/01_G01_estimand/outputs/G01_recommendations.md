# G01 Recommendations: Estimand & Identification Strategy Review

## Executive summary

This revision will get dramatically easier once you **commit to one forecasting object** and treat everything else as *inputs* to that object. My recommendation is to define the paper’s forecast target as **Census PEP net international migration to North Dakota** (annual, in persons; the “international migration” component of change used in the state population estimates). Then explicitly position DHS LPR admissions, RPC refugee arrivals, and ACS foreign-born stock as **(i) component-flow proxies, (ii) predictors/feature builders, and (iii) validation/triangulation tools**—not competing outcomes. With the estimand disciplined, you can keep the multi-method pipeline, but you must **ring-fence causal language**: strengthen DiD/shift-share inference where credible, and reframe synthetic control and “network effects” as primarily *associational/predictive* unless you add stronger identification.

---

## Prioritized recommendations

1. **[MUST] Add an “Estimand & Measurement” subsection and commit to a single forecast target.**
   - Define the target variable, unit, and time base in 3–5 sentences (ready-to-paste text below).
   - State explicitly: *PEP is the dependent variable for projection practice; the other sources inform decomposition, predictors, and scenario design.*

2. **[MUST] Insert a one-page source mapping matrix (table) and enforce consistent labels everywhere.**
   - Every table/figure that uses DHS/RPC/ACS must label **flow vs. stock**, **net vs. gross**, **FY vs. (PEP) year**, and whether it is **outcome vs predictor vs validation**.
   - Add a short “Conceptual alignment” paragraph: why these objects are related but not interchangeable.

3. **[MUST] Make the paper’s organizing logic forecasting-first, with causal modules explicitly subordinate.**
   - One sentence in the intro: *The core contribution is forecasting under short panels; causal analyses are used only to quantify policy responsiveness and to stress-test scenarios.*
   - Move any “policy caused X” headline framing to “policy-associated shifts that inform scenarios.”

4. **[MUST] Travel Ban DiD (Module 7): fix functional form + inference, and show the event study.**
   - Replace the log(y+1) OLS DiD with a **count model** (PPML with year and nationality fixed effects is the standard workhorse for migration counts with zeros).
   - Use **cluster-robust SE at the nationality level** (and consider a wild cluster bootstrap for robustness given the panel structure).
   - Include an **event-study plot** with confidence intervals; do not rely only on a single pre-trend test statistic.
   - Treatment timing: show sensitivity for **2017 vs. 2018** (treat 2017 as “transition,” or drop 2017, and show both).

5. **[MUST] Synthetic control for ND: stop presenting it as an untreated counterfactual.**
   - A national policy shock means there is no truly untreated donor pool; SCM cannot be interpreted as “ND without the policy.”
   - Recommendation: **keep only if reframed** as *descriptive peer benchmarking / divergence detection* (e.g., “post-2017 divergence from a pre-2017 peer-based synthetic”), or **drop** if you need space.
   - If you insist on causal framing, you must redesign as **exposure-based** (high vs low exposure states) with appropriate generalized synthetic control/synthetic DiD logic.

6. **[MUST] Bartik shift-share: specify construction in full and use modern shift-share inference.**
   - Clearly state baseline year, shares, shocks, and whether shocks are **leave-one-out**.
   - The reported first-stage strength (F ≈ 22.5) is encouraging, but inference must respect that shift-share induces correlated errors across states.
   - Use a **shift-share-robust inference procedure** (AKM/BHJ-style), or at minimum show robustness to two-way clustering + placebo/pre-trend checks.

7. **[MUST] Gravity “network effects”: treat them as predictive associations, not causal elasticities.**
   - Diaspora stock is mechanically related to past flows and is likely endogenous to unobservables (selection, policies, labor demand, measurement).
   - Revise language from “network effects increase migration” to “diaspora measures strongly predict migration patterns, consistent with network mechanisms but not uniquely identifying them.”
   - Add a short limitations paragraph: cross-section limits (FY2023 DHS), diaspora endogeneity, and ACS measurement error.

8. **[SHOULD] Add a transparent decomposition narrative: PEP net international migration = refugee-related + other international migration + residual.**
   - You do *not* need perfect accounting—just a defensible conceptual decomposition that explains why RPC and DHS are in the pipeline.
   - This also resolves reviewer confusion about “four different migration objects.”

9. **[SHOULD] Make fiscal-year vs PEP-year harmonization explicit and systematic.**
   - Use one consistent crosswalk (see `G01_source_mapping.md`) and label it in captions.
   - Add sensitivity: re-run key correlations/models with ±1-year shifts to show conclusions aren’t an artifact of timing.

10. **[SHOULD] Integrate ACS MOE into uncertainty where ACS is used as a predictor.**
   - At minimum: show robustness when weighting by inverse-variance or when simulating diaspora stock within MOE bounds.

11. **[COULD] Add small DAGs for the causal modules (appendix) to make assumptions auditable.**
   - Keep it minimal and module-specific (travel ban; shift-share). Do not turn the whole paper into a causal-graph treatise.

12. **[AVOID] Changes that will harm the paper.**
   - Do not keep multiple “headline outcomes” (PEP vs refugees vs LPR vs foreign-born) without an explicit hierarchy.
   - Do not claim a travel-ban “effect on total international migration” if the outcome is refugee arrivals by nationality.
   - Do not describe synthetic control as creating an untreated counterfactual for a national shock.
   - Do not describe diaspora coefficients as causal “network elasticities” without an identification story.

---

## Ready-to-paste estimand statement text

Below is text you can drop near the start of Methods (suggested header: **“Estimand & Measurement”**).

> **Forecast target (estimand).** Let \(Y_t\) denote **North Dakota’s annual net international migration** in year \(t\), measured as the Census Bureau Population Estimates Program (PEP) “international migration” component of change for North Dakota (persons; net of outflows). We treat \(Y_t\) as the **primary dependent variable** for all forecasting and projection scenarios in this paper. Our forecasting goal is to estimate the **predictive distribution** of \(Y_{t+h}\) for horizons \(h \in \{1,\ldots,H\}\) (2025–2045), conditional on information available through year \(t\).
> **Secondary data sources.** DHS lawful permanent resident (LPR) admissions, RPC refugee arrivals, and ACS foreign-born stock are not alternative targets; they are used to (i) characterize and decompose international migration, (ii) construct predictors (e.g., diaspora measures), and (iii) triangulate mechanisms and scenario sensitivity.

Optional footnote (recommended):

> *PEP “year \(t\)” corresponds to an annual component-of-change interval used in producing July 1 population estimates (a demographic year rather than a strict Jan–Dec calendar year). We refer to it as “year \(t\)” for readability and explicitly harmonize fiscal-year sources in Section X.*

---

## How each secondary data source relates to the estimand

- **RPC refugee arrivals (administrative counts, FY):** a **gross inflow subcomponent** of \(Y_t\). Used to quantify how refugee inflows respond to policy regimes and to parameterize the “refugee-driven” component in scenarios (not to redefine the forecasting target).

- **DHS LPR admissions (administrative counts, FY; intended residence):** another **gross inflow proxy** for non-refugee permanent immigration, used to estimate origin-composition and gravity-style relationships. Not net, not a full measure of total inflow, and subject to secondary migration.

- **ACS foreign-born stock (survey estimates with MOE):** a **diaspora/stock proxy** used to construct network predictors and to validate the plausibility of origin concentration. It is not a flow and cannot substitute for \(Y_t\) without strong assumptions.

---

## Language fixes for causal claims (examples you can copy)

### Travel Ban DiD

- Replace: “The Travel Ban reduced international migration to ND by 75%.”
- With: “Refugee arrivals to North Dakota from Travel-Ban nationalities declined by roughly 75% relative to other nationalities after the policy period. A causal interpretation requires the parallel-trends assumption and robustness to alternative treatment timing and count-model specifications.”

### Synthetic control

- Replace: “Synthetic control estimates the post-2017 treatment effect on ND international migration.”
- With: “A peer-weighted synthetic series closely matches ND pre-2017 and highlights a post-2017 divergence. Because the underlying policy environment changes nationally, we interpret this divergence as descriptive benchmarking rather than a clean untreated counterfactual.”

### Gravity / diaspora “network effects”

- Replace: “Diaspora networks increase migration flows (elasticity = 0.11).”
- With: “Across origins and destinations, flows are positively associated with diaspora stock measures. This pattern is consistent with network mechanisms but may also reflect selection, policy, and the mechanical persistence of past flows embedded in stock variables.”

### Bartik shift-share

- Replace: “Our Bartik IV identifies the causal impact of immigration shocks on ND migration.”
- With: “We use a shift-share (Bartik) instrument based on baseline origin shares and national origin-specific shocks. Interpreting the IV coefficient causally requires that baseline shares are predetermined and that national shocks are as-good-as-random with respect to contemporaneous state-level unobservables; inference follows modern shift-share-robust procedures.”
