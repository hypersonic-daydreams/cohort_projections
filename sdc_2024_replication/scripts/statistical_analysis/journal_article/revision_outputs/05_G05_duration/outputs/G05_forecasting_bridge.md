# Duration Analysis → Forecasting Bridge (Refugee “Waves”)

## Why the duration module matters for forecasting (and how to stop it being “orphaned”)

Module 8’s survival analysis is not merely descriptive. It is a **predictive model for the persistence of a regime**: once a state–origin series enters an elevated-arrivals “wave,” the key forecasting question becomes:

> **How long will this elevated regime persist (and with what probability), given what we can already observe about the wave?**

Survival analysis answers exactly that question by modeling the **distribution of wave duration** \(D\) and the **hazard of wave termination** \(h(\tau)\) at wave age \(\tau\). Once connected to a flow model, it becomes operational: the hazard model turns into **a time-varying probability that the wave remains active**, which directly governs expected future flows and prediction intervals.

This bridge is particularly valuable for North Dakota because small states often exhibit *baseline stability punctuated by intermittent shocks*. A standard ARIMA or trend model tends to “smear” shocks forward or backward; a wave-duration model provides a principled way to treat shocks as **temporary, stochastic spells** with empirically grounded persistence.

**Empirical anchor (pooled evidence):** Across 940 waves (56 nationalities × 48 states), the Kaplan–Meier estimate implies a **median wave duration of ~3 years** (mean ~3.54), with ~10% right censoring. The Cox model shows good discrimination (concordance ~0.77). These values support a forecasting interpretation: *most waves die out within a few years, but persistence varies systematically with wave features.*

---

## What a hazard model “means” for forecasting practice

### Translate the Cox model into a forecasting object

The fitted Cox proportional hazards model implies:

\[
h(\tau \mid x) = h_0(\tau)\,\exp(\beta^\top x)
\]

- \(h(\tau \mid x)\): instantaneous risk the wave ends at age \(\tau\), conditional on surviving to \(\tau\).
- \(h_0(\tau)\): baseline hazard over wave age (a nonparametric function).
- \(x\): covariates describing the wave (intensity, region, etc.).
- \(\beta\): estimated log-hazard effects (hazard ratios \(=\exp(\beta)\)).

For forecasting, the key derived quantity is the **survival function**:

\[
S(\tau \mid x)=\Pr(D>\tau \mid x)=\exp\left(-H_0(\tau)\exp(\beta^\top x)\right),
\qquad
H_0(\tau)=\int_0^\tau h_0(u)\,du.
\]

This is not “just statistics.” It is a **probability model for how long a detected wave will last**.

### Operational interpretation of each significant hazard ratio

Below, “hazard” means the risk a wave *ends* (i.e., falls below the wave threshold and stays there).

#### 1) `log_intensity`: HR ≈ 0.41 (strong persistence effect)

- **Meaning:** A one-unit increase in \(\log(\text{intensity})\) multiplies the termination hazard by 0.41 (≈59% reduction).
- **Forecast translation:** Higher-intensity waves are **expected to last longer**, and your forecast should place more probability mass on longer durations.
- **Practical intuition:** Because the predictor is \(\log(\text{intensity})\), the effect is multiplicative in the *intensity ratio* \(I\) (peak/baseline). The hazard scales approximately as \(I^{-0.887}\).
  - Doubling intensity (\(I \to 2I\)) implies hazard multiplier \(2^{-0.887} \approx 0.54\).
  - A 10× intensity increase implies hazard multiplier \(\approx 0.13\).

**When a new wave begins:** You don’t know the eventual peak yet, but you can compute an **initial intensity proxy** \(I_0 = y_{t_0}/b\) (first-year arrivals relative to baseline). Use \(I_0\) (or a nowcast of peak intensity) to initialize the survival distribution, then update as additional years arrive.

#### 2) `early_wave`: HR ≈ 1.36 (early-era waves ended sooner)

- **Meaning:** Waves starting in the “early” era have ≈36% higher termination hazard than later waves.
- **Forecast translation:** Treat this as a **cohort/era effect** (policy regime, measurement regime, resettlement system differences). For ND forecasting in 2010–2024 and beyond, `early_wave` is likely 0, so this mainly prevents you from over-generalizing early-period dynamics to the present.
- **When a new wave begins:** Set `early_wave=0` for modern waves; if you backcast or evaluate older eras, include it.

#### 3) `peak_arrivals`: HR ≈ 0.66 (bigger peaks persist longer)

- **Meaning:** Holding other features constant, larger peaks reduce the hazard that the wave ends. In the implementation used here, `peak_arrivals` was scaled by 1,000 for numerical stability, so HR≈0.66 is interpreted as “per additional 1,000 peak arrivals.”
- **Forecast translation:** If a wave is on track to peak high, it should receive a **longer expected duration**.
- **ND caution:** Because ND counts are often small, the “per 1,000” scaling can make the marginal effect modest in ND unless peaks are large; treat this as directional evidence rather than a precise ND lever.

**When a new wave begins:** Treat peak as *latent*. Use early growth + first-year level to nowcast a peak distribution. Update after the peak becomes observable.

#### 4) `nationality_region_Americas`: HR ≈ 1.71 (shorter waves vs reference)

- **Meaning:** For Americas-origin waves, hazard is ≈71% higher than the reference region (in this coding the omitted category is effectively the baseline, typically “Africa”).
- **Forecast translation:** Conditional on entering a wave, Americas-origin waves are **less persistent** on average; forecasts should decay earlier and carry less long-tail mass.

#### 5) `nationality_region_Europe`: HR ≈ 1.57 (shorter waves vs reference)

- **Meaning:** Europe-origin waves have ≈57% higher hazard (shorter spells) relative to the reference region.
- **Forecast translation:** Similar to Americas: shorter persistence, quicker reversion to baseline.

#### 6) `nationality_region_Other`: HR ≈ 1.66 (shorter waves vs reference; interpret cautiously)

- **Meaning:** Waves in the catch‑all “Other” origin group have a higher termination hazard (shorter expected spells) relative to the reference region.
- **Forecast translation:** If an origin is not mapped to a main region and lands in “Other,” the default forecast should **tilt toward shorter persistence**, *but with extra uncertainty* because “Other” is a heterogeneous bin.
- **Implementation note:** In forecasting code, it can be better to (i) improve the region mapping so fewer origins fall into “Other,” or (ii) treat “Other” as a **high-variance** category rather than a crisp point effect.



---

## A coherent forecasting framework that uses duration analysis

### The “wave-as-spell” view

Treat each origin–destination series as switching between:

- **Baseline regime** (routine arrivals, driven by slow-moving predictors / trend)
- **Wave regime** (elevated arrivals due to shocks: conflict, policy, resettlement scaling)

A “wave” is a **spell** of the wave regime. Survival analysis gives the spell-length distribution.

### The full pipeline

1) **Wave detection (filtering / regime identification)**
   Detect when arrivals have plausibly moved from baseline to wave regime. This can be deterministic (threshold rule) or probabilistic (state-space/HMM).

2) **Duration prediction (spell-length forecasting)**
   Once in wave regime, use the Cox model to infer a predictive distribution for total wave duration \(D\) and, more importantly, **remaining duration** \(R\) given survival to date.

3) **Flow projection (shape × persistence)**
   Convert duration/persistence into expected annual flows by combining:
   - a baseline flow model (ARIMA/trend/VAR/ensemble)
   - a wave *shape* model (initiation → peak → decline), scaled by amplitude
   - survival probabilities that determine whether the wave contributes in each future year

4) **Uncertainty propagation**
   Carry uncertainty from:
   - detection (are we “in wave” yet?)
   - duration (survival distribution + coefficient uncertainty)
   - amplitude/shape (peak, time-to-peak, decline steepness)
   - baseline forecast uncertainty
   into a single predictive distribution for annual flows.

---

## Lifecycle patterns as the missing link between “duration” and “flows”

Duration alone answers “how long,” but forecasting needs “how much each year.” The lifecycle analysis provides empirical priors for *when* peaks occur and how the wave decays:

- mean time to peak ≈ 2.2 years
- initiation phase share ≈ 28% of duration
- decline phase share ≈ 35% of duration

A practical forecasting interpretation:

- **Year 1–2 after wave start:** rising phase (initiation)
- **Around year ~2:** peak
- **After peak:** decline over the remaining years (decline phase)

This supports a parsimonious wave shape family (piecewise linear, beta-shaped, or gamma-shaped over normalized time) that maps a predicted duration \(D\) into a *trajectory* \(g(1),\dots,g(D)\) and therefore into annual forecasts.

---

## What to do about the North Dakota data limitation (few ND waves; short series)

The hazard model was estimated on 940 state–origin waves, which is excellent for learning generic duration regularities, but ND alone has few complete waves over 2010–2024. That implies:

1) **Do not treat \(\hat\beta\) as “the ND truth.”** Treat them as *portable tendencies* with uncertainty.
2) **Borrow strength via partial pooling:** operationally, use the pooled Cox model as an **informative prior** or “default hazard,” and include a wide ND-specific adjustment (frailty/random effect) that can be updated as ND accumulates waves.
3) **Prefer probabilistic outputs:** communicate *survival probabilities* and *prediction intervals* rather than crisp “expected duration = 3.2 years.”

The paper can frame this as: *“We use pooled wave dynamics to calibrate the persistence of shocks in ND forecasts, acknowledging limited ND-specific wave history.”*

---

## How this plugs into the broader forecasting framework (Module 9 scenarios)

Scenario projections already propagate uncertainty via Monte Carlo. The wave-duration module becomes an additional Monte Carlo subroutine:

- If a scenario implies a conflict/policy shock that triggers refugee inflows, represent it as one or more **waves** with:
  - initiation time distribution (scenario-driven)
  - amplitude distribution (scenario-driven or data-driven)
  - **duration distribution** (hazard model-driven)
  - trajectory shape (lifecycle-driven)

Then the scenario’s annual flows are the baseline model plus simulated wave contributions. This converts “scenario narratives” into *stochastic processes* with empirically grounded persistence.

---

## Minimal “in-paper” positioning (to fix the orphan problem)

A clean narrative move is:

> “We model refugee inflow surges as temporary spells (‘waves’). Survival analysis yields a predictive distribution for wave persistence conditional on observed wave features. We embed this distribution into the forecasting pipeline by converting wave persistence into year-ahead survival probabilities and sampling remaining wave durations inside the Monte Carlo scenario engine.”

That one paragraph is the conceptual bridge. The rest is math and implementation.
