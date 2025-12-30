# Implementation Recommendations for Claude Code

This document translates the conceptual/mathematical bridge into concrete implementation tasks and paper-ready language. It assumes the existing pipeline already produces scenario-based Monte Carlo projections.

---

## 1) Data structures Claude Code will need

### 1.1 “Wave registry” (stateful object updated each year)

A canonical structure is a table/dataframe (or a dict of dataclasses) with one row per *active or recently ended* wave:

- `wave_id` (unique)
- `state`, `origin` (nationality or region)
- `start_year` \(S_i\)
- `last_observed_year` (for updating age)
- `age` \(a_{i,t}\)
- `status` in {`candidate`, `active`, `ended`}
  - `candidate` = exceeds threshold but not yet confirmed (optional)
- `baseline` \(b_{s,o}\) (or a pointer to baseline series)
- observed partial trajectory:
  - `observed_years`: list[int]
  - `observed_arrivals`: list[float/int]
- covariates \(x_i\) and how they were constructed:
  - `log_intensity` (using observed or nowcast peak)
  - `peak_arrivals` (nowcast or realized)
  - `early_wave` (era indicator)
  - `nationality_region` (categorical)
- posterior / predictive objects:
  - `S_cond[k]` = \(\Pr(D\ge a+k \mid D\ge a, x)\) for k=0..K
  - `E_remaining` (expected remaining duration)
  - `duration_draws` (optional cached draws for speed)
  - `shape_params` posterior summary (peak time, rise/decline rates)

### 1.2 Survival model parameter bundle

Store what the forecasting engine needs, independent of estimation code:

- `beta_hat` (vector)
- `beta_vcov` (variance–covariance matrix) for sampling uncertainty
- `baseline_survival` \(S_0(\tau)\) or `baseline_cumhaz` \(H_0(\tau)\) on integer ages \(\tau=1,\dots,\tau_{\max}\)
- metadata:
  - training period
  - wave definition threshold \(\delta\), min run \(m\)
  - covariate scaling notes (e.g., `peak_arrivals` per 1000)

### 1.3 Baseline flow model objects

You likely already have these from Module 9:

- baseline forecast distribution for the target series (or for subcomponents)
- scenario parameters and weights
- Monte Carlo engine for baseline uncertainty

The bridge works best when baseline forecasts are expressed as *simulatable paths*.

---

## 2) Computational steps (implementation plan)

### Step A — Precompute survival functions usable in forecasting

1. Choose the baseline survival representation:
   - If you have Cox baseline cumulative hazard \(H_0(\tau)\): use it directly.
   - If not, use the empirical Kaplan–Meier survival curve as an approximation to \(S_0(\tau)\) and treat this as a pragmatic baseline (with a caveat).

2. Provide helper functions:
   - `linpred(x, beta) = beta^T x`
   - `S(tau, x, beta) = exp(-H0(tau) * exp(linpred))` or `S0(tau) ** exp(linpred)`
   - `S_cond(k | age, x) = S(age+k)/S(age)`
   - `p_end_next_year = 1 - S(age+1)/S(age)`
   - `sample_duration(D | survive to age)` from the discrete conditional PMF:
     \[
     \Pr(D = d \mid D\ge a) = \Pr(D\ge d \mid D\ge a) - \Pr(D\ge d+1 \mid D\ge a).
     \]

### Step B — Real-time wave detection (pragmatic to sophisticated)

**Minimum viable detection (consistent with existing definition):**
- Compute baseline \(b_{s,o}\) from a fixed historical window.
- Flag wave when \(y_{t} \ge (1+\delta)b\) for 2 consecutive years.
- For forecasting, treat the wave as “active” starting at the first exceedance year.

**Better detection (recommended):**
- Introduce a `candidate` state after the first exceedance year.
- Assign a probability of true wave start based on historical false-positive rate, or use a simple Bayesian filter.
- This prevents overreacting to single-year blips.

### Step C — Estimate wave amplitude and shape in a way that can update

You need a mapping from partial trajectory to a posterior over \(\theta_i=(P_i,T_i,\gamma)\).

Minimum viable (works with annual data):
- Use lifecycle priors for \(T_i\) (peak around year ~2).
- Use current max observed arrivals as a lower bound for \(P_i\).
- Use empirical distributions (by region, intensity quartile) to place a prior on peak size and total arrivals.
- Update \(P_i\) each year as new arrivals come in.

### Step D — Integrate with scenario Monte Carlo

Inside each scenario/Monte Carlo draw:

1. Draw baseline path \( \mu_{t+1:t+K}^{(m)}\).
2. For each active wave \(i\):
   - draw \(\beta^{(m)} \sim \mathcal{N}(\hat\beta,\widehat{\Sigma}_\beta)\) (or hold fixed if you must)
   - compute \(S^{(m)}(\tau\mid x_i)\)
   - sample \(D_i^{(m)}\) from the conditional duration distribution given survival to current age
   - sample \(\theta_i^{(m)}\) from its posterior/heuristic distribution
   - compute wave contributions for future years until the sampled duration ends
3. Sum: \(Y^{(m)}_{t+k} = \mu^{(m)}_{t+k} + \sum_i g^{(m)}_{i,t+k}\).

This yields prediction intervals that correctly widen when:
- a wave is newly detected (high uncertainty about \(D\) and peak),
- ND transferability is low (frailty variance),
- scenarios disagree.

---

## 3) How to integrate with the existing scenario framework (Module 9)

Module 9 already generates scenario paths and Monte Carlo intervals. The wave module should enter as either:

### Option 1: Additive “shock component” (cleanest)
- Baseline scenario produces \(\mu_t\) for ND net international migration (or non-refugee component).
- Wave module produces refugee-wave contributions \(\sum_i g_{i,t}\).
- Combine additively with optional scaling if the forecast target is net migration.

### Option 2: Mixture / regime adjustment (if the baseline is already for total)
- If your baseline model already includes historical shocks implicitly, treat the wave module as an **override**:
  - When active wave probability is high, replace baseline mean with a convex combination:
    \[
    \tilde{\mu}_t = (1-p_{\text{wave},t})\mu_t + p_{\text{wave},t}(\mu_t + \text{wave\_mean}_t).
    \]
- This avoids double-counting.

### Scenario knobs that pair naturally with the hazard model
- Scenario affects **wave initiation intensity** (how often new waves start)
- Scenario affects **amplitude distribution** (how big waves are)
- Hazard model affects **persistence** (how long waves last)

That division of labor is conceptually clean and easy to explain in the paper.

---

## 4) Validation checklist (must-have)

### 4.1 Internal consistency checks
- Reproduce the Kaplan–Meier median duration (~3 years) from the stored \(S_0(\tau)\).
- For a wave with covariates \(x=0\), confirm predicted survival matches the baseline survival.
- Confirm hazard-ratio behavior:
  - increase `log_intensity` by 1 → hazard multiplier ≈ 0.41
  - set region dummy → hazard multiplier consistent with HR

### 4.2 Predictive checks on the pooled wave dataset (where sample size exists)
- Time-dependent concordance / C-index (should align with ~0.77 reported).
- Calibration of survival probabilities:
  - among waves with predicted \(\Pr(D>3)\approx 0.3\), does ~30% actually exceed 3?
- Cross-validation at wave level (e.g., split by state or by origin).

### 4.3 ND-specific “sanity checks” (even with few waves)
- Sensitivity analysis:
  - vary threshold \(\delta\) (e.g., 30%, 50%, 100%)
  - vary min-run \(m\) (1 vs 2)
- Check that the wave model does not create implausible long tails for ND unless intensity is high.
- Check fiscal-year vs calendar-year alignment if integrating with ND PEP series.

### 4.4 Forecast backtesting (recommended for the paper)
- Rolling-origin evaluation for the ND series:
  - baseline-only vs baseline+wave
  - compare MAE/RMSE and interval coverage
- Even if ND has few identified waves, show at least one “event” episode where wave-informed intervals behave better.

---

## 5) Suggested paper text (drop-in ready)

### Methods (bridge paragraph)
> We treat large refugee inflow surges as temporary spells (“waves”) in which arrivals remain substantially above a historical baseline. We identify state–origin waves using a threshold rule (arrivals ≥50% above baseline for at least two consecutive years) and define wave duration as the number of consecutive above-threshold years. We estimate the distribution of wave durations using Kaplan–Meier survival curves and a Cox proportional hazards model with covariates capturing wave intensity (log intensity and peak arrivals), timing cohort, and origin-region indicators. For forecasting, we convert the estimated survival function into year-ahead survival probabilities and a predictive distribution for remaining wave duration conditional on the wave having persisted to the present. We then map this remaining-duration distribution into annual flow projections by combining (i) a baseline forecast for non-wave migration and (ii) a lifecycle-shaped wave trajectory (initiation → peak → decline), and we propagate uncertainty by sampling wave durations and amplitudes inside the Monte Carlo scenario engine.

### Results (interpretation paragraph)
> Across 940 state–origin waves, the Kaplan–Meier estimator yields a median duration of approximately three years (with a modest fraction right-censored at the end of the observation window), indicating that most refugee surges revert toward baseline within a short horizon. Wave persistence varies systematically with observable characteristics: higher-intensity waves have substantially lower termination hazards (longer expected spells), while waves associated with certain origin regions (e.g., Europe and the Americas relative to the reference region) have higher hazards (shorter spells). In forecasting terms, these effects translate into differentiated persistence profiles: newly detected high-intensity waves should be projected to last longer, with greater probability mass in the 4+ year tail, while lower-intensity or region-associated short waves should revert more quickly toward baseline.

### Limitations (explicitly ND-focused)
> The hazard estimates are derived from pooled state–origin wave histories (940 waves across 48 states and 56 origins) and therefore primarily inform typical persistence patterns rather than providing North Dakota–specific parameters. Given the short North Dakota time series and the limited number of fully observed ND-specific waves, we use the pooled survival model as a portable baseline and report probabilistic forecasts that incorporate substantial transfer uncertainty (e.g., via partial pooling or a destination-level frailty term). Accordingly, duration-based projections should be interpreted as a structured way to translate surge characteristics into forecast distributions, not as precise point predictions of wave endpoints in North Dakota.

---

## 6) How to avoid overclaiming (language guidance)

Use phrasing like:
- “provides a portable baseline for wave persistence”
- “suggests that higher-intensity surges tend to persist longer”
- “we propagate uncertainty to reflect limited ND-specific wave histories”

Avoid:
- “predicts ND wave duration precisely”
- “establishes causal effects on persistence” (unless you add stronger identification)

---

## 7) Top implementation priorities (if you only do three things)

1. **Wave registry + detection logic** that works online (candidate → active → ended).
2. **Conditional remaining-duration calculator** from \(S(a+k)/S(a)\), with coefficient uncertainty sampling.
3. **Monte Carlo integration** so wave uncertainty flows into forecast intervals rather than being a post-hoc narrative.
