# Mathematical Specifications for Hazard-Informed Forecasting

This document provides the mathematical bridge between (i) wave identification and survival analysis and (ii) operational annual flow forecasting.

---

## 0. Index and notation

### Indices
- \(t \in \{1,\dots,T\}\): calendar year index (or fiscal year; be explicit in implementation)
- \(s\): destination state (North Dakota or other)
- \(o\): origin nationality (or origin region)
- \(i\): wave index (a specific spell for a given \(s,o\))

### Observables
- \(y_{s,o,t}\): observed arrivals (refugee arrivals, or other flow measure)
- \(b_{s,o,t}\): baseline level (counterfactual “no-wave” arrivals); can be constant or time-varying
- \(x_{i}\): covariate vector for wave \(i\) (can include origin region, intensity proxies, etc.)
- \(a_{i,t}\): age of wave \(i\) at time \(t\) (years since wave start)

### Latent variables / states
- \(W_{s,o,t} \in \{0,1\}\): wave-state indicator at time \(t\) (1 = in wave regime)
- \(S_i\): start time (year) of wave \(i\)
- \(E_i\): end time (year) of wave \(i\)
- \(D_i = E_i - S_i + 1\): total duration of wave \(i\) in years
- \(R_{i,t} = D_i - a_{i,t}\): remaining duration at time \(t\) (if wave is active)

### Survival-analysis objects
- \(h(\tau \mid x)\): hazard of wave termination at wave age \(\tau\)
- \(H(\tau \mid x)=\int_0^\tau h(u\mid x)\,du\): cumulative hazard
- \(S(\tau \mid x)=\Pr(D>\tau\mid x)=\exp\{-H(\tau\mid x)\}\): survival function

---

## 1. Wave state indicator function

### 1.1 Deterministic threshold definition (offline / retrospective)

Let the baseline for a given \(s,o\) be \(b_{s,o}\) (e.g., median of a baseline window). Fix a threshold \(\delta>0\) (e.g., 0.5 for 50% above baseline).

Define the exceedance indicator:
\[
I_{s,o,t} = \mathbb{1}\left\{y_{s,o,t} \ge (1+\delta)\,b_{s,o}\right\}.
\]

With a minimum-run rule of \(m\) consecutive years (e.g., \(m=2\)), define a wave-start at year \(t\) if:
\[
\text{start at }t \iff I_{s,o,t}=1 \text{ and } I_{s,o,t+1}=1 \text{ and } (I_{s,o,t-1}=0 \text{ or } t=1).
\]

Define wave end analogously as the last year before the process falls below threshold for a sustained period (implementation-specific).

This produces a set of spells \(\{(S_i,E_i)\}_i\) and durations \(D_i\).

### 1.2 Probabilistic real-time wave state (online / forecasting)

For operational forecasting, you typically cannot wait \(m\) future years to confirm a wave. Replace the deterministic indicator with a **filtered probability**:

\[
p_{s,o,t} \equiv \Pr(W_{s,o,t}=1 \mid \mathcal{F}_t),
\]
where \(\mathcal{F}_t\) is the information set up to time \(t\).

A simple state-space version:

- Observation model:
\[
y_{s,o,t} \mid W_{s,o,t} \sim \text{CountDist}\big(\mu_{s,o,t}(W_{s,o,t})\big),
\]
with \(\mu_{s,o,t}(0)=b_{s,o,t}\) (baseline) and \(\mu_{s,o,t}(1)=b_{s,o,t}+\text{wave\_mean}_{i,t}\).

- Transition model (semi-Markov “spell” formulation): when entering the wave state, sample a spell length \(D_i\) from the survival model; remain in wave until spell ends.

This yields a forward-filtering algorithm for \(p_{s,o,t}\). A simpler approximation is to treat wave detection as deterministic but with a 1-year delay; both are acceptable as long as uncertainty is acknowledged.

---

## 2. Conditional duration distribution and expected remaining duration

### 2.1 Cox proportional hazards model

Assume the termination hazard at wave age \(\tau\) follows:

\[
h(\tau \mid x_i) = h_0(\tau)\exp(\beta^\top x_i),
\]

where:
- \(h_0(\tau)\) is the baseline hazard (unspecified, estimated nonparametrically),
- \(x_i\) are wave covariates for wave \(i\),
- \(\beta\) are log-hazard coefficients.

Then:
\[
H(\tau \mid x_i)=H_0(\tau)\exp(\beta^\top x_i),
\quad
S(\tau \mid x_i)=\exp\{-H_0(\tau)\exp(\beta^\top x_i)\}.
\]

If a baseline survival function \(S_0(\tau)=\exp\{-H_0(\tau)\}\) is available, an equivalent expression is:
\[
S(\tau \mid x_i) = \left[S_0(\tau)\right]^{\exp(\beta^\top x_i)}.
\]

### 2.2 Remaining duration distribution at time \(t\)

Let the wave start at \(S_i\). Define wave age at time \(t\) as:

\[
a_{i,t} = t - S_i + 1.
\]

Condition on the fact that the wave has survived at least to age \(a_{i,t}\) (i.e., it is still active at time \(t\)).

Define remaining time \(R_{i,t} = D_i - a_{i,t}\) (in years).

Then the conditional survival function of the remaining time is:

\[
\Pr(R_{i,t} > u \mid D_i \ge a_{i,t}, x_i)
=
\frac{S(a_{i,t}+u \mid x_i)}{S(a_{i,t}\mid x_i)}.
\]

**Expected remaining duration (continuous time):**
\[
\mathbb{E}[R_{i,t} \mid D_i \ge a_{i,t}, x_i]
=
\int_0^\infty \frac{S(a_{i,t}+u \mid x_i)}{S(a_{i,t}\mid x_i)}\,du.
\]

**Discrete annual version (recommended for annual data):**
For \(k=1,2,\dots,K\) where \(K\) is a practical truncation (e.g., 20 years),
\[
\Pr(R_{i,t} \ge k \mid D_i \ge a_{i,t}, x_i)
=
\frac{S(a_{i,t}+k \mid x_i)}{S(a_{i,t}\mid x_i)},
\]
and
\[
\mathbb{E}[R_{i,t} \mid D_i \ge a_{i,t}, x_i]
\approx \sum_{k=1}^{K} \frac{S(a_{i,t}+k \mid x_i)}{S(a_{i,t}\mid x_i)}.
\]

### 2.3 One-step-ahead termination probability

The probability the wave ends in the next year (between ages \(a\) and \(a+1\)):

\[
\Pr(\text{end at }a+1 \mid D\ge a, x) = 1 - \frac{S(a+1\mid x)}{S(a\mid x)}.
\]

This is the “forecasting-facing” hazard object: it is directly interpretable as a one-year-ahead risk that the wave stops contributing.

---

## 3. Flow-duration mapping: from duration distributions to annual flow forecasts

We need a model that converts “wave exists and persists” into annual expected arrivals.

### 3.1 Decomposition: baseline + wave contributions

Let the forecast target series be \(Y_{s,t}\) (e.g., total ND net international migration) or \(y_{s,o,t}\) (refugee arrivals by origin). For concreteness, write a generic series \(Y_t\).

Decompose:

\[
Y_t = \mu_t + \sum_{i \in \mathcal{I}_t} \underbrace{g(\tau_{i,t};\theta_i)}_{\text{wave contribution}} + \varepsilon_t,
\]

where:
- \(\mu_t\) is the baseline forecast component (time series model / regression / ensemble),
- \(\mathcal{I}_t\) indexes waves that may contribute at time \(t\),
- \(\tau_{i,t} = t - S_i + 1\) is wave age,
- \(g(\tau;\theta_i)\ge 0\) is the expected wave contribution at age \(\tau\),
- \(\varepsilon_t\) is residual noise.

Wave contributions only apply if the wave is active:
\[
g(\tau_{i,t};\theta_i)\ \text{is used only when}\ \tau_{i,t}\le D_i.
\]

### 3.2 A simple lifecycle-based wave shape family

A practical parameterization uses three phases: initiation → peak → decline.

Let:
- \(D_i\) = duration
- \(T_i\) = time-to-peak (integer, \(1\le T_i \le D_i\))
- \(P_i\) = peak excess arrivals (above baseline) at peak time
- \(\gamma_\uparrow,\gamma_\downarrow>0\) = shape exponents

Define a normalized shape with maximum 1 at \(\tau=T_i\):

\[
f(\tau \mid D_i, T_i, \gamma_\uparrow,\gamma_\downarrow)=
\begin{cases}
\left(\frac{\tau}{T_i}\right)^{\gamma_\uparrow}, & \tau \le T_i,\\[6pt]
\left(\frac{D_i-\tau+1}{D_i-T_i+1}\right)^{\gamma_\downarrow}, & \tau > T_i.
\end{cases}
\]

Then:
\[
g(\tau;\theta_i) = P_i \, f(\tau \mid D_i, T_i, \gamma_\uparrow,\gamma_\downarrow),
\quad
\theta_i = (P_i, T_i, D_i, \gamma_\uparrow,\gamma_\downarrow).
\]

Lifecycle estimates provide priors for \(T_i\) as a function of \(D_i\), e.g.
\[
T_i \sim \text{DiscreteDist}\left(\text{mean}\approx 2.2\right)
\quad \text{or} \quad
T_i = \left\lfloor \rho_{\text{peak}}D_i \right\rceil,
\]
with \(\rho_{\text{peak}}\) calibrated from initiation-share statistics.

### 3.3 Expected wave contribution at forecast horizon

At forecast origin \(t\), with current age \(a_{i,t}\), the expected contribution at \(t+k\) is:

\[
\mathbb{E}\left[g(\tau_{i,t+k};\theta_i)\,\mathbb{1}\{D_i \ge \tau_{i,t+k}\}\ \middle|\ \mathcal{F}_t\right].
\]

Closed form is rarely needed; Monte Carlo is natural:

1. Draw \(D_i^{(m)}\) from the conditional duration distribution \(p(D_i \mid D_i\ge a_{i,t}, x_i, \beta)\).
2. Draw/update \(\theta_i^{(m)}\) (at least \(P_i, T_i\)) from their posteriors given observed partial wave data.
3. Compute \(g^{(m)}_{i,t+k} = g(\tau_{i,t+k};\theta_i^{(m)})\mathbb{1}\{D_i^{(m)}\ge \tau_{i,t+k}\}\).
4. Average over \(m=1,\dots,M\) to get expected value and quantiles.

### 3.4 Mapping to aggregate ND forecasts

If the final forecast target is aggregate ND international migration \(Y_{\text{ND},t}\), and refugee flows are a component, represent:

\[
Y_{\text{ND},t} = \mu^{\text{nonref}}_t + \sum_{o} y_{\text{ND},o,t} + \varepsilon_t,
\]
where each \(y_{\text{ND},o,t}\) can be wave-modeled, or origins can be grouped into regions.

---

## 4. Bayesian updating as a wave progresses

Forecasting is sequential. The wave’s duration distribution should be updated as new data arrive.

### 4.1 Parameter uncertainty in \(\beta\) (portable uncertainty)

Treat the estimated Cox coefficients as uncertain:
\[
\beta \sim \mathcal{N}(\hat\beta,\ \widehat{\Sigma}_\beta),
\]
where \(\widehat{\Sigma}_\beta\) comes from the Cox model variance estimate (or a bootstrap).

Then survival becomes a mixture:
\[
S(\tau\mid x) = \int S(\tau\mid x,\beta)\,p(\beta)\,d\beta.
\]
In practice, sample \(\beta^{(m)}\) and compute \(S^{(m)}\).

### 4.2 ND-specific partial pooling (frailty)

To acknowledge destination-specific differences with limited ND waves, include a multiplicative frailty:

\[
h(\tau\mid x, s=\text{ND}) = h_0(\tau)\exp(\beta^\top x + u_{\text{ND}}),
\qquad
u_{\text{ND}} \sim \mathcal{N}(0,\sigma_u^2),
\]
with \(\sigma_u\) set conservatively if it cannot be estimated.

This widens intervals without overfitting.

### 4.3 Updating wave covariates and amplitude

Some predictors (peak, intensity) are not fully known at onset. Treat them as latent parameters updated by observed arrivals.

Example (peak nowcast):

- Prior: \(P_i \sim p_0(P)\) (e.g., lognormal, centered on empirical peak distribution for similar region/intensity)
- Likelihood from observed partial trajectory \(y_{S_i:S_i+a-1}\) under the shape model
- Posterior \(p(P_i \mid \text{data})\) obtained via Bayes’ rule

Then update \(x_i\) (e.g., \(\log(\text{intensity})\)) using the posterior mean/median of \(P_i\) and baseline \(b\).

### 4.4 Full sequential posterior (conceptual)

Let \(\theta_i\) be wave-shape parameters and \(\beta\) hazard coefficients. At wave age \(a\), the posterior is:

\[
p(D_i,\theta_i,\beta \mid y_{S_i:S_i+a-1})
\propto
\underbrace{p(y_{S_i:S_i+a-1} \mid D_i,\theta_i)}_{\text{flow/shape likelihood}}
\underbrace{p(D_i \mid x_i(\theta_i),\beta)}_{\text{duration model}}
\underbrace{p(\theta_i)p(\beta)}_{\text{priors}}.
\]

A computationally simple approximation is to:
- update \(\theta_i\) (amplitude/peak) from the flow model,
- update \(D_i\) from the survival model conditional on survival so far,
- propagate \(\beta\) uncertainty via sampling.

---

## 5. Pseudocode: hazard-informed wave forecasting loop

```text
Inputs:
  - Annual arrivals y[t] for each (s,o)
  - Baseline function b[s,o,t] (or constant b[s,o])
  - Cox params: beta_hat, Var(beta), baseline survival S0(tau) or H0(tau)
  - Lifecycle priors for wave shape (time-to-peak, initiation/decline shares)
  - Forecast horizon K years
  - Monte Carlo draws M

State:
  - Active waves registry Waves = {wave_i: (S_i, age a_i, covariates x_i, posterior over theta_i)}

For each forecast origin time t:
  1) Wave detection / update:
      For each (s,o):
        - compute exceedance or filtered prob p_wave
        - if new wave detected: initialize new wave_i with S_i=t (or t-1), age=1
        - if wave active: increment age, update observed partial trajectory

  2) For each active wave_i:
      - update amplitude/shape posterior theta_i | observed trajectory
      - compute covariates x_i (possibly using nowcast peak/intensity)
      - compute conditional duration distribution:
            p(D_i | D_i >= age, x_i, beta)
        using survival ratio S(age+k|x)/S(age|x)

  3) Monte Carlo forecasting:
      For m in 1..M:
        - sample baseline path mu[t+1:t+K] from baseline model / scenario engine
        - for each wave_i:
            sample beta^(m) ~ N(beta_hat, Var(beta))
            sample D_i^(m) ~ p(D_i | survive to age, x_i, beta^(m))
            sample theta_i^(m) ~ p(theta_i | observed trajectory)
            compute wave contributions for k=1..K where age+k <= D_i^(m)
        - sum contributions: Y^(m)[t+k] = mu^(m)[t+k] + sum_i wave_contrib_i^(m)[t+k]

  4) Output:
      - predictive mean and quantiles for Y[t+1:t+K]
      - diagnostics: wave survival probs, expected remaining durations
```

---

## 6. Uncertainty quantification summary

Include uncertainty from:
- Wave detection (optional but recommended)
- \(\beta\) estimation (via sampling)
- ND transfer uncertainty (frailty \(u_{\text{ND}}\))
- Shape/amplitude uncertainty (\(\theta_i\))
- Baseline model uncertainty (\(\mu_t\))
- Scenario uncertainty (policy-conditional inputs)

All uncertainties propagate naturally through the Monte Carlo loop.
