# Backtesting Specification for n=15 Annual ND International Migration (2010–2024)

## Why backtesting is non-optional here

With **15 annual observations**, asymptotic time-series inference (ADF/KPSS, AIC “optimality”, etc.) is low power and fragile—especially with a **2020 shock**. The cleanest way to support a top-tier forecasting claim is to show:

1) **Out-of-sample point accuracy**,
2) **Interval calibration** (coverage + sharpness), and
3) **Benchmark comparisons** (naive baselines must be hard to beat).

This document specifies a backtesting protocol that is feasible with n=15 and produces reviewer-friendly tables.

---

## Data and forecast target

- **Target series:** ND annual net international migration (calendar year), 2010–2024.
- **Predictor series (optional baseline/driver):** US annual net international migration, same years.
- **Forecast horizons:** prioritize **h = 1** (one-step-ahead). Optionally add **h = 2, 3** where feasible.

---

## Rolling-origin evaluation protocol

### Core protocol: expanding window, 1-step-ahead

Let the observed years be {2010, …, 2024}. Choose an initial training window that is not absurdly small.

- **Initial train:** 2010–2016 (7 observations)
- **First forecast:** predict 2017 (h=1)
- **Then expand:** train 2010–2017 → predict 2018, …, train 2010–2023 → predict 2024

This yields **8 out-of-sample forecasts** for h=1: 2017–2024.

**Why not leave-one-out?**
LOO creates training sets of size 14 but breaks time ordering and is not a forecasting evaluation. Time series validation must respect chronology.

### Optional: multi-horizon evaluation (appendix)

For each origin year T, compute forecasts for h ∈ {1,2,3}.
But usable origins shrink with horizon:

- h=1: origins 2016–2023 (8 forecasts)
- h=2: origins 2016–2022 (7 forecasts)
- h=3: origins 2016–2021 (6 forecasts)

Report h=1 as the main result; multi-horizon as appendix.

---

## Models to evaluate

### Required baselines (at least 3)

1) **Naive last observation (Random Walk / “no-change”)**
   - Forecast: \( \hat{y}_{T+1} = y_T \)

2) **Expanding mean or median benchmark**
   - Forecast: \( \hat{y}_{T+1} = \bar{y}_{2010:T} \) or median.

3) **Driver regression (simple national-driver model)**
   Minimal and interpretable:
   - Levels: \( y_t = \alpha + \beta \cdot US_t + \varepsilon_t \)
   Estimate on 2010–T, predict T+1 using observed US_{T+1}.
   If US_{T+1} would not be available in real time, label this explicitly as an **ex post driver** evaluation.

### Candidate “paper models” (include only if stable)

- **ARIMA baseline** (e.g., ARIMA(0,1,0)) with/without drift.
- **Local-level state-space model** (Kalman filter), optionally with:
  - an **intervention dummy** for 2020, and/or
  - **heavy-tailed innovations** (t errors) for robustness.

---

## Metrics to report

### Point forecast accuracy

Compute over the evaluation set \(\mathcal{T}\) (e.g., 2017–2024 for h=1):

- **MAE:** \( \frac{1}{|\mathcal{T}|} \sum_{t \in \mathcal{T}} |y_t - \hat{y}_t| \)
- **RMSE:** \( \sqrt{ \frac{1}{|\mathcal{T}|} \sum (y_t - \hat{y}_t)^2 } \)
- **MASE (recommended):** scales MAE by the in-sample naive error.

### Interval calibration (if model produces prediction intervals)

For each nominal level (e.g., 80%, 95%):

- **Empirical coverage:** fraction of times \( y_t \in [L_t, U_t] \)
- **Average interval width:** mean of \( U_t - L_t \)
- **Optional:** interval score / WIS (weighted interval score).

---

## Output table formats (recommended)

### Table A: Rolling-origin forecast log (one row per forecast)

Columns:
- origin_year (T)
- forecast_year (T+h)
- horizon (h)
- actual_y
- model_name
- point_forecast
- error (actual - forecast)
- abs_error
- lower_80, upper_80 (if available)
- lower_95, upper_95 (if available)
- covered_80, covered_95 (0/1 indicators)

### Table B: Summary metrics by model × horizon

Columns:
- model_name
- horizon
- n_forecasts
- MAE
- RMSE
- MASE
- coverage_80
- coverage_95
- avg_width_80
- avg_width_95

---

## Implementation sketch (Python snippet)

```python
import numpy as np
import pandas as pd

def expanding_window_backtest(df, y="nd_intl_migration", x="us_intl_migration",
                              start_train_end=2016, horizons=(1,)):
    years = df["year"].to_numpy()
    out = []

    for train_end in range(start_train_end, years.max()):  # up to 2023 if max is 2024
        train = df[df["year"] <= train_end].copy()

        for h in horizons:
            test_year = train_end + h
            if test_year > years.max():
                continue

            y_true = df.loc[df["year"] == test_year, y].item()

            # Baseline 1: random-walk / last observation
            y_hat_rw = train[y].iloc[-1]

            # Baseline 2: expanding mean
            y_hat_mean = train[y].mean()

            # Baseline 3: driver regression y ~ 1 + US  (ex post)
            X = np.column_stack([np.ones(len(train)), train[x].to_numpy()])
            beta = np.linalg.lstsq(X, train[y].to_numpy(), rcond=None)[0]
            us_next = df.loc[df["year"] == test_year, x].item()
            y_hat_driver = beta[0] + beta[1] * us_next

            for model, y_hat in [("naive_rw", y_hat_rw),
                                 ("expanding_mean", y_hat_mean),
                                 ("driver_ols", y_hat_driver)]:
                out.append({
                    "origin_year": train_end,
                    "forecast_year": test_year,
                    "horizon": h,
                    "model": model,
                    "actual": y_true,
                    "forecast": float(y_hat),
                    "error": float(y_true - y_hat),
                    "abs_error": float(abs(y_true - y_hat)),
                })

    return pd.DataFrame(out)
```

---

## Limitations (state explicitly in the paper)

- **Few evaluation points:** 8 one-step forecasts is not a lot; treat results as indicative, not definitive.
- **2020 dominates:** include versions with and without a 2020 intervention adjustment, and show sensitivity.
- **No strong forecast-superiority tests:** Diebold–Mariano tests will be underpowered; prefer descriptive comparisons + plots.
- **Ex post drivers:** if using US migration as a predictor, be explicit whether it would be available in real time.

---

## What reviewers usually accept as “rigorous” with n=15

- A transparent rolling-origin protocol
- Benchmark comparisons that include a naive RW
- Interval calibration evidence
- Sensitivity checks around the COVID shock
- Language that treats unit-root testing as diagnostic, not determinative
