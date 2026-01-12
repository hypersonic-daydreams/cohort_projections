"""
Unit tests for appendix covariate-conditioned near-term anchor module.

These tests focus on the small, reusable building blocks used by:
- sdc_2024_replication/scripts/statistical_analysis/module_app_covariate_anchor.py

The goal is to validate alignment logic (lag construction, LOCF extension) and
ensure the state-space regression wrapper can fit/forecast on a minimal
synthetic dataset.
"""

from __future__ import annotations

import module_app_covariate_anchor as anchor
import numpy as np
import pandas as pd


def test_series_from_grouped_sum_fills_missing_years() -> None:
    df = pd.DataFrame({"year": [2020, 2022], "value": [10.0, 5.0]})
    series = anchor._series_from_grouped_sum(
        df, year_col="year", value_col="value", year_min=2020, year_max=2022
    )

    assert list(series.index) == [2020, 2021, 2022]
    assert float(series.loc[2020]) == 10.0
    assert float(series.loc[2021]) == 0.0
    assert float(series.loc[2022]) == 5.0


def test_extend_locf_repeats_last_value() -> None:
    series = pd.Series([1.0, 2.0], index=pd.Index([2020, 2021], name="year"))
    out = anchor._extend_locf(series, end_year=2023)

    assert list(out.index) == [2020, 2021, 2022, 2023]
    assert float(out.loc[2023]) == 2.0


def test_build_exog_frame_constructs_lags() -> None:
    years = pd.Index([2020, 2021, 2022], name="year")
    # Note: `_build_exog_frame` expects covariate series to include the full
    # index up through the max forecast year so that lagged values are defined
    # after shifting.
    refugees = pd.Series(
        [100.0, 200.0, 300.0, 300.0],
        index=pd.Index([2019, 2020, 2021, 2022], name="fiscal_year"),
    )
    lpr = pd.Series(
        [10.0, 20.0, 30.0, 30.0], index=pd.Index([2019, 2020, 2021, 2022], name="fiscal_year")
    )

    exog = anchor._build_exog_frame(years=years, refugee_fy=refugees, lpr_fy=lpr, acs_year=None)

    # Lag-1 implies year 2020 uses 2019 values, etc.
    assert float(exog.loc[2020, "refugees_lag1"]) == 100.0
    assert float(exog.loc[2021, "refugees_lag1"]) == 200.0
    assert float(exog.loc[2020, "lpr_lag1"]) == 10.0
    assert float(exog.loc[2022, "lpr_lag1"]) == 30.0


def test_fit_and_forecast_local_level_regression_runs() -> None:
    rng = np.random.default_rng(123)

    years = pd.Index(range(2010, 2020), name="year")
    exog = pd.DataFrame(
        {
            "refugees_lag1": rng.normal(0, 1, size=len(years)),
            "lpr_lag1": rng.normal(0, 1, size=len(years)),
        },
        index=years,
    )
    y = pd.Series(
        100 + 5 * exog["refugees_lag1"] - 3 * exog["lpr_lag1"] + rng.normal(0, 1, size=len(years)),
        index=years,
    )

    exog_std, _ = anchor._standardize_exog(exog)
    fitted = anchor.fit_local_level_regression(y, exog_std, maxiter=200)

    future_years = pd.Index([2020, 2021, 2022], name="year")
    future_exog = pd.DataFrame(
        {
            "refugees_lag1": [0.0, 0.0, 0.0],
            "lpr_lag1": [0.0, 0.0, 0.0],
        },
        index=future_years,
    )
    # Apply standardization parameters from training window for consistency.
    _, standardization = anchor._standardize_exog(exog)
    future_exog_std = standardization.apply(future_exog)

    fc = anchor.forecast_local_level_regression(fitted, future_exog=future_exog_std)

    assert len(fc) == 3
    assert {"point", "ci80_lo", "ci80_hi", "ci95_lo", "ci95_hi"} <= set(fc.columns)
