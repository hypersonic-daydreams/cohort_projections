"""
Shared pytest fixtures for all test modules.

This conftest.py provides fixtures used across multiple test directories
for the cohort_projections test suite.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_nd_migration_n25():
    """
    Sample North Dakota international migration data with 25 observations.

    Simulates data spanning 2000-2024 with vintage labels and realistic
    variation across different methodological regimes.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: year, intl_migration, vintage
    """
    np.random.seed(42)

    # Years 2000-2024
    years = list(range(2000, 2025))

    # Migration values by regime (based on actual ND patterns)
    migration = []
    for year in years:
        if year < 2010:
            # Vintage 2009 (2000s): Lower baseline, stable
            base = 300
            noise = np.random.normal(0, 50)
        elif year < 2020:
            # Vintage 2020 (2010s): Bakken boom effect
            base = 800
            noise = np.random.normal(0, 100)
        else:
            # Vintage 2024 (2020s): Post-COVID, high variance
            if year == 2020:
                base = 30  # COVID collapse
            else:
                base = 1200  # Recovery/revision
            noise = np.random.normal(0, 200)

        migration.append(max(0, base + noise))

    # Create DataFrame
    df = pd.DataFrame(
        {
            "year": years,
            "intl_migration": migration,
        }
    )

    # Add vintage labels
    df["vintage"] = df["year"].apply(lambda y: 2009 if y < 2010 else (2020 if y < 2020 else 2024))

    return df


@pytest.fixture
def sample_state_panel():
    """
    Sample 50-state x 15-year panel data for multi-state analysis.

    Creates synthetic data for testing cross-state comparison functions.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: state, state_fips, year, intl_migration
    """
    np.random.seed(42)

    # US states (abbreviated list for testing)
    states = [
        "Alabama",
        "Alaska",
        "Arizona",
        "Arkansas",
        "California",
        "Colorado",
        "Connecticut",
        "Delaware",
        "Florida",
        "Georgia",
        "Hawaii",
        "Idaho",
        "Illinois",
        "Indiana",
        "Iowa",
        "Kansas",
        "Kentucky",
        "Louisiana",
        "Maine",
        "Maryland",
        "Massachusetts",
        "Michigan",
        "Minnesota",
        "Mississippi",
        "Missouri",
        "Montana",
        "Nebraska",
        "Nevada",
        "New Hampshire",
        "New Jersey",
        "New Mexico",
        "New York",
        "North Carolina",
        "North Dakota",
        "Ohio",
        "Oklahoma",
        "Oregon",
        "Pennsylvania",
        "Rhode Island",
        "South Carolina",
        "South Dakota",
        "Tennessee",
        "Texas",
        "Utah",
        "Vermont",
        "Virginia",
        "Washington",
        "West Virginia",
        "Wisconsin",
        "Wyoming",
    ]

    state_fips = list(range(1, 51))

    years = list(range(2010, 2025))  # 15 years

    data = []
    for i, state in enumerate(states):
        # Base migration varies by state population
        base_migration = np.random.uniform(100, 10000)

        # Oil states get a boom effect
        is_oil_state = state in [
            "North Dakota",
            "Texas",
            "Oklahoma",
            "Colorado",
            "New Mexico",
            "Wyoming",
            "Louisiana",
            "Montana",
            "Alaska",
            "California",
            "Kansas",
        ]

        for year in years:
            if year < 2020:
                # Pre-2020 period
                if is_oil_state:
                    boom_effect = base_migration * 0.3
                else:
                    boom_effect = 0
                migration = base_migration + boom_effect + np.random.normal(0, base_migration * 0.1)
            else:
                # Post-2020 period
                if year == 2020:
                    # COVID year
                    migration = base_migration * 0.1
                else:
                    # Recovery with potential revision effect
                    if is_oil_state:
                        revision = base_migration * 0.5
                    else:
                        revision = base_migration * 0.2
                    migration = (
                        base_migration + revision + np.random.normal(0, base_migration * 0.15)
                    )

            data.append(
                {
                    "state": state,
                    "state_fips": state_fips[i],
                    "year": year,
                    "intl_migration": max(0, migration),
                }
            )

    return pd.DataFrame(data)


@pytest.fixture
def synthetic_regime_data():
    """
    Synthetic data with known structural breaks for testing regime detection.

    Creates data with clear level shifts at 2010 and 2020.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: year, y, regime
    """
    np.random.seed(42)

    years = list(range(2000, 2025))
    y = []
    regime = []

    for year in years:
        if year < 2010:
            # Regime 1: level=100, trend=5
            t = year - 2000
            value = 100 + 5 * t + np.random.normal(0, 10)
            regime.append("2000s")
        elif year < 2020:
            # Regime 2: level=200 (shift), trend=10
            t = year - 2010
            value = 200 + 10 * t + np.random.normal(0, 15)
            regime.append("2010s")
        else:
            # Regime 3: level=400 (shift), trend=-5
            t = year - 2020
            value = 400 - 5 * t + np.random.normal(0, 20)
            regime.append("2020s")

        y.append(value)

    return pd.DataFrame({"year": years, "y": y, "regime": regime})


@pytest.fixture
def sample_var_data():
    """
    Sample multivariate time series data for VAR testing.

    Creates two correlated time series with known structure.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: year, var1, var2
    """
    np.random.seed(42)
    n = 50

    # Generate VAR(1) data with known coefficients
    # y1_t = 0.8*y1_{t-1} + 0.1*y2_{t-1} + e1
    # y2_t = 0.2*y1_{t-1} + 0.6*y2_{t-1} + e2

    var1 = np.zeros(n)
    var2 = np.zeros(n)

    var1[0] = 100
    var2[0] = 50

    for t in range(1, n):
        e1 = np.random.normal(0, 10)
        e2 = np.random.normal(0, 5)

        var1[t] = 0.8 * var1[t - 1] + 0.1 * var2[t - 1] + e1
        var2[t] = 0.2 * var1[t - 1] + 0.6 * var2[t - 1] + e2

    years = list(range(2000, 2000 + n))

    return pd.DataFrame({"year": years, "var1": var1, "var2": var2})
