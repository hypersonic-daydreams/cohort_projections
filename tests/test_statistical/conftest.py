"""
Statistical test-specific fixtures.

Provides fixtures specific to statistical analysis testing, building on
the shared fixtures from the parent conftest.py.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add the module path for imports
MODULE_PATH = (
    Path(__file__).parent.parent.parent
    / "sdc_2024_replication"
    / "scripts"
    / "statistical_analysis"
)
if str(MODULE_PATH) not in sys.path:
    sys.path.insert(0, str(MODULE_PATH))


@pytest.fixture
def nd_migration_with_covid():
    """
    North Dakota migration data with explicit COVID year (2020).

    Used for testing COVID intervention modeling.

    Returns
    -------
    pd.DataFrame
        DataFrame with year, intl_migration, includes 2020 COVID outlier
    """
    np.random.seed(42)

    data = []
    for year in range(2015, 2025):
        if year < 2020:
            migration = 800 + np.random.normal(0, 100)
        elif year == 2020:
            migration = 30  # COVID collapse
        else:
            migration = 1200 + np.random.normal(0, 150)

        data.append({"year": year, "intl_migration": migration})

    return pd.DataFrame(data)


@pytest.fixture
def small_regime_data():
    """
    Small dataset for testing edge cases with regime analysis.

    Returns
    -------
    pd.DataFrame
        Minimal data with year and y columns
    """
    return pd.DataFrame(
        {
            "year": [2005, 2006, 2007, 2015, 2016, 2017, 2021, 2022, 2023],
            "y": [100, 105, 110, 200, 210, 220, 400, 390, 380],
        }
    )


@pytest.fixture
def homogeneous_variance_data():
    """
    Data with equal variance across regimes (for testing variance tests).

    Returns
    -------
    pd.DataFrame
        Data with year, y, vintage columns, homogeneous variance
    """
    np.random.seed(42)

    data = []
    for year in range(2000, 2025):
        if year < 2010:
            vintage = 2009
            y = 100 + np.random.normal(0, 10)
        elif year < 2020:
            vintage = 2020
            y = 150 + np.random.normal(0, 10)
        else:
            vintage = 2024
            y = 200 + np.random.normal(0, 10)

        data.append({"year": year, "y": y, "vintage": vintage})

    return pd.DataFrame(data)


@pytest.fixture
def high_variance_ratio_data():
    """
    Data with highly heterogeneous variance across regimes.

    Mimics the 29:1 variance ratio observed in actual ND data.

    Returns
    -------
    pd.DataFrame
        Data with year, y, vintage columns, heterogeneous variance
    """
    np.random.seed(42)

    data = []
    for year in range(2000, 2025):
        if year < 2010:
            vintage = 2009
            y = 100 + np.random.normal(0, 20)  # Low variance
        elif year < 2020:
            vintage = 2020
            y = 500 + np.random.normal(0, 50)  # Medium variance
        else:
            vintage = 2024
            y = 800 + np.random.normal(0, 200)  # High variance

        data.append({"year": year, "y": y, "vintage": vintage})

    return pd.DataFrame(data)


@pytest.fixture
def shift_df_50_states(sample_state_panel):
    """
    Pre-calculated shift statistics for 50 states.

    This fixture computes shift statistics from the sample_state_panel
    for use in hypothesis testing functions.

    Returns
    -------
    pd.DataFrame
        DataFrame with shift statistics per state
    """
    # Calculate shift statistics for each state
    results = []

    for state in sample_state_panel["state"].unique():
        state_data = sample_state_panel[sample_state_panel["state"] == state]
        state_fips = state_data["state_fips"].iloc[0]

        # Pre-2020 mean (2010-2019)
        pre = state_data[(state_data["year"] >= 2010) & (state_data["year"] <= 2019)][
            "intl_migration"
        ].values

        # Post-2020 mean (2021-2024, excluding 2020)
        post = state_data[state_data["year"] >= 2021]["intl_migration"].values

        mean_pre = np.mean(pre) if len(pre) > 0 else np.nan
        mean_post = np.mean(post) if len(post) > 0 else np.nan

        shift = mean_post - mean_pre
        relative_shift = (shift / abs(mean_pre)) * 100 if mean_pre != 0 else np.nan

        # Cohen's d
        if len(pre) >= 2 and len(post) >= 2:
            pooled_std = np.sqrt(
                ((len(pre) - 1) * np.var(pre, ddof=1) + (len(post) - 1) * np.var(post, ddof=1))
                / (len(pre) + len(post) - 2)
            )
            cohens_d = shift / pooled_std if pooled_std > 0 else np.nan
        else:
            cohens_d = np.nan

        results.append(
            {
                "state": state,
                "state_fips": state_fips,
                "mean_2010s": mean_pre,
                "mean_2020s": mean_post,
                "shift_magnitude": shift,
                "relative_shift": relative_shift,
                "cohens_d": cohens_d,
                "n_2010s": len(pre),
                "n_2020s": len(post),
            }
        )

    return pd.DataFrame(results)
