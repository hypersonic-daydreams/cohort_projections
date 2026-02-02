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
def nd_migration_with_covid() -> pd.DataFrame:
    """
    North Dakota migration data with explicit COVID year (2020).

    Used for testing COVID intervention modeling.

    Synthetic Data Characteristics
    ------------------------------
    - Years: 2015-2024 (n=10 observations)
    - Pre-COVID (2015-2019): ~800, noise std=100
    - COVID year (2020): Fixed value 30 (outlier)
    - Post-COVID (2021-2024): ~1200, noise std=150
    - Random seed: 42 for reproducibility

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
def small_regime_data() -> pd.DataFrame:
    """
    Small dataset for testing edge cases with regime analysis.

    Synthetic Data Characteristics
    ------------------------------
    - n=9 observations (small sample for edge case testing)
    - Three regimes: 2005-2007 (~100), 2015-2017 (~200), 2021-2023 (~400)
    - Clear level shifts for testing with minimal data
    - Deterministic values (no random noise)

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
def homogeneous_variance_data() -> pd.DataFrame:
    """
    Data with equal variance across regimes (for testing variance tests).

    Synthetic Data Characteristics
    ------------------------------
    - Years: 2000-2024 (n=25 observations)
    - Regime 1 (2000-2009, vintage 2009): Level=100, noise std=10
    - Regime 2 (2010-2019, vintage 2020): Level=150, noise std=10
    - Regime 3 (2020-2024, vintage 2024): Level=200, noise std=10
    - Equal variance (std=10) across all regimes for Levene test validation
    - Random seed: 42 for reproducibility

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
def high_variance_ratio_data() -> pd.DataFrame:
    """
    Data with highly heterogeneous variance across regimes.

    Mimics the 29:1 variance ratio observed in actual ND data.

    Synthetic Data Characteristics
    ------------------------------
    - Years: 2000-2024 (n=25 observations)
    - Regime 1 (2000-2009, vintage 2009): Level=100, noise std=20 (low)
    - Regime 2 (2010-2019, vintage 2020): Level=500, noise std=50 (medium)
    - Regime 3 (2020-2024, vintage 2024): Level=800, noise std=200 (high)
    - Variance ratio: ~100:1 (200^2/20^2) for heteroscedasticity testing
    - Random seed: 42 for reproducibility

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
def shift_df_50_states(sample_state_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-calculated shift statistics for 50 states.

    This fixture computes shift statistics from the sample_state_panel
    for use in hypothesis testing functions.

    Synthetic Data Characteristics
    ------------------------------
    - Derived from sample_state_panel fixture (50 states)
    - Calculates pre-2020 (2010-2019) vs post-2020 (2021-2024) comparisons
    - Includes Cohen's d effect size for each state
    - Relative shift percentages for magnitude comparison

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: state, state_fips, mean_2010s, mean_2020s,
        shift_magnitude, relative_shift, cohens_d, n_2010s, n_2020s
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
