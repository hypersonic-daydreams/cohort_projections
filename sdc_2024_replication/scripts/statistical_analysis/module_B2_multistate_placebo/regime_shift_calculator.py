"""
Regime Shift Calculator
=======================

Computes shift statistics at vintage transition boundaries for each state.
The primary comparison is between Vintage 2020 (2010-2019) and
Vintage 2024 (2020-2024), excluding 2020 as a COVID outlier.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class StateShiftResult:
    """Container for a single state's shift analysis."""

    state: str
    state_fips: int
    mean_2010s: float  # 2010-2019 mean
    mean_2020s: float  # 2021-2024 mean (excluding 2020)
    shift_magnitude: float  # Absolute difference
    relative_shift: float  # Percentage change
    t_statistic: float
    p_value: float
    cohens_d: float
    n_2010s: int
    n_2020s: int

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "state": self.state,
            "state_fips": self.state_fips,
            "mean_2010s": self.mean_2010s,
            "mean_2020s": self.mean_2020s,
            "shift_magnitude": self.shift_magnitude,
            "relative_shift": self.relative_shift,
            "t_statistic": self.t_statistic,
            "p_value": self.p_value,
            "cohens_d": self.cohens_d,
            "n_2010s": self.n_2010s,
            "n_2020s": self.n_2020s,
            "significant_at_05": self.p_value < 0.05,
        }


def calculate_state_shift(
    df: pd.DataFrame,
    state: str,
    exclude_2020: bool = True,
) -> StateShiftResult:
    """
    Calculate regime shift statistics for one state.

    Compares Vintage 2020 (2010-2019) to Vintage 2024 (2020-2024).

    Parameters
    ----------
    df : pd.DataFrame
        Full panel data with 'state', 'year', 'intl_migration' columns
    state : str
        State name to analyze
    exclude_2020 : bool
        If True, excludes 2020 from post-period (COVID outlier)

    Returns
    -------
    StateShiftResult
        Shift statistics for the state
    """
    state_df = df[df["state"] == state].copy()

    if len(state_df) == 0:
        raise ValueError(f"No data found for state: {state}")

    # Get state FIPS
    state_fips = state_df["state_fips"].iloc[0]

    # Pre-period: 2010-2019
    pre = state_df[(state_df["year"] >= 2010) & (state_df["year"] <= 2019)][
        "intl_migration"
    ].values

    # Post-period: 2020-2024 (or 2021-2024 if excluding 2020)
    if exclude_2020:
        post = state_df[state_df["year"] >= 2021]["intl_migration"].values
    else:
        post = state_df[state_df["year"] >= 2020]["intl_migration"].values

    # Calculate means
    mean_pre = np.mean(pre) if len(pre) > 0 else np.nan
    mean_post = np.mean(post) if len(post) > 0 else np.nan

    # Shift magnitude
    shift = mean_post - mean_pre

    # Relative shift (handle division by zero)
    if mean_pre != 0 and not np.isnan(mean_pre):
        relative_shift = (shift / abs(mean_pre)) * 100
    else:
        relative_shift = np.nan

    # Two-sample t-test (Welch's, allowing unequal variances)
    if len(pre) >= 2 and len(post) >= 2:
        t_stat, p_val = stats.ttest_ind(post, pre, equal_var=False)
    else:
        t_stat, p_val = np.nan, np.nan

    # Cohen's d (effect size)
    if len(pre) >= 2 and len(post) >= 2:
        pooled_std = np.sqrt(
            (
                (len(pre) - 1) * np.var(pre, ddof=1)
                + (len(post) - 1) * np.var(post, ddof=1)
            )
            / (len(pre) + len(post) - 2)
        )
        cohens_d = shift / pooled_std if pooled_std > 0 else np.nan
    else:
        cohens_d = np.nan

    return StateShiftResult(
        state=state,
        state_fips=int(state_fips),
        mean_2010s=float(mean_pre),
        mean_2020s=float(mean_post),
        shift_magnitude=float(shift),
        relative_shift=float(relative_shift),
        t_statistic=float(t_stat) if not np.isnan(t_stat) else np.nan,
        p_value=float(p_val) if not np.isnan(p_val) else np.nan,
        cohens_d=float(cohens_d) if not np.isnan(cohens_d) else np.nan,
        n_2010s=len(pre),
        n_2020s=len(post),
    )


def calculate_all_state_shifts(
    df: pd.DataFrame,
    exclude_2020: bool = True,
) -> pd.DataFrame:
    """
    Calculate shift statistics for all states in the panel.

    Parameters
    ----------
    df : pd.DataFrame
        Full panel data
    exclude_2020 : bool
        If True, excludes 2020 from post-period

    Returns
    -------
    pd.DataFrame
        One row per state with shift statistics
    """
    states = sorted(df["state"].unique())
    results = []

    for state in states:
        try:
            shift = calculate_state_shift(df, state, exclude_2020)
            results.append(shift.to_dict())
        except Exception as e:
            print(f"Warning: Could not calculate shift for {state}: {e}")

    return pd.DataFrame(results)


def rank_states_by_shift(
    shift_df: pd.DataFrame,
    metric: str = "shift_magnitude",
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Rank states by a shift metric.

    Parameters
    ----------
    shift_df : pd.DataFrame
        Output from calculate_all_state_shifts()
    metric : str
        Column to rank by: "shift_magnitude", "relative_shift", "cohens_d"
    ascending : bool
        If True, smallest values get rank 1

    Returns
    -------
    pd.DataFrame
        Ranked DataFrame with added 'rank' and 'percentile' columns
    """
    df = shift_df.copy()

    # Sort and rank
    df = df.sort_values(metric, ascending=ascending).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    # Calculate percentile (what percent of states have lower values)
    df["percentile"] = (df["rank"] / len(df)) * 100

    return df


def get_nd_percentile(
    shift_df: pd.DataFrame,
    metric: str = "shift_magnitude",
) -> dict:
    """
    Get North Dakota's position in the national distribution.

    Parameters
    ----------
    shift_df : pd.DataFrame
        Output from calculate_all_state_shifts()
    metric : str
        Metric to use for ranking

    Returns
    -------
    dict
        ND's rank, percentile, and interpretation
    """
    ranked = rank_states_by_shift(shift_df, metric, ascending=False)

    nd_row = ranked[ranked["state"] == "North Dakota"]

    if len(nd_row) == 0:
        return {"error": "North Dakota not found in data"}

    nd_row = nd_row.iloc[0]
    n_states = len(ranked)

    # Interpretation
    pct = nd_row["percentile"]
    if pct <= 10:
        interpretation = "ND in top 10% - strongly supports real driver hypothesis"
    elif pct <= 25:
        interpretation = "ND in top 25% - supports real driver hypothesis"
    elif pct <= 50:
        interpretation = "ND in upper half - moderate evidence for real driver"
    elif pct <= 75:
        interpretation = "ND in lower half - weak evidence for real driver"
    else:
        interpretation = (
            "ND in bottom quartile - favors methodology artifact explanation"
        )

    return {
        "state": "North Dakota",
        "metric": metric,
        "value": float(nd_row[metric]),
        "rank": int(nd_row["rank"]),
        "n_states": n_states,
        "percentile": float(pct),
        "percentile_from_top": float(100 - pct),
        "interpretation": interpretation,
    }
