"""
Oil/Energy State Hypothesis Testing
====================================

Tests whether oil-producing states have systematically different
vintage transition patterns, supporting the Bakken boom explanation
for North Dakota's shifts.

Classification is based on OIL_STATE_RESEARCH.md analysis:
- Boom timing matters more than static production levels
- The Bakken boom (2008-2015) had different timing than Permian (2015+)
- Grouping by boom period enables more meaningful comparisons

Classification scheme (Hybrid Approach per research):
1. Bakken Boom States: ND, MT (2008-2015 boom, ~40% CAGR)
2. Permian Boom States: TX, NM (two-phase boom, 2010-2014 and 2015+)
3. Other Shale States: CO, OK, LA (various formations, moderate growth)
4. Mature Oil States: CA, AK, WY, KS (declining or stable production)
5. Non-Oil States: All remaining states

Sources:
- EIA State Production Rankings (https://www.eia.gov/state/rankings/)
- Wilson (2022) "Moving to Economic Opportunity" J Human Resources
- See docs/adr/020-reports/phase_b_plans/OIL_STATE_RESEARCH.md
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

# =============================================================================
# BOOM-TIMING BASED CLASSIFICATION (Recommended by OIL_STATE_RESEARCH.md)
# =============================================================================

# Bakken Formation boom (2008-2015): North Dakota had ~40% CAGR
# This is the core hypothesis: ND's boom timing aligns with vintage transitions
BAKKEN_BOOM_STATES = [
    "North Dakota",
    "Montana",  # Western ND/Eastern MT Bakken region
]

# Permian Basin boom (two-phase: 2010-2014, then accelerating 2015+)
# Different timing than Bakken - major acceleration after 2015
PERMIAN_BOOM_STATES = [
    "Texas",
    "New Mexico",  # Southeast NM Delaware Basin
]

# Other shale formations with moderate growth during 2008-2015
OTHER_SHALE_STATES = [
    "Colorado",  # Niobrara/DJ Basin
    "Oklahoma",  # Various plays
    "Louisiana",  # Haynesville
]

# Mature oil states: high production but low/no growth during boom period
# These provide a control group - oil states without boom dynamics
MATURE_OIL_STATES = [
    "California",  # Declining production
    "Alaska",  # Declining since 1988
    "Wyoming",  # Moderate, stable
    "Kansas",  # Small, stable
]

# All boom states (experienced significant production growth 2008-2015)
ALL_BOOM_STATES = BAKKEN_BOOM_STATES + PERMIAN_BOOM_STATES + OTHER_SHALE_STATES

# =============================================================================
# LEGACY CLASSIFICATION (for backwards compatibility)
# =============================================================================
# Original ad-hoc classification based on EIA production rankings
# DEPRECATED: Use boom-timing classification for new analyses

OIL_STATES = [
    "Texas",
    "North Dakota",
    "New Mexico",
    "Oklahoma",
    "Colorado",
    "Alaska",
]

SECONDARY_OIL_STATES = [
    "Wyoming",
    "Louisiana",
    "Kansas",
    "California",
]

# Combined oil states for analysis (legacy)
ALL_OIL_STATES = OIL_STATES + SECONDARY_OIL_STATES


def run_oil_state_hypothesis_test(
    shift_df: pd.DataFrame,
    metric: str = "relative_shift",
    oil_states: Optional[list[str]] = None,
) -> dict:
    """
    Test whether oil states have systematically different shifts.

    Parameters
    ----------
    shift_df : pd.DataFrame
        Output from calculate_all_state_shifts()
    metric : str
        Metric to compare: "shift_magnitude", "relative_shift", "cohens_d"
    oil_states : list[str], optional
        Custom list of oil states. If None, uses ALL_OIL_STATES.

    Returns
    -------
    dict
        Test results including means, difference, and p-value
    """
    if oil_states is None:
        oil_states = ALL_OIL_STATES

    # Split into oil vs non-oil
    oil_df = shift_df[shift_df["state"].isin(oil_states)]
    non_oil_df = shift_df[~shift_df["state"].isin(oil_states)]

    oil_values = oil_df[metric].dropna().values
    non_oil_values = non_oil_df[metric].dropna().values

    # Calculate means
    oil_mean = np.mean(oil_values) if len(oil_values) > 0 else np.nan
    non_oil_mean = np.mean(non_oil_values) if len(non_oil_values) > 0 else np.nan

    # Two-sample t-test (Welch's)
    if len(oil_values) >= 2 and len(non_oil_values) >= 2:
        t_stat, p_val = stats.ttest_ind(oil_values, non_oil_values, equal_var=False)
    else:
        t_stat, p_val = np.nan, np.nan

    # Mann-Whitney U (non-parametric alternative)
    if len(oil_values) >= 2 and len(non_oil_values) >= 2:
        u_stat, u_p_val = stats.mannwhitneyu(
            oil_values, non_oil_values, alternative="two-sided"
        )
    else:
        u_stat, u_p_val = np.nan, np.nan

    # Interpretation
    if p_val < 0.05 and oil_mean > non_oil_mean:
        interpretation = "Oil states have significantly higher shifts - supports real driver hypothesis"
    elif p_val < 0.05 and oil_mean < non_oil_mean:
        interpretation = (
            "Oil states have significantly lower shifts - unexpected result"
        )
    elif p_val < 0.10 and oil_mean > non_oil_mean:
        interpretation = (
            "Oil states show marginally higher shifts - weak support for real driver"
        )
    else:
        interpretation = "No significant difference between oil and non-oil states"

    return {
        "metric": metric,
        "oil_states_used": oil_states,
        "n_oil_states": len(oil_values),
        "n_non_oil_states": len(non_oil_values),
        "oil_mean": float(oil_mean),
        "non_oil_mean": float(non_oil_mean),
        "difference": float(oil_mean - non_oil_mean),
        "t_test": {
            "statistic": float(t_stat) if not np.isnan(t_stat) else None,
            "p_value": float(p_val) if not np.isnan(p_val) else None,
        },
        "mann_whitney": {
            "statistic": float(u_stat) if not np.isnan(u_stat) else None,
            "p_value": float(u_p_val) if not np.isnan(u_p_val) else None,
        },
        "significant_at_05": bool(p_val < 0.05) if not np.isnan(p_val) else False,
        "interpretation": interpretation,
    }


def compare_oil_vs_non_oil(
    shift_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create comparison table of oil vs non-oil states.

    Parameters
    ----------
    shift_df : pd.DataFrame
        Output from calculate_all_state_shifts()

    Returns
    -------
    pd.DataFrame
        Summary comparison table
    """
    # Add oil state indicator
    df = shift_df.copy()
    df["oil_state"] = df["state"].isin(ALL_OIL_STATES)
    df["oil_category"] = df["state"].apply(
        lambda s: "Primary Oil"
        if s in OIL_STATES
        else ("Secondary Oil" if s in SECONDARY_OIL_STATES else "Non-Oil")
    )

    # Summary by category
    summary = (
        df.groupby("oil_category")
        .agg(
            n_states=("state", "count"),
            mean_shift_magnitude=("shift_magnitude", "mean"),
            std_shift_magnitude=("shift_magnitude", "std"),
            mean_relative_shift=("relative_shift", "mean"),
            std_relative_shift=("relative_shift", "std"),
            mean_cohens_d=("cohens_d", "mean"),
        )
        .reset_index()
    )

    return summary


def get_nd_rank_among_oil_states(
    shift_df: pd.DataFrame,
    metric: str = "shift_magnitude",
) -> dict:
    """
    Rank North Dakota within the oil state subset.

    This answers: "Is ND unusual even among oil states?"

    Parameters
    ----------
    shift_df : pd.DataFrame
        Output from calculate_all_state_shifts()
    metric : str
        Metric to rank by

    Returns
    -------
    dict
        ND's rank and position among oil states
    """
    # Filter to oil states only
    oil_df = shift_df[shift_df["state"].isin(ALL_OIL_STATES)].copy()

    if len(oil_df) == 0:
        return {"error": "No oil states found in data"}

    # Sort by metric (descending)
    oil_df = oil_df.sort_values(metric, ascending=False).reset_index(drop=True)
    oil_df["rank"] = range(1, len(oil_df) + 1)

    # Find ND
    nd_row = oil_df[oil_df["state"] == "North Dakota"]

    if len(nd_row) == 0:
        return {"error": "North Dakota not found among oil states"}

    nd_row = nd_row.iloc[0]
    n_oil = len(oil_df)

    # Interpretation
    if nd_row["rank"] == 1:
        interpretation = (
            "ND has largest shift among oil states - Bakken-specific effect"
        )
    elif nd_row["rank"] <= 3:
        interpretation = "ND in top 3 among oil states - supports Bakken hypothesis"
    elif nd_row["rank"] <= n_oil / 2:
        interpretation = "ND in upper half among oil states"
    else:
        interpretation = "ND in lower half among oil states - unexpected"

    return {
        "state": "North Dakota",
        "metric": metric,
        "value": float(nd_row[metric]),
        "rank_among_oil": int(nd_row["rank"]),
        "n_oil_states": n_oil,
        "percentile_among_oil": float((1 - (nd_row["rank"] - 1) / n_oil) * 100),
        "all_oil_rankings": oil_df[["state", metric, "rank"]].to_dict("records"),
        "interpretation": interpretation,
    }


# =============================================================================
# BOOM-TIMING BASED ANALYSIS FUNCTIONS (New per OIL_STATE_RESEARCH.md)
# =============================================================================


def get_boom_category(state: str) -> str:
    """
    Classify a state into boom-timing category.

    Categories:
    - Bakken Boom: ND, MT (2008-2015 boom)
    - Permian Boom: TX, NM (2015+ acceleration)
    - Other Shale: CO, OK, LA (moderate growth)
    - Mature Oil: CA, AK, WY, KS (stable/declining)
    - Non-Oil: All others

    Parameters
    ----------
    state : str
        State name

    Returns
    -------
    str
        Category name
    """
    if state in BAKKEN_BOOM_STATES:
        return "Bakken Boom"
    elif state in PERMIAN_BOOM_STATES:
        return "Permian Boom"
    elif state in OTHER_SHALE_STATES:
        return "Other Shale"
    elif state in MATURE_OIL_STATES:
        return "Mature Oil"
    else:
        return "Non-Oil"


def run_boom_state_hypothesis_test(
    shift_df: pd.DataFrame,
    metric: str = "relative_shift",
) -> dict:
    """
    Test whether boom states have systematically different shifts.

    This is the improved test that uses boom-timing classification
    instead of static oil state classification.

    Key comparison: Boom states (experienced 2008-2015 growth) vs. Non-oil states

    Parameters
    ----------
    shift_df : pd.DataFrame
        Output from calculate_all_state_shifts()
    metric : str
        Metric to compare: "shift_magnitude", "relative_shift", "cohens_d"

    Returns
    -------
    dict
        Test results including means, difference, and p-value
    """
    # Add boom category
    df = shift_df.copy()
    df["boom_category"] = df["state"].apply(get_boom_category)

    # Split into boom vs non-oil (exclude mature oil for cleaner test)
    boom_df = df[df["state"].isin(ALL_BOOM_STATES)]
    non_oil_df = df[df["boom_category"] == "Non-Oil"]

    boom_values = boom_df[metric].dropna().values
    non_oil_values = non_oil_df[metric].dropna().values

    # Calculate means
    boom_mean = np.mean(boom_values) if len(boom_values) > 0 else np.nan
    non_oil_mean = np.mean(non_oil_values) if len(non_oil_values) > 0 else np.nan

    # Two-sample t-test (Welch's)
    if len(boom_values) >= 2 and len(non_oil_values) >= 2:
        t_stat, p_val = stats.ttest_ind(boom_values, non_oil_values, equal_var=False)
    else:
        t_stat, p_val = np.nan, np.nan

    # Mann-Whitney U (non-parametric alternative)
    if len(boom_values) >= 2 and len(non_oil_values) >= 2:
        u_stat, u_p_val = stats.mannwhitneyu(
            boom_values, non_oil_values, alternative="two-sided"
        )
    else:
        u_stat, u_p_val = np.nan, np.nan

    # Interpretation
    if p_val < 0.05 and boom_mean > non_oil_mean:
        interpretation = "Boom states have significantly higher shifts - supports real driver hypothesis"
    elif p_val < 0.05 and boom_mean < non_oil_mean:
        interpretation = (
            "Boom states have significantly lower shifts - unexpected result"
        )
    elif p_val < 0.10 and boom_mean > non_oil_mean:
        interpretation = (
            "Boom states show marginally higher shifts - weak support for real driver"
        )
    else:
        interpretation = "No significant difference between boom and non-oil states"

    return {
        "metric": metric,
        "classification": "boom_timing",
        "boom_states_used": ALL_BOOM_STATES,
        "n_boom_states": len(boom_values),
        "n_non_oil_states": len(non_oil_values),
        "boom_mean": float(boom_mean),
        "non_oil_mean": float(non_oil_mean),
        "difference": float(boom_mean - non_oil_mean),
        "t_test": {
            "statistic": float(t_stat) if not np.isnan(t_stat) else None,
            "p_value": float(p_val) if not np.isnan(p_val) else None,
        },
        "mann_whitney": {
            "statistic": float(u_stat) if not np.isnan(u_stat) else None,
            "p_value": float(u_p_val) if not np.isnan(u_p_val) else None,
        },
        "significant_at_05": bool(p_val < 0.05) if not np.isnan(p_val) else False,
        "interpretation": interpretation,
    }


def run_bakken_specific_hypothesis_test(
    shift_df: pd.DataFrame,
    metric: str = "relative_shift",
) -> dict:
    """
    Test whether Bakken boom states differ from Permian boom states.

    This addresses the key question: Is ND's pattern due to Bakken-specific
    timing (2008-2015 boom) rather than general oil state effects?

    Parameters
    ----------
    shift_df : pd.DataFrame
        Output from calculate_all_state_shifts()
    metric : str
        Metric to compare

    Returns
    -------
    dict
        Comparison of Bakken vs Permian boom states
    """
    df = shift_df.copy()
    df["boom_category"] = df["state"].apply(get_boom_category)

    bakken = df[df["state"].isin(BAKKEN_BOOM_STATES)][metric].dropna().values
    permian = df[df["state"].isin(PERMIAN_BOOM_STATES)][metric].dropna().values
    other_shale = df[df["state"].isin(OTHER_SHALE_STATES)][metric].dropna().values
    mature = df[df["state"].isin(MATURE_OIL_STATES)][metric].dropna().values
    non_oil = df[df["boom_category"] == "Non-Oil"][metric].dropna().values

    def safe_mean(arr):
        return float(np.mean(arr)) if len(arr) > 0 else None

    def safe_std(arr):
        return float(np.std(arr, ddof=1)) if len(arr) > 1 else None

    # Build group summaries
    groups = {
        "Bakken Boom": {"states": BAKKEN_BOOM_STATES, "values": bakken},
        "Permian Boom": {"states": PERMIAN_BOOM_STATES, "values": permian},
        "Other Shale": {"states": OTHER_SHALE_STATES, "values": other_shale},
        "Mature Oil": {"states": MATURE_OIL_STATES, "values": mature},
        "Non-Oil": {"states": "all others", "values": non_oil},
    }

    group_stats = {}
    for name, data in groups.items():
        group_stats[name] = {
            "n": len(data["values"]),
            "mean": safe_mean(data["values"]),
            "std": safe_std(data["values"]),
            "states": data["states"] if isinstance(data["states"], list) else [],
        }

    # Key comparison: Bakken vs Permian
    if len(bakken) >= 1 and len(permian) >= 1:
        # With small samples, use descriptive comparison
        bakken_mean = safe_mean(bakken)
        permian_mean = safe_mean(permian)
        difference = (
            bakken_mean - permian_mean if bakken_mean and permian_mean else None
        )

        # Mann-Whitney for small samples
        if len(bakken) >= 2 and len(permian) >= 2:
            _, u_p = stats.mannwhitneyu(bakken, permian, alternative="two-sided")
        else:
            u_p = np.nan
    else:
        difference = None
        u_p = np.nan

    # Interpretation
    if difference is not None:
        if difference > 0:
            interp = "Bakken states have higher shifts than Permian - timing matters"
        elif difference < 0:
            interp = "Permian states have higher shifts than Bakken"
        else:
            interp = "Bakken and Permian have similar shifts"
    else:
        interp = "Insufficient data for comparison"

    # ND-specific finding
    nd_row = df[df["state"] == "North Dakota"]
    nd_value = float(nd_row[metric].iloc[0]) if len(nd_row) > 0 else None

    return {
        "metric": metric,
        "group_statistics": group_stats,
        "bakken_vs_permian": {
            "bakken_mean": safe_mean(bakken),
            "permian_mean": safe_mean(permian),
            "difference": difference,
            "mann_whitney_p": float(u_p) if not np.isnan(u_p) else None,
        },
        "nd_value": nd_value,
        "interpretation": interp,
    }


def get_nd_rank_among_boom_states(
    shift_df: pd.DataFrame,
    metric: str = "relative_shift",
) -> dict:
    """
    Rank North Dakota within boom states only.

    This is the key test: Is ND unusual among states that experienced
    the same type of boom during 2008-2015?

    Parameters
    ----------
    shift_df : pd.DataFrame
        Output from calculate_all_state_shifts()
    metric : str
        Metric to rank by. Recommend "relative_shift" to normalize for size.

    Returns
    -------
    dict
        ND's rank and position among boom states
    """
    # Filter to boom states only
    boom_df = shift_df[shift_df["state"].isin(ALL_BOOM_STATES)].copy()

    if len(boom_df) == 0:
        return {"error": "No boom states found in data"}

    # Add boom category
    boom_df["boom_category"] = boom_df["state"].apply(get_boom_category)

    # Sort by metric (descending - largest shift first)
    boom_df = boom_df.sort_values(metric, ascending=False).reset_index(drop=True)
    boom_df["rank"] = range(1, len(boom_df) + 1)

    # Find ND
    nd_row = boom_df[boom_df["state"] == "North Dakota"]

    if len(nd_row) == 0:
        return {"error": "North Dakota not found among boom states"}

    nd_row = nd_row.iloc[0]
    n_boom = len(boom_df)

    # Interpretation
    if nd_row["rank"] == 1:
        interpretation = "ND has largest shift among ALL boom states - Bakken-specific effect confirmed"
    elif nd_row["rank"] <= 2:
        interpretation = (
            "ND in top 2 among boom states - strong support for Bakken hypothesis"
        )
    elif nd_row["rank"] <= n_boom / 2:
        interpretation = "ND in upper half among boom states"
    else:
        interpretation = "ND in lower half among boom states - pattern not exceptional"

    return {
        "state": "North Dakota",
        "metric": metric,
        "value": float(nd_row[metric]),
        "rank_among_boom": int(nd_row["rank"]),
        "n_boom_states": n_boom,
        "percentile_among_boom": float((1 - (nd_row["rank"] - 1) / n_boom) * 100),
        "all_boom_rankings": boom_df[
            ["state", "boom_category", metric, "rank"]
        ].to_dict("records"),
        "interpretation": interpretation,
    }


def compare_boom_categories(
    shift_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create comparison table across all boom-timing categories.

    Parameters
    ----------
    shift_df : pd.DataFrame
        Output from calculate_all_state_shifts()

    Returns
    -------
    pd.DataFrame
        Summary comparison table by boom category
    """
    df = shift_df.copy()
    df["boom_category"] = df["state"].apply(get_boom_category)

    # Order categories meaningfully
    category_order = [
        "Bakken Boom",
        "Permian Boom",
        "Other Shale",
        "Mature Oil",
        "Non-Oil",
    ]

    # Summary by category
    summary = (
        df.groupby("boom_category")
        .agg(
            n_states=("state", "count"),
            states=("state", lambda x: ", ".join(sorted(x))),
            mean_shift_magnitude=("shift_magnitude", "mean"),
            std_shift_magnitude=("shift_magnitude", "std"),
            mean_relative_shift=("relative_shift", "mean"),
            std_relative_shift=("relative_shift", "std"),
            mean_cohens_d=("cohens_d", "mean"),
        )
        .reset_index()
    )

    # Reorder
    summary["boom_category"] = pd.Categorical(
        summary["boom_category"], categories=category_order, ordered=True
    )
    summary = summary.sort_values("boom_category").reset_index(drop=True)

    return summary
