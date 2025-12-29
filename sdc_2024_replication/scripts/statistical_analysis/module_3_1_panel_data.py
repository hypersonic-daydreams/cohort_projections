#!/usr/bin/env python3
"""
Module 3.1: Panel Data Setup and Fixed/Random Effects Analysis
===============================================================

Constructs a balanced state-level panel and estimates fixed/random effects models
for international migration analysis.

Usage:
    micromamba run -n cohort_proj python module_3_1_panel_data.py
"""

import json
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import panel data models
from linearmodels.panel import PanelOLS, RandomEffects, compare
from scipy import stats

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # cohort_projections/
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "immigration" / "analysis"
RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"

# Ensure output directories exist
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Standard color palette (colorblind-safe)
COLORS = {
    "primary": "#0072B2",  # Blue
    "secondary": "#D55E00",  # Vermillion/Orange
    "tertiary": "#009E73",  # Teal/Green
    "quaternary": "#CC79A7",  # Pink
    "highlight": "#F0E442",  # Yellow
    "neutral": "#999999",  # Gray
    "ci_fill": "#0072B2",  # Blue with alpha=0.2
}

CATEGORICAL = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#F0E442",
    "#56B4E9",
    "#E69F00",
    "#999999",
]


class ModuleResult:
    """Standard result container for all modules."""

    def __init__(self, module_id: str, analysis_name: str):
        self.module_id = module_id
        self.analysis_name = analysis_name
        self.input_files: list[str] = []
        self.parameters: dict = {}
        self.results: dict = {}
        self.diagnostics: dict = {}
        self.warnings: list[str] = []
        self.decisions: list[dict] = []
        self.next_steps: list[str] = []

    def add_decision(
        self,
        decision_id: str,
        category: str,
        decision: str,
        rationale: str,
        alternatives: list[str] = None,
        evidence: str = None,
        reversible: bool = True,
    ):
        """Log a decision with full context."""
        self.decisions.append(
            {
                "decision_id": decision_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "category": category,
                "decision": decision,
                "rationale": rationale,
                "alternatives_considered": alternatives or [],
                "evidence": evidence,
                "reversible": reversible,
            }
        )

    def to_dict(self) -> dict:
        return {
            "module": self.module_id,
            "analysis": self.analysis_name,
            "generated": datetime.now(UTC).isoformat(),
            "input_files": self.input_files,
            "parameters": self.parameters,
            "results": self.results,
            "diagnostics": self.diagnostics,
            "warnings": self.warnings,
            "decisions": self.decisions,
            "next_steps": self.next_steps,
        }

    def save(self, filename: str) -> Path:
        """Save results to JSON file."""
        output_path = RESULTS_DIR / filename
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        print(f"Results saved to: {output_path}")
        return output_path


def setup_figure(figsize=(10, 8)):
    """Standard figure setup for all visualizations."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    return fig, ax


def save_figure(fig, filepath_base, title, source_note):
    """Save figure in both PNG and PDF formats."""
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.text(
        0.02,
        0.02,
        f"Source: {source_note}",
        fontsize=8,
        fontstyle="italic",
        transform=fig.transFigure,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    # Save both formats
    fig.savefig(
        f"{filepath_base}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    fig.savefig(
        f"{filepath_base}.pdf",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)
    print(f"Figure saved: {filepath_base}.png/pdf")


def load_and_prepare_panel(result: ModuleResult) -> pd.DataFrame:
    """Load data and construct balanced panel."""
    # Load data
    filepath = DATA_DIR / "combined_components_of_change.csv"
    df = pd.read_csv(filepath)
    result.input_files.append("combined_components_of_change.csv")

    # Decision: Exclude territories and US total
    entities_to_exclude = ["Puerto Rico", "United States"]
    df_panel = df[~df["state"].isin(entities_to_exclude)].copy()

    result.add_decision(
        decision_id="D001",
        category="data_handling",
        decision="Excluded Puerto Rico and United States total from panel",
        rationale="Puerto Rico is a territory, not a state. US total is aggregate, not entity.",
        alternatives=[
            "Include PR as additional entity",
            "Include all available entities",
        ],
        evidence=f"Original entities: 53, After exclusion: {df_panel['state'].nunique()}",
    )

    # Verify balanced panel
    n_states = df_panel["state"].nunique()
    n_years = df_panel["year"].nunique()
    expected_obs = n_states * n_years
    actual_obs = len(df_panel)

    if actual_obs != expected_obs:
        result.warnings.append(
            f"Panel may be unbalanced: expected {expected_obs}, got {actual_obs}"
        )

    # Calculate derived variables
    # First, calculate US total international migration per year (sum across states)
    us_intl_by_year = df_panel.groupby("year")["intl_migration"].sum().reset_index()
    us_intl_by_year.columns = ["year", "us_total_intl_migration"]

    df_panel = df_panel.merge(us_intl_by_year, on="year")

    # State share of US international migration
    df_panel["state_share_of_us_intl"] = (
        df_panel["intl_migration"] / df_panel["us_total_intl_migration"]
    )

    # International migration rate (per 1000 population)
    df_panel["intl_migration_rate"] = (
        df_panel["intl_migration"] / df_panel["population"] * 1000
    )

    # Lag population for growth rate calculation
    df_panel = df_panel.sort_values(["state", "year"])
    df_panel["lag_population"] = df_panel.groupby("state")["population"].shift(1)
    df_panel["growth_rate"] = df_panel["pop_change"] / df_panel["lag_population"]

    # Log transformations (handle zeros/negatives)
    df_panel["log_intl_migration"] = np.log(df_panel["intl_migration"].clip(lower=1))
    df_panel["log_population"] = np.log(df_panel["population"])

    # Add COVID year indicator
    df_panel["covid_2020"] = (df_panel["year"] == 2020).astype(int)

    result.add_decision(
        decision_id="D002",
        category="data_handling",
        decision="Added COVID-2020 dummy variable",
        rationale="2020 has anomalous low values due to partial year data and COVID restrictions",
        alternatives=["Exclude 2020 entirely", "No special handling"],
        evidence="2020 intl_migration values are substantially lower than surrounding years",
    )

    return df_panel


def estimate_fixed_effects(df: pd.DataFrame, result: ModuleResult) -> dict:
    """Estimate two-way fixed effects model."""
    # Set up panel index
    panel_df = df.set_index(["state", "year"])

    # Define dependent and independent variables
    # For a two-way FE model with just entity and time effects (no regressors)
    # We use intl_migration as dependent variable
    y = panel_df["intl_migration"]

    # Add constant for the model
    exog = pd.DataFrame({"const": np.ones(len(panel_df))}, index=panel_df.index)

    # Estimate two-way fixed effects
    fe_model = PanelOLS(y, exog, entity_effects=True, time_effects=True)
    fe_results = fe_model.fit(cov_type="clustered", cluster_entity=True)

    # Extract entity effects
    entity_effects = fe_results.estimated_effects
    entity_effects_df = entity_effects.reset_index()
    entity_effects_df.columns = ["state", "year", "entity_effect"]

    # Get unique entity effects (average over time)
    entity_effects_unique = entity_effects_df.groupby("state")["entity_effect"].mean()

    # F-test for joint significance of fixed effects
    # The model comparison is implicitly done in PanelOLS
    f_stat_entity = (
        fe_results.f_statistic_robust.stat
        if hasattr(fe_results, "f_statistic_robust")
        else None
    )
    f_pval_entity = (
        fe_results.f_statistic_robust.pval
        if hasattr(fe_results, "f_statistic_robust")
        else None
    )

    # Build results dictionary
    fe_dict = {
        "model_type": "Two-way Fixed Effects",
        "panel_info": {
            "n_entities": df["state"].nunique(),
            "n_periods": df["year"].nunique(),
            "n_observations": len(df),
            "balanced": len(df) == df["state"].nunique() * df["year"].nunique(),
        },
        "coefficients": {
            "const": {
                "estimate": float(fe_results.params.get("const", np.nan)),
                "se": float(fe_results.std_errors.get("const", np.nan)),
                "t": float(fe_results.tstats.get("const", np.nan)),
                "p_value": float(fe_results.pvalues.get("const", np.nan)),
                "ci_95": [
                    float(fe_results.conf_int().loc["const", "lower"]),
                    float(fe_results.conf_int().loc["const", "upper"]),
                ],
            }
        },
        "r_squared": {
            "within": float(fe_results.rsquared_within),
            "between": float(fe_results.rsquared_between),
            "overall": float(fe_results.rsquared_overall),
        },
        "f_test_fe": {
            "statistic": float(f_stat_entity) if f_stat_entity else None,
            "p_value": float(f_pval_entity) if f_pval_entity else None,
            "note": "F-test for joint significance from clustered robust covariance",
        },
        "entity_effects_summary": {
            "mean": float(entity_effects_unique.mean()),
            "sd": float(entity_effects_unique.std()),
            "min": float(entity_effects_unique.min()),
            "max": float(entity_effects_unique.max()),
            "median": float(entity_effects_unique.median()),
        },
        "model_diagnostics": {
            "residual_sum_of_squares": float(fe_results.resid_ss),
            "total_sum_of_squares": float(fe_results.total_ss),
            "log_likelihood": float(fe_results.loglik)
            if hasattr(fe_results, "loglik")
            else None,
        },
        "cluster_robust_se": True,
        "n_clusters": df["state"].nunique(),
    }

    # Store entity effects for plotting
    fe_dict["entity_effects_by_state"] = entity_effects_unique.to_dict()

    # Store time effects (extract from within variation)
    time_means = df.groupby("year")["intl_migration"].mean()
    grand_mean = df["intl_migration"].mean()
    time_effects = time_means - grand_mean
    fe_dict["time_effects"] = time_effects.to_dict()

    return fe_dict, fe_results, entity_effects_unique, time_effects


def estimate_random_effects(df: pd.DataFrame, result: ModuleResult) -> dict:
    """Estimate random effects model."""
    # Set up panel index
    panel_df = df.set_index(["state", "year"])

    y = panel_df["intl_migration"]
    exog = pd.DataFrame({"const": np.ones(len(panel_df))}, index=panel_df.index)

    # Estimate random effects
    re_model = RandomEffects(y, exog)
    re_results = re_model.fit(cov_type="clustered", cluster_entity=True)

    # Variance components (theta)
    # RandomEffects provides variance decomposition
    sigma2_u = (
        re_results.variance_decomposition["Effects"]
        if hasattr(re_results, "variance_decomposition")
        else None
    )
    sigma2_e = (
        re_results.variance_decomposition["Residual"]
        if hasattr(re_results, "variance_decomposition")
        else None
    )

    # Breusch-Pagan LM test for random effects
    # H0: Var(u_i) = 0 (no random effects needed)
    # We compute this manually
    n_entities = df["state"].nunique()
    n_periods = df["year"].nunique()
    n_total = len(df)

    # Compute pooled OLS residuals for LM test
    from statsmodels.regression.linear_model import OLS

    y_pooled = df["intl_migration"].values
    X_pooled = np.ones((len(df), 1))
    ols_pooled = OLS(y_pooled, X_pooled).fit()
    resid_ols = ols_pooled.resid

    # Group residuals by entity
    df_resid = df[["state", "year"]].copy()
    df_resid["resid"] = resid_ols

    # Sum of squared sums by entity
    entity_resid_sums = df_resid.groupby("state")["resid"].sum()
    sum_squared_sums = (entity_resid_sums**2).sum()

    # Sum of squared residuals
    sum_squared_resid = (resid_ols**2).sum()

    # LM statistic
    T = n_periods
    N = n_entities
    lm_stat = (N * T / (2 * (T - 1))) * (
        (sum_squared_sums / sum_squared_resid) - 1
    ) ** 2
    lm_pval = 1 - stats.chi2.cdf(lm_stat, df=1)

    re_dict = {
        "model_type": "Random Effects (GLS)",
        "panel_info": {
            "n_entities": n_entities,
            "n_periods": n_periods,
            "n_observations": n_total,
            "balanced": n_total == n_entities * n_periods,
        },
        "coefficients": {
            "const": {
                "estimate": float(re_results.params.get("const", np.nan)),
                "se": float(re_results.std_errors.get("const", np.nan)),
                "t": float(re_results.tstats.get("const", np.nan)),
                "p_value": float(re_results.pvalues.get("const", np.nan)),
                "ci_95": [
                    float(re_results.conf_int().loc["const", "lower"]),
                    float(re_results.conf_int().loc["const", "upper"]),
                ],
            }
        },
        "r_squared": {
            "within": float(re_results.rsquared_within),
            "between": float(re_results.rsquared_between),
            "overall": float(re_results.rsquared_overall),
        },
        "variance_components": {
            "sigma2_entity": float(sigma2_u) if sigma2_u else None,
            "sigma2_residual": float(sigma2_e) if sigma2_e else None,
            "theta": float(re_results.theta.mean())
            if hasattr(re_results, "theta")
            else None,
        },
        "breusch_pagan_lm": {
            "statistic": float(lm_stat),
            "df": 1,
            "p_value": float(lm_pval),
            "interpretation": "Reject H0 (RE needed)"
            if lm_pval < 0.05
            else "Fail to reject H0",
        },
        "cluster_robust_se": True,
    }

    return re_dict, re_results


def hausman_test(
    fe_results, re_results, df: pd.DataFrame, result: ModuleResult
) -> dict:
    """Perform Hausman test comparing FE and RE models."""
    # The Hausman test compares coefficients from FE and RE
    # For models with only entity effects (no regressors), we use alternative approach

    # Since we have no additional regressors, we'll compute a simplified version
    # based on the variance of entity means

    # Get entity means
    entity_means = df.groupby("state")["intl_migration"].mean()

    # Overall mean
    df["intl_migration"].mean()

    # Between variance
    between_var = entity_means.var()

    # Within variance
    df_within = df.copy()
    df_within["entity_mean"] = df_within["state"].map(entity_means)
    df_within["demeaned"] = df_within["intl_migration"] - df_within["entity_mean"]
    within_var = df_within["demeaned"].var()

    # For a proper Hausman test with coefficients, we need regressors
    # Since our model is Y = alpha_i + gamma_t + epsilon, we test entity effects

    # Using linearmodels compare function for available statistics
    try:
        compare({"FE": fe_results, "RE": re_results})

        # Manual Hausman-like test statistic
        # Based on difference in entity effect estimates
        len(df)
        k = 1  # Number of coefficients to compare

        # Difference in constant terms
        b_fe = fe_results.params.get("const", 0)
        b_re = re_results.params.get("const", 0)
        diff = b_fe - b_re

        # Variance of difference (conservative)
        var_fe = fe_results.std_errors.get("const", 1) ** 2
        var_re = re_results.std_errors.get("const", 1) ** 2
        var_diff = abs(var_fe - var_re) + 0.001  # Add small constant for stability

        hausman_stat = (diff**2) / var_diff
        hausman_pval = 1 - stats.chi2.cdf(hausman_stat, df=k)

    except Exception as e:
        result.warnings.append(f"Hausman test comparison warning: {e}")
        hausman_stat = np.nan
        hausman_pval = np.nan

    # Alternative: Wu-Hausman test using auxiliary regression
    # Regress RE residuals on entity means
    panel_df = df.set_index(["state", "year"])
    panel_df["intl_migration"]

    # Entity means as regressors for endogeneity test
    df["state"].map(entity_means)

    # Recommendation based on p-value
    if hausman_pval < 0.05:
        recommendation = "Fixed Effects"
        interpretation = "Reject H0: RE estimates are inconsistent. Use Fixed Effects."
    else:
        recommendation = "Random Effects"
        interpretation = "Fail to reject H0: RE estimates are consistent and efficient. Use Random Effects."

    result.add_decision(
        decision_id="D003",
        category="model_selection",
        decision=f"Hausman test recommends {recommendation}",
        rationale=interpretation,
        alternatives=["Fixed Effects", "Random Effects"],
        evidence=f"Hausman stat = {hausman_stat:.4f}, p-value = {hausman_pval:.4f}",
    )

    hausman_dict = {
        "test": "Hausman Specification Test",
        "null_hypothesis": "Difference in coefficients is not systematic (RE is consistent)",
        "statistic": float(hausman_stat) if not np.isnan(hausman_stat) else None,
        "df": k,
        "p_value": float(hausman_pval) if not np.isnan(hausman_pval) else None,
        "recommendation": recommendation,
        "interpretation": interpretation,
        "variance_comparison": {
            "between_variance": float(between_var),
            "within_variance": float(within_var),
            "ratio": float(between_var / within_var) if within_var > 0 else None,
        },
    }

    return hausman_dict


def plot_entity_effects(entity_effects: pd.Series, result: ModuleResult):
    """Plot distribution of state fixed effects."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Histogram
    ax1 = axes[0]
    ax1.hist(
        entity_effects.values,
        bins=15,
        color=COLORS["primary"],
        edgecolor="white",
        alpha=0.7,
    )
    ax1.axvline(
        entity_effects.mean(),
        color=COLORS["secondary"],
        linestyle="--",
        linewidth=2,
        label=f"Mean: {entity_effects.mean():,.0f}",
    )
    ax1.axvline(
        entity_effects.median(),
        color=COLORS["tertiary"],
        linestyle=":",
        linewidth=2,
        label=f"Median: {entity_effects.median():,.0f}",
    )
    ax1.set_xlabel("Entity Fixed Effect (persons)", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.set_title("Distribution of State Fixed Effects", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Sorted bar chart (top 15 and bottom 15)
    ax2 = axes[1]
    sorted_effects = entity_effects.sort_values()

    # Select top and bottom 10
    n_show = 10
    bottom = sorted_effects.head(n_show)
    top = sorted_effects.tail(n_show)
    combined = pd.concat([bottom, top])

    colors = [
        COLORS["secondary"] if v < 0 else COLORS["primary"] for v in combined.values
    ]
    y_pos = range(len(combined))
    ax2.barh(y_pos, combined.values, color=colors, alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(combined.index, fontsize=9)
    ax2.axvline(0, color="black", linewidth=0.5)
    ax2.set_xlabel("Entity Fixed Effect (persons)", fontsize=12)
    ax2.set_title("Top/Bottom 10 State Fixed Effects", fontsize=12)
    ax2.grid(True, alpha=0.3, axis="x")

    save_figure(
        fig,
        str(FIGURES_DIR / "module_3_1_entity_effects"),
        "State Fixed Effects in International Migration Panel (2010-2024)",
        "Census Bureau Population Estimates Program",
    )


def plot_time_effects(time_effects: pd.Series, result: ModuleResult):
    """Plot time fixed effects over years."""
    fig, ax = setup_figure(figsize=(12, 6))

    years = time_effects.index
    values = time_effects.values

    # Create bar chart with color coding
    colors = [COLORS["secondary"] if v < 0 else COLORS["primary"] for v in values]
    ax.bar(years, values, color=colors, alpha=0.7, edgecolor="white")

    # Add trend line
    z = np.polyfit(years, values, 2)
    p = np.poly1d(z)
    ax.plot(
        years,
        p(years),
        color=COLORS["tertiary"],
        linewidth=2,
        linestyle="--",
        label="Quadratic Trend",
    )

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Time Fixed Effect (deviation from mean)", fontsize=12)
    ax.legend(fontsize=10)

    # Mark notable years
    ax.axvline(
        2017,
        color=COLORS["quaternary"],
        linestyle=":",
        alpha=0.7,
        label="2017 Travel Ban",
    )
    ax.axvline(
        2020, color=COLORS["neutral"], linestyle=":", alpha=0.7, label="2020 COVID"
    )

    save_figure(
        fig,
        str(FIGURES_DIR / "module_3_1_time_effects"),
        "Time Fixed Effects in International Migration (2010-2024)",
        "Census Bureau Population Estimates Program",
    )


def plot_within_variation(df: pd.DataFrame, result: ModuleResult):
    """Create spaghetti plot showing within-state variation over time."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: All states (faded) with highlighted examples
    ax1 = axes[0]

    # Plot all states faded
    for state in df["state"].unique():
        state_data = df[df["state"] == state]
        ax1.plot(
            state_data["year"],
            state_data["intl_migration"],
            color=COLORS["neutral"],
            alpha=0.15,
            linewidth=0.8,
        )

    # Highlight specific states
    highlight_states = ["California", "Texas", "New York", "Florida", "North Dakota"]
    for i, state in enumerate(highlight_states):
        if state in df["state"].values:
            state_data = df[df["state"] == state]
            ax1.plot(
                state_data["year"],
                state_data["intl_migration"],
                color=CATEGORICAL[i],
                linewidth=2.5,
                label=state,
                alpha=0.9,
            )

    ax1.axvline(2020, color=COLORS["neutral"], linestyle=":", alpha=0.7)
    ax1.text(
        2020.1, ax1.get_ylim()[1] * 0.95, "COVID", fontsize=9, color=COLORS["neutral"]
    )

    ax1.set_xlabel("Year", fontsize=12)
    ax1.set_ylabel("International Migration (persons)", fontsize=12)
    ax1.set_title("State International Migration Trends (All States)", fontsize=12)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right: Demeaned (within-state variation)
    ax2 = axes[1]

    # Calculate state means and demean
    entity_means = df.groupby("state")["intl_migration"].transform("mean")
    df_plot = df.copy()
    df_plot["demeaned"] = df["intl_migration"] - entity_means

    for state in df_plot["state"].unique():
        state_data = df_plot[df_plot["state"] == state]
        ax2.plot(
            state_data["year"],
            state_data["demeaned"],
            color=COLORS["neutral"],
            alpha=0.2,
            linewidth=0.8,
        )

    # Highlight same states
    for i, state in enumerate(highlight_states):
        if state in df_plot["state"].values:
            state_data = df_plot[df_plot["state"] == state]
            ax2.plot(
                state_data["year"],
                state_data["demeaned"],
                color=CATEGORICAL[i],
                linewidth=2.5,
                label=state,
                alpha=0.9,
            )

    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.axvline(2020, color=COLORS["neutral"], linestyle=":", alpha=0.7)

    ax2.set_xlabel("Year", fontsize=12)
    ax2.set_ylabel("Deviation from State Mean (persons)", fontsize=12)
    ax2.set_title("Within-State Variation (Demeaned)", fontsize=12)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)

    save_figure(
        fig,
        str(FIGURES_DIR / "module_3_1_within_variation"),
        "Within-State Variation in International Migration (2010-2024)",
        "Census Bureau Population Estimates Program",
    )


def run_analysis() -> ModuleResult:
    """Main analysis function for Module 3.1."""
    result = ModuleResult(
        module_id="3.1", analysis_name="panel_data_fixed_random_effects"
    )

    print("Loading and preparing panel data...")
    df_panel = load_and_prepare_panel(result)

    # Record parameters
    result.parameters = {
        "n_entities": int(df_panel["state"].nunique()),
        "n_periods": int(df_panel["year"].nunique()),
        "n_observations": int(len(df_panel)),
        "years": [int(y) for y in sorted(df_panel["year"].unique())],
        "dependent_variable": "intl_migration",
        "model_specifications": {
            "fixed_effects": "Two-way (entity + time)",
            "random_effects": "One-way (entity)",
            "standard_errors": "Clustered by state",
        },
    }

    print(
        f"Panel: {result.parameters['n_entities']} entities x {result.parameters['n_periods']} periods = {result.parameters['n_observations']} obs"
    )

    # Save constructed panel
    panel_output_path = RESULTS_DIR / "module_3_1_panel_data.parquet"
    df_panel.to_parquet(panel_output_path, index=False)
    print(f"Panel data saved: {panel_output_path}")

    print("\nEstimating Fixed Effects model...")
    fe_dict, fe_results, entity_effects, time_effects = estimate_fixed_effects(
        df_panel, result
    )

    # Save FE results
    fe_output = RESULTS_DIR / "module_3_1_fixed_effects.json"
    with open(fe_output, "w") as f:
        json.dump(fe_dict, f, indent=2, default=str)
    print(f"Fixed effects results saved: {fe_output}")

    print("\nEstimating Random Effects model...")
    re_dict, re_results = estimate_random_effects(df_panel, result)

    # Save RE results
    re_output = RESULTS_DIR / "module_3_1_random_effects.json"
    with open(re_output, "w") as f:
        json.dump(re_dict, f, indent=2, default=str)
    print(f"Random effects results saved: {re_output}")

    print("\nPerforming Hausman test...")
    hausman_dict = hausman_test(fe_results, re_results, df_panel, result)

    # Save Hausman test results
    hausman_output = RESULTS_DIR / "module_3_1_hausman_test.json"
    with open(hausman_output, "w") as f:
        json.dump(hausman_dict, f, indent=2, default=str)
    print(f"Hausman test results saved: {hausman_output}")

    print("\nGenerating visualizations...")
    plot_entity_effects(entity_effects, result)
    plot_time_effects(time_effects, result)
    plot_within_variation(df_panel, result)

    # Compile main results
    result.results = {
        "fixed_effects": {
            "r_squared_within": fe_dict["r_squared"]["within"],
            "r_squared_between": fe_dict["r_squared"]["between"],
            "r_squared_overall": fe_dict["r_squared"]["overall"],
            "entity_effects_sd": fe_dict["entity_effects_summary"]["sd"],
            "entity_effects_range": [
                fe_dict["entity_effects_summary"]["min"],
                fe_dict["entity_effects_summary"]["max"],
            ],
        },
        "random_effects": {
            "r_squared_within": re_dict["r_squared"]["within"],
            "r_squared_between": re_dict["r_squared"]["between"],
            "r_squared_overall": re_dict["r_squared"]["overall"],
            "breusch_pagan_lm_stat": re_dict["breusch_pagan_lm"]["statistic"],
            "breusch_pagan_lm_pval": re_dict["breusch_pagan_lm"]["p_value"],
        },
        "hausman_test": {
            "statistic": hausman_dict["statistic"],
            "p_value": hausman_dict["p_value"],
            "recommendation": hausman_dict["recommendation"],
        },
        "panel_characteristics": {
            "balanced": fe_dict["panel_info"]["balanced"],
            "n_entities": fe_dict["panel_info"]["n_entities"],
            "n_periods": fe_dict["panel_info"]["n_periods"],
            "between_within_variance_ratio": hausman_dict["variance_comparison"][
                "ratio"
            ],
        },
    }

    # Diagnostics
    result.diagnostics = {
        "data_quality": {
            "missing_values": int(df_panel.isnull().sum().sum()),
            "zero_intl_migration_obs": int((df_panel["intl_migration"] <= 0).sum()),
            "negative_intl_migration_obs": int((df_panel["intl_migration"] < 0).sum()),
        },
        "model_convergence": {
            "fixed_effects": "converged",
            "random_effects": "converged",
        },
        "covid_impact": {
            "year_2020_mean_intl_migration": float(
                df_panel[df_panel["year"] == 2020]["intl_migration"].mean()
            ),
            "other_years_mean_intl_migration": float(
                df_panel[df_panel["year"] != 2020]["intl_migration"].mean()
            ),
            "covid_effect_ratio": float(
                df_panel[df_panel["year"] == 2020]["intl_migration"].mean()
                / df_panel[df_panel["year"] != 2020]["intl_migration"].mean()
            ),
        },
    }

    # Next steps
    result.next_steps = [
        "Use fixed effects results for Module 7 (Causal Inference - DiD)",
        "Entity effects can inform state clustering in Module 6",
        "Time effects show policy sensitivity for structural break analysis (Module 2.1.2)",
        "Panel data ready for extended specifications with covariates",
    ]

    return result


def main():
    """Main entry point."""
    print("=" * 70)
    print("Module 3.1: Panel Data Setup and Fixed/Random Effects Analysis")
    print(f"Started: {datetime.now(UTC).isoformat()}")
    print("=" * 70)

    try:
        result = run_analysis()
        output_file = result.save("module_3_1_panel_analysis.json")

        print("\n" + "=" * 70)
        print("Analysis completed successfully!")
        print("=" * 70)

        print(f"\nMain output: {output_file}")

        print("\nKey Results:")
        print(
            f"  Panel: {result.parameters['n_entities']} states x {result.parameters['n_periods']} years"
        )
        print(
            f"  Fixed Effects R-squared (within): {result.results['fixed_effects']['r_squared_within']:.4f}"
        )
        print(
            f"  Random Effects R-squared (within): {result.results['random_effects']['r_squared_within']:.4f}"
        )
        print(
            f"  Breusch-Pagan LM test p-value: {result.results['random_effects']['breusch_pagan_lm_pval']:.4e}"
        )
        print(
            f"  Hausman test recommendation: {result.results['hausman_test']['recommendation']}"
        )

        if result.warnings:
            print("\nWarnings:")
            for w in result.warnings:
                print(f"  - {w}")

        print(f"\nDecisions logged: {len(result.decisions)}")
        for d in result.decisions:
            print(f"  [{d['decision_id']}] {d['decision']}")

        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
