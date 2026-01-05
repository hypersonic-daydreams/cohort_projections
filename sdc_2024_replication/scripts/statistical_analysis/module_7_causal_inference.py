#!/usr/bin/env python3
"""
Module 7: Causal Inference Agent - Difference-in-Differences and Event Studies
================================================================================

Implements causal inference methods for policy event analysis:
1. Difference-in-Differences (DiD) for 2017 Travel Ban and 2020 COVID
2. Event Study specification with dynamic treatment effects
3. Synthetic comparator (descriptive benchmark) for North Dakota
4. Shift-share (Bartik) instrument approach

Usage:
    micromamba run -n cohort_proj python module_7_causal_inference.py
"""

import json
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Statsmodels imports
import statsmodels.api as sm

# Panel data models
from scipy import stats
from statsmodels.regression.linear_model import OLS

# Add scripts directory to path to find db_config
sys.path.append(str(Path(__file__).parent.parent))
from database import db_config

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # cohort_projections/
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "immigration" / "analysis"
RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"
ARTICLE_FIGURES_DIR = Path(__file__).parent / "journal_article" / "figures"

# Ensure output directories exist
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
ARTICLE_FIGURES_DIR.mkdir(exist_ok=True)

# Standard color palette (colorblind-safe)
COLORS = {
    "primary": "#0072B2",  # Blue
    "secondary": "#D55E00",  # Vermillion/Orange
    "tertiary": "#009E73",  # Teal/Green
    "quaternary": "#CC79A7",  # Pink
    "highlight": "#F0E442",  # Yellow
    "neutral": "#999999",  # Gray
    "ci_fill": "#0072B2",  # Blue with alpha=0.2
    "treatment": "#E31A1C",  # Red for treatment effect
    "control": "#1F78B4",  # Blue for control
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


class DiDResult(NamedTuple):
    """Container for DiD estimation results."""

    att: float
    se: float
    t_stat: float
    p_value: float
    ci_lower: float
    ci_upper: float
    n_treatment: int
    n_control: int
    n_pre: int
    n_post: int
    pre_trend_f: float
    pre_trend_p: float


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

    if isinstance(filepath_base, (str, Path)):
        filepath_bases = [filepath_base]
    else:
        filepath_bases = list(filepath_base)

    # Save both formats
    for base in filepath_bases:
        fig.savefig(
            f"{base}.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        fig.savefig(
            f"{base}.pdf",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
    plt.close(fig)
    if len(filepath_bases) == 1:
        print(f"Figure saved: {filepath_bases[0]}.png/pdf")
    else:
        saved_bases = ", ".join(str(base) for base in filepath_bases)
        print(f"Figure saved: {saved_bases}.png/pdf")


# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================


from data_loader import load_refugee_arrivals, load_state_components


def load_data(result: ModuleResult) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all required data for causal inference analysis from shared loader."""

    # 1. Load Census Components (Generic)
    df_components = load_state_components()
    result.input_files.append("census.state_components (PostgreSQL via data_loader)")
    print(f"Loaded components of change (DB): {df_components.shape}")

    # 2. Load Refugee Arrivals
    df_refugee = load_refugee_arrivals()
    result.input_files.append("rpc.refugee_arrivals (PostgreSQL via data_loader)")
    print(f"Loaded refugee arrivals (DB): {df_refugee.shape}")

    # 3. Create Panel Data
    # Filter out aggregates
    df_panel = df_components[
        ~df_components["state"].isin(
            ["Puerto Rico", "United States", "US Region", "US Division"]
        )
    ].copy()

    return df_components, df_refugee, df_panel


def prepare_travel_ban_did_data(
    df_refugee: pd.DataFrame, result: ModuleResult
) -> pd.DataFrame:
    """
    Prepare data for Travel Ban DiD analysis.

    Treatment: Nationalities affected by Travel Ban (Iran, Iraq, Libya, Somalia, Sudan, Syria, Yemen)
    Event year: 2017 (Executive Order signed January 2017, took effect March 2017)
    """
    # Travel Ban affected countries (original 7 from EO 13769)
    travel_ban_countries = [
        "Iran",
        "Iraq",
        "Libya",
        "Somalia",
        "Sudan",
        "Syria",
        "Yemen",
    ]

    # Aggregate refugee arrivals by nationality and year
    df_nat_year = (
        df_refugee.groupby(["fiscal_year", "nationality"])["arrivals"]
        .sum()
        .reset_index()
    )
    df_nat_year.columns = ["year", "nationality", "arrivals"]

    # Create treatment indicator
    df_nat_year["treated"] = (
        df_nat_year["nationality"].isin(travel_ban_countries).astype(int)
    )

    # Post indicator (Travel Ban effective in 2017, but full year effect in 2018+)
    # Using 2018 as post period since 2017 was partial year
    df_nat_year["post"] = (df_nat_year["year"] >= 2018).astype(int)

    # Interaction term (DiD estimator)
    df_nat_year["treated_x_post"] = df_nat_year["treated"] * df_nat_year["post"]

    # Log arrivals (add 1 to handle zeros)
    df_nat_year["log_arrivals"] = np.log(df_nat_year["arrivals"] + 1)

    # Relative time to treatment
    df_nat_year["rel_time"] = df_nat_year["year"] - 2018  # 2018 is first full post year

    result.add_decision(
        decision_id="D001",
        category="causal_identification",
        decision="Use 2018 as first post-treatment year for Travel Ban DiD",
        rationale="Executive Order 13769 signed Jan 27, 2017, repeatedly blocked. "
        "2017 reflects partial implementation; 2018 is first full year of effect.",
        alternatives=["Use 2017 as first post year", "Use phased treatment intensity"],
        evidence=f"Affected countries: {travel_ban_countries}",
    )

    return df_nat_year


def prepare_covid_did_data(
    df_panel: pd.DataFrame, result: ModuleResult
) -> pd.DataFrame:
    """
    Prepare data for COVID-19 DiD analysis.

    This uses state-level international migration as outcome.
    Treatment: All states (universal shock), so we compare to synthetic counterfactual
    Alternative: Use pre-COVID trend as control
    """
    df = df_panel.copy()

    # Post indicator for COVID
    df["post_covid"] = (df["year"] >= 2020).astype(int)

    # For universal treatment, we can still identify effect using:
    # 1. Pre-trend interruption
    # 2. Comparison with domestic migration (less affected pathway)

    # Create relative time
    df["rel_time_covid"] = df["year"] - 2020

    # Log transformation
    df["log_intl_migration"] = np.log(df["intl_migration"].clip(lower=1))
    df["log_domestic_migration"] = np.where(
        df["domestic_migration"] > 0,
        np.log(df["domestic_migration"]),
        -np.log(-df["domestic_migration"] + 1),  # Handle negative values
    )

    result.add_decision(
        decision_id="D002",
        category="causal_identification",
        decision="Use 2020 as COVID treatment year for state-level DiD",
        rationale="COVID-19 travel restrictions began March 2020, affecting international migration universally",
        alternatives=[
            "Use 2021 to capture full-year effect",
            "Exclude 2020 due to partial year data",
        ],
        evidence="Census PEP shows 2020 as partial year with anomalous low values",
    )

    return df


# =============================================================================
# DIFFERENCE-IN-DIFFERENCES ESTIMATION
# =============================================================================


def estimate_did_travel_ban(df: pd.DataFrame, result: ModuleResult) -> dict:
    """
    Estimate DiD for Travel Ban effect on refugee arrivals.

    Model: log(arrivals_it) = alpha + beta*Treated_i + gamma*Post_t + delta*(Treated*Post) + epsilon

    ATT = delta (coefficient on interaction term)
    """
    print("\n" + "=" * 60)
    print("DIFFERENCE-IN-DIFFERENCES: 2017 TRAVEL BAN")
    print("=" * 60)

    # Exclude 2020 due to COVID confounding
    df_analysis = df[df["year"] < 2020].copy()

    # Sample info
    n_treated = df_analysis[df_analysis["treated"] == 1]["nationality"].nunique()
    n_control = df_analysis[df_analysis["treated"] == 0]["nationality"].nunique()
    n_pre = len(df_analysis[df_analysis["post"] == 0]["year"].unique())
    n_post = len(df_analysis[df_analysis["post"] == 1]["year"].unique())

    print("\nSample Information:")
    print(f"  Treatment units (nationalities): {n_treated}")
    print(f"  Control units (nationalities): {n_control}")
    print(f"  Pre-treatment periods: {n_pre} (2002-2017)")
    print(f"  Post-treatment periods: {n_post} (2018-2019)")
    print(f"  Total observations: {len(df_analysis)}")

    # Create design matrix - reset index to avoid alignment issues
    df_analysis = df_analysis.reset_index(drop=True)
    y = df_analysis["log_arrivals"].values

    # Note: In TWFE DiD, the 'treated' main effect is absorbed by entity (nationality) FE
    # and the 'post' main effect is absorbed by time (year) FE.
    # We only need the interaction term (ATT estimator).
    X = pd.DataFrame({"treated_x_post": df_analysis["treated_x_post"].values})

    # Add nationality fixed effects (absorbs 'treated')
    nationality_dummies = pd.get_dummies(
        df_analysis["nationality"], prefix="nat", drop_first=True
    )
    X_fe = pd.concat(
        [X.reset_index(drop=True), nationality_dummies.reset_index(drop=True)], axis=1
    )

    # Add year fixed effects (absorbs 'post')
    year_dummies = pd.get_dummies(df_analysis["year"], prefix="year", drop_first=True)
    X_twfe = pd.concat(
        [X_fe.reset_index(drop=True), year_dummies.reset_index(drop=True)], axis=1
    )

    # Add constant
    X_twfe = sm.add_constant(X_twfe)

    # Ensure all numeric types
    X_twfe = X_twfe.astype(float)

    # Estimate with clustered standard errors (nationality-level)
    model_twfe = OLS(y, X_twfe).fit(
        cov_type="cluster", cov_kwds={"groups": df_analysis["nationality"]}
    )

    # Extract ATT
    att = model_twfe.params["treated_x_post"]
    se = model_twfe.bse["treated_x_post"]
    t_stat = model_twfe.tvalues["treated_x_post"]
    p_value = model_twfe.pvalues["treated_x_post"]
    ci = model_twfe.conf_int().loc["treated_x_post"]

    # Convert from log to percentage change
    pct_effect = (np.exp(att) - 1) * 100

    print(f"\n{'='*60}")
    print("TWO-WAY FIXED EFFECTS DiD RESULTS")
    print("=" * 60)
    print("Dependent Variable: log(refugee arrivals + 1)")
    print("Fixed Effects: Nationality, Year")
    print("Standard Errors: Clustered by nationality")
    print("-" * 60)
    print("\nATT (Treated x Post coefficient):")
    print(f"  Estimate:     {att:>10.4f}")
    print(f"  Std. Error:   {se:>10.4f}")
    print(f"  t-statistic:  {t_stat:>10.4f}")
    print(
        f"  p-value:      {p_value:>10.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''}"
    )
    print(f"  95% CI:       [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"\n  Percentage effect: {pct_effect:.1f}%")
    print("-" * 60)

    # Pre-trend test: Test if treatment effects exist before treatment
    pre_data = df_analysis[df_analysis["post"] == 0].copy()
    pre_data["time_trend"] = pre_data["year"] - pre_data["year"].min()
    pre_data["treated_x_trend"] = pre_data["treated"] * pre_data["time_trend"]

    y_pre = pre_data["log_arrivals"].values
    X_pre = sm.add_constant(pre_data[["treated", "time_trend", "treated_x_trend"]])
    pre_trend_model = OLS(y_pre, X_pre).fit(
        cov_type="cluster", cov_kwds={"groups": pre_data["nationality"]}
    )

    # F-test for pre-trend
    pre_trend_coef = pre_trend_model.params["treated_x_trend"]
    pre_trend_t = pre_trend_model.tvalues["treated_x_trend"]
    pre_trend_p = pre_trend_model.pvalues["treated_x_trend"]

    print("\nPRE-TREND TEST:")
    print("  H0: Treatment and control have parallel trends pre-treatment")
    print(f"  Treated x Trend coefficient: {pre_trend_coef:.4f}")
    print(f"  t-statistic: {pre_trend_t:.4f}")
    print(f"  p-value: {pre_trend_p:.4f}")
    print(
        f"  Interpretation: {'FAIL - Pre-trends may differ' if pre_trend_p < 0.05 else 'PASS - Parallel trends supported'}"
    )

    result.add_decision(
        decision_id="D003",
        category="model_specification",
        decision="Use two-way fixed effects with nationality-clustered standard errors",
        rationale="Nationality FE controls for time-invariant heterogeneity; Year FE controls for common shocks",
        alternatives=["HC1/HC3 robust SE", "Wild bootstrap", "Random effects"],
        evidence=f"ATT = {att:.4f} (SE = {se:.4f})",
    )

    # Build results dictionary
    did_results = {
        "analysis": "Travel Ban DiD",
        "treatment_definition": {
            "treated_countries": [
                "Iran",
                "Iraq",
                "Libya",
                "Somalia",
                "Sudan",
                "Syria",
                "Yemen",
            ],
            "treatment_year": 2018,
            "policy": "Executive Order 13769 (Travel Ban)",
        },
        "sample_info": {
            "n_treatment_units": n_treated,
            "n_control_units": n_control,
            "n_pre_periods": n_pre,
            "n_post_periods": n_post,
            "n_observations": len(df_analysis),
            "years_analyzed": sorted(df_analysis["year"].unique().tolist()),
        },
        "att_estimate": {
            "coefficient": float(att),
            "std_error": float(se),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "ci_95_lower": float(ci[0]),
            "ci_95_upper": float(ci[1]),
            "significance": "***"
            if p_value < 0.001
            else "**"
            if p_value < 0.01
            else "*"
            if p_value < 0.05
            else "ns",
        },
        "percentage_effect": {
            "estimate": float(pct_effect),
            "interpretation": f"Refugee arrivals from banned countries decreased by approximately {abs(pct_effect):.1f}% post-ban",
        },
        "pre_trend_test": {
            "treated_x_trend_coef": float(pre_trend_coef),
            "t_statistic": float(pre_trend_t),
            "p_value": float(pre_trend_p),
            "parallel_trends_supported": pre_trend_p >= 0.05,
        },
        "model_specification": {
            "dependent_variable": "log(arrivals + 1)",
            "fixed_effects": ["nationality", "year"],
            "standard_errors": "clustered by nationality",
        },
        "model_fit": {
            "r_squared": float(model_twfe.rsquared),
            "r_squared_adj": float(model_twfe.rsquared_adj),
            "n_observations": int(model_twfe.nobs),
            "df_model": int(model_twfe.df_model),
            "df_resid": int(model_twfe.df_resid),
        },
    }

    return did_results, df_analysis


def estimate_did_covid(df: pd.DataFrame, result: ModuleResult) -> dict:
    """
    Estimate COVID effect on international migration using interrupted time series.

    Since COVID is a universal shock, we use:
    1. Pre-COVID trend as counterfactual
    2. State fixed effects for heterogeneity
    """
    print("\n" + "=" * 60)
    print("INTERRUPTED TIME SERIES: 2020 COVID IMPACT")
    print("=" * 60)

    # Create panel structure
    df_analysis = df[~df["state"].isin(["Puerto Rico", "United States"])].copy()

    n_states = df_analysis["state"].nunique()
    df_analysis["year"].nunique()
    n_pre = len(df_analysis[df_analysis["year"] < 2020]["year"].unique())
    n_post = len(df_analysis[df_analysis["year"] >= 2020]["year"].unique())

    print("\nSample Information:")
    print(f"  States: {n_states}")
    print(f"  Pre-COVID periods: {n_pre} (2010-2019)")
    print(f"  Post-COVID periods: {n_post} (2020-2024)")
    print(f"  Total observations: {len(df_analysis)}")

    # Time trend variable
    df_analysis["time"] = df_analysis["year"] - df_analysis["year"].min()
    df_analysis["post_covid"] = (df_analysis["year"] >= 2020).astype(int)
    df_analysis["time_since_covid"] = np.maximum(0, df_analysis["year"] - 2020)

    y = df_analysis["intl_migration"].values.astype(float)
    X = pd.DataFrame(
        {
            "time": df_analysis["time"].values.astype(float),
            "post_covid": df_analysis["post_covid"].values.astype(float),
            "time_since_covid": df_analysis["time_since_covid"].values.astype(float),
        }
    )

    # Add state fixed effects
    state_dummies = pd.get_dummies(
        df_analysis["state"], prefix="state", drop_first=True
    )
    X = pd.concat(
        [X.reset_index(drop=True), state_dummies.reset_index(drop=True)], axis=1
    )
    X = sm.add_constant(X).astype(float)

    its_results = OLS(y, X).fit(
        cov_type="cluster", cov_kwds={"groups": df_analysis["state"]}
    )

    # Extract COVID effect
    level_effect = its_results.params["post_covid"]
    level_se = its_results.bse["post_covid"]
    level_t = its_results.tvalues["post_covid"]
    level_p = its_results.pvalues["post_covid"]
    level_ci = its_results.conf_int().loc["post_covid"].to_numpy()
    level_ci_lower = float(level_ci[0])
    level_ci_upper = float(level_ci[1])

    trend_change = its_results.params["time_since_covid"]
    trend_se = its_results.bse["time_since_covid"]
    trend_t = its_results.tvalues["time_since_covid"]
    trend_p = its_results.pvalues["time_since_covid"]

    print(f"\n{'='*60}")
    print("INTERRUPTED TIME SERIES RESULTS")
    print("=" * 60)
    print("Dependent Variable: International migration (persons)")
    print("Fixed Effects: State")
    print("Standard Errors: Clustered by state")
    print("-" * 60)

    print("\nLevel Shift (immediate COVID effect):")
    print(f"  Estimate:     {level_effect:>12,.0f}")
    print(f"  Std. Error:   {level_se:>12,.0f}")
    print(f"  t-statistic:  {level_t:>12.4f}")
    print(
        f"  p-value:      {level_p:>12.4f} {'***' if level_p < 0.001 else '**' if level_p < 0.01 else '*' if level_p < 0.05 else ''}"
    )
    print(f"  95% CI:       [{level_ci_lower:,.0f}, {level_ci_upper:,.0f}]")

    print("\nTrend Change (post-COVID recovery):")
    print(f"  Estimate:     {trend_change:>12,.0f} per year")
    print(f"  Std. Error:   {trend_se:>12,.0f}")
    print(f"  t-statistic:  {trend_t:>12.4f}")
    print(
        f"  p-value:      {trend_p:>12.4f} {'***' if trend_p < 0.001 else '**' if trend_p < 0.01 else '*' if trend_p < 0.05 else ''}"
    )

    # Calculate mean pre-COVID as baseline
    pre_covid_mean = df_analysis[df_analysis["year"] < 2020]["intl_migration"].mean()
    pct_level_effect = (
        (level_effect / pre_covid_mean) * 100 if pre_covid_mean != 0 else 0
    )

    print(f"\n  Pre-COVID mean: {pre_covid_mean:,.0f}")
    print(f"  Level effect as % of mean: {pct_level_effect:.1f}%")

    covid_results = {
        "analysis": "COVID-19 Interrupted Time Series",
        "treatment_definition": {
            "event": "COVID-19 pandemic",
            "treatment_year": 2020,
            "policy": "Travel restrictions and border closures",
        },
        "sample_info": {
            "n_states": n_states,
            "n_pre_periods": n_pre,
            "n_post_periods": n_post,
            "n_observations": len(df_analysis),
            "years_analyzed": sorted(df_analysis["year"].unique().tolist()),
        },
        "level_effect": {
            "coefficient": float(level_effect),
            "std_error": float(level_se),
            "t_statistic": float(level_t),
            "p_value": float(level_p),
            "ci_95_lower": level_ci_lower,
            "ci_95_upper": level_ci_upper,
            "significance": "***"
            if level_p < 0.001
            else "**"
            if level_p < 0.01
            else "*"
            if level_p < 0.05
            else "ns",
        },
        "trend_change": {
            "coefficient": float(trend_change),
            "std_error": float(trend_se),
            "t_statistic": float(trend_t),
            "p_value": float(trend_p),
            "interpretation": f"International migration changes by {trend_change:,.0f} per year post-COVID",
        },
        "percentage_effect": {
            "pre_covid_mean": float(pre_covid_mean),
            "level_effect_pct": float(pct_level_effect),
        },
        "model_specification": {
            "dependent_variable": "intl_migration",
            "fixed_effects": ["state"],
            "standard_errors": "clustered by state",
        },
        "model_fit": {
            "r_squared": float(its_results.rsquared),
            "r_squared_adj": float(its_results.rsquared_adj),
        },
    }

    return covid_results, df_analysis


# =============================================================================
# EVENT STUDY SPECIFICATION
# =============================================================================


def estimate_event_study(
    df: pd.DataFrame, result: ModuleResult
) -> tuple[dict, pd.DataFrame]:
    """
    Estimate event study for Travel Ban with dynamic treatment effects.

    Model: log(arrivals_it) = alpha_i + gamma_t + sum_k(beta_k * 1{t=k} * Treated_i) + epsilon

    where k indexes relative time periods.
    """
    print("\n" + "=" * 60)
    print("EVENT STUDY: DYNAMIC TREATMENT EFFECTS")
    print("=" * 60)

    # Exclude 2020 due to COVID
    df_analysis = df[df["year"] < 2020].copy()

    # Create relative time dummies (omit -1 as reference)
    df_analysis["year"].min()
    df_analysis["year"].max()
    df_analysis["rel_time"] = df_analysis["year"] - 2018  # 2018 is first post year

    # Get unique relative times
    rel_times = sorted(df_analysis["rel_time"].unique())

    # Remove reference period (-1)
    rel_times_no_ref = [t for t in rel_times if t != -1]

    print(f"\nRelative time periods: {rel_times}")
    print("Reference period: -1 (2017)")
    print(f"Periods estimated: {rel_times_no_ref}")

    # Reset index for clean operations
    df_analysis = df_analysis.reset_index(drop=True)

    # Convert rel_times to int to avoid numpy int64 issues
    rel_times = [int(t) for t in rel_times]
    rel_times_no_ref = [int(t) for t in rel_times_no_ref]

    # Create interaction dummies
    for t in rel_times_no_ref:
        df_analysis[f"treated_x_t{t}"] = (
            df_analysis["treated"] * (df_analysis["rel_time"] == t)
        ).astype(float)

    # Design matrix
    y = df_analysis["log_arrivals"].values

    # Interaction columns only (entity FE absorbs 'treated' main effect)
    interaction_cols = [f"treated_x_t{t}" for t in rel_times_no_ref]

    # Start building X with interaction terms
    X_dict = {}
    for col in interaction_cols:
        X_dict[col] = df_analysis[col].values
    X = pd.DataFrame(X_dict)

    # Add nationality fixed effects (absorbs 'treated')
    nat_dummies = pd.get_dummies(
        df_analysis["nationality"], prefix="nat", drop_first=True
    )
    X = pd.concat(
        [X.reset_index(drop=True), nat_dummies.reset_index(drop=True)], axis=1
    )

    # Add year fixed effects (absorbs 'post')
    year_dummies = pd.get_dummies(df_analysis["year"], prefix="year", drop_first=True)
    X = pd.concat(
        [X.reset_index(drop=True), year_dummies.reset_index(drop=True)], axis=1
    )

    # Add constant
    X = sm.add_constant(X)

    # Ensure all numeric types
    X = X.astype(float)

    # Estimate with nationality-clustered SE
    es_model = OLS(y, X).fit(
        cov_type="cluster", cov_kwds={"groups": df_analysis["nationality"]}
    )

    # Extract event study coefficients
    es_coefs = []
    for t in rel_times:
        if t == -1:
            # Reference period
            es_coefs.append(
                {
                    "rel_time": t,
                    "year": 2018 + t,
                    "coefficient": 0.0,
                    "std_error": 0.0,
                    "ci_lower": 0.0,
                    "ci_upper": 0.0,
                    "is_reference": True,
                }
            )
        else:
            col = f"treated_x_t{t}"
            if col in es_model.params:
                ci = es_model.conf_int().loc[col]
                es_coefs.append(
                    {
                        "rel_time": t,
                        "year": 2018 + t,
                        "coefficient": float(es_model.params[col]),
                        "std_error": float(es_model.bse[col]),
                        "t_statistic": float(es_model.tvalues[col]),
                        "p_value": float(es_model.pvalues[col]),
                        "ci_lower": float(ci[0]),
                        "ci_upper": float(ci[1]),
                        "is_reference": False,
                    }
                )

    es_df = pd.DataFrame(es_coefs)

    # Print results
    print(f"\n{'='*60}")
    print("EVENT STUDY COEFFICIENTS")
    print("=" * 60)
    print(f"{'Rel Time':<10} {'Year':<8} {'Coef':<12} {'SE':<10} {'95% CI':<25}")
    print("-" * 60)

    for _, row in es_df.iterrows():
        if row["is_reference"]:
            print(
                f"{row['rel_time']:<10} {row['year']:<8} {'[ref]':<12} {'-':<10} {'-':<25}"
            )
        else:
            ci_str = f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
            sig = (
                "***"
                if row["p_value"] < 0.001
                else "**"
                if row["p_value"] < 0.01
                else "*"
                if row["p_value"] < 0.05
                else ""
            )
            print(
                f"{row['rel_time']:<10} {row['year']:<8} {row['coefficient']:<12.4f} {row['std_error']:<10.4f} {ci_str:<25} {sig}"
            )

    # Pre-trend F-test
    pre_trend_cols = [f"treated_x_t{t}" for t in rel_times_no_ref if t < 0]
    if pre_trend_cols:
        # Wald test for joint significance of pre-trend coefficients
        r_matrix = np.zeros((len(pre_trend_cols), len(es_model.params)))
        for i, col in enumerate(pre_trend_cols):
            if col in es_model.params.index:
                r_matrix[i, es_model.params.index.get_loc(col)] = 1

        try:
            f_test = es_model.f_test(r_matrix)
            pre_trend_f = float(f_test.fvalue)
            pre_trend_p = float(f_test.pvalue)
        except Exception:
            # Fallback: simple joint test
            pre_trend_coefs = es_df[es_df["rel_time"] < 0]["coefficient"].values
            pre_trend_ses = es_df[es_df["rel_time"] < 0]["std_error"].values
            # Chi-squared test
            chi2_stat = np.sum((pre_trend_coefs / (pre_trend_ses + 1e-10)) ** 2)
            pre_trend_f = chi2_stat / len(pre_trend_coefs)
            pre_trend_p = 1 - stats.f.cdf(
                pre_trend_f, len(pre_trend_coefs), len(y) - len(es_model.params)
            )
    else:
        pre_trend_f = np.nan
        pre_trend_p = np.nan

    print("\nPre-Trend Joint F-test:")
    print(f"  F-statistic: {pre_trend_f:.4f}")
    print(f"  p-value: {pre_trend_p:.4f}")
    print(
        f"  Interpretation: {'Pre-trends differ (parallel trends violated)' if pre_trend_p < 0.05 else 'No significant pre-trends (parallel trends supported)'}"
    )

    es_results = {
        "analysis": "Event Study - Travel Ban",
        "specification": {
            "event_year": 2018,
            "reference_period": -1,
            "periods_estimated": rel_times_no_ref,
        },
        "coefficients": es_coefs,
        "pre_trend_test": {
            "f_statistic": float(pre_trend_f) if not np.isnan(pre_trend_f) else None,
            "p_value": float(pre_trend_p) if not np.isnan(pre_trend_p) else None,
            "parallel_trends_supported": pre_trend_p >= 0.05
            if not np.isnan(pre_trend_p)
            else None,
        },
        "model_fit": {
            "r_squared": float(es_model.rsquared),
            "r_squared_adj": float(es_model.rsquared_adj),
        },
    }

    return es_results, es_df


# =============================================================================
# SYNTHETIC CONTROL METHOD
# =============================================================================


def estimate_synthetic_control(df_panel: pd.DataFrame, result: ModuleResult) -> dict:
    """
    Estimate a synthetic comparator for North Dakota (descriptive only).

    Creates a weighted average of control states that best matches ND's pre-treatment trajectory.
    The post-period gap is reported as a descriptive benchmark, not a causal effect.
    """
    print("\n" + "=" * 60)
    print("SYNTHETIC COMPARATOR (DESCRIPTIVE): NORTH DAKOTA")
    print("=" * 60)

    # Filter to states and relevant years
    df = df_panel[~df_panel["state"].isin(["Puerto Rico", "United States"])].copy()

    # Focus on international migration rate for comparability
    df["intl_rate"] = df["intl_migration"] / df["population"] * 1000

    # Pivot to state x year matrix
    outcome_matrix = df.pivot_table(
        index="year", columns="state", values="intl_rate", aggfunc="first"
    )

    # Define treatment and control
    treated_unit = "North Dakota"
    treatment_year = 2017

    if treated_unit not in outcome_matrix.columns:
        result.warnings.append(f"{treated_unit} not found in data")
        return {"error": f"{treated_unit} not found", "feasible": False}

    # Pre-treatment period
    pre_years = outcome_matrix.index[outcome_matrix.index < treatment_year]
    post_years = outcome_matrix.index[outcome_matrix.index >= treatment_year]

    print(f"\nTreated unit: {treated_unit}")
    print(f"Treatment year: {treatment_year}")
    print(
        f"Pre-treatment periods: {len(pre_years)} ({pre_years.min()}-{pre_years.max()})"
    )
    print(
        f"Post-treatment periods: {len(post_years)} ({post_years.min()}-{post_years.max()})"
    )

    # Donor pool (exclude treated unit and states with similar characteristics to ND)
    # Also exclude states with very different sizes or migration patterns
    donor_states = [s for s in outcome_matrix.columns if s != treated_unit]

    # Get pre-treatment values
    Y_treated_pre = outcome_matrix.loc[pre_years, treated_unit].values
    Y_donors_pre = outcome_matrix.loc[pre_years, donor_states].values

    # Handle missing values
    valid_donors = ~np.any(np.isnan(Y_donors_pre), axis=0)
    donor_states = [s for i, s in enumerate(donor_states) if valid_donors[i]]
    Y_donors_pre = Y_donors_pre[:, valid_donors]

    print(f"Donor pool size: {len(donor_states)} states")

    # Solve for synthetic control weights using constrained least squares
    # min ||Y_treated - Y_donors * w||^2 s.t. w >= 0, sum(w) = 1

    from scipy.optimize import minimize

    def objective(w):
        synthetic = Y_donors_pre @ w
        return np.sum((Y_treated_pre - synthetic) ** 2)

    # Constraints
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1}  # weights sum to 1
    ]
    bounds = [(0, 1) for _ in range(len(donor_states))]  # non-negative weights

    # Initial guess (uniform weights)
    w0 = np.ones(len(donor_states)) / len(donor_states)

    # Optimize
    result_opt = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000},
    )

    weights = result_opt.x

    # Get states with positive weights
    weight_threshold = 0.01
    significant_weights = [
        (s, w)
        for s, w in zip(donor_states, weights, strict=False)
        if w > weight_threshold
    ]
    significant_weights.sort(key=lambda x: -x[1])

    print(f"\nSynthetic comparator weights (>{weight_threshold*100}%):")
    for state, weight in significant_weights[:10]:
        print(f"  {state}: {weight:.3f} ({weight*100:.1f}%)")

    # Construct synthetic ND
    Y_donors_all = outcome_matrix[donor_states].values
    synthetic_nd = Y_donors_all @ weights

    # Calculate descriptive gap
    actual_nd = outcome_matrix[treated_unit].values
    gap = actual_nd - synthetic_nd

    # Pre-treatment fit (RMSPE)
    pre_rmspe = np.sqrt(np.mean((Y_treated_pre - Y_donors_pre @ weights) ** 2))

    # Post-treatment gap
    post_idx = outcome_matrix.index >= treatment_year
    post_gap_mean = np.mean(gap[post_idx])
    post_gap_std = np.std(gap[post_idx])

    print(f"\nPre-treatment RMSPE: {pre_rmspe:.4f}")
    print("\nPost-period gap (descriptive):")
    print(f"  Mean: {post_gap_mean:.4f}")
    print(f"  Std: {post_gap_std:.4f}")

    # Ratio of post/pre RMSPE (descriptive only)
    post_rmspe = np.sqrt(np.mean(gap[post_idx] ** 2))
    rmspe_ratio = post_rmspe / pre_rmspe if pre_rmspe > 0 else np.inf

    print(f"\nRMSPE Ratio (post/pre): {rmspe_ratio:.2f}")
    print("  Interpretation: Larger ratios indicate larger post-period divergence")

    # Build time series for output
    sc_time_series = pd.DataFrame(
        {
            "year": outcome_matrix.index,
            "actual": actual_nd,
            "synthetic": synthetic_nd,
            "gap": gap,
        }
    )

    sc_results = {
        "analysis": "Synthetic Comparator (descriptive)",
        "treated_unit": treated_unit,
        "treatment_year": treatment_year,
        "feasible": True,
        "donor_pool_size": len(donor_states),
        "weights": {
            s: float(w)
            for s, w in zip(donor_states, weights, strict=False)
            if w > weight_threshold
        },
        "pre_treatment_fit": {"rmspe": float(pre_rmspe), "n_periods": len(pre_years)},
        "post_period_gap": {
            "mean_gap": float(post_gap_mean),
            "std_gap": float(post_gap_std),
            "rmspe": float(post_rmspe),
            "rmspe_ratio": float(rmspe_ratio),
        },
        "time_series": sc_time_series.to_dict("records"),
        "optimization": {
            "converged": result_opt.success,
            "objective_value": float(result_opt.fun),
        },
    }

    result.add_decision(
        decision_id="D004",
        category="causal_identification",
        decision="Use international migration rate (per 1000) for synthetic comparator",
        rationale="Rate normalizes for population differences across states",
        alternatives=["Use raw counts", "Use log transformation", "Use growth rates"],
        evidence=f"Pre-treatment RMSPE = {pre_rmspe:.4f}",
    )

    return sc_results, sc_time_series


# =============================================================================
# SHIFT-SHARE (BARTIK) INSTRUMENT
# =============================================================================


def estimate_bartik_instrument(
    df_refugee: pd.DataFrame, df_panel: pd.DataFrame, result: ModuleResult
) -> dict:
    """
    Implement shift-share (Bartik) instrument for immigration analysis.

    Instrument: National immigration × State's historical share of immigration by nationality

    B_st = sum_n (share_sn,t0 × Delta_national_n,t)

    where:
    - share_sn,t0: State s's share of nationality n at baseline
    - Delta_national_n,t: Change in national immigration of nationality n
    """
    print("\n" + "=" * 60)
    print("SHIFT-SHARE (BARTIK) INSTRUMENT")
    print("=" * 60)

    # Use refugee data for shift-share construction
    df_ref = df_refugee.copy()
    df_ref = df_ref.rename(columns={"fiscal_year": "year"})

    # Drop states missing post-2020 totals (unknown coverage)
    total_rows = df_ref[df_ref["nationality"] == "Total"]
    post_years = sorted(y for y in total_rows["year"].unique() if y >= 2021)
    if post_years:
        coverage = (
            total_rows[total_rows["year"].isin(post_years)]
            .groupby("state")["year"]
            .nunique()
        )
        complete_states = set(coverage[coverage == len(post_years)].index)
        missing_states = sorted(set(total_rows["state"].unique()) - complete_states)
        if missing_states:
            msg = (
                "Dropping states missing post-2020 refugee totals for Bartik: "
                + ", ".join(missing_states)
            )
            print(f"WARNING: {msg}")
            result.warnings.append(msg)
        df_ref = df_ref[df_ref["state"].isin(complete_states)]

    # Baseline year for shares
    baseline_year = 2010

    # Calculate baseline shares: State's share of each nationality
    baseline_data = df_ref[df_ref["year"] == baseline_year].copy()

    # Total arrivals by nationality in baseline year
    nat_totals_baseline = baseline_data.groupby("nationality")["arrivals"].sum()

    # State's share of each nationality
    baseline_shares = baseline_data.merge(
        nat_totals_baseline.reset_index().rename(columns={"arrivals": "nat_total"}),
        on="nationality",
    )
    baseline_shares["state_share"] = (
        baseline_shares["arrivals"] / baseline_shares["nat_total"]
    )
    baseline_shares = baseline_shares[["state", "nationality", "state_share"]]

    print(f"\nBaseline year: {baseline_year}")
    print(
        f"Nationalities with baseline data: {baseline_shares['nationality'].nunique()}"
    )
    print(f"States: {baseline_shares['state'].nunique()}")
    print("Shift definition: leave-one-out national changes relative to baseline")

    # National totals by year and nationality
    nat_by_year_state = (
        df_ref.groupby(["year", "nationality", "state"])["arrivals"].sum().reset_index()
    )
    nat_totals_by_year = (
        nat_by_year_state.groupby(["year", "nationality"])["arrivals"]
        .sum()
        .reset_index()
        .rename(columns={"arrivals": "nat_total"})
    )

    # Build state-year-nationality panel using baseline shares
    years = sorted(nat_totals_by_year["year"].unique())
    state_nat = baseline_shares.copy()
    state_nat["key"] = 1
    years_df = pd.DataFrame({"year": years, "key": 1})
    panel = state_nat.merge(years_df, on="key", how="left").drop(columns=["key"])

    # Merge national totals and state-specific arrivals for leave-one-out shifts
    panel = panel.merge(nat_totals_by_year, on=["year", "nationality"], how="left")
    panel = panel.merge(
        nat_by_year_state.rename(columns={"arrivals": "state_arrivals"}),
        on=["year", "nationality", "state"],
        how="left",
    )
    # Missing nationality rows within covered states are treated as zero arrivals.
    panel["state_arrivals"] = panel["state_arrivals"].fillna(0)

    # Baseline totals and state arrivals (for leave-one-out baseline)
    baseline_totals = nat_totals_by_year[nat_totals_by_year["year"] == baseline_year][
        ["nationality", "nat_total"]
    ].rename(columns={"nat_total": "nat_total_baseline"})
    baseline_state = nat_by_year_state[nat_by_year_state["year"] == baseline_year][
        ["state", "nationality", "arrivals"]
    ].rename(columns={"arrivals": "state_arrivals_baseline"})
    panel = panel.merge(baseline_totals, on="nationality", how="left")
    panel = panel.merge(baseline_state, on=["state", "nationality"], how="left")
    panel["state_arrivals_baseline"] = panel["state_arrivals_baseline"].fillna(0)

    # Leave-one-out national change for each state
    panel["nat_total_excl_state"] = panel["nat_total"] - panel["state_arrivals"]
    panel["nat_total_baseline_excl_state"] = (
        panel["nat_total_baseline"] - panel["state_arrivals_baseline"]
    )
    panel["delta_shift"] = (
        panel["nat_total_excl_state"] - panel["nat_total_baseline_excl_state"]
    )

    # Construct Bartik instrument for each state-year
    bartik_df = (
        panel.groupby(["state", "year"])
        .apply(lambda g: np.sum(g["state_share"] * g["delta_shift"]))
        .reset_index(name="bartik_instrument")
    )

    print("\nBartik instrument constructed:")
    print(f"  Observations: {len(bartik_df)}")
    print(f"  Mean: {bartik_df['bartik_instrument'].mean():.2f}")
    print(f"  Std: {bartik_df['bartik_instrument'].std():.2f}")

    # Merge with panel data for IV estimation
    # Need to align years (refugee data is 2002-2020, panel is 2010-2024)
    df_merged = df_panel.merge(bartik_df, on=["state", "year"], how="inner")

    if len(df_merged) == 0:
        result.warnings.append(
            "No overlapping years between refugee and panel data for Bartik"
        )
        return {
            "analysis": "Shift-Share (Bartik) Instrument",
            "feasible": False,
            "reason": "No overlapping years",
        }

    print(f"\nMerged data: {len(df_merged)} observations")
    print(f"Years: {sorted(df_merged['year'].unique())}")

    # First stage: Regress actual immigration on Bartik instrument
    df_analysis = df_merged.dropna(
        subset=["intl_migration", "bartik_instrument"]
    ).copy()
    df_analysis = df_analysis.reset_index(drop=True)

    # Add state and year fixed effects
    state_dummies = pd.get_dummies(
        df_analysis["state"], prefix="state", drop_first=True
    )
    year_dummies = pd.get_dummies(df_analysis["year"], prefix="year", drop_first=True)

    # Build design matrix properly
    X_first = pd.DataFrame(
        {"bartik": df_analysis["bartik_instrument"].values.astype(float)}
    )
    X_first = pd.concat(
        [X_first.reset_index(drop=True), state_dummies.reset_index(drop=True)], axis=1
    )
    X_first = pd.concat(
        [X_first.reset_index(drop=True), year_dummies.reset_index(drop=True)], axis=1
    )
    X_first = sm.add_constant(X_first)
    X_first = X_first.astype(float)

    y_first = df_analysis["intl_migration"].values.astype(float)

    first_stage = OLS(y_first, X_first).fit(
        cov_type="cluster", cov_kwds={"groups": df_analysis["state"]}
    )

    bartik_coef = first_stage.params["bartik"]
    bartik_se = first_stage.bse["bartik"]
    bartik_t = first_stage.tvalues["bartik"]
    bartik_p = first_stage.pvalues["bartik"]

    # F-statistic for instrument strength
    f_stat = bartik_t**2

    print(f"\n{'='*60}")
    print("FIRST STAGE REGRESSION")
    print("=" * 60)
    print("Dependent Variable: International migration")
    print("Instrument: Bartik (shift-share)")
    print("-" * 60)
    print(f"  Bartik coefficient: {bartik_coef:.4f}")
    print(f"  Std. Error: {bartik_se:.4f}")
    print(f"  t-statistic: {bartik_t:.4f}")
    print(f"  p-value: {bartik_p:.4f}")
    print(f"\n  First-stage F-statistic: {f_stat:.2f}")
    print("  Strong instrument threshold: F > 10")
    print(f"  Assessment: {'Strong' if f_stat > 10 else 'Weak'} instrument")

    bartik_results = {
        "analysis": "Shift-Share (Bartik) Instrument",
        "feasible": True,
        "construction": {
            "baseline_year": baseline_year,
            "share_component": "State share of nationality at baseline",
            "shift_component": "Leave-one-out national change in nationality arrivals",
        },
        "sample_info": {
            "n_observations": len(df_analysis),
            "n_states": df_analysis["state"].nunique(),
            "n_years": df_analysis["year"].nunique(),
            "years": sorted(df_analysis["year"].unique().tolist()),
        },
        "first_stage": {
            "bartik_coefficient": float(bartik_coef),
            "std_error": float(bartik_se),
            "t_statistic": float(bartik_t),
            "p_value": float(bartik_p),
            "f_statistic": float(f_stat),
            "strong_instrument": f_stat > 10,
        },
        "instrument_statistics": {
            "mean": float(bartik_df["bartik_instrument"].mean()),
            "std": float(bartik_df["bartik_instrument"].std()),
            "min": float(bartik_df["bartik_instrument"].min()),
            "max": float(bartik_df["bartik_instrument"].max()),
        },
        "model_fit": {
            "r_squared": float(first_stage.rsquared),
            "r_squared_adj": float(first_stage.rsquared_adj),
        },
    }

    result.add_decision(
        decision_id="D005",
        category="causal_identification",
        decision=f"Use {baseline_year} baseline with leave-one-out shifts for shift-share construction",
        rationale="2010 is first year with reliable state-level refugee data and aligns with panel start",
        alternatives=[
            "Use 2005 as baseline",
            "Use average of early years",
            "Time-varying shares",
        ],
        evidence=f"First-stage F = {f_stat:.2f}",
    )

    return bartik_results, bartik_df


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================


def plot_parallel_trends(df: pd.DataFrame, result: ModuleResult):
    """Plot parallel trends for Travel Ban DiD."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Aggregate by treatment status and year
    trends = (
        df.groupby(["year", "treated"])
        .agg({"arrivals": "sum", "log_arrivals": "mean"})
        .reset_index()
    )

    # Left panel: Raw arrivals
    ax1 = axes[0]
    for treated, label, color in [
        (0, "Control", COLORS["control"]),
        (1, "Treatment", COLORS["treatment"]),
    ]:
        data = trends[trends["treated"] == treated]
        ax1.plot(
            data["year"],
            data["arrivals"],
            "o-",
            color=color,
            label=f"{label} countries",
            linewidth=2,
            markersize=6,
        )

    ax1.axvline(2017.5, color="gray", linestyle="--", alpha=0.7, label="Travel Ban")
    ax1.set_xlabel("Year", fontsize=12)
    ax1.set_ylabel("Total Refugee Arrivals", fontsize=12)
    ax1.set_title("Raw Arrivals by Treatment Status", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right panel: Log arrivals (normalized)
    ax2 = axes[1]
    for treated, label, color in [
        (0, "Control", COLORS["control"]),
        (1, "Treatment", COLORS["treatment"]),
    ]:
        data = trends[trends["treated"] == treated]
        # Normalize to pre-treatment mean
        pre_mean = data[data["year"] < 2018]["log_arrivals"].mean()
        normalized = data["log_arrivals"] - pre_mean
        ax2.plot(
            data["year"],
            normalized,
            "o-",
            color=color,
            label=f"{label} countries",
            linewidth=2,
            markersize=6,
        )

    ax2.axvline(2017.5, color="gray", linestyle="--", alpha=0.7, label="Travel Ban")
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_xlabel("Year", fontsize=12)
    ax2.set_ylabel("Log Arrivals (normalized to pre-period)", fontsize=12)
    ax2.set_title("Normalized Trends (for parallel trends assessment)", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    save_figure(
        fig,
        str(FIGURES_DIR / "module_7_parallel_trends"),
        "Parallel Trends Assessment: 2017 Travel Ban",
        "State Department Refugee Processing Center",
    )


def plot_event_study(es_df: pd.DataFrame, result: ModuleResult):
    """Plot event study coefficients."""
    fig, ax = setup_figure(figsize=(12, 7))

    # Plot coefficients with confidence intervals
    ax.errorbar(
        es_df["rel_time"],
        es_df["coefficient"],
        yerr=[
            es_df["coefficient"] - es_df["ci_lower"],
            es_df["ci_upper"] - es_df["coefficient"],
        ],
        fmt="o",
        color=COLORS["primary"],
        markersize=8,
        capsize=4,
        capthick=2,
        linewidth=2,
        label="Point estimate (95% CI)",
    )

    # Reference line at zero
    ax.axhline(0, color="black", linewidth=0.5)

    # Treatment line
    ax.axvline(-0.5, color="gray", linestyle="--", alpha=0.7, label="Treatment (2018)")

    # Shade post-treatment period
    ax.axvspan(-0.5, ax.get_xlim()[1], alpha=0.1, color=COLORS["secondary"])

    ax.set_xlabel("Relative Time (years since 2018)", fontsize=12)
    ax.set_ylabel("Treatment Effect (log arrivals)", fontsize=12)
    ax.legend(fontsize=10, loc="upper right")

    # Add year labels on secondary axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    years = [2018 + t for t in range(int(ax.get_xlim()[0]), int(ax.get_xlim()[1]) + 1)]
    ax2.set_xticks(list(range(int(ax.get_xlim()[0]), int(ax.get_xlim()[1]) + 1)))
    ax2.set_xticklabels([str(y) for y in years], fontsize=9)
    ax2.set_xlabel("Calendar Year", fontsize=10)

    save_figure(
        fig,
        [
            str(FIGURES_DIR / "module_7_event_study_plot"),
            str(ARTICLE_FIGURES_DIR / "event_study_travel_ban"),
        ],
        "Event Study: Dynamic Treatment Effects of Travel Ban",
        "State Department Refugee Processing Center",
    )


def plot_did_means(df: pd.DataFrame, result: ModuleResult):
    """Plot DiD group means over time."""
    fig, ax = setup_figure(figsize=(12, 7))

    # Calculate group means
    group_means = df.groupby(["year", "treated"])["log_arrivals"].mean().reset_index()

    # Plot treatment group
    treat_data = group_means[group_means["treated"] == 1]
    ax.plot(
        treat_data["year"],
        treat_data["log_arrivals"],
        "o-",
        color=COLORS["treatment"],
        linewidth=2.5,
        markersize=8,
        label="Treatment (Travel Ban countries)",
    )

    # Plot control group
    control_data = group_means[group_means["treated"] == 0]
    ax.plot(
        control_data["year"],
        control_data["log_arrivals"],
        "s-",
        color=COLORS["control"],
        linewidth=2.5,
        markersize=8,
        label="Control (Other countries)",
    )

    # Treatment line
    ax.axvline(2017.5, color="gray", linestyle="--", alpha=0.7)
    ax.text(2017.6, ax.get_ylim()[1] * 0.95, "Travel Ban", fontsize=10, color="gray")

    # Shade pre/post
    ax.axvspan(ax.get_xlim()[0], 2017.5, alpha=0.05, color=COLORS["primary"])
    ax.axvspan(2017.5, ax.get_xlim()[1], alpha=0.05, color=COLORS["secondary"])

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Mean Log Arrivals", fontsize=12)
    ax.legend(fontsize=10)

    save_figure(
        fig,
        str(FIGURES_DIR / "module_7_did_means"),
        "Difference-in-Differences: Group Means Over Time",
        "State Department Refugee Processing Center",
    )


def plot_treatment_effect_time_series(
    sc_df: pd.DataFrame, treatment_year: int, result: ModuleResult
):
    """Plot synthetic comparator gap over time."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: Actual vs Synthetic
    ax1 = axes[0]
    ax1.plot(
        sc_df["year"],
        sc_df["actual"],
        "o-",
        color=COLORS["primary"],
        linewidth=2,
        label="North Dakota (actual)",
    )
    ax1.plot(
        sc_df["year"],
        sc_df["synthetic"],
        "s--",
        color=COLORS["secondary"],
        linewidth=2,
        label="Synthetic comparator",
    )

    ax1.axvline(treatment_year - 0.5, color="gray", linestyle="--", alpha=0.7)
    ax1.set_xlabel("Year", fontsize=12)
    ax1.set_ylabel("International Migration Rate (per 1000)", fontsize=12)
    ax1.set_title("Actual vs Synthetic Comparator", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right panel: Gap (actual - synthetic)
    ax2 = axes[1]
    pre_years = sc_df["year"] < treatment_year
    post_years = sc_df["year"] >= treatment_year

    ax2.bar(
        sc_df.loc[pre_years, "year"],
        sc_df.loc[pre_years, "gap"],
        color=COLORS["neutral"],
        alpha=0.7,
        label="Pre-treatment",
    )
    ax2.bar(
        sc_df.loc[post_years, "year"],
        sc_df.loc[post_years, "gap"],
        color=COLORS["treatment"],
        alpha=0.7,
        label="Post-treatment",
    )

    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.axvline(treatment_year - 0.5, color="gray", linestyle="--", alpha=0.7)

    ax2.set_xlabel("Year", fontsize=12)
    ax2.set_ylabel("Gap (Actual - Synthetic)", fontsize=12)
    ax2.set_title("Synthetic Comparator Gap", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    save_figure(
        fig,
        str(FIGURES_DIR / "module_7_treatment_effect_time_series"),
        "Synthetic Comparator: North Dakota Gap",
        "Census Bureau Population Estimates Program",
    )


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================


def run_analysis() -> ModuleResult:
    """Main analysis function for Module 7."""
    result = ModuleResult(
        module_id="7", analysis_name="causal_inference_did_event_study"
    )

    # Load data
    print("Loading data...")
    df_components, df_refugee, df_panel = load_data(result)

    # Record parameters
    result.parameters = {
        "analyses": [
            "Difference-in-Differences (Travel Ban)",
            "Interrupted Time Series (COVID)",
            "Event Study (Travel Ban)",
            "Synthetic Comparator (ND, descriptive)",
            "Shift-Share (Bartik) Instrument",
        ],
        "travel_ban_treatment_year": 2018,
        "covid_treatment_year": 2020,
        "synthetic_control_unit": "North Dakota",
    }

    # ==========================================================================
    # 1. DiD for Travel Ban
    # ==========================================================================
    print("\n" + "#" * 70)
    print("# ANALYSIS 1: DIFFERENCE-IN-DIFFERENCES - TRAVEL BAN")
    print("#" * 70)

    df_travel_ban = prepare_travel_ban_did_data(df_refugee, result)
    did_travel_ban, df_did = estimate_did_travel_ban(df_travel_ban, result)

    # ==========================================================================
    # 2. DiD/ITS for COVID
    # ==========================================================================
    print("\n" + "#" * 70)
    print("# ANALYSIS 2: INTERRUPTED TIME SERIES - COVID")
    print("#" * 70)

    df_covid = prepare_covid_did_data(df_panel, result)
    did_covid, df_its = estimate_did_covid(df_covid, result)

    # ==========================================================================
    # 3. Event Study for Travel Ban
    # ==========================================================================
    print("\n" + "#" * 70)
    print("# ANALYSIS 3: EVENT STUDY - TRAVEL BAN")
    print("#" * 70)

    event_study, es_df = estimate_event_study(df_travel_ban, result)

    # ==========================================================================
    # 4. Synthetic Control for ND
    # ==========================================================================
    print("\n" + "#" * 70)
    print("# ANALYSIS 4: SYNTHETIC COMPARATOR (DESCRIPTIVE)")
    print("#" * 70)

    try:
        synthetic_control, sc_df = estimate_synthetic_control(df_panel, result)
    except Exception as e:
        result.warnings.append(f"Synthetic control estimation failed: {e}")
        synthetic_control = {"feasible": False, "error": str(e)}
        sc_df = None

    # ==========================================================================
    # 5. Bartik Instrument
    # ==========================================================================
    print("\n" + "#" * 70)
    print("# ANALYSIS 5: SHIFT-SHARE (BARTIK) INSTRUMENT")
    print("#" * 70)

    try:
        bartik, bartik_df = estimate_bartik_instrument(df_refugee, df_panel, result)
    except Exception as e:
        result.warnings.append(f"Bartik instrument estimation failed: {e}")
        bartik = {"feasible": False, "error": str(e)}

    # ==========================================================================
    # Generate Visualizations
    # ==========================================================================
    print("\n" + "#" * 70)
    print("# GENERATING VISUALIZATIONS")
    print("#" * 70)

    plot_parallel_trends(df_did, result)
    plot_event_study(es_df, result)
    plot_did_means(df_did, result)

    if sc_df is not None:
        plot_treatment_effect_time_series(sc_df, 2017, result)

    # ==========================================================================
    # Compile Results
    # ==========================================================================
    result.results = {
        "did_travel_ban": did_travel_ban,
        "did_covid": did_covid,
        "event_study": event_study,
        "synthetic_control": synthetic_control,
        "bartik_instrument": bartik,
    }

    # Save individual result files
    print("\nSaving individual result files...")

    # DiD estimates
    did_output = {"travel_ban": did_travel_ban, "covid": did_covid}
    with open(RESULTS_DIR / "module_7_did_estimates.json", "w") as f:
        json.dump(did_output, f, indent=2, default=str)
    print(f"  Saved: {RESULTS_DIR / 'module_7_did_estimates.json'}")

    # Event study parquet
    es_df.to_parquet(RESULTS_DIR / "module_7_event_study.parquet", index=False)
    print(f"  Saved: {RESULTS_DIR / 'module_7_event_study.parquet'}")

    # Synthetic control (if feasible)
    if synthetic_control.get("feasible", False):
        with open(RESULTS_DIR / "module_7_synthetic_control.json", "w") as f:
            json.dump(synthetic_control, f, indent=2, default=str)
        print(f"  Saved: {RESULTS_DIR / 'module_7_synthetic_control.json'}")

    # Diagnostics
    result.diagnostics = {
        "parallel_trends": {
            "travel_ban_supported": did_travel_ban.get("pre_trend_test", {}).get(
                "parallel_trends_supported", None
            ),
            "event_study_pre_trend_p": event_study.get("pre_trend_test", {}).get(
                "p_value", None
            ),
        },
        "instrument_strength": {
            "bartik_f_statistic": bartik.get("first_stage", {}).get("f_statistic", None)
            if isinstance(bartik, dict)
            else None,
            "strong_instrument": bartik.get("first_stage", {}).get(
                "strong_instrument", None
            )
            if isinstance(bartik, dict)
            else None,
        },
        "synthetic_control_fit": {
            "pre_treatment_rmspe": synthetic_control.get("pre_treatment_fit", {}).get(
                "rmspe", None
            ),
            "rmspe_ratio": synthetic_control.get("post_period_gap", {}).get(
                "rmspe_ratio", None
            ),
        },
    }

    # Next steps
    result.next_steps = [
        "Module 8: Conduct robustness checks (alternative specifications, placebo tests)",
        "Consider heterogeneous treatment effects by state/region",
        "Extend event study window with more recent data when available",
        "Validate synthetic control with leave-one-out placebo analysis",
    ]

    return result


def main():
    """Main entry point."""
    print("=" * 70)
    print("Module 7: Causal Inference Agent")
    print("Difference-in-Differences, Event Studies, and Synthetic Comparator")
    print(f"Started: {datetime.now(UTC).isoformat()}")
    print("=" * 70)

    try:
        result = run_analysis()
        output_file = result.save("module_7_causal_inference.json")

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("=" * 70)

        print(f"\nMain output: {output_file}")

        print("\n" + "-" * 70)
        print("KEY FINDINGS SUMMARY")
        print("-" * 70)

        # Travel Ban DiD
        tb = result.results.get("did_travel_ban", {})
        if tb:
            att = tb.get("att_estimate", {})
            print("\n1. TRAVEL BAN DiD:")
            print(
                f"   ATT: {att.get('coefficient', 'N/A'):.4f} (SE: {att.get('std_error', 'N/A'):.4f})"
            )
            print(
                f"   p-value: {att.get('p_value', 'N/A'):.4f} {att.get('significance', '')}"
            )
            print(
                f"   % effect: {tb.get('percentage_effect', {}).get('estimate', 'N/A'):.1f}%"
            )
            print(
                f"   Parallel trends: {'Supported' if tb.get('pre_trend_test', {}).get('parallel_trends_supported') else 'May be violated'}"
            )

        # COVID ITS
        cov = result.results.get("did_covid", {})
        if cov:
            level = cov.get("level_effect", {})
            print("\n2. COVID INTERRUPTED TIME SERIES:")
            print(
                f"   Level shift: {level.get('coefficient', 'N/A'):,.0f} (SE: {level.get('std_error', 'N/A'):,.0f})"
            )
            print(
                f"   p-value: {level.get('p_value', 'N/A'):.4f} {level.get('significance', '')}"
            )

        # Synthetic Comparator (descriptive)
        sc = result.results.get("synthetic_control", {})
        if sc and sc.get("feasible"):
            print("\n3. SYNTHETIC COMPARATOR (North Dakota, descriptive):")
            pre_rmspe = sc.get("pre_treatment_fit", {}).get("rmspe")
            post_gap = sc.get("post_period_gap", {})
            mean_gap = post_gap.get("mean_gap")
            rmspe_ratio = post_gap.get("rmspe_ratio")
            if pre_rmspe is not None:
                print(f"   Pre-treatment RMSPE: {pre_rmspe:.4f}")
            else:
                print("   Pre-treatment RMSPE: N/A")
            if mean_gap is not None:
                print(f"   Post-period mean gap: {mean_gap:.4f}")
            else:
                print("   Post-period mean gap: N/A")
            if rmspe_ratio is not None:
                print(f"   RMSPE ratio: {rmspe_ratio:.2f}")
            else:
                print("   RMSPE ratio: N/A")

        # Bartik
        bartik = result.results.get("bartik_instrument", {})
        if bartik and bartik.get("feasible"):
            fs = bartik.get("first_stage", {})
            print("\n4. BARTIK INSTRUMENT:")
            print(f"   First-stage F: {fs.get('f_statistic', 'N/A'):.2f}")
            print(
                f"   Strong instrument: {'Yes' if fs.get('strong_instrument') else 'No'}"
            )

        print("\n" + "-" * 70)
        print("OUTPUT FILES")
        print("-" * 70)
        print(f"  Results: {RESULTS_DIR / 'module_7_did_estimates.json'}")
        print(f"           {RESULTS_DIR / 'module_7_event_study.parquet'}")
        print(f"           {RESULTS_DIR / 'module_7_synthetic_control.json'}")
        print(f"  Figures: {FIGURES_DIR / 'module_7_parallel_trends.png/pdf'}")
        print(f"           {FIGURES_DIR / 'module_7_event_study_plot.png/pdf'}")
        print(f"           {FIGURES_DIR / 'module_7_did_means.png/pdf'}")
        print(
            f"           {FIGURES_DIR / 'module_7_treatment_effect_time_series.png/pdf'}"
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
