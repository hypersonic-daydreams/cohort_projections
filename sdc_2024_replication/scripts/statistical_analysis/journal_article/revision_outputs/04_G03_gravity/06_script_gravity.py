#!/usr/bin/env python3
"""
Module 5: Gravity Model and Network Analysis
=============================================

Implements gravity model specification for immigration flows and analyzes
diaspora network effects using Poisson Pseudo-Maximum Likelihood (PPML) estimation.

This module:
1. Specifies gravity models for immigration flows (origin/destination characteristics)
2. Estimates PPML models (appropriate for count data with zeros)
3. Analyzes diaspora network elasticity with controls

Usage:
    micromamba run -n cohort_proj python module_5_gravity_network.py
"""

import json
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Statsmodels imports
import statsmodels.api as sm
from scipy import stats
from statsmodels.genmod.families import Poisson
from statsmodels.genmod.generalized_linear_model import GLM

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


def load_data(result: ModuleResult) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all required data files."""
    print("\n" + "=" * 70)
    print("MODULE 5: GRAVITY MODEL AND NETWORK ANALYSIS")
    print("=" * 70)
    print(f"\nAnalysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 70)

    print("\n[1] Loading Data Files...")

    # Load DHS LPR by state/country
    dhs_lpr = pd.read_parquet(DATA_DIR / "dhs_lpr_by_state_country.parquet")
    print(f"    - DHS LPR by state/country: {len(dhs_lpr):,} rows")
    result.input_files.append("dhs_lpr_by_state_country.parquet")

    # Load ACS foreign-born by state/origin
    acs_origin = pd.read_parquet(DATA_DIR / "acs_foreign_born_by_state_origin.parquet")
    print(f"    - ACS foreign-born by state/origin: {len(acs_origin):,} rows")
    result.input_files.append("acs_foreign_born_by_state_origin.parquet")

    # Load Module 3.2 origin panel data
    origin_panel = pd.read_parquet(RESULTS_DIR / "module_3_2_origin_panel.parquet")
    print(f"    - Module 3.2 origin panel: {len(origin_panel):,} rows")
    result.input_files.append("results/module_3_2_origin_panel.parquet")

    return dhs_lpr, acs_origin, origin_panel


def prepare_gravity_data(
    dhs_lpr: pd.DataFrame, acs_origin: pd.DataFrame, result: ModuleResult
) -> pd.DataFrame:
    """
    Prepare data for gravity model estimation.

    Creates origin-destination flows with characteristics for gravity model:
    - Flows: LPR admissions by state and country of origin
    - Origin characteristics: Existing diaspora stock (ACS foreign-born)
    - State/destination characteristics: Total foreign-born population
    """
    print("\n[2] Preparing Gravity Model Data...")
    print("-" * 70)

    # Filter DHS to country level only (not regions)
    dhs_countries = dhs_lpr[~dhs_lpr["is_region"]].copy()
    dhs_countries = dhs_countries[dhs_countries["region_country_of_birth"] != "Total"]
    print(f"    - DHS country-level observations: {len(dhs_countries):,}")

    # Get ACS data for 2023 at country level
    acs_2023 = acs_origin[
        (acs_origin["year"] == 2023) & (acs_origin["level"] == "country")
    ].copy()
    print(f"    - ACS 2023 country-level observations: {len(acs_2023):,}")

    # Get state totals (destination mass)
    state_totals = acs_origin[
        (acs_origin["year"] == 2023) & (acs_origin["level"] == "total")
    ][["state_name", "foreign_born_pop"]].copy()
    state_totals.columns = ["state", "state_foreign_born_total"]

    # Create country name harmonization mapping
    # DHS uses different naming conventions than ACS
    country_mapping = {
        # ACS name -> DHS name
        "China": "China, People's Republic",
        "Korea": "Korea, South",
        "Burma": "Burma (Myanmar)",
        "Czechoslovakia (includes Czech Republic and Slovakia)": "Czech Republic",
        "United Kingdom (inc. Crown Dependencies)": "United Kingdom",
        "Russia": "Russia (Russian Federation)",
        "Serbia": "Serbia and Montenegro",
        "Congo (Kinshasa)": "Congo, Democratic Republic",
        "Congo (Brazzaville)": "Congo, Republic",
    }

    # Create reverse mapping for DHS to match ACS
    reverse_mapping = {v: k for k, v in country_mapping.items()}

    # Standardize DHS country names
    dhs_countries["country_std"] = dhs_countries["region_country_of_birth"].replace(
        reverse_mapping
    )

    # Standardize ACS country names
    acs_2023["country_std"] = acs_2023["country"]

    # Merge DHS flows with ACS diaspora stock
    gravity_df = dhs_countries.merge(
        acs_2023[["state_name", "country_std", "foreign_born_pop", "margin_of_error"]],
        left_on=["state", "country_std"],
        right_on=["state_name", "country_std"],
        how="left",
    )

    # Rename columns for clarity
    gravity_df = gravity_df.rename(
        columns={
            "lpr_count": "flow",
            "foreign_born_pop": "diaspora_stock",
            "region_country_of_birth": "origin_country",
        }
    )

    # Add state foreign-born totals (destination mass)
    gravity_df = gravity_df.merge(state_totals, on="state", how="left")

    # Calculate national totals by origin (origin mass)
    national_by_origin = (
        acs_2023.groupby("country_std")["foreign_born_pop"].sum().reset_index()
    )
    national_by_origin.columns = ["country_std", "national_origin_total"]
    gravity_df = gravity_df.merge(national_by_origin, on="country_std", how="left")

    # Filter to observations with valid flow and stock data
    gravity_df = gravity_df.dropna(subset=["flow", "diaspora_stock"])
    gravity_df = gravity_df[gravity_df["flow"] > 0]

    print(f"    - Merged gravity observations: {len(gravity_df):,}")
    print(f"    - Unique states: {gravity_df['state'].nunique()}")
    print(f"    - Unique origin countries: {gravity_df['country_std'].nunique()}")

    # Create log-transformed variables for gravity estimation
    gravity_df["log_flow"] = np.log(gravity_df["flow"] + 1)
    gravity_df["log_diaspora"] = np.log(gravity_df["diaspora_stock"] + 1)
    gravity_df["log_state_total"] = np.log(gravity_df["state_foreign_born_total"] + 1)
    gravity_df["log_origin_total"] = np.log(gravity_df["national_origin_total"] + 1)

    # Create diaspora share variable
    gravity_df["diaspora_share"] = (
        gravity_df["diaspora_stock"] / gravity_df["national_origin_total"]
    )
    gravity_df["diaspora_share"] = gravity_df["diaspora_share"].fillna(0)

    result.add_decision(
        decision_id="D001",
        category="data_preparation",
        decision="Harmonized country names between DHS LPR and ACS datasets",
        rationale="Different data sources use different country naming conventions",
        alternatives=["Fuzzy matching", "Manual mapping only"],
        evidence=f"Matched {gravity_df['country_std'].nunique()} countries across datasets",
    )

    decisions = {
        "flow_variable": "DHS LPR admissions (FY2023)",
        "diaspora_stock_variable": "ACS foreign-born by origin country (2023)",
        "destination_mass": "ACS total foreign-born by state (2023)",
        "origin_mass": "ACS national total foreign-born by origin country (2023)",
        "observations_final": len(gravity_df),
        "unique_states": gravity_df["state"].nunique(),
        "unique_origins": gravity_df["country_std"].nunique(),
    }

    return gravity_df, decisions


def estimate_gravity_ppml(gravity_df: pd.DataFrame, result: ModuleResult) -> dict:
    """
    Estimate gravity model using Poisson Pseudo-Maximum Likelihood (PPML).

    PPML is preferred for gravity models because:
    1. Handles zero flows naturally (unlike log-linear OLS)
    2. Consistent under heteroskedasticity (Santos Silva & Tenreyro, 2006)
    3. Multiplicative form matches theoretical gravity model

    Models estimated:
    1. Basic gravity: Flow ~ Diaspora_Stock
    2. Full gravity: Flow ~ Diaspora_Stock + Origin_Mass + Destination_Mass
    3. Network only: Flow ~ Diaspora_Stock (with state FE)
    """
    print("\n[3] Estimating PPML Gravity Models...")
    print("-" * 70)

    # Prepare data
    df = gravity_df.copy()

    # Ensure no NaN or Inf values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(
        subset=[
            "flow",
            "diaspora_stock",
            "state_foreign_born_total",
            "national_origin_total",
        ]
    )

    # Model 1: Simple network effect model
    print("\n" + "=" * 70)
    print("MODEL 1: SIMPLE NETWORK EFFECT (PPML)")
    print("Flow_ij = exp(beta * log(Diaspora_ij) + alpha)")
    print("=" * 70)

    y_simple = df["flow"]
    X_simple = sm.add_constant(df["log_diaspora"])

    ppml_simple = GLM(y_simple, X_simple, family=Poisson())
    ppml_simple_result = ppml_simple.fit()

    print(ppml_simple_result.summary())

    # Model 2: Full gravity model
    print("\n" + "=" * 70)
    print("MODEL 2: FULL GRAVITY MODEL (PPML)")
    print(
        "Flow_ij = exp(beta1*log(Diaspora) + beta2*log(Origin_Mass) + beta3*log(Dest_Mass) + alpha)"
    )
    print("=" * 70)

    y_full = df["flow"]
    X_full = sm.add_constant(
        df[["log_diaspora", "log_origin_total", "log_state_total"]]
    )

    ppml_full = GLM(y_full, X_full, family=Poisson())
    ppml_full_result = ppml_full.fit()

    print(ppml_full_result.summary())

    # Model 3: With state fixed effects (using dummies)
    print("\n" + "=" * 70)
    print("MODEL 3: NETWORK EFFECT WITH STATE FIXED EFFECTS (PPML)")
    print("Flow_ij = exp(beta*log(Diaspora) + state_i + alpha)")
    print("=" * 70)

    # Create state dummies (excluding reference state for identification)
    state_dummies = pd.get_dummies(
        df["state"], prefix="state", drop_first=True, dtype=float
    )
    X_fe = pd.concat(
        [
            df[["log_diaspora"]].reset_index(drop=True),
            state_dummies.reset_index(drop=True),
        ],
        axis=1,
    )
    X_fe = sm.add_constant(X_fe)

    ppml_fe = GLM(df["flow"].reset_index(drop=True), X_fe, family=Poisson())
    ppml_fe_result = ppml_fe.fit()

    # Only print summary for main coefficient
    print(
        f"\nNetwork Elasticity (log_diaspora): {ppml_fe_result.params['log_diaspora']:.4f}"
    )
    print(f"  Std. Error: {ppml_fe_result.bse['log_diaspora']:.4f}")
    print(f"  z-statistic: {ppml_fe_result.tvalues['log_diaspora']:.4f}")
    print(f"  p-value: {ppml_fe_result.pvalues['log_diaspora']:.6f}")
    print(f"  N state fixed effects: {len(state_dummies.columns)}")
    print(f"  AIC: {ppml_fe_result.aic:.2f}")
    print(f"  BIC: {ppml_fe_result.bic:.2f}")

    # Calculate model comparison statistics
    # Pseudo R-squared (McFadden)
    null_model = GLM(y_simple, np.ones(len(y_simple)), family=Poisson())
    null_result = null_model.fit()
    null_ll = null_result.llf

    pseudo_r2_simple = 1 - (ppml_simple_result.llf / null_ll)
    pseudo_r2_full = 1 - (ppml_full_result.llf / null_ll)
    pseudo_r2_fe = 1 - (ppml_fe_result.llf / null_ll)

    # Calculate fitted values and residuals for diagnostics
    df["fitted_simple"] = ppml_simple_result.fittedvalues
    df["fitted_full"] = ppml_full_result.fittedvalues
    df["fitted_fe"] = ppml_fe_result.fittedvalues

    df["resid_simple"] = df["flow"] - df["fitted_simple"]
    df["resid_full"] = df["flow"] - df["fitted_full"]
    df["resid_fe"] = df["flow"] - df["fitted_fe"]

    # Pearson residuals
    df["pearson_simple"] = (df["flow"] - df["fitted_simple"]) / np.sqrt(
        df["fitted_simple"]
    )
    df["pearson_full"] = (df["flow"] - df["fitted_full"]) / np.sqrt(df["fitted_full"])

    # Correlation between actual and fitted
    corr_simple = np.corrcoef(df["flow"], df["fitted_simple"])[0, 1]
    corr_full = np.corrcoef(df["flow"], df["fitted_full"])[0, 1]
    corr_fe = np.corrcoef(df["flow"], df["fitted_fe"])[0, 1]

    result.add_decision(
        decision_id="D002",
        category="methodology",
        decision="Used PPML for gravity estimation instead of log-linear OLS",
        rationale="PPML handles zeros naturally and is consistent under heteroskedasticity (Santos Silva & Tenreyro 2006)",
        alternatives=[
            "Log-linear OLS with log(flow+1)",
            "Negative binomial regression",
            "Zero-inflated Poisson",
        ],
        evidence=f"Full model pseudo-R2: {pseudo_r2_full:.4f}",
    )

    # Build comprehensive results
    print("\n" + "=" * 70)
    print("                    GRAVITY MODEL COMPARISON")
    print("=" * 70)
    print(
        f"\n{'Model':<25} {'Network Elast.':>15} {'SE':>10} {'Pseudo R2':>12} {'AIC':>12}"
    )
    print("-" * 75)
    print(
        f"{'Simple Network':<25} {ppml_simple_result.params['log_diaspora']:>15.4f} {ppml_simple_result.bse['log_diaspora']:>10.4f} {pseudo_r2_simple:>12.4f} {ppml_simple_result.aic:>12.2f}"
    )
    print(
        f"{'Full Gravity':<25} {ppml_full_result.params['log_diaspora']:>15.4f} {ppml_full_result.bse['log_diaspora']:>10.4f} {pseudo_r2_full:>12.4f} {ppml_full_result.aic:>12.2f}"
    )
    print(
        f"{'State Fixed Effects':<25} {ppml_fe_result.params['log_diaspora']:>15.4f} {ppml_fe_result.bse['log_diaspora']:>10.4f} {pseudo_r2_fe:>12.4f} {ppml_fe_result.aic:>12.2f}"
    )

    print("\n--- Interpretation ---")
    network_elast = ppml_full_result.params["log_diaspora"]
    print(f"Network Elasticity: {network_elast:.4f}")
    print(
        f"  A 1% increase in diaspora stock is associated with a {network_elast:.4f}% increase in new LPR admissions"
    )
    if network_elast > 0.5:
        print(
            "  -> Strong network effects: Existing diaspora substantially attracts new immigrants"
        )
    elif network_elast > 0:
        print("  -> Positive but moderate network effects")
    else:
        print("  -> Weak or negative network effects (unexpected)")

    gravity_results = {
        "model_1_simple_network": {
            "specification": "Flow ~ log(Diaspora_Stock)",
            "n_observations": int(len(df)),
            "coefficients": {
                "const": {
                    "estimate": float(ppml_simple_result.params["const"]),
                    "std_error": float(ppml_simple_result.bse["const"]),
                    "z_statistic": float(ppml_simple_result.tvalues["const"]),
                    "p_value": float(ppml_simple_result.pvalues["const"]),
                },
                "log_diaspora": {
                    "estimate": float(ppml_simple_result.params["log_diaspora"]),
                    "std_error": float(ppml_simple_result.bse["log_diaspora"]),
                    "z_statistic": float(ppml_simple_result.tvalues["log_diaspora"]),
                    "p_value": float(ppml_simple_result.pvalues["log_diaspora"]),
                    "ci_95_lower": float(
                        ppml_simple_result.conf_int().loc["log_diaspora", 0]
                    ),
                    "ci_95_upper": float(
                        ppml_simple_result.conf_int().loc["log_diaspora", 1]
                    ),
                },
            },
            "fit_statistics": {
                "log_likelihood": float(ppml_simple_result.llf),
                "aic": float(ppml_simple_result.aic),
                "bic": float(ppml_simple_result.bic),
                "pseudo_r2_mcfadden": float(pseudo_r2_simple),
                "correlation_actual_fitted": float(corr_simple),
                "deviance": float(ppml_simple_result.deviance),
                "pearson_chi2": float(ppml_simple_result.pearson_chi2),
            },
        },
        "model_2_full_gravity": {
            "specification": "Flow ~ log(Diaspora) + log(Origin_Mass) + log(Dest_Mass)",
            "n_observations": int(len(df)),
            "coefficients": {
                var: {
                    "estimate": float(ppml_full_result.params[var]),
                    "std_error": float(ppml_full_result.bse[var]),
                    "z_statistic": float(ppml_full_result.tvalues[var]),
                    "p_value": float(ppml_full_result.pvalues[var]),
                    "ci_95_lower": float(ppml_full_result.conf_int().loc[var, 0]),
                    "ci_95_upper": float(ppml_full_result.conf_int().loc[var, 1]),
                }
                for var in ppml_full_result.params.index
            },
            "fit_statistics": {
                "log_likelihood": float(ppml_full_result.llf),
                "aic": float(ppml_full_result.aic),
                "bic": float(ppml_full_result.bic),
                "pseudo_r2_mcfadden": float(pseudo_r2_full),
                "correlation_actual_fitted": float(corr_full),
                "deviance": float(ppml_full_result.deviance),
                "pearson_chi2": float(ppml_full_result.pearson_chi2),
            },
        },
        "model_3_state_fixed_effects": {
            "specification": "Flow ~ log(Diaspora) + State_FE",
            "n_observations": int(len(df)),
            "n_state_dummies": int(len(state_dummies.columns)),
            "network_elasticity": {
                "estimate": float(ppml_fe_result.params["log_diaspora"]),
                "std_error": float(ppml_fe_result.bse["log_diaspora"]),
                "z_statistic": float(ppml_fe_result.tvalues["log_diaspora"]),
                "p_value": float(ppml_fe_result.pvalues["log_diaspora"]),
                "ci_95_lower": float(ppml_fe_result.conf_int().loc["log_diaspora", 0]),
                "ci_95_upper": float(ppml_fe_result.conf_int().loc["log_diaspora", 1]),
            },
            "fit_statistics": {
                "log_likelihood": float(ppml_fe_result.llf),
                "aic": float(ppml_fe_result.aic),
                "bic": float(ppml_fe_result.bic),
                "pseudo_r2_mcfadden": float(pseudo_r2_fe),
                "correlation_actual_fitted": float(corr_fe),
            },
        },
        "model_comparison": {
            "preferred_model": "Full Gravity"
            if ppml_full_result.aic < ppml_simple_result.aic
            else "Simple Network",
            "aic_comparison": {
                "simple": float(ppml_simple_result.aic),
                "full": float(ppml_full_result.aic),
                "state_fe": float(ppml_fe_result.aic),
            },
            "network_elasticity_comparison": {
                "simple": float(ppml_simple_result.params["log_diaspora"]),
                "full": float(ppml_full_result.params["log_diaspora"]),
                "state_fe": float(ppml_fe_result.params["log_diaspora"]),
            },
        },
        "interpretation": {
            "network_elasticity_full": float(ppml_full_result.params["log_diaspora"]),
            "interpretation": f"A 1% increase in diaspora stock is associated with a {ppml_full_result.params['log_diaspora']:.4f}% increase in new LPR admissions",
            "effect_strength": "strong"
            if ppml_full_result.params["log_diaspora"] > 0.5
            else "moderate"
            if ppml_full_result.params["log_diaspora"] > 0.2
            else "weak",
        },
    }

    return gravity_results, df, ppml_full_result


def analyze_nd_network_effects(
    gravity_df: pd.DataFrame, origin_panel: pd.DataFrame, result: ModuleResult
) -> dict:
    """
    Analyze network effects specifically for North Dakota immigration.

    Uses both cross-sectional (gravity) and panel (time series) approaches
    to estimate how diaspora stocks affect new immigrant flows to ND.
    """
    print("\n[4] Analyzing North Dakota Network Effects...")
    print("-" * 70)

    # Filter gravity data to North Dakota
    nd_gravity = gravity_df[gravity_df["state"] == "North Dakota"].copy()
    print(f"    - ND gravity observations: {len(nd_gravity):,}")

    # Cross-sectional PPML for ND
    print("\n" + "=" * 70)
    print("ND-SPECIFIC NETWORK EFFECT (CROSS-SECTIONAL PPML)")
    print("=" * 70)

    nd_valid = nd_gravity.dropna(subset=["flow", "diaspora_stock"])
    nd_valid = nd_valid[nd_valid["flow"] > 0]

    if len(nd_valid) >= 10:
        y_nd = nd_valid["flow"]
        X_nd = sm.add_constant(nd_valid["log_diaspora"])

        ppml_nd = GLM(y_nd, X_nd, family=Poisson())
        ppml_nd_result = ppml_nd.fit()

        print(ppml_nd_result.summary())

        nd_valid["fitted"] = ppml_nd_result.fittedvalues
        nd_valid["residuals"] = nd_valid["flow"] - nd_valid["fitted"]

        # Correlation
        nd_corr = np.corrcoef(nd_valid["flow"], nd_valid["fitted"])[0, 1]

        # Null model for pseudo R2
        null_nd = GLM(y_nd, np.ones(len(y_nd)), family=Poisson())
        null_nd_result = null_nd.fit()
        pseudo_r2_nd = 1 - (ppml_nd_result.llf / null_nd_result.llf)

        nd_ppml_results = {
            "n_observations": int(len(nd_valid)),
            "n_origin_countries": int(nd_valid["country_std"].nunique()),
            "coefficients": {
                "const": {
                    "estimate": float(ppml_nd_result.params["const"]),
                    "std_error": float(ppml_nd_result.bse["const"]),
                    "p_value": float(ppml_nd_result.pvalues["const"]),
                },
                "log_diaspora": {
                    "estimate": float(ppml_nd_result.params["log_diaspora"]),
                    "std_error": float(ppml_nd_result.bse["log_diaspora"]),
                    "z_statistic": float(ppml_nd_result.tvalues["log_diaspora"]),
                    "p_value": float(ppml_nd_result.pvalues["log_diaspora"]),
                    "ci_95_lower": float(
                        ppml_nd_result.conf_int().loc["log_diaspora", 0]
                    ),
                    "ci_95_upper": float(
                        ppml_nd_result.conf_int().loc["log_diaspora", 1]
                    ),
                },
            },
            "fit_statistics": {
                "pseudo_r2": float(pseudo_r2_nd),
                "correlation_actual_fitted": float(nd_corr),
                "aic": float(ppml_nd_result.aic),
                "deviance": float(ppml_nd_result.deviance),
            },
        }
    else:
        print("    WARNING: Insufficient ND observations for PPML estimation")
        nd_ppml_results = {
            "error": "Insufficient observations",
            "n_observations": len(nd_valid),
        }
        nd_valid = None

    # Panel-based network elasticity from Module 3.2 panel
    print("\n" + "=" * 70)
    print("ND PANEL-BASED NETWORK ELASTICITY (OLS)")
    print("Model: log(Stock_t) ~ log(Stock_t-1)")
    print("=" * 70)

    panel_valid = origin_panel.dropna(subset=["log_stock", "log_stock_lag1"])

    if len(panel_valid) >= 10:
        y_panel = panel_valid["log_stock"]
        X_panel = sm.add_constant(panel_valid["log_stock_lag1"])

        ols_panel = sm.OLS(y_panel, X_panel)
        ols_panel_result = ols_panel.fit()

        print(ols_panel_result.summary())

        panel_results = {
            "n_observations": int(len(panel_valid)),
            "n_countries": int(panel_valid["country"].nunique()),
            "years": [int(y) for y in sorted(panel_valid["year"].unique())],
            "coefficients": {
                "const": {
                    "estimate": float(ols_panel_result.params["const"]),
                    "std_error": float(ols_panel_result.bse["const"]),
                    "p_value": float(ols_panel_result.pvalues["const"]),
                },
                "network_elasticity": {
                    "estimate": float(ols_panel_result.params["log_stock_lag1"]),
                    "std_error": float(ols_panel_result.bse["log_stock_lag1"]),
                    "t_statistic": float(ols_panel_result.tvalues["log_stock_lag1"]),
                    "p_value": float(ols_panel_result.pvalues["log_stock_lag1"]),
                    "ci_95_lower": float(
                        ols_panel_result.conf_int().loc["log_stock_lag1", 0]
                    ),
                    "ci_95_upper": float(
                        ols_panel_result.conf_int().loc["log_stock_lag1", 1]
                    ),
                },
            },
            "fit_statistics": {
                "r_squared": float(ols_panel_result.rsquared),
                "adj_r_squared": float(ols_panel_result.rsquared_adj),
                "f_statistic": float(ols_panel_result.fvalue),
                "f_p_value": float(ols_panel_result.f_pvalue),
            },
            "interpretation": {
                "elasticity": float(ols_panel_result.params["log_stock_lag1"]),
                "description": f"A 1% increase in lagged diaspora stock is associated with a {ols_panel_result.params['log_stock_lag1']:.4f}% increase in current stock",
            },
        }
    else:
        print("    WARNING: Insufficient panel observations")
        panel_results = {
            "error": "Insufficient observations",
            "n_observations": len(panel_valid),
        }

    # Compare network effects with controls
    print("\n" + "=" * 70)
    print("NETWORK EFFECT WITH SIZE CONTROLS (OLS PANEL)")
    print("Model: log(Stock_t) ~ log(Stock_t-1) + log(initial_size)")
    print("=" * 70)

    # Get initial size for each country (earliest year in panel)
    initial_size = (
        origin_panel.groupby("country")
        .apply(lambda x: x.sort_values("year")["foreign_born_pop"].iloc[0])
        .reset_index()
    )
    initial_size.columns = ["country", "initial_pop"]
    initial_size["log_initial"] = np.log(initial_size["initial_pop"] + 1)

    panel_with_controls = panel_valid.merge(
        initial_size[["country", "log_initial"]], on="country", how="left"
    )
    panel_with_controls = panel_with_controls.dropna(subset=["log_initial"])

    if len(panel_with_controls) >= 10:
        y_ctrl = panel_with_controls["log_stock"]
        X_ctrl = sm.add_constant(panel_with_controls[["log_stock_lag1", "log_initial"]])

        ols_ctrl = sm.OLS(y_ctrl, X_ctrl)
        ols_ctrl_result = ols_ctrl.fit()

        print(ols_ctrl_result.summary())

        controlled_results = {
            "n_observations": int(len(panel_with_controls)),
            "coefficients": {
                var: {
                    "estimate": float(ols_ctrl_result.params[var]),
                    "std_error": float(ols_ctrl_result.bse[var]),
                    "t_statistic": float(ols_ctrl_result.tvalues[var]),
                    "p_value": float(ols_ctrl_result.pvalues[var]),
                }
                for var in ols_ctrl_result.params.index
            },
            "fit_statistics": {
                "r_squared": float(ols_ctrl_result.rsquared),
                "adj_r_squared": float(ols_ctrl_result.rsquared_adj),
            },
            "interpretation": {
                "network_elasticity_with_controls": float(
                    ols_ctrl_result.params["log_stock_lag1"]
                ),
                "initial_size_effect": float(ols_ctrl_result.params["log_initial"]),
            },
        }
    else:
        controlled_results = {"error": "Insufficient observations"}

    result.add_decision(
        decision_id="D003",
        category="methodology",
        decision="Estimated network effects using both cross-sectional PPML and panel OLS",
        rationale="Cross-sectional captures contemporaneous flow-stock relationship; panel captures dynamic persistence",
        alternatives=["GMM panel estimation", "Instrumental variables"],
        evidence=f"Cross-sectional N={len(nd_valid) if nd_valid is not None else 0}, Panel N={len(panel_valid)}",
    )

    network_results = {
        "nd_cross_sectional_ppml": nd_ppml_results,
        "nd_panel_ols": panel_results,
        "nd_panel_with_controls": controlled_results,
        "summary": {
            "cross_sectional_elasticity": nd_ppml_results.get("coefficients", {})
            .get("log_diaspora", {})
            .get("estimate", None),
            "panel_elasticity": panel_results.get("coefficients", {})
            .get("network_elasticity", {})
            .get("estimate", None),
            "controlled_elasticity": controlled_results.get("interpretation", {}).get(
                "network_elasticity_with_controls", None
            ),
        },
    }

    return network_results, nd_valid


def plot_gravity_fit(gravity_df: pd.DataFrame, result: ModuleResult):
    """Create visualization of gravity model predicted vs actual flows."""
    print("\n[5] Creating Gravity Model Fit Visualization...")
    print("-" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Predicted vs Actual (log scale)
    ax1 = axes[0]

    # Filter to valid fitted values
    plot_df = gravity_df.dropna(subset=["fitted_full", "flow"])
    plot_df = plot_df[plot_df["flow"] > 0]

    ax1.scatter(
        plot_df["fitted_full"],
        plot_df["flow"],
        alpha=0.5,
        s=40,
        color=COLORS["primary"],
        edgecolors="white",
        linewidth=0.5,
    )

    # 45-degree line
    max_val = max(plot_df["flow"].max(), plot_df["fitted_full"].max())
    min_val = min(plot_df["flow"].min(), plot_df["fitted_full"].min())
    ax1.plot(
        [min_val, max_val],
        [min_val, max_val],
        "k--",
        linewidth=1.5,
        label="Perfect Fit",
    )

    # Add correlation
    corr = np.corrcoef(plot_df["flow"], plot_df["fitted_full"])[0, 1]
    ax1.text(
        0.05,
        0.95,
        f"r = {corr:.3f}",
        transform=ax1.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    ax1.set_xlabel("Predicted LPR Admissions (PPML)", fontsize=12)
    ax1.set_ylabel("Actual LPR Admissions", fontsize=12)
    ax1.set_title("Gravity Model Fit: Full Specification", fontsize=12)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. Residuals by diaspora size
    ax2 = axes[1]

    plot_df["pearson_resid"] = (plot_df["flow"] - plot_df["fitted_full"]) / np.sqrt(
        plot_df["fitted_full"].clip(lower=0.1)
    )

    ax2.scatter(
        plot_df["log_diaspora"],
        plot_df["pearson_resid"],
        alpha=0.5,
        s=40,
        color=COLORS["secondary"],
        edgecolors="white",
        linewidth=0.5,
    )

    ax2.axhline(0, color="black", linewidth=1)
    ax2.axhline(2, color=COLORS["neutral"], linestyle=":", linewidth=1)
    ax2.axhline(-2, color=COLORS["neutral"], linestyle=":", linewidth=1)

    # Add LOESS-like smooth
    z = np.polyfit(plot_df["log_diaspora"], plot_df["pearson_resid"], 2)
    p = np.poly1d(z)
    x_smooth = np.linspace(
        plot_df["log_diaspora"].min(), plot_df["log_diaspora"].max(), 100
    )
    ax2.plot(
        x_smooth,
        p(x_smooth),
        color=COLORS["tertiary"],
        linewidth=2,
        label="Quadratic Fit",
    )

    ax2.set_xlabel("log(Diaspora Stock)", fontsize=12)
    ax2.set_ylabel("Pearson Residuals", fontsize=12)
    ax2.set_title("Residual Diagnostics by Diaspora Size", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    save_figure(
        fig,
        str(FIGURES_DIR / "module_5_gravity_fit"),
        "Gravity Model Fit: Immigration Flows vs Diaspora Networks",
        "DHS LPR FY2023, ACS 2023",
    )


def plot_network_elasticity(
    nd_data: pd.DataFrame, gravity_df: pd.DataFrame, result: ModuleResult
):
    """Create visualization of network elasticity effects."""
    print("\n[6] Creating Network Elasticity Visualization...")
    print("-" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Log-log scatter of flow vs diaspora (all states)
    ax1 = axes[0]

    # Sample for visibility
    plot_df = gravity_df.dropna(subset=["flow", "diaspora_stock"])
    plot_df = plot_df[(plot_df["flow"] > 0) & (plot_df["diaspora_stock"] > 0)]

    ax1.scatter(
        plot_df["log_diaspora"],
        plot_df["log_flow"],
        alpha=0.4,
        s=30,
        color=COLORS["primary"],
        edgecolors="none",
        label="All states",
    )

    # Highlight ND
    nd_plot = plot_df[plot_df["state"] == "North Dakota"]
    if len(nd_plot) > 0:
        ax1.scatter(
            nd_plot["log_diaspora"],
            nd_plot["log_flow"],
            alpha=0.8,
            s=80,
            color=COLORS["secondary"],
            edgecolors="black",
            linewidth=1,
            label="North Dakota",
            zorder=5,
        )

    # Add regression line
    slope, intercept, r, p, se = stats.linregress(
        plot_df["log_diaspora"], plot_df["log_flow"]
    )
    x_line = np.linspace(
        plot_df["log_diaspora"].min(), plot_df["log_diaspora"].max(), 100
    )
    ax1.plot(
        x_line,
        intercept + slope * x_line,
        color=COLORS["tertiary"],
        linewidth=2,
        linestyle="--",
        label=f"OLS: slope={slope:.3f}",
    )

    ax1.set_xlabel("log(Diaspora Stock)", fontsize=12)
    ax1.set_ylabel("log(LPR Admissions)", fontsize=12)
    ax1.set_title("Network Effect: Diaspora Stock vs New Arrivals", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Add annotation
    ax1.text(
        0.05,
        0.95,
        f"Network Elasticity = {slope:.3f}\n(SE = {se:.3f})\nR = {r:.3f}",
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    # 2. Elasticity by state
    ax2 = axes[1]

    # Calculate state-level elasticities (simplified - correlation coefficient)
    state_elasticities = []
    for state in plot_df["state"].unique():
        state_data = plot_df[plot_df["state"] == state]
        if len(state_data) >= 5:
            try:
                slope_st, _, r_st, p_st, _ = stats.linregress(
                    state_data["log_diaspora"], state_data["log_flow"]
                )
                state_elasticities.append(
                    {
                        "state": state,
                        "elasticity": slope_st,
                        "r_squared": r_st**2,
                        "n_obs": len(state_data),
                        "significant": p_st < 0.05,
                    }
                )
            except Exception:
                pass

    state_elast_df = pd.DataFrame(state_elasticities)
    state_elast_df = state_elast_df.sort_values("elasticity", ascending=True)

    # Select top and bottom states
    if len(state_elast_df) > 20:
        display_df = pd.concat([state_elast_df.head(10), state_elast_df.tail(10)])
    else:
        display_df = state_elast_df

    colors = [
        COLORS["secondary"] if s == "North Dakota" else COLORS["primary"]
        for s in display_df["state"]
    ]

    ax2.barh(range(len(display_df)), display_df["elasticity"], color=colors, alpha=0.7)

    ax2.set_yticks(range(len(display_df)))
    ax2.set_yticklabels(display_df["state"], fontsize=9)
    ax2.axvline(
        slope,
        color=COLORS["neutral"],
        linestyle="--",
        linewidth=2,
        label=f"National avg: {slope:.3f}",
    )

    ax2.set_xlabel("Network Elasticity (OLS slope)", fontsize=12)
    ax2.set_title("Network Elasticity by State", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis="x")

    # Highlight ND position
    nd_idx = display_df[display_df["state"] == "North Dakota"].index
    if len(nd_idx) > 0:
        nd_elast = display_df.loc[nd_idx[0], "elasticity"]
        ax2.annotate(
            f"ND: {nd_elast:.3f}",
            xy=(nd_elast, list(display_df["state"]).index("North Dakota")),
            fontsize=10,
            fontweight="bold",
            color=COLORS["secondary"],
        )

    save_figure(
        fig,
        str(FIGURES_DIR / "module_5_network_elasticity"),
        "Network Elasticity: How Existing Diasporas Attract New Immigrants",
        "DHS LPR FY2023, ACS 2023",
    )


def run_analysis() -> ModuleResult:
    """Main analysis function for Module 5."""
    result = ModuleResult(module_id="5", analysis_name="gravity_model_network_analysis")

    # Load data
    dhs_lpr, acs_origin, origin_panel = load_data(result)

    # Prepare gravity data
    gravity_df, prep_decisions = prepare_gravity_data(dhs_lpr, acs_origin, result)

    # Estimate PPML gravity models
    gravity_results, gravity_df, ppml_model = estimate_gravity_ppml(gravity_df, result)

    # Analyze ND-specific network effects
    network_results, nd_data = analyze_nd_network_effects(
        gravity_df, origin_panel, result
    )

    # Create visualizations
    plot_gravity_fit(gravity_df, result)
    plot_network_elasticity(nd_data, gravity_df, result)

    # Record parameters
    result.parameters = {
        "data_preparation": prep_decisions,
        "gravity_model": {
            "estimator": "PPML (Poisson Pseudo-Maximum Likelihood)",
            "family": "Poisson",
            "link": "log",
            "models_estimated": 3,
        },
        "network_analysis": {
            "cross_sectional_method": "PPML",
            "panel_method": "OLS",
            "controls": ["initial_size"],
        },
    }

    # Compile results
    result.results = {
        "gravity_model": gravity_results,
        "network_effects": network_results,
    }

    # Diagnostics
    result.diagnostics = {
        "gravity_model": {
            "n_observations": gravity_results["model_2_full_gravity"]["n_observations"],
            "deviance": gravity_results["model_2_full_gravity"]["fit_statistics"][
                "deviance"
            ],
            "pearson_chi2": gravity_results["model_2_full_gravity"]["fit_statistics"][
                "pearson_chi2"
            ],
            "correlation_actual_fitted": gravity_results["model_2_full_gravity"][
                "fit_statistics"
            ]["correlation_actual_fitted"],
        },
        "nd_network": {
            "cross_sectional_n": network_results["nd_cross_sectional_ppml"].get(
                "n_observations", 0
            ),
            "panel_n": network_results["nd_panel_ols"].get("n_observations", 0),
        },
    }

    # Warnings
    if (
        gravity_results["model_2_full_gravity"]["fit_statistics"][
            "correlation_actual_fitted"
        ]
        < 0.7
    ):
        result.warnings.append(
            "Gravity model fit is moderate - consider additional controls"
        )

    if network_results["nd_cross_sectional_ppml"].get("n_observations", 0) < 20:
        result.warnings.append("Limited ND cross-sectional observations for PPML")

    # Next steps
    result.next_steps = [
        "Compare gravity model predictions with Module 2 time series forecasts",
        "Use network elasticity for projection scenarios",
        "Investigate origin-specific network effects for major sending countries",
        "Consider instrumental variable estimation to address endogeneity",
    ]

    # Save gravity model results separately
    gravity_output = RESULTS_DIR / "module_5_gravity_model.json"
    with open(gravity_output, "w") as f:
        json.dump(gravity_results, f, indent=2, default=str)
    print(f"\nGravity model results saved: {gravity_output}")

    # Save network effects results separately
    network_output = RESULTS_DIR / "module_5_network_effects.json"
    with open(network_output, "w") as f:
        json.dump(network_results, f, indent=2, default=str)
    print(f"Network effects results saved: {network_output}")

    return result


def main():
    """Main entry point."""
    print("=" * 70)
    print("Module 5: Gravity Model and Network Analysis")
    print(f"Started: {datetime.now(UTC).isoformat()}")
    print("=" * 70)

    try:
        result = run_analysis()
        output_file = result.save("module_5_gravity_network.json")

        print("\n" + "=" * 70)
        print("Analysis completed successfully!")
        print("=" * 70)

        print(f"\nMain output: {output_file}")

        print("\n--- Key Results ---")

        # Gravity model results
        if "gravity_model" in result.results:
            gm = result.results["gravity_model"]
            print("\nGravity Model (Full Specification):")
            full_model = gm.get("model_2_full_gravity", {})
            network_coef = full_model.get("coefficients", {}).get("log_diaspora", {})
            print(f"  Network Elasticity: {network_coef.get('estimate', 'N/A'):.4f}")
            print(f"  Standard Error: {network_coef.get('std_error', 'N/A'):.4f}")
            print(f"  p-value: {network_coef.get('p_value', 'N/A'):.6f}")
            print(
                f"  Pseudo R2: {full_model.get('fit_statistics', {}).get('pseudo_r2_mcfadden', 'N/A'):.4f}"
            )

            interpretation = gm.get("interpretation", {})
            print(f"\n  Interpretation: {interpretation.get('interpretation', 'N/A')}")

        # ND network effects
        if "network_effects" in result.results:
            ne = result.results["network_effects"]
            summary = ne.get("summary", {})
            print("\nND Network Effects:")
            print(
                f"  Cross-sectional elasticity: {summary.get('cross_sectional_elasticity', 'N/A')}"
            )
            print(f"  Panel elasticity: {summary.get('panel_elasticity', 'N/A')}")
            print(
                f"  Controlled elasticity: {summary.get('controlled_elasticity', 'N/A')}"
            )

        if result.warnings:
            print("\nWarnings:")
            for w in result.warnings:
                print(f"  - {w}")

        print(f"\nDecisions logged: {len(result.decisions)}")
        for d in result.decisions:
            print(f"  [{d['decision_id']}] {d['decision']}")

        print("\nFigures generated:")
        print("  - module_5_gravity_fit.png/pdf")
        print("  - module_5_network_elasticity.png/pdf")

        print("\nNext steps:")
        for step in result.next_steps:
            print(f"  - {step}")

        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
