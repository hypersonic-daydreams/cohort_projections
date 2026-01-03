#!/usr/bin/env python3
"""
Module 7b: LSSND Closure Synthetic Control Analysis
====================================================

Implements synthetic control methods to estimate the causal effect of Lutheran
Social Services of North Dakota (LSSND) closure in January 2021 on refugee
resettlement capacity.

ADR-021 Recommendation #3: Quantify ND reception capacity effect via synthetic
control analysis.

Methodology:
1. Pre-treatment synthetic control (Abadie et al. 2010) using donor pool
2. National share-based counterfactual for post-treatment projection
3. Both methods combined for robust capacity estimation

Key Features:
- Treatment: North Dakota after January 2021 (LSSND closure)
- Donor pool: SD, NE, ID, ME, VT, NH (similar low-flow states with stable infrastructure)
- Pre-treatment period: FY2010-2020
- Post-treatment period: FY2021-2024
- National share method: ND's pre-treatment share applied to national post-treatment

Data Note:
- FY2021-2024 state-level data only available for ND
- National totals available for all years
- Counterfactual constructed using ND's historical share of national arrivals

Usage:
    uv run python module_7b_lssnd_synthetic_control.py
"""

import json
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Add scripts directory to path to find db_config
sys.path.append(str(Path(__file__).parent.parent))
from database import db_config

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # cohort_projections/
RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"
ADR_RESULTS_DIR = PROJECT_ROOT / "docs" / "governance" / "adrs" / "021-reports" / "results"
ADR_FIGURES_DIR = PROJECT_ROOT / "docs" / "governance" / "adrs" / "021-reports" / "figures"

# Ensure output directories exist
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
ADR_RESULTS_DIR.mkdir(exist_ok=True)
ADR_FIGURES_DIR.mkdir(exist_ok=True)

# Standard color palette (colorblind-safe)
COLORS = {
    "primary": "#0072B2",  # Blue
    "secondary": "#D55E00",  # Vermillion/Orange
    "tertiary": "#009E73",  # Teal/Green
    "quaternary": "#CC79A7",  # Pink
    "highlight": "#F0E442",  # Yellow
    "neutral": "#999999",  # Gray
    "nd_actual": "#E31A1C",  # Red
    "synthetic": "#1F78B4",  # Blue
}


class SyntheticControlResult(NamedTuple):
    """Container for synthetic control estimation results."""

    weights: dict  # State -> weight mapping
    pre_rmspe: float  # Pre-treatment root mean squared prediction error
    post_rmspe: float  # Post-treatment RMSPE
    rmspe_ratio: float  # post_rmspe / pre_rmspe
    att_mean: float  # Average treatment effect (mean gap)
    att_by_year: dict  # Year -> treatment effect
    capacity_multiplier: float  # actual / synthetic ratio (post-treatment)


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
        alternatives: list[str] | None = None,
        evidence: str | None = None,
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

    def save(self, filename: str, output_dir: Path = RESULTS_DIR) -> Path:
        """Save results to JSON file."""
        output_path = output_dir / filename
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        print(f"Results saved to: {output_path}")
        return output_path


# =============================================================================
# DATA LOADING
# =============================================================================


def load_refugee_data() -> pd.DataFrame:
    """Load refugee arrivals data from PostgreSQL."""
    conn = db_config.get_db_connection()
    try:
        query = """
        SELECT
            fiscal_year as year,
            destination_state as state,
            SUM(arrivals) as arrivals
        FROM rpc.refugee_arrivals
        GROUP BY fiscal_year, destination_state
        ORDER BY destination_state, fiscal_year
        """
        df = pd.read_sql(query, conn)
        print(f"Loaded refugee arrivals: {df.shape[0]} state-year observations")
        return df
    finally:
        conn.close()


def load_census_data() -> pd.DataFrame:
    """Load census population and components data."""
    conn = db_config.get_db_connection()
    try:
        query = """
        SELECT
            year,
            state_name as state,
            population,
            intl_migration
        FROM census.state_components
        WHERE state_name IS NOT NULL
          AND state_name NOT IN ('Puerto Rico', 'United States')
        ORDER BY state_name, year
        """
        df = pd.read_sql(query, conn)
        print(f"Loaded census data: {df.shape[0]} state-year observations")
        return df
    finally:
        conn.close()


def load_acs_foreign_born() -> pd.DataFrame:
    """Load ACS foreign-born population data."""
    conn = db_config.get_db_connection()
    try:
        query = """
        SELECT
            calendar_year as year,
            state_name as state,
            SUM(estimate) as foreign_born
        FROM acs.foreign_born
        WHERE state_name IS NOT NULL
        GROUP BY calendar_year, state_name
        ORDER BY state_name, calendar_year
        """
        df = pd.read_sql(query, conn)
        print(f"Loaded ACS foreign-born: {df.shape[0]} state-year observations")
        return df
    finally:
        conn.close()


def load_national_refugee_totals() -> pd.DataFrame:
    """
    Load national (US total) refugee arrivals by year.

    Note: For FY2021-2024, the database only contains ND data since post-2020
    data for other states has not been ingested. We supplement with official
    national totals from the Refugee Processing Center and DHS.

    Official sources:
    - FY2021: 11,411 (DHS Yearbook 2021)
    - FY2022: 25,519 (DHS Yearbook 2022)
    - FY2023: 60,014 (DHS/RPC)
    - FY2024: 100,034 (RPC report as of 30 Oct 2024)
    """
    conn = db_config.get_db_connection()
    try:
        # Load pre-2021 from database (complete state coverage)
        query = """
        SELECT
            fiscal_year as year,
            SUM(arrivals) as national_arrivals
        FROM rpc.refugee_arrivals
        WHERE fiscal_year <= 2020
        GROUP BY fiscal_year
        ORDER BY fiscal_year
        """
        df = pd.read_sql(query, conn)
        print(f"Loaded national totals from DB: {len(df)} years (FY2002-2020)")
    finally:
        conn.close()

    # Add FY2021-2024 national totals from official sources
    # Sources: DHS Yearbooks, RPC archives, ADR-021 data acquisition report
    official_post_2020 = pd.DataFrame({
        "year": [2021, 2022, 2023, 2024],
        "national_arrivals": [11411, 25519, 60014, 100034],
    })

    df = pd.concat([df, official_post_2020], ignore_index=True)
    df = df.sort_values("year").reset_index(drop=True)

    print("Added official FY2021-2024 national totals")
    print(f"Total: {len(df)} years (FY2002-2024)")

    return df


def prepare_panel_data(
    df_refugee: pd.DataFrame,
    df_census: pd.DataFrame,
    df_acs: pd.DataFrame,
    result: ModuleResult,
) -> pd.DataFrame:
    """
    Prepare panel data for synthetic control analysis.

    Returns DataFrame with:
    - year, state
    - arrivals (refugee arrivals)
    - population
    - foreign_born_share
    - arrivals_rate (per 100,000 population)
    """
    # Merge refugee and census data
    df = df_refugee.merge(df_census, on=["year", "state"], how="outer")

    # Merge ACS foreign-born
    df = df.merge(df_acs, on=["year", "state"], how="left")

    # Calculate rates and shares
    df["arrivals"] = df["arrivals"].fillna(0)
    df["arrivals_rate"] = (df["arrivals"] / df["population"]) * 100000
    df["foreign_born_share"] = (df["foreign_born"] / df["population"]) * 100

    result.input_files.extend([
        "rpc.refugee_arrivals (PostgreSQL)",
        "census.state_components (PostgreSQL)",
        "acs.foreign_born (PostgreSQL)",
    ])

    return df


# =============================================================================
# SYNTHETIC CONTROL ESTIMATION
# =============================================================================


def estimate_synthetic_control(
    df: pd.DataFrame,
    treated_unit: str,
    donor_states: list[str],
    pre_years: list[int],
    post_years: list[int],
    outcome_var: str = "arrivals",
    covariates: list[str] | None = None,
    result: ModuleResult | None = None,
) -> tuple[SyntheticControlResult, pd.DataFrame]:
    """
    Estimate synthetic control weights using constrained optimization.

    Implements the Abadie et al. (2010) synthetic control method:
    - Minimize pre-treatment MSPE subject to convexity constraints
    - Weights sum to 1, all weights >= 0

    Args:
        df: Panel data with year, state, and outcome
        treated_unit: Name of treated state (e.g., "North Dakota")
        donor_states: List of potential donor state names
        pre_years: List of pre-treatment years
        post_years: List of post-treatment years
        outcome_var: Outcome variable name (default: "arrivals")
        covariates: Optional list of covariates for matching
        result: ModuleResult for logging decisions

    Returns:
        SyntheticControlResult with weights and diagnostics
        DataFrame with actual, synthetic, and gap by year
    """
    print("\n" + "=" * 60)
    print(f"SYNTHETIC CONTROL ESTIMATION: {treated_unit}")
    print("=" * 60)

    # Filter to relevant states and years
    all_years = pre_years + post_years
    relevant_states = [treated_unit] + donor_states

    df_analysis = df[
        (df["state"].isin(relevant_states)) &
        (df["year"].isin(all_years))
    ].copy()

    # Pivot to wide format: rows = years, columns = states
    outcome_matrix = df_analysis.pivot_table(
        index="year",
        columns="state",
        values=outcome_var,
        aggfunc="first"
    )

    # Check data availability
    if treated_unit not in outcome_matrix.columns:
        raise ValueError(f"Treated unit '{treated_unit}' not found in data")

    available_donors = [s for s in donor_states if s in outcome_matrix.columns]
    if len(available_donors) < 2:
        raise ValueError(f"Insufficient donors: only {len(available_donors)} found")

    print(f"\nTreated unit: {treated_unit}")
    print(f"Donor pool: {len(available_donors)} states")
    print(f"Pre-treatment periods: {len(pre_years)} ({min(pre_years)}-{max(pre_years)})")
    print(f"Post-treatment periods: {len(post_years)} ({min(post_years)}-{max(post_years)})")

    # Extract pre-treatment data
    pre_idx = outcome_matrix.index.isin(pre_years)
    Y_treated_pre = outcome_matrix.loc[pre_idx, treated_unit].values
    Y_donors_pre = outcome_matrix.loc[pre_idx, available_donors].values

    # Handle missing values in pre-period
    valid_donors_mask = ~np.any(np.isnan(Y_donors_pre), axis=0)
    valid_donors = [s for i, s in enumerate(available_donors) if valid_donors_mask[i]]
    Y_donors_pre = Y_donors_pre[:, valid_donors_mask]

    if len(valid_donors) < 2:
        raise ValueError("Too few donors with complete pre-treatment data")

    print(f"Valid donors (complete pre-treatment data): {len(valid_donors)}")

    # Handle missing values in treated unit pre-period
    if np.any(np.isnan(Y_treated_pre)):
        nan_years = [pre_years[i] for i, v in enumerate(Y_treated_pre) if np.isnan(v)]
        raise ValueError(f"Treated unit has missing data in years: {nan_years}")

    # Optimization: minimize pre-treatment MSPE
    def objective(w):
        synthetic = Y_donors_pre @ w
        return np.mean((Y_treated_pre - synthetic) ** 2)

    # Constraints: weights sum to 1
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    # Bounds: non-negative weights
    bounds = [(0, 1) for _ in range(len(valid_donors))]

    # Initial guess: uniform weights
    w0 = np.ones(len(valid_donors)) / len(valid_donors)

    # Optimize
    opt_result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10},
    )

    if not opt_result.success:
        print(f"WARNING: Optimization did not fully converge: {opt_result.message}")

    weights = opt_result.x

    # Report significant weights
    weight_threshold = 0.01
    significant_weights = {
        s: float(w)
        for s, w in zip(valid_donors, weights)
        if w > weight_threshold
    }
    significant_weights = dict(sorted(significant_weights.items(), key=lambda x: -x[1]))

    print(f"\nSynthetic control weights (>{weight_threshold*100}%):")
    for state, weight in list(significant_weights.items())[:10]:
        print(f"  {state}: {weight:.3f} ({weight*100:.1f}%)")

    # Construct synthetic control for all years
    Y_donors_all = outcome_matrix[valid_donors].values
    synthetic_all = Y_donors_all @ weights
    actual_all = outcome_matrix[treated_unit].values

    # Calculate gap (treatment effect)
    gap = actual_all - synthetic_all

    # Pre-treatment fit metrics
    pre_gap = gap[outcome_matrix.index.isin(pre_years)]
    pre_rmspe = np.sqrt(np.mean(pre_gap ** 2))
    pre_mae = np.mean(np.abs(pre_gap))

    # Post-treatment effect metrics
    post_idx = outcome_matrix.index.isin(post_years)
    post_gap = gap[post_idx]
    post_rmspe = np.sqrt(np.mean(post_gap ** 2))

    # RMSPE ratio (key diagnostic)
    rmspe_ratio = post_rmspe / pre_rmspe if pre_rmspe > 0 else np.inf

    # Average treatment effect
    att_mean = np.mean(post_gap)

    # Year-by-year effects
    post_actual = actual_all[post_idx]
    post_synthetic = synthetic_all[post_idx]
    post_years_list = outcome_matrix.index[post_idx].tolist()

    att_by_year = {
        int(year): float(gap_val)
        for year, gap_val in zip(post_years_list, post_gap)
    }

    # Capacity multiplier: actual / synthetic ratio
    # This tells us what fraction of expected arrivals actually occurred
    with np.errstate(divide="ignore", invalid="ignore"):
        capacity_by_year = np.where(
            post_synthetic > 0,
            post_actual / post_synthetic,
            np.nan
        )
    capacity_multiplier = float(np.nanmean(capacity_by_year))

    print(f"\n{'='*60}")
    print("SYNTHETIC CONTROL RESULTS")
    print("=" * 60)
    print("\nPre-treatment fit:")
    print(f"  RMSPE: {pre_rmspe:.2f}")
    print(f"  MAE: {pre_mae:.2f}")
    print("\nPost-treatment (LSSND closure effect):")
    print(f"  Mean gap (ATT): {att_mean:.2f}")
    print(f"  RMSPE: {post_rmspe:.2f}")
    print(f"  RMSPE ratio: {rmspe_ratio:.2f}")
    print(f"\nCapacity multiplier: {capacity_multiplier:.3f}")
    print(f"  Interpretation: ND received {capacity_multiplier*100:.1f}% of expected arrivals")

    print("\nYear-by-year effects:")
    for year in post_years_list:
        actual_y = actual_all[outcome_matrix.index == year][0]
        synth_y = synthetic_all[outcome_matrix.index == year][0]
        gap_y = att_by_year[int(year)]
        pct_y = (actual_y / synth_y * 100) if synth_y > 0 else np.nan
        print(f"  {year}: Actual={actual_y:.0f}, Synthetic={synth_y:.0f}, "
              f"Gap={gap_y:.0f} ({pct_y:.1f}% of expected)")

    # Create output DataFrame
    output_df = pd.DataFrame({
        "year": outcome_matrix.index,
        "actual": actual_all,
        "synthetic": synthetic_all,
        "gap": gap,
        "period": ["pre" if y in pre_years else "post" for y in outcome_matrix.index],
    })

    # Log decision
    if result is not None:
        result.add_decision(
            decision_id="SC001",
            category="causal_identification",
            decision=f"Synthetic control for {treated_unit} using {len(valid_donors)} donors",
            rationale="Donors selected based on similar size and stable resettlement infrastructure",
            alternatives=["Larger donor pool", "Covariate matching", "Different pre-period"],
            evidence=f"Pre-RMSPE={pre_rmspe:.2f}, RMSPE ratio={rmspe_ratio:.2f}",
        )

    sc_result = SyntheticControlResult(
        weights=significant_weights,
        pre_rmspe=float(pre_rmspe),
        post_rmspe=float(post_rmspe),
        rmspe_ratio=float(rmspe_ratio),
        att_mean=float(att_mean),
        att_by_year=att_by_year,
        capacity_multiplier=float(capacity_multiplier),
    )

    return sc_result, output_df


# =============================================================================
# NATIONAL SHARE-BASED SYNTHETIC CONTROL
# =============================================================================


def estimate_national_share_synthetic(
    df_nd: pd.DataFrame,
    df_national: pd.DataFrame,
    pre_years: list[int],
    post_years: list[int],
    result: ModuleResult | None = None,
) -> tuple[dict, pd.DataFrame]:
    """
    Estimate synthetic control using national share method.

    This approach constructs a counterfactual based on ND's historical share
    of national refugee arrivals, then projects what ND would have received
    post-treatment if that share had been maintained.

    This is appropriate when:
    - Post-treatment donor state data is unavailable
    - National trends are observable
    - Pre-treatment share is relatively stable

    Args:
        df_nd: DataFrame with ND arrivals by year
        df_national: DataFrame with national arrivals by year
        pre_years: Pre-treatment years
        post_years: Post-treatment years
        result: ModuleResult for logging

    Returns:
        Result dictionary and time series DataFrame
    """
    print("\n" + "=" * 60)
    print("NATIONAL SHARE-BASED SYNTHETIC CONTROL")
    print("=" * 60)

    # Merge ND and national data
    df = df_nd.merge(df_national, on="year", how="outer")
    df = df.sort_values("year")

    # Calculate ND share of national arrivals
    df["nd_share"] = df["arrivals"] / df["national_arrivals"]

    print(f"\nData range: {df['year'].min()}-{df['year'].max()}")
    print(f"Pre-treatment: {min(pre_years)}-{max(pre_years)}")
    print(f"Post-treatment: {min(post_years)}-{max(post_years)}")

    # Calculate pre-treatment average share
    pre_mask = df["year"].isin(pre_years)
    pre_df = df[pre_mask].dropna(subset=["nd_share"])

    pre_share_mean = pre_df["nd_share"].mean()
    pre_share_std = pre_df["nd_share"].std()

    # Also calculate share excluding 2017-2020 (Travel Ban period)
    stable_pre_years = [y for y in pre_years if y < 2017]
    stable_pre_mask = df["year"].isin(stable_pre_years)
    stable_pre_df = df[stable_pre_mask].dropna(subset=["nd_share"])

    if len(stable_pre_df) > 0:
        stable_share_mean = stable_pre_df["nd_share"].mean()
        stable_share_std = stable_pre_df["nd_share"].std()
    else:
        stable_share_mean = pre_share_mean
        stable_share_std = pre_share_std

    print("\nPre-treatment ND share of national arrivals:")
    print(f"  Full pre-period ({min(pre_years)}-{max(pre_years)}): "
          f"{pre_share_mean*100:.3f}% (SD={pre_share_std*100:.3f}%)")

    if stable_pre_years:
        print(f"  Pre-Travel Ban ({min(stable_pre_years)}-{max(stable_pre_years)}): "
              f"{stable_share_mean*100:.3f}% (SD={stable_share_std*100:.3f}%)")

    # Construct synthetic counterfactual
    # Use stable pre-period share (before Travel Ban) as baseline
    df["synthetic"] = df["national_arrivals"] * stable_share_mean
    df["gap"] = df["arrivals"] - df["synthetic"]
    df["capacity_ratio"] = df["arrivals"] / df["synthetic"]

    # Mark periods
    df["period"] = df["year"].apply(
        lambda y: "pre" if y in pre_years else "post" if y in post_years else "other"
    )

    # Pre-treatment fit metrics
    pre_rmspe = np.sqrt(np.mean((pre_df["arrivals"] - pre_df["national_arrivals"] * stable_share_mean) ** 2))
    pre_mae = np.mean(np.abs(pre_df["arrivals"] - pre_df["national_arrivals"] * stable_share_mean))

    # Post-treatment effects
    post_mask = df["year"].isin(post_years)
    post_df = df[post_mask].dropna(subset=["arrivals", "synthetic"])

    if len(post_df) > 0:
        post_rmspe = np.sqrt(np.mean(post_df["gap"] ** 2))
        att_mean = post_df["gap"].mean()
        capacity_multiplier = post_df["capacity_ratio"].mean()

        att_by_year = {
            int(row["year"]): float(row["gap"])
            for _, row in post_df.iterrows()
        }

        capacity_by_year = {
            int(row["year"]): float(row["capacity_ratio"])
            for _, row in post_df.iterrows()
        }
    else:
        post_rmspe = np.nan
        att_mean = np.nan
        capacity_multiplier = np.nan
        att_by_year = {}
        capacity_by_year = {}

    rmspe_ratio = post_rmspe / pre_rmspe if pre_rmspe > 0 else np.nan

    print(f"\n{'='*60}")
    print("RESULTS: NATIONAL SHARE METHOD")
    print("=" * 60)
    print(f"\nCounterfactual share used: {stable_share_mean*100:.3f}%")
    print(f"  (Based on {min(stable_pre_years)}-{max(stable_pre_years)} average)")

    print("\nPre-treatment fit:")
    print(f"  RMSPE: {pre_rmspe:.2f}")
    print(f"  MAE: {pre_mae:.2f}")

    print("\nPost-treatment (LSSND closure effect):")
    print(f"  Mean gap (ATT): {att_mean:.1f}")
    print(f"  RMSPE: {post_rmspe:.2f}")
    print(f"  RMSPE ratio: {rmspe_ratio:.2f}")
    print(f"\n  Capacity multiplier: {capacity_multiplier:.3f}")
    print(f"  Interpretation: ND received {capacity_multiplier*100:.1f}% of expected arrivals")

    print("\nYear-by-year comparison:")
    for _, row in post_df.iterrows():
        year = int(row["year"])
        actual = row["arrivals"]
        synthetic = row["synthetic"]
        gap = row["gap"]
        ratio = row["capacity_ratio"]
        print(f"  FY{year}: Actual={actual:.0f}, Expected={synthetic:.0f}, "
              f"Gap={gap:.0f} ({ratio*100:.1f}% of expected)")

    # Build result
    ns_result = {
        "method": "national_share_synthetic",
        "counterfactual_share": {
            "value": float(stable_share_mean),
            "based_on": f"FY{min(stable_pre_years)}-{max(stable_pre_years)} average",
            "pre_period_mean": float(pre_share_mean),
            "pre_period_std": float(pre_share_std),
            "stable_period_mean": float(stable_share_mean),
            "stable_period_std": float(stable_share_std),
        },
        "pre_treatment_fit": {
            "rmspe": float(pre_rmspe),
            "mae": float(pre_mae),
        },
        "post_treatment_effect": {
            "att_mean": float(att_mean) if not np.isnan(att_mean) else None,
            "rmspe": float(post_rmspe) if not np.isnan(post_rmspe) else None,
            "rmspe_ratio": float(rmspe_ratio) if not np.isnan(rmspe_ratio) else None,
            "capacity_multiplier": float(capacity_multiplier) if not np.isnan(capacity_multiplier) else None,
        },
        "att_by_year": att_by_year,
        "capacity_by_year": capacity_by_year,
        "interpretation": {
            "att_text": f"ND received {abs(att_mean):.0f} {'fewer' if att_mean < 0 else 'more'} "
                       f"refugees per year than expected post-LSSND closure",
            "capacity_text": f"ND operated at {capacity_multiplier*100:.1f}% of expected capacity",
        },
    }

    # Output DataFrame
    output_df = df[["year", "arrivals", "national_arrivals", "nd_share",
                    "synthetic", "gap", "capacity_ratio", "period"]].copy()
    output_df = output_df.rename(columns={"arrivals": "actual"})

    if result is not None:
        result.add_decision(
            decision_id="NS001",
            category="methodology",
            decision=f"National share synthetic control using {stable_share_mean*100:.3f}% share",
            rationale="Post-treatment donor state data unavailable; use ND's historical share "
                     "of national arrivals as counterfactual",
            alternatives=["Use full pre-period share", "Time-varying share model",
                         "Regression-based projection"],
            evidence=f"Pre-Travel Ban share = {stable_share_mean*100:.3f}% "
                    f"(SD={stable_share_std*100:.3f}%)",
        )

    return ns_result, output_df


# =============================================================================
# ROBUSTNESS CHECKS
# =============================================================================


def run_placebo_tests(
    df: pd.DataFrame,
    treated_unit: str,
    donor_states: list[str],
    pre_years: list[int],
    post_years: list[int],
    outcome_var: str = "arrivals",
    result: ModuleResult | None = None,
) -> dict:
    """
    Run placebo tests: apply synthetic control to each donor state.

    A valid treatment effect should be larger than most placebo effects.
    We calculate the p-value as: proportion of placebos with RMSPE ratio >= treated.
    """
    print("\n" + "=" * 60)
    print("PLACEBO TESTS (In-space)")
    print("=" * 60)

    # Get treated unit result
    try:
        treated_result, _ = estimate_synthetic_control(
            df, treated_unit, donor_states, pre_years, post_years, outcome_var
        )
        treated_rmspe_ratio = treated_result.rmspe_ratio
    except Exception as e:
        print(f"ERROR: Could not estimate treated unit: {e}")
        return {"feasible": False, "error": str(e)}

    # Run placebo for each donor
    placebo_results = []

    for placebo_unit in donor_states:
        # Donor pool is treated + other donors (excluding placebo unit)
        placebo_donors = [treated_unit] + [d for d in donor_states if d != placebo_unit]

        try:
            placebo_result, _ = estimate_synthetic_control(
                df, placebo_unit, placebo_donors, pre_years, post_years, outcome_var
            )
            placebo_results.append({
                "state": placebo_unit,
                "pre_rmspe": placebo_result.pre_rmspe,
                "post_rmspe": placebo_result.post_rmspe,
                "rmspe_ratio": placebo_result.rmspe_ratio,
                "att_mean": placebo_result.att_mean,
            })
            print(f"  {placebo_unit}: RMSPE ratio = {placebo_result.rmspe_ratio:.2f}")
        except Exception as e:
            print(f"  {placebo_unit}: FAILED - {e}")
            placebo_results.append({
                "state": placebo_unit,
                "error": str(e),
            })

    # Calculate p-value
    valid_placebos = [p for p in placebo_results if "rmspe_ratio" in p]
    n_placebos = len(valid_placebos)

    if n_placebos > 0:
        n_larger = len([
            p for p in valid_placebos
            if float(p["rmspe_ratio"]) >= treated_rmspe_ratio  # type: ignore[arg-type]
        ])
        p_value = (n_larger + 1) / (n_placebos + 1)  # +1 includes treated unit

        print(f"\n{'='*60}")
        print("PLACEBO TEST RESULTS")
        print("=" * 60)
        print(f"Treated unit ({treated_unit}) RMSPE ratio: {treated_rmspe_ratio:.2f}")
        print(f"Valid placebos: {n_placebos}")
        print(f"Placebos with larger RMSPE ratio: {n_larger}")
        print(f"Placebo p-value: {p_value:.3f}")
        print(f"Interpretation: {'Significant' if p_value < 0.10 else 'Not significant'} "
              f"at 10% level")
    else:
        p_value = np.nan
        print("WARNING: No valid placebo tests completed")

    if result is not None:
        result.add_decision(
            decision_id="SC002",
            category="robustness",
            decision=f"Placebo tests with {n_placebos} donor states",
            rationale="In-space placebo validates that treatment effect is not spurious",
            evidence=f"p-value = {p_value:.3f}" if not np.isnan(p_value) else "No valid placebos",
        )

    return {
        "feasible": True,
        "treated_rmspe_ratio": treated_rmspe_ratio,
        "n_placebos": n_placebos,
        "n_larger": n_larger if n_placebos > 0 else 0,
        "p_value": float(p_value) if not np.isnan(p_value) else None,
        "placebo_details": placebo_results,
    }


def run_leave_one_out(
    df: pd.DataFrame,
    treated_unit: str,
    donor_states: list[str],
    pre_years: list[int],
    post_years: list[int],
    outcome_var: str = "arrivals",
    result: ModuleResult | None = None,
) -> dict:
    """
    Run leave-one-out analysis: re-estimate excluding each donor.

    Checks sensitivity to any single donor state.
    """
    print("\n" + "=" * 60)
    print("LEAVE-ONE-OUT ANALYSIS")
    print("=" * 60)

    loo_results = []

    for excluded_donor in donor_states:
        remaining_donors = [d for d in donor_states if d != excluded_donor]

        try:
            loo_result, loo_df = estimate_synthetic_control(
                df, treated_unit, remaining_donors, pre_years, post_years, outcome_var
            )

            loo_results.append({
                "excluded": excluded_donor,
                "pre_rmspe": loo_result.pre_rmspe,
                "rmspe_ratio": loo_result.rmspe_ratio,
                "att_mean": loo_result.att_mean,
                "capacity_multiplier": loo_result.capacity_multiplier,
            })
            print(f"  Excluding {excluded_donor}: ATT = {loo_result.att_mean:.1f}, "
                  f"capacity = {loo_result.capacity_multiplier:.3f}")
        except Exception as e:
            print(f"  Excluding {excluded_donor}: FAILED - {e}")
            loo_results.append({
                "excluded": excluded_donor,
                "error": str(e),
            })

    # Calculate sensitivity metrics
    valid_loo = [result for result in loo_results if "att_mean" in result]

    att_range: float
    att_std: float
    capacity_range_val: float
    capacity_std: float

    if len(valid_loo) > 0:
        att_values: list[float] = [
            float(r["att_mean"]) for r in valid_loo  # type: ignore[arg-type]
        ]
        capacity_values: list[float] = [
            float(r["capacity_multiplier"]) for r in valid_loo  # type: ignore[arg-type]
        ]

        att_range = max(att_values) - min(att_values)
        att_std = float(np.std(att_values))
        capacity_range_val = max(capacity_values) - min(capacity_values)
        capacity_std = float(np.std(capacity_values))

        print(f"\n{'='*60}")
        print("LEAVE-ONE-OUT SENSITIVITY")
        print("=" * 60)
        print(f"ATT range: {min(att_values):.1f} to {max(att_values):.1f} (SD={att_std:.1f})")
        print(f"Capacity multiplier range: {min(capacity_values):.3f} to "
              f"{max(capacity_values):.3f} (SD={capacity_std:.3f})")
    else:
        att_range = att_std = capacity_range_val = capacity_std = float("nan")

    if result is not None:
        result.add_decision(
            decision_id="SC003",
            category="robustness",
            decision="Leave-one-out sensitivity analysis",
            rationale="Validates that results are not driven by any single donor",
            evidence=f"ATT SD = {att_std:.1f}" if not np.isnan(att_std) else "N/A",
        )

    return {
        "n_tests": len(loo_results),
        "n_valid": len(valid_loo),
        "att_range": float(att_range) if not np.isnan(att_range) else None,
        "att_std": float(att_std) if not np.isnan(att_std) else None,
        "capacity_range": float(capacity_range_val) if not np.isnan(capacity_range_val) else None,
        "capacity_std": float(capacity_std) if not np.isnan(capacity_std) else None,
        "loo_details": loo_results,
    }


def run_pre_period_sensitivity(
    df: pd.DataFrame,
    treated_unit: str,
    donor_states: list[str],
    post_years: list[int],
    outcome_var: str = "arrivals",
    result: ModuleResult | None = None,
) -> dict:
    """
    Test sensitivity to different pre-treatment windows.
    """
    print("\n" + "=" * 60)
    print("PRE-PERIOD SENSITIVITY ANALYSIS")
    print("=" * 60)

    # Different pre-period windows
    windows = [
        ("Full (2010-2020)", list(range(2010, 2021))),
        ("Recent (2015-2020)", list(range(2015, 2021))),
        ("Pre-TravelBan (2010-2016)", list(range(2010, 2017))),
    ]

    sensitivity_results = []

    for window_name, pre_years in windows:
        try:
            sens_result, _ = estimate_synthetic_control(
                df, treated_unit, donor_states, pre_years, post_years, outcome_var
            )

            sensitivity_results.append({
                "window": window_name,
                "pre_years": pre_years,
                "pre_rmspe": sens_result.pre_rmspe,
                "rmspe_ratio": sens_result.rmspe_ratio,
                "att_mean": sens_result.att_mean,
                "capacity_multiplier": sens_result.capacity_multiplier,
                "weights": sens_result.weights,
            })
            print(f"  {window_name}: ATT = {sens_result.att_mean:.1f}, "
                  f"capacity = {sens_result.capacity_multiplier:.3f}")
        except Exception as e:
            print(f"  {window_name}: FAILED - {e}")
            sensitivity_results.append({
                "window": window_name,
                "error": str(e),
            })

    if result is not None:
        result.add_decision(
            decision_id="SC004",
            category="robustness",
            decision="Pre-period sensitivity with 3 window specifications",
            rationale="Validates stability across different matching windows",
        )

    return {"windows": sensitivity_results}


# =============================================================================
# VISUALIZATION
# =============================================================================


def plot_synthetic_control(
    sc_df: pd.DataFrame,
    treated_unit: str,
    treatment_year: int,
    result: ModuleResult,
    save_path: Path | None = None,
):
    """Plot synthetic control main results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: Actual vs Synthetic
    ax1 = axes[0]
    ax1.plot(
        sc_df["year"],
        sc_df["actual"],
        "o-",
        color=COLORS["nd_actual"],
        linewidth=2.5,
        markersize=6,
        label=f"{treated_unit} (actual)",
    )
    ax1.plot(
        sc_df["year"],
        sc_df["synthetic"],
        "s--",
        color=COLORS["synthetic"],
        linewidth=2,
        markersize=5,
        label="Synthetic control",
    )

    ax1.axvline(treatment_year - 0.5, color="gray", linestyle="--", alpha=0.7)
    ax1.text(
        treatment_year - 0.3, ax1.get_ylim()[1] * 0.95,
        "LSSND\nClosure",
        fontsize=9,
        color="gray",
        va="top",
    )

    ax1.set_xlabel("Fiscal Year", fontsize=12)
    ax1.set_ylabel("Refugee Arrivals", fontsize=12)
    ax1.set_title("Actual vs Synthetic Control", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10, loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Right panel: Gap (treatment effect)
    ax2 = axes[1]
    pre_mask = sc_df["period"] == "pre"
    post_mask = sc_df["period"] == "post"

    ax2.bar(
        sc_df.loc[pre_mask, "year"],
        sc_df.loc[pre_mask, "gap"],
        color=COLORS["neutral"],
        alpha=0.7,
        label="Pre-treatment",
        width=0.8,
    )
    ax2.bar(
        sc_df.loc[post_mask, "year"],
        sc_df.loc[post_mask, "gap"],
        color=COLORS["nd_actual"],
        alpha=0.7,
        label="Post-treatment (LSSND effect)",
        width=0.8,
    )

    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.axvline(treatment_year - 0.5, color="gray", linestyle="--", alpha=0.7)

    ax2.set_xlabel("Fiscal Year", fontsize=12)
    ax2.set_ylabel("Gap (Actual - Synthetic)", fontsize=12)
    ax2.set_title("Treatment Effect (Gap)", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"Synthetic Control Analysis: LSSND Closure Impact on {treated_unit}",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))

    # Save to multiple locations
    if save_path is None:
        save_path = FIGURES_DIR / "module_7b_synthetic_control"

    for ext in [".png", ".pdf"]:
        fig.savefig(str(save_path) + ext, dpi=300, bbox_inches="tight")

    # Also save to ADR figures directory
    adr_path = ADR_FIGURES_DIR / "rec3_lssnd_synthetic_control"
    for ext in [".png", ".pdf"]:
        fig.savefig(str(adr_path) + ext, dpi=300, bbox_inches="tight")

    plt.close(fig)
    print(f"Figure saved: {save_path}.png/.pdf and {adr_path}.png/.pdf")


def plot_placebo_distribution(
    treated_ratio: float,
    placebo_ratios: list[float],
    result: ModuleResult,
    save_path: Path | None = None,
):
    """Plot distribution of placebo RMSPE ratios."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram of placebo ratios
    ax.hist(
        placebo_ratios,
        bins=10,
        color=COLORS["neutral"],
        alpha=0.7,
        edgecolor="black",
        label="Placebo states",
    )

    # Vertical line for treated unit
    ax.axvline(
        treated_ratio,
        color=COLORS["nd_actual"],
        linewidth=3,
        linestyle="--",
        label=f"North Dakota (ratio={treated_ratio:.2f})",
    )

    # Calculate rank
    n_larger = sum(1 for r in placebo_ratios if r >= treated_ratio)
    rank = n_larger + 1
    total = len(placebo_ratios) + 1

    ax.set_xlabel("RMSPE Ratio (Post / Pre)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(
        f"Placebo Test: North Dakota Ranks {rank}/{total}\n"
        f"(p-value = {rank/total:.3f})",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is None:
        save_path = FIGURES_DIR / "module_7b_placebo_distribution"

    for ext in [".png", ".pdf"]:
        fig.savefig(str(save_path) + ext, dpi=300, bbox_inches="tight")

    adr_path = ADR_FIGURES_DIR / "rec3_placebo_distribution"
    for ext in [".png", ".pdf"]:
        fig.savefig(str(adr_path) + ext, dpi=300, bbox_inches="tight")

    plt.close(fig)
    print(f"Figure saved: {save_path}.png/.pdf")


# =============================================================================
# MAIN ANALYSIS
# =============================================================================


def run_analysis() -> ModuleResult:
    """Main analysis function for LSSND synthetic control."""
    result = ModuleResult(
        module_id="7b",
        analysis_name="LSSND_synthetic_control",
    )

    print("=" * 70)
    print("Module 7b: LSSND Closure Synthetic Control Analysis")
    print("ADR-021 Recommendation #3")
    print(f"Started: {datetime.now(UTC).isoformat()}")
    print("=" * 70)

    # ==========================================================================
    # 1. Load Data
    # ==========================================================================
    print("\n[1/6] Loading data...")

    df_refugee = load_refugee_data()
    df_census = load_census_data()
    df_acs = load_acs_foreign_born()
    df_national = load_national_refugee_totals()

    df = prepare_panel_data(df_refugee, df_census, df_acs, result)

    # Get ND-specific refugee data
    df_nd = df_refugee[df_refugee["state"] == "North Dakota"].copy()
    result.input_files.append("National refugee totals (PostgreSQL)")

    # ==========================================================================
    # 2. Define Analysis Parameters
    # ==========================================================================
    print("\n[2/6] Setting up analysis parameters...")

    treated_unit = "North Dakota"

    # Donor states from Phase A Agent 2 analysis (similar low-flow states)
    donor_states = [
        "South Dakota",
        "Nebraska",
        "Idaho",
        "Maine",
        "Vermont",
        "New Hampshire",
    ]

    # Treatment: LSSND closed January 2021
    # FY2021 starts October 2020, so FY2021 is first full post-treatment year
    treatment_year = 2021

    # Pre-treatment: FY2010-2020 (11 years)
    pre_years = list(range(2010, 2021))

    # Post-treatment: FY2021-2024 (4 years)
    post_years = list(range(2021, 2025))

    result.parameters = {
        "treated_unit": treated_unit,
        "donor_states": donor_states,
        "treatment_event": "LSSND closure",
        "treatment_date": "January 2021",
        "treatment_year": treatment_year,
        "pre_treatment_years": pre_years,
        "post_treatment_years": post_years,
        "outcome_variable": "arrivals",
        "methodology": [
            "National share-based synthetic control (primary)",
            "Abadie et al. (2010) synthetic control (pre-treatment validation)",
        ],
    }

    result.add_decision(
        decision_id="SC000",
        category="study_design",
        decision="Treatment year = FY2021 (LSSND closed January 2021)",
        rationale="LSSND closure occurred mid-FY2021; first full fiscal year "
                  "without LSSND is FY2021 (Oct 2020 - Sep 2021)",
        alternatives=["Use FY2022 as first post-treatment year",
                     "Partial year adjustment for FY2021"],
        evidence="LSSND announced closure January 2021",
    )

    # ==========================================================================
    # 3. NATIONAL SHARE-BASED SYNTHETIC CONTROL (Primary Method)
    # ==========================================================================
    print("\n[3/6] National share-based synthetic control (PRIMARY)...")

    # This is the primary method since we have post-treatment ND data and
    # national totals, but not post-treatment donor state data.
    try:
        ns_result, ns_df = estimate_national_share_synthetic(
            df_nd=df_nd,
            df_national=df_national,
            pre_years=pre_years,
            post_years=post_years,
            result=result,
        )

        result.results["national_share_estimate"] = ns_result

        # Generate visualization for national share method
        plot_synthetic_control(
            ns_df,
            treated_unit,
            treatment_year,
            result,
            save_path=FIGURES_DIR / "module_7b_national_share_synthetic",
        )

    except Exception as e:
        result.warnings.append(f"National share SC estimation failed: {e}")
        traceback.print_exc()
        ns_result = None

    # ==========================================================================
    # 4. TRADITIONAL SYNTHETIC CONTROL (Pre-treatment validation)
    # ==========================================================================
    print("\n[4/6] Traditional synthetic control (pre-treatment validation)...")

    # This method validates that ND can be matched by donors in pre-treatment,
    # but cannot provide post-treatment estimates without donor data.
    try:
        sc_result, sc_df = estimate_synthetic_control(
            df=df,
            treated_unit=treated_unit,
            donor_states=donor_states,
            pre_years=pre_years,
            post_years=post_years,
            outcome_var="arrivals",
            result=result,
        )

        result.results["traditional_sc_estimate"] = {
            "method": "traditional_synthetic_control",
            "note": "Post-treatment values unreliable due to missing donor data FY2021+",
            "weights": sc_result.weights,
            "pre_rmspe": sc_result.pre_rmspe,
            "pre_treatment_fit_quality": "good" if sc_result.pre_rmspe < 200 else "moderate",
        }

    except Exception as e:
        result.warnings.append(f"Traditional SC estimation failed: {e}")
        traceback.print_exc()
        sc_result = None

    # ==========================================================================
    # 5. Robustness Checks (using national share method)
    # ==========================================================================
    print("\n[5/6] Running robustness checks...")

    # 5a. Sensitivity to share calculation window
    robustness: dict[str, Any] = {}

    try:
        # Test different base periods for share calculation
        share_windows = [
            ("Full pre-period (2010-2020)", list(range(2010, 2021))),
            ("Pre-Travel Ban (2010-2016)", list(range(2010, 2017))),
            ("Recent stable (2012-2016)", list(range(2012, 2017))),
        ]

        share_sensitivity: list[dict[str, str | list[int] | float]] = []
        for window_name, window_years in share_windows:
            # Calculate share for this window
            window_mask = df_nd["year"].isin(window_years)
            window_nd = df_nd[window_mask].merge(df_national, on="year")
            if len(window_nd) > 0:
                share = (window_nd["arrivals"] / window_nd["national_arrivals"]).mean()

                # Calculate post-treatment capacity with this share
                post_national = df_national[df_national["year"].isin(post_years)]
                expected = post_national["national_arrivals"] * share
                actual = df_nd[df_nd["year"].isin(post_years)]["arrivals"].values

                if len(expected) > 0 and len(actual) > 0:
                    capacity = (actual.sum() / expected.sum())
                    share_sensitivity.append({
                        "window": window_name,
                        "years": window_years,
                        "share": float(share),
                        "capacity_multiplier": float(capacity),
                    })
                    print(f"  {window_name}: share={share*100:.3f}%, capacity={capacity*100:.1f}%")

        robustness["share_sensitivity"] = share_sensitivity

        if len(share_sensitivity) > 1:
            capacities: list[float] = [
                float(s["capacity_multiplier"])  # type: ignore[arg-type]
                for s in share_sensitivity
            ]
            robustness["capacity_range"] = {
                "min": float(min(capacities)),
                "max": float(max(capacities)),
                "mean": float(np.mean(capacities)),
                "std": float(np.std(capacities)),
            }
            print(f"\n  Capacity range: {min(capacities)*100:.1f}% - {max(capacities)*100:.1f}%")

    except Exception as e:
        result.warnings.append(f"Share sensitivity failed: {e}")
        traceback.print_exc()

    result.results["robustness"] = robustness

    # ==========================================================================
    # 6. Derive Capacity Parameter for Scenario Integration
    # ==========================================================================
    print("\n[6/6] Deriving capacity parameter...")

    # Use national share estimate as primary
    if ns_result is not None:
        capacity_value = ns_result["post_treatment_effect"]["capacity_multiplier"]
        att_mean = ns_result["post_treatment_effect"]["att_mean"]

        # Get uncertainty from robustness checks
        capacity_range_dict: dict[str, float] = robustness.get("capacity_range", {})

        capacity_param = {
            "parameter_name": "lssnd_capacity_multiplier",
            "value": capacity_value,
            "interpretation": f"Post-LSSND closure, ND received {capacity_value*100:.1f}% "
                             f"of what national share predicts",
            "usage": "Multiply baseline projections by this factor to account for "
                    "reduced local resettlement infrastructure",
            "method": "national_share_synthetic",
            "uncertainty": {
                "capacity_range_min": capacity_range_dict.get("min"),
                "capacity_range_max": capacity_range_dict.get("max"),
                "capacity_std": capacity_range_dict.get("std"),
            },
            "att_mean": att_mean,
            "att_interpretation": f"ND received ~{abs(att_mean):.0f} fewer refugees per year "
                                 f"than expected based on national trends",
        }

        result.results["capacity_parameter"] = capacity_param
    else:
        result.warnings.append("Could not derive capacity parameter - analysis failed")

    # ==========================================================================
    # Diagnostics and Summary
    # ==========================================================================
    result.diagnostics = {
        "national_share_method": {
            "pre_rmspe": ns_result["pre_treatment_fit"]["rmspe"] if ns_result else None,
            "post_rmspe": ns_result["post_treatment_effect"]["rmspe"] if ns_result else None,
            "rmspe_ratio": ns_result["post_treatment_effect"]["rmspe_ratio"] if ns_result else None,
            "interpretation": "National share method provides reliable post-treatment estimates",
        },
        "traditional_sc_validation": {
            "pre_rmspe": sc_result.pre_rmspe if sc_result else None,
            "interpretation": "Traditional SC validates pre-treatment matching capability",
        },
        "sensitivity": {
            "capacity_std": dict(robustness.get("capacity_range", {})).get("std"),
            "interpretation": "Lower SD = more stable estimate across share windows",
        },
    }

    result.next_steps = [
        "Integrate capacity_multiplier into ADR-021 Recommendation #6 scenario models",
        "Consider time-varying recovery as Global Refuge expands ND operations",
        "Monitor FY2025+ data for capacity normalization",
        "Acquire FY2021-2024 donor state data for traditional SC validation",
        "Explore heterogeneous effects by nationality/origin country",
    ]

    return result


def main():
    """Main entry point."""
    try:
        result = run_analysis()

        # Save main results
        output_file = result.save("module_7b_lssnd_synthetic_control.json")

        # Also save to ADR-021 results directory
        adr_output = result.save(
            "rec3_lssnd_synthetic_control.json",
            output_dir=ADR_RESULTS_DIR
        )

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("=" * 70)

        print(f"\nMain output: {output_file}")
        print(f"ADR output: {adr_output}")

        # Summary
        if "national_share_estimate" in result.results:
            ns = result.results["national_share_estimate"]
            print("\n" + "-" * 70)
            print("KEY FINDINGS (National Share Method)")
            print("-" * 70)

            post_effect = ns["post_treatment_effect"]
            print("\n1. CAPACITY IMPACT:")
            print(f"   Capacity multiplier: {post_effect['capacity_multiplier']:.3f}")
            print(f"   {ns['interpretation']['capacity_text']}")
            print(f"\n   ATT (mean gap): {post_effect['att_mean']:.1f} arrivals/year")
            print(f"   {ns['interpretation']['att_text']}")

            print("\n2. COUNTERFACTUAL SHARE:")
            share_info = ns["counterfactual_share"]
            print(f"   Pre-Travel Ban share: {share_info['stable_period_mean']*100:.3f}%")
            print(f"   Based on: {share_info['based_on']}")

            print("\n3. FIT QUALITY:")
            print(f"   Pre-RMSPE: {ns['pre_treatment_fit']['rmspe']:.2f}")
            print(f"   RMSPE ratio: {post_effect['rmspe_ratio']:.2f}")

            print("\n4. YEAR-BY-YEAR EFFECTS:")
            for year, gap in ns["att_by_year"].items():
                capacity = ns["capacity_by_year"].get(year, np.nan)
                print(f"   FY{year}: Gap = {gap:.0f} ({capacity*100:.1f}% of expected)")

        if "robustness" in result.results and "share_sensitivity" in result.results["robustness"]:
            sens = result.results["robustness"]["share_sensitivity"]
            print("\n5. ROBUSTNESS (Share Window Sensitivity):")
            for s in sens:
                print(f"   {s['window']}: capacity = {s['capacity_multiplier']*100:.1f}%")

        if "capacity_parameter" in result.results:
            cap = result.results["capacity_parameter"]
            print("\n6. DERIVED PARAMETER FOR SCENARIO INTEGRATION:")
            print(f"   Parameter: {cap['parameter_name']}")
            print(f"   Value: {cap['value']:.3f}")
            print(f"   Usage: {cap['usage']}")

        if result.warnings:
            print("\nWarnings:")
            for w in result.warnings:
                print(f"  - {w}")

        print(f"\nDecisions logged: {len(result.decisions)}")

        return 0

    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
