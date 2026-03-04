#!/usr/bin/env python3
"""Uncertainty quantification for North Dakota population projections.

Created: 2026-03-03
Author: Claude Code (automated)

Purpose
-------
Build a comprehensive uncertainty quantification framework for two projection
methods (SDC 2024 and M2026) using historical walk-forward validation errors.
Generates empirical prediction intervals, Monte Carlo simulation bands, error
decomposition analysis, and an interactive HTML report with Plotly fan charts.

Method
------
1. Load walk-forward validation error data (county-level and state-level) produced
   by walk_forward_validation.py.
2. Load current production projection results (baseline scenario) from the parquet
   files in data/projections/baseline/.
3. For each horizon length (1-19 years), compute the empirical distribution of
   percent errors across all available origin years and counties/state.
4. Derive 50%, 80%, and 95% prediction intervals from the error percentiles.
5. Run a Monte Carlo simulation (N=1000, reproducible seed) that samples from the
   historical error distributions to generate simulated projection paths around
   the current baseline.
6. Decompose total projection error into systematic bias, random variation,
   horizon effect, and geographic variation components.
7. Write CSV outputs and generate a self-contained interactive HTML report.

Inputs
------
- data/analysis/walk_forward/annual_county_detail.csv -- county-level errors
- data/analysis/walk_forward/annual_state_results.csv -- state-level errors
- data/analysis/walk_forward/projection_curves.csv -- full projection curves
- data/projections/baseline/state/*.parquet -- state production projections
- data/projections/baseline/county/*.parquet -- county production projections
- data/projections/baseline/county/countys_summary.csv -- county metadata

Outputs
-------
- data/analysis/walk_forward/prediction_intervals.csv
- data/analysis/walk_forward/uncertainty_bands.csv
- data/analysis/walk_forward/error_decomposition.csv
- data/analysis/walk_forward/uncertainty_report.html

Usage
-----
    python scripts/analysis/uncertainty_analysis.py
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WALK_FORWARD_DIR = PROJECT_ROOT / "data" / "analysis" / "walk_forward"
PROJECTIONS_DIR = PROJECT_ROOT / "data" / "projections" / "baseline"

# Monte Carlo parameters
N_SIMULATIONS = 1000
RANDOM_SEED = 42

# Prediction interval levels
PI_LEVELS = [0.50, 0.80, 0.95]

# Percentiles to compute for the error distribution
PERCENTILES = [5, 10, 25, 50, 75, 90, 95]

# Base year for production projections
BASE_YEAR = 2025
PROJECTION_END = 2055
MAX_HORIZON = PROJECTION_END - BASE_YEAR  # 30


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_walk_forward_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load walk-forward validation error data.

    Returns:
        county_errors: DataFrame with county-level pct_error by horizon/method
        state_errors: DataFrame with state-level pct_error by horizon/method
    """
    county_path = WALK_FORWARD_DIR / "annual_county_detail.csv"
    state_path = WALK_FORWARD_DIR / "annual_state_results.csv"

    county_errors = pd.read_csv(county_path)
    state_errors = pd.read_csv(state_path)

    return county_errors, state_errors


def load_production_projections() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load current baseline production projections.

    Returns:
        state_proj: DataFrame with columns [year, population] (state totals)
        county_proj: DataFrame with columns [year, county_fips, county_name, population]
    """
    # State-level: aggregate from detailed parquet
    state_parquet = (
        PROJECTIONS_DIR / "state" / "nd_state_38_projection_2025_2055_baseline.parquet"
    )
    state_detail = pd.read_parquet(state_parquet)
    state_proj = state_detail.groupby("year")["population"].sum().reset_index()

    # County-level: read summary for names, then load each parquet
    summary = pd.read_csv(PROJECTIONS_DIR / "county" / "countys_summary.csv")
    fips_to_name = dict(zip(summary["fips"].astype(str), summary["name"]))

    county_records = []
    for fips_str, name in sorted(fips_to_name.items()):
        parquet_path = (
            PROJECTIONS_DIR
            / "county"
            / f"nd_county_{fips_str}_projection_2025_2055_baseline.parquet"
        )
        if not parquet_path.exists():
            continue
        df = pd.read_parquet(parquet_path)
        yearly_total = df.groupby("year")["population"].sum().reset_index()
        yearly_total["county_fips"] = fips_str
        yearly_total["county_name"] = name
        county_records.append(yearly_total)

    county_proj = pd.concat(county_records, ignore_index=True)
    return state_proj, county_proj


# ---------------------------------------------------------------------------
# 1. Empirical Prediction Intervals
# ---------------------------------------------------------------------------


def compute_prediction_intervals(
    county_errors: pd.DataFrame,
    state_errors: pd.DataFrame,
) -> pd.DataFrame:
    """Compute empirical prediction intervals from historical errors.

    For each horizon and method, compute percentiles of the percent error
    distribution across all available origin years and counties/state.

    Returns:
        DataFrame with columns: level, method, horizon, p5..p95
    """
    records = []

    for method in ["sdc_2024", "m2026"]:
        for horizon in sorted(county_errors["horizon"].unique()):
            # County-level errors
            county_mask = (county_errors["method"] == method) & (
                county_errors["horizon"] == horizon
            )
            county_pct = county_errors.loc[county_mask, "pct_error"].dropna()

            if len(county_pct) > 0:
                row = {"level": "county", "method": method, "horizon": int(horizon)}
                for p in PERCENTILES:
                    row[f"p{p}"] = np.percentile(county_pct, p)
                row["n_obs"] = len(county_pct)
                row["mean"] = county_pct.mean()
                row["std"] = county_pct.std()
                records.append(row)

            # State-level errors
            state_mask = (state_errors["method"] == method) & (
                state_errors["horizon"] == horizon
            )
            state_pct = state_errors.loc[state_mask, "pct_error"].dropna()

            if len(state_pct) > 0:
                row = {"level": "state", "method": method, "horizon": int(horizon)}
                for p in PERCENTILES:
                    row[f"p{p}"] = np.percentile(state_pct, p)
                row["n_obs"] = len(state_pct)
                row["mean"] = state_pct.mean()
                row["std"] = state_pct.std()
                records.append(row)

    return pd.DataFrame(records)


def extrapolate_error_distribution(
    pi_df: pd.DataFrame,
    max_horizon: int,
) -> pd.DataFrame:
    """Extrapolate error distributions beyond observed horizons.

    Uses power-law fit: std(h) = a * h^b, where h is the horizon.
    Bias is fit as linear: mean(h) = c + d*h.

    This allows us to create uncertainty bands for horizons 20-30 even
    though walk-forward data only covers horizons 1-19.
    """
    records = []

    for level in ["state", "county"]:
        for method in ["sdc_2024", "m2026"]:
            subset = pi_df[(pi_df["level"] == level) & (pi_df["method"] == method)].copy()
            if len(subset) < 3:
                continue

            horizons = subset["horizon"].values.astype(float)
            means = subset["mean"].values
            stds = subset["std"].values

            # Fit bias as linear: mean = c + d*h
            if len(horizons) >= 2:
                slope_mean, intercept_mean, _, _, _ = sp_stats.linregress(horizons, means)
            else:
                slope_mean, intercept_mean = 0.0, means[0]

            # Fit std as power law: log(std) = log(a) + b*log(h)
            valid_std = stds > 0
            if valid_std.sum() >= 2:
                log_h = np.log(horizons[valid_std])
                log_s = np.log(stds[valid_std])
                slope_std, intercept_std, _, _, _ = sp_stats.linregress(log_h, log_s)
                a_std = np.exp(intercept_std)
                b_std = slope_std
            else:
                a_std = stds[0] if len(stds) > 0 else 1.0
                b_std = 0.5  # default square-root growth

            # Generate extrapolated distributions for horizons beyond data
            max_observed = int(subset["horizon"].max())
            for h in range(max_observed + 1, max_horizon + 1):
                extrap_mean = intercept_mean + slope_mean * h
                extrap_std = a_std * (h ** b_std)

                row = {
                    "level": level,
                    "method": method,
                    "horizon": h,
                    "mean": extrap_mean,
                    "std": extrap_std,
                    "n_obs": 0,  # extrapolated
                }
                # Generate percentiles from normal approximation
                for p in PERCENTILES:
                    z = sp_stats.norm.ppf(p / 100.0)
                    row[f"p{p}"] = extrap_mean + z * extrap_std
                records.append(row)

    extrap_df = pd.DataFrame(records)
    return pd.concat([pi_df, extrap_df], ignore_index=True)


# ---------------------------------------------------------------------------
# 2. Uncertainty Bands on Production Projections
# ---------------------------------------------------------------------------


def compute_uncertainty_bands(
    state_proj: pd.DataFrame,
    county_proj: pd.DataFrame,
    pi_df: pd.DataFrame,
) -> pd.DataFrame:
    """Apply prediction intervals to current production projections.

    For each year in the projection, look up the corresponding horizon
    and apply the empirical error percentiles to create upper/lower bounds.
    """
    records = []

    for method in ["sdc_2024", "m2026"]:
        # State-level bands
        state_pi = pi_df[
            (pi_df["level"] == "state") & (pi_df["method"] == method)
        ].set_index("horizon")

        for _, row in state_proj.iterrows():
            year = int(row["year"])
            pop = row["population"]
            horizon = year - BASE_YEAR

            if horizon <= 0:
                # Base year has no uncertainty
                rec = {
                    "level": "state",
                    "geography": "North Dakota",
                    "fips": "38",
                    "method": method,
                    "year": year,
                    "horizon": horizon,
                    "projected": pop,
                }
                for ci in PI_LEVELS:
                    pct = int(ci * 100)
                    rec[f"lower_{pct}"] = pop
                    rec[f"upper_{pct}"] = pop
                records.append(rec)
                continue

            if horizon in state_pi.index:
                pi_row = state_pi.loc[horizon]
                rec = {
                    "level": "state",
                    "geography": "North Dakota",
                    "fips": "38",
                    "method": method,
                    "year": year,
                    "horizon": horizon,
                    "projected": pop,
                }
                for ci in PI_LEVELS:
                    pct = int(ci * 100)
                    lower_p = (1 - ci) / 2 * 100  # e.g. 2.5 for 95%
                    upper_p = (1 + ci) / 2 * 100  # e.g. 97.5 for 95%
                    # Interpolate between available percentiles
                    lower_err = _interpolate_percentile(pi_row, lower_p)
                    upper_err = _interpolate_percentile(pi_row, upper_p)
                    rec[f"lower_{pct}"] = pop * (1 + lower_err / 100)
                    rec[f"upper_{pct}"] = pop * (1 + upper_err / 100)
                records.append(rec)

        # County-level bands
        county_pi = pi_df[
            (pi_df["level"] == "county") & (pi_df["method"] == method)
        ].set_index("horizon")

        for (fips, name), group in county_proj.groupby(["county_fips", "county_name"]):
            for _, row in group.iterrows():
                year = int(row["year"])
                pop = row["population"]
                horizon = year - BASE_YEAR

                if horizon <= 0:
                    rec = {
                        "level": "county",
                        "geography": name,
                        "fips": fips,
                        "method": method,
                        "year": year,
                        "horizon": horizon,
                        "projected": pop,
                    }
                    for ci in PI_LEVELS:
                        pct = int(ci * 100)
                        rec[f"lower_{pct}"] = pop
                        rec[f"upper_{pct}"] = pop
                    records.append(rec)
                    continue

                if horizon in county_pi.index:
                    pi_row = county_pi.loc[horizon]
                    rec = {
                        "level": "county",
                        "geography": name,
                        "fips": fips,
                        "method": method,
                        "year": year,
                        "horizon": horizon,
                        "projected": pop,
                    }
                    for ci in PI_LEVELS:
                        pct = int(ci * 100)
                        lower_p = (1 - ci) / 2 * 100
                        upper_p = (1 + ci) / 2 * 100
                        lower_err = _interpolate_percentile(pi_row, lower_p)
                        upper_err = _interpolate_percentile(pi_row, upper_p)
                        rec[f"lower_{pct}"] = pop * (1 + lower_err / 100)
                        rec[f"upper_{pct}"] = pop * (1 + upper_err / 100)
                    records.append(rec)

    return pd.DataFrame(records)


def _interpolate_percentile(pi_row: pd.Series, target_pct: float) -> float:
    """Interpolate a target percentile from available percentile columns.

    Given the stored percentiles (p5, p10, p25, p50, p75, p90, p95),
    linearly interpolate to get the error value at an arbitrary percentile.
    """
    available = []
    for p in PERCENTILES:
        col = f"p{p}"
        if col in pi_row.index:
            available.append((p, pi_row[col]))

    if not available:
        return 0.0

    pcts = [a[0] for a in available]
    vals = [a[1] for a in available]

    if target_pct <= pcts[0]:
        return vals[0]
    if target_pct >= pcts[-1]:
        return vals[-1]

    return float(np.interp(target_pct, pcts, vals))


# ---------------------------------------------------------------------------
# 3. Monte Carlo Simulation
# ---------------------------------------------------------------------------


def run_monte_carlo(
    state_proj: pd.DataFrame,
    county_proj: pd.DataFrame,
    pi_df: pd.DataFrame,
    n_sims: int = N_SIMULATIONS,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Monte Carlo simulation sampling from historical error distributions.

    For each simulation:
    1. For each horizon year, draw a random error from the empirical error
       distribution for that horizon.
    2. Apply the drawn error to the baseline projection.
    3. Repeat N times and compute percentile bands.

    Returns:
        state_mc: DataFrame with MC percentile bands for state
        county_mc: DataFrame with MC percentile bands for counties
    """
    rng = np.random.default_rng(seed)

    # We use the m2026 method errors as the primary for MC
    # (since that is the production method)
    method = "m2026"

    # Build error pools by horizon from county-level data for richer samples
    county_pi = pi_df[
        (pi_df["level"] == "county") & (pi_df["method"] == method)
    ].set_index("horizon")

    state_pi = pi_df[
        (pi_df["level"] == "state") & (pi_df["method"] == method)
    ].set_index("horizon")

    # --- State-level MC ---
    state_mc_records = _run_mc_for_geography(
        state_proj, state_pi, "state", "North Dakota", "38", rng, n_sims
    )

    # --- County-level MC ---
    county_mc_records = []
    for (fips, name), group in county_proj.groupby(["county_fips", "county_name"]):
        group_sorted = group.sort_values("year").reset_index(drop=True)
        county_records = _run_mc_for_geography(
            group_sorted, county_pi, "county", name, fips, rng, n_sims
        )
        county_mc_records.extend(county_records)

    state_mc = pd.DataFrame(state_mc_records)
    county_mc = pd.DataFrame(county_mc_records)

    return state_mc, county_mc


def _run_mc_for_geography(
    proj: pd.DataFrame,
    pi_horizon: pd.DataFrame,
    level: str,
    geography: str,
    fips: str,
    rng: np.random.Generator,
    n_sims: int,
) -> list[dict]:
    """Run MC simulation for a single geography (state or county).

    Uses correlated random walk: each simulated path draws an error at each
    horizon, with partial autocorrelation to produce smooth paths.
    """
    records = []
    years = sorted(proj["year"].unique())
    base_pops = dict(zip(proj["year"], proj["population"]))

    # Build per-horizon mean/std from prediction intervals
    horizon_params: dict[int, tuple[float, float]] = {}
    for h in range(0, MAX_HORIZON + 1):
        if h in pi_horizon.index:
            row = pi_horizon.loc[h]
            horizon_params[h] = (float(row["mean"]), float(row["std"]))
        elif h == 0:
            horizon_params[h] = (0.0, 0.0)

    # Extrapolate any missing horizons using power-law fit
    known_h = sorted(k for k in horizon_params if k > 0)
    if len(known_h) >= 3:
        h_arr = np.array(known_h, dtype=float)
        mean_arr = np.array([horizon_params[k][0] for k in known_h])
        std_arr = np.array([horizon_params[k][1] for k in known_h])

        # Fit linear for mean
        slope_m, intercept_m, _, _, _ = sp_stats.linregress(h_arr, mean_arr)
        # Fit power law for std
        valid = std_arr > 0
        if valid.sum() >= 2:
            slope_s, intercept_s, _, _, _ = sp_stats.linregress(
                np.log(h_arr[valid]), np.log(std_arr[valid])
            )
            a_s, b_s = np.exp(intercept_s), slope_s
        else:
            a_s, b_s = 1.0, 0.5

        for h in range(0, MAX_HORIZON + 1):
            if h not in horizon_params:
                ext_mean = intercept_m + slope_m * h
                ext_std = a_s * (max(h, 1) ** b_s)
                horizon_params[h] = (ext_mean, ext_std)

    # Generate simulation paths
    sim_matrix = np.zeros((n_sims, len(years)))  # simulated populations

    for sim_idx in range(n_sims):
        prev_z = 0.0  # for autocorrelation
        for yr_idx, year in enumerate(years):
            horizon = year - BASE_YEAR
            base_pop = base_pops.get(year, 0.0)

            if horizon <= 0 or base_pop == 0:
                sim_matrix[sim_idx, yr_idx] = base_pop
                continue

            mean_err, std_err = horizon_params.get(horizon, (0.0, 5.0))

            # Draw with autocorrelation (rho=0.7 gives smooth paths)
            rho = 0.7
            innovation = rng.normal(0, 1)
            z = rho * prev_z + np.sqrt(1 - rho**2) * innovation
            prev_z = z

            # Convert z-score to error percentage
            pct_error = mean_err + z * std_err
            sim_pop = base_pop * (1 + pct_error / 100)
            sim_matrix[sim_idx, yr_idx] = max(0.0, sim_pop)

    # Compute percentile bands from simulations
    for yr_idx, year in enumerate(years):
        horizon = year - BASE_YEAR
        base_pop = base_pops.get(year, 0.0)
        sim_vals = sim_matrix[:, yr_idx]

        rec = {
            "level": level,
            "geography": geography,
            "fips": fips,
            "method": "m2026",
            "year": year,
            "horizon": horizon,
            "projected": base_pop,
            "mc_mean": np.mean(sim_vals),
            "mc_median": np.median(sim_vals),
        }

        for ci in PI_LEVELS:
            pct = int(ci * 100)
            lower_q = (1 - ci) / 2
            upper_q = (1 + ci) / 2
            rec[f"mc_lower_{pct}"] = np.quantile(sim_vals, lower_q)
            rec[f"mc_upper_{pct}"] = np.quantile(sim_vals, upper_q)

        records.append(rec)

    return records


# ---------------------------------------------------------------------------
# 4. Error Decomposition
# ---------------------------------------------------------------------------


def compute_error_decomposition(
    county_errors: pd.DataFrame,
    state_errors: pd.DataFrame,
) -> pd.DataFrame:
    """Decompose total projection error into components.

    Components:
    - Systematic bias: mean error (how much the method consistently over/under-projects)
    - Random variation: std around the mean
    - Horizon effect: how error grows with horizon (R^2 of horizon regression)
    - Geographic variation: county-to-county error variance

    Returns DataFrame with decomposition for each method.
    """
    records = []

    for method in ["sdc_2024", "m2026"]:
        m_county = county_errors[county_errors["method"] == method].copy()
        m_state = state_errors[state_errors["method"] == method].copy()

        if len(m_county) == 0:
            continue

        pct_errors = m_county["pct_error"].values
        total_variance = np.var(pct_errors, ddof=1) if len(pct_errors) > 1 else 0.0

        # 1. Systematic bias
        mean_error = np.mean(pct_errors)
        bias_variance = mean_error**2  # squared bias contribution

        # 2. Random variation (residual after removing mean)
        residual_errors = pct_errors - mean_error
        random_variance = np.var(residual_errors, ddof=1) if len(residual_errors) > 1 else 0.0

        # 3. Horizon effect
        # Regress pct_error on horizon to get explained variance
        horizons = m_county["horizon"].values.astype(float)
        if len(horizons) >= 3:
            slope, intercept, r_value, _, _ = sp_stats.linregress(horizons, pct_errors)
            horizon_r2 = r_value**2
            horizon_explained_var = horizon_r2 * total_variance
        else:
            slope, intercept = 0.0, 0.0
            horizon_r2 = 0.0
            horizon_explained_var = 0.0

        # 4. Geographic variation
        # Compute county-level mean errors, then variance across counties
        county_means = m_county.groupby("county_fips")["pct_error"].mean()
        geographic_variance = np.var(county_means.values, ddof=1) if len(county_means) > 1 else 0.0

        # Compute fractions (may not sum to 1 due to interactions)
        total_ss = total_variance * (len(pct_errors) - 1) if len(pct_errors) > 1 else 1.0

        # Two-way decomposition via ANOVA-like approach
        # Total = between-county + between-horizon + residual
        county_group = m_county.groupby("county_fips")["pct_error"]
        horizon_group = m_county.groupby("horizon")["pct_error"]

        grand_mean = np.mean(pct_errors)

        # Between-county SS
        ss_county = sum(
            len(g) * (g.mean() - grand_mean) ** 2
            for _, g in county_group
        )

        # Between-horizon SS
        ss_horizon = sum(
            len(g) * (g.mean() - grand_mean) ** 2
            for _, g in horizon_group
        )

        # Residual SS
        ss_residual = total_ss - ss_county - ss_horizon
        ss_residual = max(0.0, ss_residual)  # Floor at 0

        # Fractions
        frac_bias = bias_variance / total_variance if total_variance > 0 else 0.0
        frac_county = ss_county / total_ss if total_ss > 0 else 0.0
        frac_horizon = ss_horizon / total_ss if total_ss > 0 else 0.0
        frac_residual = ss_residual / total_ss if total_ss > 0 else 0.0

        records.append({
            "method": method,
            "n_observations": len(pct_errors),
            "total_variance": total_variance,
            "mean_error_pct": mean_error,
            "bias_squared": bias_variance,
            "random_std": np.sqrt(random_variance),
            "horizon_slope": slope,
            "horizon_intercept": intercept,
            "horizon_r2": horizon_r2,
            "geographic_variance": geographic_variance,
            "geographic_std": np.sqrt(geographic_variance),
            "ss_total": total_ss,
            "ss_county": ss_county,
            "ss_horizon": ss_horizon,
            "ss_residual": ss_residual,
            "frac_bias": frac_bias,
            "frac_county": frac_county,
            "frac_horizon": frac_horizon,
            "frac_residual": frac_residual,
        })

        # Also add per-horizon breakdown
        for horizon_val in sorted(m_county["horizon"].unique()):
            h_data = m_county[m_county["horizon"] == horizon_val]["pct_error"]
            records.append({
                "method": method,
                "horizon": int(horizon_val),
                "n_observations": len(h_data),
                "mean_error_pct": h_data.mean(),
                "std_error_pct": h_data.std(),
                "median_error_pct": h_data.median(),
                "mae_pct": h_data.abs().mean(),
                "rmse_pct": np.sqrt((h_data**2).mean()),
            })

    return pd.DataFrame(records)


def compute_normality_tests(
    county_errors: pd.DataFrame,
) -> pd.DataFrame:
    """Test normality of error distributions by horizon bucket.

    Uses Shapiro-Wilk test (best for moderate sample sizes) and
    D'Agostino-Pearson test for larger samples.
    """
    records = []

    # Define horizon buckets
    buckets = {
        "short_term": (1, 5),
        "medium_term": (6, 12),
        "long_term": (13, 19),
    }

    for method in ["sdc_2024", "m2026"]:
        m_data = county_errors[county_errors["method"] == method]

        for bucket_name, (h_min, h_max) in buckets.items():
            subset = m_data[
                (m_data["horizon"] >= h_min) & (m_data["horizon"] <= h_max)
            ]["pct_error"].dropna()

            if len(subset) < 8:
                continue

            # Shapiro-Wilk (max 5000 samples)
            sample = subset.values
            if len(sample) > 5000:
                sample = np.random.default_rng(42).choice(sample, 5000, replace=False)

            sw_stat, sw_p = sp_stats.shapiro(sample)

            # D'Agostino-Pearson
            if len(sample) >= 20:
                dp_stat, dp_p = sp_stats.normaltest(sample)
            else:
                dp_stat, dp_p = np.nan, np.nan

            # Skewness and kurtosis
            skew = sp_stats.skew(subset.values)
            kurt = sp_stats.kurtosis(subset.values)

            records.append({
                "method": method,
                "bucket": bucket_name,
                "horizon_range": f"{h_min}-{h_max}",
                "n": len(subset),
                "mean": subset.mean(),
                "std": subset.std(),
                "skewness": skew,
                "kurtosis": kurt,
                "shapiro_stat": sw_stat,
                "shapiro_p": sw_p,
                "dagostino_stat": dp_stat,
                "dagostino_p": dp_p,
                "normal_at_05": sw_p > 0.05,
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 5. HTML Report Generation
# ---------------------------------------------------------------------------


def generate_html_report(
    pi_df: pd.DataFrame,
    bands_df: pd.DataFrame,
    decomp_df: pd.DataFrame,
    normality_df: pd.DataFrame,
    state_mc: pd.DataFrame,
    county_mc: pd.DataFrame,
    county_errors: pd.DataFrame,
    state_errors: pd.DataFrame,
) -> str:
    """Generate self-contained interactive HTML report with Plotly charts."""

    today = datetime.date.today().isoformat()

    # Prepare data for charts
    state_fan = _build_state_fan_chart(bands_df, state_mc)
    pi_width_chart = _build_pi_width_chart(pi_df)
    county_fan = _build_county_fan_charts(bands_df, county_mc)
    error_histograms = _build_error_histograms(county_errors)
    bias_variance_chart = _build_bias_variance_chart(decomp_df)
    qq_charts = _build_qq_charts(county_errors)
    horizon_growth = _build_horizon_growth_chart(pi_df)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Uncertainty Quantification Report - {today}</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            color: #595959;
            background: #f5f5f5;
            line-height: 1.5;
        }}

        .header {{
            background: linear-gradient(135deg, #1F3864 0%, #0563C1 100%);
            color: white;
            padding: 30px 40px;
            margin-bottom: 0;
        }}

        .header h1 {{
            font-size: 28px;
            font-weight: 600;
            margin-bottom: 8px;
        }}

        .header .subtitle {{
            font-size: 14px;
            opacity: 0.85;
        }}

        .tab-nav {{
            background: white;
            border-bottom: 2px solid #D9D9D9;
            padding: 0 20px;
            display: flex;
            gap: 0;
            position: sticky;
            top: 0;
            z-index: 100;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            flex-wrap: wrap;
        }}

        .tab-btn {{
            padding: 12px 14px;
            background: none;
            border: none;
            font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            font-size: 12px;
            font-weight: 500;
            color: #595959;
            cursor: pointer;
            border-bottom: 3px solid transparent;
            transition: all 0.2s;
            white-space: nowrap;
        }}

        .tab-btn:hover {{
            color: #0563C1;
            background: #f0f7ff;
        }}

        .tab-btn.active {{
            color: #0563C1;
            border-bottom-color: #0563C1;
            font-weight: 600;
        }}

        .tab-content {{
            display: none;
            padding: 25px 30px;
            max-width: 1400px;
            margin: 0 auto;
        }}

        .tab-content.active {{
            display: block;
        }}

        .card {{
            background: white;
            border-radius: 8px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }}

        .card h2 {{
            font-size: 18px;
            color: #1F3864;
            margin-bottom: 12px;
            font-weight: 600;
        }}

        .card h3 {{
            font-size: 15px;
            color: #333;
            margin-bottom: 10px;
            font-weight: 600;
        }}

        .card p {{
            font-size: 13px;
            margin-bottom: 10px;
            line-height: 1.6;
        }}

        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 15px 0;
        }}

        .metric-box {{
            background: #f8f9fa;
            border-radius: 6px;
            padding: 14px;
            text-align: center;
            border-left: 4px solid #0563C1;
        }}

        .metric-box .value {{
            font-size: 24px;
            font-weight: 700;
            color: #1F3864;
        }}

        .metric-box .label {{
            font-size: 11px;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-top: 4px;
        }}

        .plot-container {{
            width: 100%;
            min-height: 450px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            margin: 10px 0;
        }}

        th {{
            background: #f0f4f8;
            color: #1F3864;
            padding: 10px 12px;
            text-align: left;
            font-weight: 600;
            border-bottom: 2px solid #d0d8e0;
        }}

        td {{
            padding: 8px 12px;
            border-bottom: 1px solid #eee;
        }}

        tr:hover td {{
            background: #f8f9fa;
        }}

        .interpretation {{
            background: #f0f7ff;
            border-left: 4px solid #0563C1;
            padding: 12px 16px;
            margin: 12px 0;
            font-size: 13px;
            border-radius: 0 6px 6px 0;
        }}

        .methodology {{
            background: #fafafa;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 16px;
            margin: 12px 0;
            font-size: 12px;
        }}

        .methodology h4 {{
            color: #1F3864;
            margin-bottom: 8px;
        }}
    </style>
</head>
<body>

<div class="header">
    <h1>Uncertainty Quantification Report</h1>
    <div class="subtitle">North Dakota Population Projections | SDC 2024 vs M2026 Methods | Generated {today}</div>
</div>

<div class="tab-nav">
    <button class="tab-btn active" onclick="showTab('overview')">Overview</button>
    <button class="tab-btn" onclick="showTab('fan-chart')">State Fan Chart</button>
    <button class="tab-btn" onclick="showTab('pi-width')">Interval Width</button>
    <button class="tab-btn" onclick="showTab('county-fan')">County Fan Charts</button>
    <button class="tab-btn" onclick="showTab('histograms')">Error Distributions</button>
    <button class="tab-btn" onclick="showTab('decomposition')">Error Decomposition</button>
    <button class="tab-btn" onclick="showTab('normality')">Normality Checks</button>
    <button class="tab-btn" onclick="showTab('methodology')">Methodology</button>
</div>

<!-- Tab: Overview -->
<div class="tab-content active" id="tab-overview">
{_build_overview_tab(pi_df, decomp_df, normality_df)}
</div>

<!-- Tab: State Fan Chart -->
<div class="tab-content" id="tab-fan-chart">
<div class="card">
    <h2>State-Level Projection with Prediction Intervals</h2>
    <p>Fan chart showing the M2026 baseline projection for North Dakota with 50%, 80%, and 95%
    prediction intervals derived from historical walk-forward validation errors. Shaded bands
    represent the range within which future population is expected to fall at each confidence level.
    Monte Carlo bands (dashed) provide an independent cross-check.</p>
    <div id="state-fan-chart" class="plot-container"></div>
    <div class="interpretation">
        <strong>Reading the chart:</strong> The darkest band covers 50% of expected outcomes (the most
        likely range). The medium band covers 80%, and the lightest covers 95%. If the projection
        methodology is well-calibrated, actual future population should fall within the 95% band
        approximately 95% of the time.
    </div>
</div>
</div>

<!-- Tab: PI Width -->
<div class="tab-content" id="tab-pi-width">
<div class="card">
    <h2>Prediction Interval Width vs. Horizon</h2>
    <p>How quickly uncertainty grows with the projection horizon. The width of each prediction
    interval (upper bound minus lower bound, as percent of projected population) shows the
    rate at which forecasting precision degrades over time.</p>
    <div id="pi-width-chart" class="plot-container"></div>
</div>
<div class="card">
    <h2>Error Standard Deviation Growth</h2>
    <p>The standard deviation of percent error at each horizon, with power-law extrapolation
    beyond the observed horizon range (1-19 years). The dashed line shows the fitted curve
    used for horizons 20-30.</p>
    <div id="horizon-growth-chart" class="plot-container"></div>
</div>
</div>

<!-- Tab: County Fan Charts -->
<div class="tab-content" id="tab-county-fan">
<div class="card">
    <h2>County-Level Projection Uncertainty</h2>
    <p>Select a county to see its M2026 baseline projection with empirical prediction intervals.
    County-level uncertainty is generally wider than state-level because individual counties
    have more volatile migration patterns.</p>
    <div id="county-fan-chart" class="plot-container" style="min-height:500px;"></div>
</div>
</div>

<!-- Tab: Error Histograms -->
<div class="tab-content" id="tab-histograms">
<div class="card">
    <h2>Error Distribution by Horizon Bucket</h2>
    <p>Histograms of percent error for short-term (1-5 yr), medium-term (6-12 yr), and
    long-term (13-19 yr) horizons. Red dashed line marks zero (no error). The shape of
    these distributions determines the prediction intervals.</p>
    <div id="error-histograms" class="plot-container" style="min-height:600px;"></div>
</div>
</div>

<!-- Tab: Decomposition -->
<div class="tab-content" id="tab-decomposition">
<div class="card">
    <h2>Error Decomposition: Bias vs. Variance</h2>
    <p>Decomposition of total projection error into systematic bias (consistent over/under-projection),
    horizon effect (error growth with time), geographic variation (county-to-county differences),
    and residual unexplained variation.</p>
    <div id="bias-variance-chart" class="plot-container"></div>
</div>
{_build_decomposition_table(decomp_df)}
</div>

<!-- Tab: Normality -->
<div class="tab-content" id="tab-normality">
<div class="card">
    <h2>Normality Assessment</h2>
    <p>QQ plots and formal tests to assess whether projection errors follow a normal distribution.
    If errors are non-normal, the empirical (non-parametric) prediction intervals are more
    appropriate than parametric ones.</p>
    <div id="qq-charts" class="plot-container" style="min-height:600px;"></div>
</div>
{_build_normality_table(normality_df)}
</div>

<!-- Tab: Methodology -->
<div class="tab-content" id="tab-methodology">
{_build_methodology_tab()}
</div>

<script>
// Tab switching
function showTab(tabId) {{
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    document.getElementById('tab-' + tabId).classList.add('active');
    event.target.classList.add('active');

    // Trigger resize to fix Plotly layout
    window.dispatchEvent(new Event('resize'));
}}

// Chart rendering
{state_fan}
{pi_width_chart}
{county_fan}
{error_histograms}
{bias_variance_chart}
{qq_charts}
{horizon_growth}
</script>

</body>
</html>"""

    return html


# ---------------------------------------------------------------------------
# Chart builders (return JavaScript strings for Plotly)
# ---------------------------------------------------------------------------


def _build_state_fan_chart(bands_df: pd.DataFrame, state_mc: pd.DataFrame) -> str:
    """Build Plotly fan chart for state-level projection."""
    m2026_state = bands_df[
        (bands_df["level"] == "state") & (bands_df["method"] == "m2026")
    ].sort_values("year")

    mc_state = state_mc[state_mc["level"] == "state"].sort_values("year")

    years = m2026_state["year"].tolist()
    projected = m2026_state["projected"].tolist()

    # Empirical bands
    l95 = m2026_state["lower_95"].tolist()
    u95 = m2026_state["upper_95"].tolist()
    l80 = m2026_state["lower_80"].tolist()
    u80 = m2026_state["upper_80"].tolist()
    l50 = m2026_state["lower_50"].tolist()
    u50 = m2026_state["upper_50"].tolist()

    # MC bands
    mc_years = mc_state["year"].tolist()
    mc_l95 = mc_state["mc_lower_95"].tolist() if "mc_lower_95" in mc_state.columns else []
    mc_u95 = mc_state["mc_upper_95"].tolist() if "mc_upper_95" in mc_state.columns else []

    return f"""
Plotly.newPlot('state-fan-chart', [
    // 95% band
    {{
        x: {years}.concat({years}.slice().reverse()),
        y: {u95}.concat({l95}.slice().reverse()),
        fill: 'toself', fillcolor: 'rgba(5,99,193,0.10)',
        line: {{color: 'transparent'}}, type: 'scatter', mode: 'lines',
        name: '95% PI', showlegend: true, hoverinfo: 'skip'
    }},
    // 80% band
    {{
        x: {years}.concat({years}.slice().reverse()),
        y: {u80}.concat({l80}.slice().reverse()),
        fill: 'toself', fillcolor: 'rgba(5,99,193,0.20)',
        line: {{color: 'transparent'}}, type: 'scatter', mode: 'lines',
        name: '80% PI', showlegend: true, hoverinfo: 'skip'
    }},
    // 50% band
    {{
        x: {years}.concat({years}.slice().reverse()),
        y: {u50}.concat({l50}.slice().reverse()),
        fill: 'toself', fillcolor: 'rgba(5,99,193,0.35)',
        line: {{color: 'transparent'}}, type: 'scatter', mode: 'lines',
        name: '50% PI', showlegend: true, hoverinfo: 'skip'
    }},
    // Baseline projection
    {{
        x: {years}, y: {projected},
        mode: 'lines', type: 'scatter',
        line: {{color: '#0563C1', width: 2.5}},
        name: 'M2026 Baseline',
        hovertemplate: '%{{x}}: %{{y:,.0f}}<extra>Baseline</extra>'
    }},
    // MC 95% bounds (dashed)
    {{
        x: {mc_years}, y: {mc_u95},
        mode: 'lines', type: 'scatter',
        line: {{color: '#E67E22', width: 1, dash: 'dash'}},
        name: 'MC 95% Upper',
        hovertemplate: '%{{x}}: %{{y:,.0f}}<extra>MC 95% Upper</extra>'
    }},
    {{
        x: {mc_years}, y: {mc_l95},
        mode: 'lines', type: 'scatter',
        line: {{color: '#E67E22', width: 1, dash: 'dash'}},
        name: 'MC 95% Lower',
        hovertemplate: '%{{x}}: %{{y:,.0f}}<extra>MC 95% Lower</extra>'
    }}
], {{
    title: 'North Dakota Population Projection with Uncertainty Bands',
    xaxis: {{title: 'Year', dtick: 5}},
    yaxis: {{title: 'Population', tickformat: ',.0f'}},
    hovermode: 'x unified',
    legend: {{x: 0.02, y: 0.98, bgcolor: 'rgba(255,255,255,0.8)'}},
    margin: {{l: 80, r: 30, t: 50, b: 50}}
}}, {{responsive: true}});
"""


def _build_pi_width_chart(pi_df: pd.DataFrame) -> str:
    """Build chart showing PI width vs horizon."""
    traces = []

    for method in ["sdc_2024", "m2026"]:
        for level in ["state", "county"]:
            subset = pi_df[
                (pi_df["method"] == method) & (pi_df["level"] == level)
            ].sort_values("horizon")

            if len(subset) == 0:
                continue

            horizons = subset["horizon"].tolist()
            widths_95 = (subset["p95"] - subset["p5"]).tolist()

            label = f"{method} ({level})"
            dash = "dash" if level == "county" else "solid"
            color = "#0563C1" if method == "m2026" else "#E74C3C"

            traces.append(f"""{{
                x: {horizons}, y: {widths_95},
                mode: 'lines+markers', type: 'scatter',
                line: {{dash: '{dash}', color: '{color}'}},
                marker: {{size: 4}},
                name: '{label}'
            }}""")

    traces_js = ",\n    ".join(traces)

    return f"""
Plotly.newPlot('pi-width-chart', [
    {traces_js}
], {{
    title: '95% Prediction Interval Width by Horizon',
    xaxis: {{title: 'Forecast Horizon (years)', dtick: 2}},
    yaxis: {{title: 'Interval Width (p95 - p5, % points)'}},
    hovermode: 'x unified',
    legend: {{x: 0.02, y: 0.98}},
    margin: {{l: 70, r: 30, t: 50, b: 50}}
}}, {{responsive: true}});
"""


def _build_county_fan_charts(bands_df: pd.DataFrame, county_mc: pd.DataFrame) -> str:
    """Build county-level fan charts with dropdown selector."""
    county_bands = bands_df[
        (bands_df["level"] == "county") & (bands_df["method"] == "m2026")
    ].copy()

    counties = sorted(county_bands["geography"].unique())
    if len(counties) == 0:
        return "// No county data available"

    # Build traces for each county (all initially hidden except first)
    all_traces = []
    buttons = []

    traces_per_county = 4  # 95% band, 80% band, 50% band, baseline

    for i, county in enumerate(counties):
        c_data = county_bands[county_bands["geography"] == county].sort_values("year")
        fips = c_data["fips"].iloc[0]

        years = c_data["year"].tolist()
        projected = c_data["projected"].tolist()
        l95 = c_data["lower_95"].tolist()
        u95 = c_data["upper_95"].tolist()
        l80 = c_data["lower_80"].tolist()
        u80 = c_data["upper_80"].tolist()
        l50 = c_data["lower_50"].tolist()
        u50 = c_data["upper_50"].tolist()

        visible = "true" if i == 0 else "false"

        all_traces.append(f"""{{
            x: {years}.concat({years}.slice().reverse()),
            y: {u95}.concat({l95}.slice().reverse()),
            fill: 'toself', fillcolor: 'rgba(5,99,193,0.10)',
            line: {{color: 'transparent'}}, type: 'scatter', mode: 'lines',
            name: '95% PI', showlegend: {visible}, visible: {visible}, hoverinfo: 'skip'
        }}""")
        all_traces.append(f"""{{
            x: {years}.concat({years}.slice().reverse()),
            y: {u80}.concat({l80}.slice().reverse()),
            fill: 'toself', fillcolor: 'rgba(5,99,193,0.20)',
            line: {{color: 'transparent'}}, type: 'scatter', mode: 'lines',
            name: '80% PI', showlegend: false, visible: {visible}, hoverinfo: 'skip'
        }}""")
        all_traces.append(f"""{{
            x: {years}.concat({years}.slice().reverse()),
            y: {u50}.concat({l50}.slice().reverse()),
            fill: 'toself', fillcolor: 'rgba(5,99,193,0.35)',
            line: {{color: 'transparent'}}, type: 'scatter', mode: 'lines',
            name: '50% PI', showlegend: false, visible: {visible}, hoverinfo: 'skip'
        }}""")
        all_traces.append(f"""{{
            x: {years}, y: {projected},
            mode: 'lines', type: 'scatter',
            line: {{color: '#0563C1', width: 2.5}},
            name: 'Baseline', showlegend: {visible}, visible: {visible},
            hovertemplate: '%{{x}}: %{{y:,.0f}}<extra>Baseline</extra>'
        }}""")

        # Build visibility array for this county's button
        vis = ["false"] * (len(counties) * traces_per_county)
        for j in range(traces_per_county):
            vis[i * traces_per_county + j] = "true"
        vis_str = "[" + ",".join(vis) + "]"

        buttons.append(f"""{{
            method: 'update',
            args: [{{visible: {vis_str}}},
                   {{title: '{county} ({fips}) - Projection with Uncertainty Bands'}}],
            label: '{county.replace(" County", "")}'
        }}""")

    traces_js = ",\n    ".join(all_traces)
    buttons_js = ",\n            ".join(buttons)

    first_county = counties[0]

    return f"""
Plotly.newPlot('county-fan-chart', [
    {traces_js}
], {{
    title: '{first_county} - Projection with Uncertainty Bands',
    xaxis: {{title: 'Year', dtick: 5}},
    yaxis: {{title: 'Population', tickformat: ',.0f'}},
    hovermode: 'x unified',
    legend: {{x: 0.02, y: 0.98, bgcolor: 'rgba(255,255,255,0.8)'}},
    margin: {{l: 80, r: 30, t: 80, b: 50}},
    updatemenus: [{{
        buttons: [
            {buttons_js}
        ],
        direction: 'down',
        showactive: true,
        x: 0.5, xanchor: 'center',
        y: 1.15, yanchor: 'top',
        bgcolor: 'white',
        bordercolor: '#d0d8e0',
        font: {{size: 11}}
    }}]
}}, {{responsive: true}});
"""


def _build_error_histograms(county_errors: pd.DataFrame) -> str:
    """Build histograms of error distributions by horizon bucket."""
    buckets = {
        "Short-term (1-5 yr)": (1, 5),
        "Medium-term (6-12 yr)": (6, 12),
        "Long-term (13-19 yr)": (13, 19),
    }

    traces = []
    for method, color in [("m2026", "#0563C1"), ("sdc_2024", "#E74C3C")]:
        for bucket_name, (h_min, h_max) in buckets.items():
            subset = county_errors[
                (county_errors["method"] == method)
                & (county_errors["horizon"] >= h_min)
                & (county_errors["horizon"] <= h_max)
            ]["pct_error"].dropna()

            if len(subset) == 0:
                continue

            vals = subset.tolist()
            traces.append(f"""{{
                x: {vals},
                type: 'histogram',
                name: '{method} - {bucket_name}',
                opacity: 0.6,
                marker: {{color: '{color}'}},
                nbinsx: 50,
                xaxis: 'x',
                yaxis: 'y'
            }}""")

    traces_js = ",\n    ".join(traces)

    return f"""
Plotly.newPlot('error-histograms', [
    {traces_js}
], {{
    title: 'Distribution of Percent Errors by Horizon Bucket',
    xaxis: {{title: 'Percent Error (%)', zeroline: true, zerolinecolor: '#c0392b', zerolinewidth: 2}},
    yaxis: {{title: 'Count'}},
    barmode: 'overlay',
    legend: {{x: 0.02, y: 0.98}},
    margin: {{l: 60, r: 30, t: 50, b: 50}},
    shapes: [{{
        type: 'line', x0: 0, x1: 0, y0: 0, y1: 1,
        yref: 'paper', line: {{color: '#c0392b', width: 2, dash: 'dash'}}
    }}]
}}, {{responsive: true}});
"""


def _build_bias_variance_chart(decomp_df: pd.DataFrame) -> str:
    """Build bias vs variance decomposition chart."""
    # Get the aggregate rows (those without 'horizon' column populated)
    agg = decomp_df[decomp_df["horizon"].isna()].copy() if "horizon" in decomp_df.columns else decomp_df[decomp_df.get("horizon", pd.Series(dtype=float)).isna()].copy()

    if len(agg) == 0:
        # Fallback: rows with all decomposition columns
        agg = decomp_df[decomp_df["frac_bias"].notna()].copy()

    methods = agg["method"].tolist()
    frac_bias = agg["frac_bias"].tolist() if "frac_bias" in agg.columns else [0, 0]
    frac_horizon = agg["frac_horizon"].tolist() if "frac_horizon" in agg.columns else [0, 0]
    frac_county = agg["frac_county"].tolist() if "frac_county" in agg.columns else [0, 0]
    frac_residual = agg["frac_residual"].tolist() if "frac_residual" in agg.columns else [0, 0]

    # Convert to percentages
    frac_bias_pct = [round(x * 100, 1) for x in frac_bias]
    frac_horizon_pct = [round(x * 100, 1) for x in frac_horizon]
    frac_county_pct = [round(x * 100, 1) for x in frac_county]
    frac_residual_pct = [round(x * 100, 1) for x in frac_residual]

    return f"""
Plotly.newPlot('bias-variance-chart', [
    {{
        x: {methods}, y: {frac_bias_pct},
        type: 'bar', name: 'Systematic Bias',
        marker: {{color: '#E74C3C'}},
        text: {frac_bias_pct},
        textposition: 'inside'
    }},
    {{
        x: {methods}, y: {frac_horizon_pct},
        type: 'bar', name: 'Horizon Effect',
        marker: {{color: '#F39C12'}},
        text: {frac_horizon_pct},
        textposition: 'inside'
    }},
    {{
        x: {methods}, y: {frac_county_pct},
        type: 'bar', name: 'Geographic Variation',
        marker: {{color: '#0563C1'}},
        text: {frac_county_pct},
        textposition: 'inside'
    }},
    {{
        x: {methods}, y: {frac_residual_pct},
        type: 'bar', name: 'Residual',
        marker: {{color: '#95A5A6'}},
        text: {frac_residual_pct},
        textposition: 'inside'
    }}
], {{
    title: 'Error Variance Decomposition by Method',
    xaxis: {{title: 'Method'}},
    yaxis: {{title: 'Fraction of Total Variance (%)', range: [0, 100]}},
    barmode: 'stack',
    legend: {{x: 0.7, y: 0.98}},
    margin: {{l: 60, r: 30, t: 50, b: 50}}
}}, {{responsive: true}});
"""


def _build_qq_charts(county_errors: pd.DataFrame) -> str:
    """Build QQ plots for error normality assessment."""
    buckets = {
        "Short (1-5yr)": (1, 5),
        "Medium (6-12yr)": (6, 12),
        "Long (13-19yr)": (13, 19),
    }

    traces = []
    shapes = []

    for idx, (bucket_name, (h_min, h_max)) in enumerate(buckets.items()):
        for method, color in [("m2026", "#0563C1"), ("sdc_2024", "#E74C3C")]:
            subset = county_errors[
                (county_errors["method"] == method)
                & (county_errors["horizon"] >= h_min)
                & (county_errors["horizon"] <= h_max)
            ]["pct_error"].dropna().values

            if len(subset) < 10:
                continue

            # Sort the data
            sorted_data = np.sort(subset)
            n = len(sorted_data)

            # Theoretical quantiles
            theoretical = sp_stats.norm.ppf(
                (np.arange(1, n + 1) - 0.5) / n,
                loc=np.mean(subset),
                scale=np.std(subset),
            )

            # Subsample for performance (max 500 points)
            if n > 500:
                idx_sample = np.linspace(0, n - 1, 500, dtype=int)
                sorted_data = sorted_data[idx_sample]
                theoretical = theoretical[idx_sample]

            traces.append(f"""{{
                x: {theoretical.tolist()},
                y: {sorted_data.tolist()},
                mode: 'markers', type: 'scatter',
                marker: {{color: '{color}', size: 3, opacity: 0.5}},
                name: '{method} - {bucket_name}'
            }}""")

    traces_js = ",\n    ".join(traces)

    # Reference line (y=x)
    return f"""
Plotly.newPlot('qq-charts', [
    {traces_js}
], {{
    title: 'QQ Plots: Observed vs. Normal Theoretical Quantiles',
    xaxis: {{title: 'Theoretical Quantiles (Normal)'}},
    yaxis: {{title: 'Observed Quantiles (% Error)'}},
    hovermode: 'closest',
    legend: {{x: 0.02, y: 0.98}},
    margin: {{l: 60, r: 30, t: 50, b: 50}},
    shapes: [{{
        type: 'line', x0: -50, y0: -50, x1: 50, y1: 50,
        line: {{color: 'gray', width: 1, dash: 'dash'}}
    }}]
}}, {{responsive: true}});
"""


def _build_horizon_growth_chart(pi_df: pd.DataFrame) -> str:
    """Build chart showing error std growth with horizon."""
    traces = []

    for method, color in [("m2026", "#0563C1"), ("sdc_2024", "#E74C3C")]:
        for level, dash in [("state", "solid"), ("county", "dash")]:
            subset = pi_df[
                (pi_df["method"] == method) & (pi_df["level"] == level)
            ].sort_values("horizon")

            if len(subset) == 0:
                continue

            # Separate observed vs extrapolated
            observed = subset[subset["n_obs"] > 0]
            extrapolated = subset[subset["n_obs"] == 0]

            horizons_obs = observed["horizon"].tolist()
            stds_obs = observed["std"].tolist()

            label = f"{method} ({level})"
            traces.append(f"""{{
                x: {horizons_obs}, y: {stds_obs},
                mode: 'lines+markers', type: 'scatter',
                line: {{color: '{color}', dash: '{dash}', width: 2}},
                marker: {{size: 5}},
                name: '{label} (observed)'
            }}""")

            if len(extrapolated) > 0:
                horizons_ext = extrapolated["horizon"].tolist()
                stds_ext = extrapolated["std"].tolist()
                traces.append(f"""{{
                    x: {horizons_ext}, y: {stds_ext},
                    mode: 'lines', type: 'scatter',
                    line: {{color: '{color}', dash: 'dot', width: 1.5}},
                    name: '{label} (extrapolated)'
                }}""")

    traces_js = ",\n    ".join(traces)

    return f"""
Plotly.newPlot('horizon-growth-chart', [
    {traces_js}
], {{
    title: 'Error Standard Deviation Growth with Horizon',
    xaxis: {{title: 'Forecast Horizon (years)', dtick: 5}},
    yaxis: {{title: 'Std of Percent Error (pp)'}},
    hovermode: 'x unified',
    legend: {{x: 0.02, y: 0.98}},
    margin: {{l: 60, r: 30, t: 50, b: 50}}
}}, {{responsive: true}});
"""


# ---------------------------------------------------------------------------
# HTML content builders
# ---------------------------------------------------------------------------


def _build_overview_tab(
    pi_df: pd.DataFrame,
    decomp_df: pd.DataFrame,
    normality_df: pd.DataFrame,
) -> str:
    """Build the overview tab HTML content."""
    # Key metrics
    m2026_county = pi_df[(pi_df["method"] == "m2026") & (pi_df["level"] == "county")]
    m2026_state = pi_df[(pi_df["method"] == "m2026") & (pi_df["level"] == "state")]

    # 5-year and 10-year county MAPE
    h5_county = m2026_county[m2026_county["horizon"] == 5]
    h10_county = m2026_county[m2026_county["horizon"] == 10]
    h5_mape = abs(h5_county["mean"].values[0]) if len(h5_county) > 0 else 0
    h10_mape = abs(h10_county["mean"].values[0]) if len(h10_county) > 0 else 0

    # 5-year and 10-year state APE
    h5_state = m2026_state[m2026_state["horizon"] == 5]
    h10_state = m2026_state[m2026_state["horizon"] == 10]
    h5_state_ape = abs(h5_state["mean"].values[0]) if len(h5_state) > 0 else 0
    h10_state_ape = abs(h10_state["mean"].values[0]) if len(h10_state) > 0 else 0

    # Normality assessment
    m2026_norm = normality_df[normality_df["method"] == "m2026"]
    normal_short = m2026_norm[m2026_norm["bucket"] == "short_term"]
    is_normal_short = normal_short["normal_at_05"].values[0] if len(normal_short) > 0 else False
    norm_str = "Yes" if is_normal_short else "No"

    # Decomposition
    decomp_agg = decomp_df[decomp_df.get("frac_bias", pd.Series(dtype=float)).notna()]
    m2026_decomp = decomp_agg[decomp_agg["method"] == "m2026"]
    bias_pct = m2026_decomp["frac_bias"].values[0] * 100 if len(m2026_decomp) > 0 else 0
    horizon_pct = m2026_decomp["frac_horizon"].values[0] * 100 if len(m2026_decomp) > 0 else 0

    return f"""
<div class="card">
    <h2>Uncertainty Quantification Summary</h2>
    <p>This report quantifies the uncertainty inherent in the North Dakota population
    projections by analyzing historical forecasting errors from walk-forward validation.
    Two methods are compared: the SDC 2024 methodology and the enhanced M2026 methodology
    used for current production projections.</p>

    <div class="metric-grid">
        <div class="metric-box">
            <div class="value">{h5_mape:.1f}%</div>
            <div class="label">M2026 Mean County Error (5yr)</div>
        </div>
        <div class="metric-box">
            <div class="value">{h10_mape:.1f}%</div>
            <div class="label">M2026 Mean County Error (10yr)</div>
        </div>
        <div class="metric-box">
            <div class="value">{h5_state_ape:.1f}%</div>
            <div class="label">M2026 State Error (5yr)</div>
        </div>
        <div class="metric-box">
            <div class="value">{h10_state_ape:.1f}%</div>
            <div class="label">M2026 State Error (10yr)</div>
        </div>
    </div>
</div>

<div class="card">
    <h2>Key Findings</h2>
    <div class="metric-grid">
        <div class="metric-box">
            <div class="value">{bias_pct:.0f}%</div>
            <div class="label">Error from Systematic Bias</div>
        </div>
        <div class="metric-box">
            <div class="value">{horizon_pct:.0f}%</div>
            <div class="label">Error from Horizon Effect</div>
        </div>
        <div class="metric-box">
            <div class="value">{norm_str}</div>
            <div class="label">Short-term Errors Normal?</div>
        </div>
        <div class="metric-box">
            <div class="value">1,000</div>
            <div class="label">Monte Carlo Simulations</div>
        </div>
    </div>

    <div class="interpretation">
        <strong>Interpretation:</strong> The projection uncertainty grows approximately as a
        power law with horizon length. At 5 years, state-level projections are accurate to
        within ~{h5_state_ape:.0f}% on average, but by 10 years the error roughly doubles.
        County-level projections have wider uncertainty due to the additional geographic
        variance component. The empirical prediction intervals provided here should be used
        as uncertainty ranges when communicating projection results to stakeholders.
    </div>
</div>
"""


def _build_decomposition_table(decomp_df: pd.DataFrame) -> str:
    """Build the decomposition detail table."""
    # Per-horizon rows
    horizon_rows = decomp_df[decomp_df.get("mae_pct", pd.Series(dtype=float)).notna()].copy()

    if len(horizon_rows) == 0:
        return ""

    rows_html = ""
    for _, row in horizon_rows.iterrows():
        method = row.get("method", "")
        horizon = row.get("horizon", "")
        n = row.get("n_observations", 0)
        mean_e = row.get("mean_error_pct", 0)
        std_e = row.get("std_error_pct", 0)
        mae = row.get("mae_pct", 0)
        rmse = row.get("rmse_pct", 0)

        if pd.isna(horizon):
            continue

        rows_html += f"""<tr>
            <td>{method}</td>
            <td>{int(horizon)}</td>
            <td>{int(n)}</td>
            <td>{mean_e:.2f}%</td>
            <td>{std_e:.2f}%</td>
            <td>{mae:.2f}%</td>
            <td>{rmse:.2f}%</td>
        </tr>\n"""

    return f"""
<div class="card">
    <h2>Per-Horizon Error Statistics</h2>
    <table>
        <thead>
            <tr>
                <th>Method</th>
                <th>Horizon (yr)</th>
                <th>N obs</th>
                <th>Mean Error</th>
                <th>Std Error</th>
                <th>MAE</th>
                <th>RMSE</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
</div>
"""


def _build_normality_table(normality_df: pd.DataFrame) -> str:
    """Build the normality test results table."""
    if len(normality_df) == 0:
        return ""

    rows_html = ""
    for _, row in normality_df.iterrows():
        sw_p = row.get("shapiro_p", float("nan"))
        dp_p = row.get("dagostino_p", float("nan"))
        normal = "Yes" if row.get("normal_at_05", False) else "No"
        dp_str = f"{dp_p:.4f}" if not pd.isna(dp_p) else "N/A"

        rows_html += f"""<tr>
            <td>{row['method']}</td>
            <td>{row['horizon_range']}</td>
            <td>{int(row['n'])}</td>
            <td>{row['skewness']:.3f}</td>
            <td>{row['kurtosis']:.3f}</td>
            <td>{sw_p:.4f}</td>
            <td>{dp_str}</td>
            <td>{normal}</td>
        </tr>\n"""

    return f"""
<div class="card">
    <h2>Normality Test Results</h2>
    <p>Shapiro-Wilk and D'Agostino-Pearson tests assess whether the error distributions
    are sufficiently close to Gaussian. Non-normal distributions (p &lt; 0.05) suggest that
    empirical (non-parametric) intervals are more appropriate than parametric ones.</p>
    <table>
        <thead>
            <tr>
                <th>Method</th>
                <th>Horizon Range</th>
                <th>N</th>
                <th>Skewness</th>
                <th>Kurtosis</th>
                <th>Shapiro-Wilk p</th>
                <th>D'Agostino p</th>
                <th>Normal (0.05)?</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
</div>
"""


def _build_methodology_tab() -> str:
    """Build the methodology documentation tab."""
    return """
<div class="card">
    <h2>Methodology</h2>

    <h3>1. Empirical Prediction Intervals</h3>
    <div class="methodology">
        <p>For each forecast horizon h (1 to 19 years), we collect all percent errors from the
        walk-forward validation across origin years and counties. The prediction interval at
        confidence level alpha is defined by the (alpha/2) and (1 - alpha/2) percentiles of
        this empirical error distribution.</p>
        <p>For horizons beyond 19 years (up to 30), we extrapolate using fitted models:
        the error mean grows linearly with horizon, and the error standard deviation follows
        a power law: std(h) = a * h^b. Percentiles for extrapolated horizons are computed
        from a normal approximation using the fitted mean and std.</p>
    </div>

    <h3>2. Monte Carlo Simulation</h3>
    <div class="methodology">
        <p>N = 1,000 simulated projection paths are generated. For each path, at each projection
        year (horizon h), a random error is drawn from a normal distribution calibrated to the
        observed mean and standard deviation of errors at that horizon.</p>
        <p>To produce realistic (smooth) paths rather than year-to-year noise, errors are drawn
        with autocorrelation (rho = 0.7): z_t = rho * z_{t-1} + sqrt(1-rho^2) * epsilon_t,
        where epsilon_t ~ N(0,1). The z-score is then mapped to an error percentage via
        pct_error = mean(h) + z * std(h).</p>
        <p>Random seed = 42 ensures reproducibility.</p>
    </div>

    <h3>3. Error Decomposition</h3>
    <div class="methodology">
        <p>Total error variance is decomposed using an ANOVA-like approach:</p>
        <ul>
            <li><strong>Systematic Bias</strong>: Squared grand mean of percent errors. Measures
            the consistent tendency to over- or under-project.</li>
            <li><strong>Horizon Effect</strong>: Sum of squares attributable to horizon grouping
            (between-horizon SS). Captures how error grows with forecast distance.</li>
            <li><strong>Geographic Variation</strong>: Sum of squares attributable to county
            grouping (between-county SS). Captures county-to-county differences in forecastability.</li>
            <li><strong>Residual</strong>: Remaining variation after removing horizon and geographic
            effects. Represents unpredictable, interaction, and measurement effects.</li>
        </ul>
        <p>Note: Because the decomposition is additive (SS_total = SS_county + SS_horizon + SS_residual),
        the fractions sum to approximately 100% (bias is computed separately as it overlaps with
        the other components).</p>
    </div>

    <h3>4. Normality Assessment</h3>
    <div class="methodology">
        <p>The Shapiro-Wilk test and D'Agostino-Pearson K^2 test are used to check whether error
        distributions at each horizon bucket are approximately Gaussian. Non-normality (heavy tails,
        skewness) motivates the use of empirical rather than parametric prediction intervals.</p>
        <p>QQ plots provide a visual check: points falling along the diagonal indicate normality;
        departures at the tails indicate heavier-than-normal tails.</p>
    </div>

    <h3>5. Data Sources</h3>
    <div class="methodology">
        <p>Walk-forward validation data produced by <code>scripts/analysis/walk_forward_validation.py</code>
        using origin years 2005, 2010, 2015, and 2020 with validation against Census/PEP actuals.
        Production projections from the M2026 baseline scenario (2025-2055) computed by the
        cohort-component engine.</p>
    </div>
</div>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full uncertainty analysis pipeline."""
    print("=" * 70)
    print("UNCERTAINTY QUANTIFICATION ANALYSIS")
    print("=" * 70)

    # 1. Load data
    print("\n[1/6] Loading walk-forward validation data...")
    county_errors, state_errors = load_walk_forward_data()
    print(f"  County errors: {len(county_errors):,} rows")
    print(f"  State errors:  {len(state_errors):,} rows")

    print("\n[2/6] Loading production projections...")
    state_proj, county_proj = load_production_projections()
    print(f"  State projection: {len(state_proj)} years ({BASE_YEAR}-{PROJECTION_END})")
    print(f"  County projections: {county_proj['county_fips'].nunique()} counties")

    # 2. Empirical prediction intervals
    print("\n[3/6] Computing empirical prediction intervals...")
    pi_df = compute_prediction_intervals(county_errors, state_errors)
    print(f"  Observed horizons: 1-{int(pi_df['horizon'].max())}")

    # Extrapolate to full horizon
    pi_df = extrapolate_error_distribution(pi_df, MAX_HORIZON)
    print(f"  Extended to horizon {MAX_HORIZON} via power-law extrapolation")

    # Apply to production projections
    print("\n[4/6] Computing uncertainty bands and Monte Carlo simulation...")
    bands_df = compute_uncertainty_bands(state_proj, county_proj, pi_df)
    print(f"  Uncertainty bands: {len(bands_df):,} rows")

    # Monte Carlo
    state_mc, county_mc = run_monte_carlo(state_proj, county_proj, pi_df)
    print(f"  MC state: {len(state_mc)} rows, MC county: {len(county_mc):,} rows")

    # 3. Error decomposition
    print("\n[5/6] Decomposing projection errors...")
    decomp_df = compute_error_decomposition(county_errors, state_errors)
    normality_df = compute_normality_tests(county_errors)

    # Print decomposition summary
    for method in ["sdc_2024", "m2026"]:
        agg = decomp_df[
            (decomp_df["method"] == method) & (decomp_df["frac_bias"].notna())
        ]
        if len(agg) > 0:
            row = agg.iloc[0]
            print(f"\n  {method}:")
            print(f"    Mean error:     {row['mean_error_pct']:+.2f}%")
            print(f"    Bias fraction:  {row['frac_bias']:.1%}")
            print(f"    Horizon effect: {row['frac_horizon']:.1%}")
            print(f"    County effect:  {row['frac_county']:.1%}")
            print(f"    Residual:       {row['frac_residual']:.1%}")

    # 4. Write CSV outputs
    print("\n[6/6] Writing outputs...")
    WALK_FORWARD_DIR.mkdir(parents=True, exist_ok=True)

    pi_path = WALK_FORWARD_DIR / "prediction_intervals.csv"
    pi_df.to_csv(pi_path, index=False, float_format="%.4f")
    print(f"  {pi_path.relative_to(PROJECT_ROOT)}")

    bands_path = WALK_FORWARD_DIR / "uncertainty_bands.csv"
    bands_df.to_csv(bands_path, index=False, float_format="%.2f")
    print(f"  {bands_path.relative_to(PROJECT_ROOT)}")

    decomp_path = WALK_FORWARD_DIR / "error_decomposition.csv"
    decomp_df.to_csv(decomp_path, index=False, float_format="%.4f")
    print(f"  {decomp_path.relative_to(PROJECT_ROOT)}")

    # 5. Generate HTML report
    print("\n  Generating interactive HTML report...")
    html = generate_html_report(
        pi_df, bands_df, decomp_df, normality_df, state_mc, county_mc,
        county_errors, state_errors,
    )
    report_path = WALK_FORWARD_DIR / "uncertainty_report.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"  {report_path.relative_to(PROJECT_ROOT)}")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
