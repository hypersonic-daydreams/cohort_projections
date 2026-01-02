#!/usr/bin/env python3
"""
Module 9: Scenario Modeling Agent - Combined Forecasts and Monte Carlo Simulation
==================================================================================

Synthesizes results from all previous modules (1-8) to generate scenario-based
projections for North Dakota immigration. Implements:

1. Model averaging using fit statistics (R-squared, AIC weights)
2. Scenario generation:
   - CBO Full: Full immigration policy scenario
   - Moderate: Middle-ground assumptions
   - Zero: Zero net immigration scenario
   - Pre-2020 Trend: Continue historical trend
3. Monte Carlo simulation (1000 draws) for uncertainty quantification
4. Prediction intervals and fan chart data for visualization

Usage:
    micromamba run -n cohort_proj python module_9_scenario_modeling.py
"""

import json
import logging
import sys
import traceback
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from module_08_duration.wave_registry import (
    DurationModelBundle,
    WaveRegistry,
    load_duration_model_bundle,
    simulate_wave_contributions,
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

LOGGER = logging.getLogger(__name__)

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

SCENARIO_COLORS = {
    "cbo_full": "#0072B2",  # Blue
    "moderate": "#009E73",  # Teal
    "zero": "#D55E00",  # Orange
    "pre_2020_trend": "#CC79A7",  # Pink
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


from data_loader import load_migration_summary


def load_previous_results(result: ModuleResult) -> dict:
    """Load all previous module results for aggregation."""
    print("\n--- Loading Previous Module Results ---")

    all_results = {}

    # Module result file patterns to load
    module_patterns = [
        "module_1_1*.json",
        "module_1_2*.json",
        "module_2_1*.json",
        "module_2_2*.json",
        "module_3_1*.json",
        "module_3_2*.json",
        "module_4*.json",
        "module_5*.json",
        "module_6*.json",
        "module_7*.json",
        "module_8*.json",
    ]

    for pattern in module_patterns:
        for f in RESULTS_DIR.glob(pattern):
            try:
                with open(f) as fp:
                    all_results[f.stem] = json.load(fp)
                result.input_files.append(f.name)
            except Exception as e:
                print(f"  Warning: Could not load {f.name}: {e}")

    print(f"  Loaded {len(all_results)} result files")
    return all_results


def load_migration_data(result: ModuleResult) -> pd.DataFrame:
    """Load ND migration summary data from PostgreSQL."""
    df = load_migration_summary()
    result.input_files.append("census.state_components (PostgreSQL)")
    print(f"  Loaded migration data (DB): {len(df)} years")
    return df


def load_duration_bundle(result: ModuleResult) -> Optional[DurationModelBundle]:
    """Load hazard-based duration model inputs, if available."""
    hazard_path = RESULTS_DIR / "module_8_hazard_model.json"
    duration_path = RESULTS_DIR / "module_8_duration_analysis.json"
    if not hazard_path.exists() or not duration_path.exists():
        result.warnings.append(
            "Duration model outputs missing; wave adjustments skipped."
        )
        LOGGER.warning("Duration model outputs missing; wave adjustments skipped.")
        return None
    try:
        return load_duration_model_bundle(RESULTS_DIR)
    except Exception as exc:  # pragma: no cover - defensive guard
        result.warnings.append(f"Duration model load failed: {exc}")
        LOGGER.exception("Duration model load failed.", exc_info=exc)
        return None


def build_nd_wave_registry(
    df_migration: pd.DataFrame,
    *,
    baseline_year_end: int = 2019,
    threshold_pct: float = 50.0,
    min_wave_years: int = 2,
) -> tuple[WaveRegistry, float]:
    """Construct a wave registry for ND aggregate series."""
    baseline_window = df_migration[df_migration["year"] <= baseline_year_end][
        "nd_intl_migration"
    ]
    if len(baseline_window) > 0:
        baseline = float(np.median(baseline_window))
    else:
        baseline = float(df_migration["nd_intl_migration"].median())

    registry = WaveRegistry.from_series(
        state="North Dakota",
        origin="All",
        years=df_migration["year"].tolist(),
        arrivals=df_migration["nd_intl_migration"].tolist(),
        baseline=baseline,
        threshold_pct=threshold_pct,
        min_wave_years=min_wave_years,
    )
    return registry, baseline


def extract_model_estimates(all_results: dict, result: ModuleResult) -> dict:
    """
    Extract key estimates from all previous modules.

    Returns a dictionary of model predictions, coefficients, and fit statistics.
    """
    print("\n--- Extracting Model Estimates ---")

    estimates = {
        "arima": {},
        "var": {},
        "panel": {},
        "quantile": {},
        "robust": {},
        "gravity": {},
        "machine_learning": {},
        "causal": {},
        "duration": {},
    }

    # Module 2.1 - ARIMA
    arima_key = "module_2_1_arima_model"
    if arima_key in all_results:
        arima = all_results[arima_key]
        if "results" in arima and "forecasts" in arima["results"]:
            forecasts = arima["results"]["forecasts"]
            estimates["arima"] = {
                "model": arima["results"].get("model", "ARIMA"),
                "aic": arima["results"].get("fit_statistics", {}).get("aic"),
                "forecasts": [
                    {
                        "horizon": f["horizon"],
                        "year": 2024 + f["horizon"],
                        "point": f["point"],
                        "se": f["se"],
                        "ci_80": f["ci_80"],
                        "ci_95": f["ci_95"],
                    }
                    for f in forecasts
                ],
            }
            print(f"  ARIMA: {len(forecasts)} forecasts extracted")

    # Module 2.2 - VAR/Cointegration
    var_key = "module_2_2_var_cointegration"
    if var_key in all_results:
        var = all_results[var_key]
        if "results" in var:
            res = var["results"]
            # Extract long-run relationship
            eg = res.get("engle_granger_cointegration", {})
            step1 = eg.get("step1_ols_regression", {})
            estimates["var"] = {
                "cointegration_slope": step1.get("slope", {}).get("coefficient"),
                "cointegration_r2": step1.get("r_squared"),
                "var_aic": res.get("var_model", {}).get("model_fit", {}).get("aic"),
            }
            print(
                f"  VAR: Cointegration slope = {estimates['var']['cointegration_slope']:.6f}"
            )

    # Module 3.1 - Panel Data
    panel_key = "module_3_1_panel_analysis"
    if panel_key in all_results:
        panel = all_results[panel_key]
        if "results" in panel:
            estimates["panel"] = {
                "fixed_r2_within": panel["results"]
                .get("fixed_effects", {})
                .get("r_squared_within"),
                "hausman_recommendation": panel["results"]
                .get("hausman_test", {})
                .get("recommendation"),
            }
            print(
                f"  Panel: Hausman recommends {estimates['panel']['hausman_recommendation']}"
            )

    # Module 4 - Regression Extensions
    reg_key = "module_4_regression_extensions"
    if reg_key in all_results:
        reg = all_results[reg_key]
        if "results" in reg:
            res = reg["results"]
            comparison = res.get("comparison_table", {}).get(
                "ols_vs_quantile_vs_robust", {}
            )
            estimates["quantile"] = {
                "median_trend": comparison.get("time_trend", {}).get("q50"),
                "q10_trend": comparison.get("time_trend", {}).get("q10"),
                "q90_trend": comparison.get("time_trend", {}).get("q90"),
                "covid_effect_median": comparison.get("covid_2020", {}).get("q50"),
            }
            estimates["robust"] = {
                "huber_trend": comparison.get("time_trend", {}).get("huber"),
                "tukey_trend": comparison.get("time_trend", {}).get("tukey"),
                "ols_r2": res.get("robust_regression", {}).get("ols_r_squared"),
            }
            print(
                f"  Quantile: Median trend = {estimates['quantile']['median_trend']:.2f}"
            )
            print(f"  Robust: Huber trend = {estimates['robust']['huber_trend']:.2f}")

    # Module 5 - Gravity Model
    gravity_key = "module_5_gravity_model"
    if gravity_key in all_results:
        gravity = all_results[gravity_key]
        if "model_2_full_gravity" in gravity:
            full = gravity["model_2_full_gravity"]
            estimates["gravity"] = {
                "network_elasticity": full.get("coefficients", {})
                .get("log_diaspora", {})
                .get("estimate"),
                "pseudo_r2": full.get("fit_statistics", {}).get("pseudo_r2_mcfadden"),
            }
            print(
                f"  Gravity: Network elasticity = {estimates['gravity']['network_elasticity']:.4f}"
            )

    # Module 6 - Machine Learning
    ml_key = "module_6_machine_learning"
    if ml_key in all_results:
        ml = all_results[ml_key]
        if "results" in ml:
            res = ml["results"]
            estimates["machine_learning"] = {
                "elastic_net_cv_r2": res.get("elastic_net", {}).get("cv_r2"),
                "rf_cv_r2": res.get("random_forest", {}).get("cv_r2"),
                "clustering_silhouette": res.get("clustering", {}).get(
                    "silhouette_score"
                ),
                "nd_cluster": res.get("clustering", {}).get("nd_cluster"),
            }
            print(
                f"  ML: Elastic Net CV R2 = {estimates['machine_learning']['elastic_net_cv_r2']:.4f}"
            )

    # Module 7 - Causal Inference
    causal_key = "module_7_causal_inference"
    if causal_key in all_results:
        causal = all_results[causal_key]
        if "results" in causal:
            estimates["causal"] = {
                "treatment_effect": causal["results"]
                .get("did_analysis", {})
                .get("treatment_effect"),
            }
            print("  Causal: Loaded DiD results")

    # Module 8 - Duration Analysis
    duration_key = "module_8_duration_analysis"
    if duration_key in all_results:
        duration = all_results[duration_key]
        if "results" in duration:
            res = duration["results"]
            estimates["duration"] = {
                "median_wave_duration": res.get("kaplan_meier", {})
                .get("overall_summary", {})
                .get("median_survival_years"),
                "cox_concordance": res.get("cox_proportional_hazards", {})
                .get("fit_statistics", {})
                .get("concordance_index"),
            }
            print(
                f"  Duration: Median wave duration = {estimates['duration']['median_wave_duration']}"
            )

    result.add_decision(
        decision_id="D001",
        category="data_integration",
        decision="Extracted estimates from 9 module categories",
        rationale="Consolidate all statistical findings for scenario modeling",
        alternatives=[
            "Use only time series models",
            "Use only cross-sectional results",
        ],
        evidence=f"Loaded {len(result.input_files)} result files",
    )

    return estimates


def calculate_model_weights(estimates: dict, result: ModuleResult) -> dict:
    """
    Calculate model weights for forecast averaging.

    Uses AIC weights where available, otherwise equal weights.
    """
    print("\n--- Calculating Model Weights ---")

    weights = {}

    # Collect AICs from models with forecasts
    aics = {}
    if estimates.get("arima", {}).get("aic"):
        aics["arima"] = estimates["arima"]["aic"]
    if estimates.get("var", {}).get("var_aic"):
        aics["var"] = estimates["var"]["var_aic"]

    # If we have AICs, calculate AIC weights
    if len(aics) >= 2:
        min_aic = min(aics.values())
        delta_aic = {k: v - min_aic for k, v in aics.items()}
        exp_weights = {k: np.exp(-0.5 * v) for k, v in delta_aic.items()}
        sum_exp = sum(exp_weights.values())
        weights["aic_based"] = {k: v / sum_exp for k, v in exp_weights.items()}
        print(f"  AIC-based weights: {weights['aic_based']}")
    else:
        # Fall back to equal weights
        weights["aic_based"] = {"arima": 0.5, "var": 0.5}
        print("  Using equal weights (insufficient AICs for comparison)")

    # Also compute R-squared weights for cross-sectional models
    r2_vals = {}
    if estimates.get("robust", {}).get("ols_r2"):
        r2_vals["ols"] = estimates["robust"]["ols_r2"]
    if estimates.get("machine_learning", {}).get("elastic_net_cv_r2"):
        r2_vals["elastic_net"] = estimates["machine_learning"]["elastic_net_cv_r2"]
    if estimates.get("machine_learning", {}).get("rf_cv_r2"):
        r2_vals["random_forest"] = estimates["machine_learning"]["rf_cv_r2"]

    if r2_vals:
        total_r2 = sum(r2_vals.values())
        weights["r2_based"] = {k: v / total_r2 for k, v in r2_vals.items()}
        print(f"  R2-based weights: {weights['r2_based']}")

    result.add_decision(
        decision_id="D002",
        category="methodology",
        decision="Computed AIC and R-squared based model weights",
        rationale="AIC weights for time series averaging; R2 weights for cross-sectional insight",
        alternatives=["Equal weights", "Expert-assigned weights", "BIC weights"],
        evidence=f"AIC models: {list(aics.keys())}, R2 models: {list(r2_vals.keys())}",
    )

    return weights


def generate_scenarios(
    df_migration: pd.DataFrame, estimates: dict, result: ModuleResult
) -> dict:
    """
    Generate projection scenarios for 2025-2045.

    Scenarios:
    1. CBO Full: Based on CBO immigration projections (high growth)
    2. Moderate: Weighted average of model estimates
    3. Zero: Zero net immigration
    4. Pre-2020 Trend: Continue 2010-2019 trend
    """
    print("\n--- Generating Scenarios ---")

    # Projection parameters
    base_year = 2024
    horizon_end = 2045
    projection_years = list(range(base_year + 1, horizon_end + 1))
    len(projection_years)

    # Get 2024 baseline value
    baseline_2024 = df_migration[df_migration["year"] == 2024][
        "nd_intl_migration"
    ].values
    if len(baseline_2024) > 0:
        baseline = float(baseline_2024[0])
    else:
        # Use last available year
        baseline = float(df_migration["nd_intl_migration"].iloc[-1])
    print(f"  Baseline (2024): {baseline:.0f}")

    # Pre-2020 data for trend estimation
    pre_2020 = df_migration[df_migration["year"] < 2020]["nd_intl_migration"]
    pre_2020_years = df_migration[df_migration["year"] < 2020]["year"]

    # Estimate pre-2020 trend via OLS
    if len(pre_2020) > 2:
        X = pre_2020_years.values - pre_2020_years.min()
        y = pre_2020.values
        slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
        pre_2020_trend = slope
    else:
        pre_2020_trend = 0

    print(f"  Pre-2020 trend: {pre_2020_trend:.2f} per year")

    # ARIMA forecasts (if available)
    arima_forecasts = estimates.get("arima", {}).get("forecasts", [])

    # Get trend estimates from various models
    median_trend = estimates.get("quantile", {}).get("median_trend", pre_2020_trend)
    huber_trend = estimates.get("robust", {}).get("huber_trend", pre_2020_trend)
    tukey_trend = estimates.get("robust", {}).get("tukey_trend", pre_2020_trend)

    # Weight average trend
    avg_trend = np.mean(
        [t for t in [median_trend, huber_trend, tukey_trend] if t is not None]
    )

    scenarios = {}

    # Scenario 1: CBO Full (high immigration assumption)
    # Assume immigration continues to grow at elevated rates post-2024
    cbo_growth_rate = 0.08  # 8% annual growth (aggressive)
    cbo_projections = []
    current = baseline
    for i, year in enumerate(projection_years):
        if i < len(arima_forecasts):
            # Use ARIMA for first few years
            current = arima_forecasts[i]["point"] * 1.1  # 10% above ARIMA
        else:
            current = current * (1 + cbo_growth_rate)
        cbo_projections.append({"year": year, "value": current})

    scenarios["cbo_full"] = {
        "name": "CBO Full Immigration",
        "description": "2025--2029 at 10% above ARIMA, then 8% compound growth",
        "assumptions": {
            "growth_rate": cbo_growth_rate,
            "arima_multiplier": 1.1,
            "arima_years": len(arima_forecasts),
        },
        "projections": cbo_projections,
    }
    print(f"  CBO Full 2045: {cbo_projections[-1]['value']:.0f}")

    # Scenario 2: Moderate (weighted average)
    moderate_projections = []
    current = baseline
    for i, year in enumerate(projection_years):
        if i < len(arima_forecasts):
            # Use ARIMA directly
            current = arima_forecasts[i]["point"]
        else:
            # Use average trend
            current = current + avg_trend * 0.5  # Dampened trend
        moderate_projections.append({"year": year, "value": max(0, current)})

    scenarios["moderate"] = {
        "name": "Moderate Scenario",
        "description": "Middle-ground assumption with dampened historical trend",
        "assumptions": {
            "trend_dampening": 0.5,
            "average_trend_used": avg_trend,
        },
        "projections": moderate_projections,
    }
    print(f"  Moderate 2045: {moderate_projections[-1]['value']:.0f}")

    # Scenario 3: Zero Net Immigration
    zero_projections = []
    for year in projection_years:
        zero_projections.append({"year": year, "value": 0.0})

    scenarios["zero"] = {
        "name": "Zero Net Immigration",
        "description": "Hypothetical scenario with no international migration",
        "assumptions": {
            "immigration_level": 0,
        },
        "projections": zero_projections,
    }
    print(f"  Zero 2045: {zero_projections[-1]['value']:.0f}")

    # Scenario 4: Pre-2020 Trend
    pre2020_projections = []
    start_val = float(pre_2020.iloc[-1]) if len(pre_2020) > 0 else baseline
    for i, year in enumerate(projection_years):
        # Continue linear trend
        val = start_val + pre_2020_trend * (year - 2019)
        pre2020_projections.append({"year": year, "value": max(0, val)})

    scenarios["pre_2020_trend"] = {
        "name": "Pre-2020 Trend",
        "description": "Counterfactual anchored to 2019 with 2010--2019 slope",
        "assumptions": {
            "trend_slope": pre_2020_trend,
            "start_value": start_val,
        },
        "projections": pre2020_projections,
    }
    print(f"  Pre-2020 Trend 2045: {pre2020_projections[-1]['value']:.0f}")

    result.add_decision(
        decision_id="D003",
        category="scenario_design",
        decision="Generated 4 scenarios: CBO Full, Moderate, Zero, Pre-2020 Trend",
        rationale="Cover range from optimistic to restrictive immigration policy outcomes",
        alternatives=["5-scenario approach", "Probabilistic weighting of scenarios"],
        evidence=f"Baseline 2024 = {baseline:.0f}, Pre-2020 trend = {pre_2020_trend:.2f}/yr",
    )

    return scenarios


def monte_carlo_simulation(
    df_migration: pd.DataFrame,
    estimates: dict,
    result: ModuleResult,
    n_draws: int = 1000,
) -> dict:
    """
    Run Monte Carlo simulation to quantify forecast uncertainty.

    Uses parameter uncertainty from model estimates to generate
    distribution of future outcomes.
    """
    print(f"\n--- Monte Carlo Simulation ({n_draws} draws) ---")

    np.random.seed(42)  # Reproducibility

    # Projection parameters
    base_year = 2024
    horizon_end = 2045
    projection_years = list(range(base_year + 1, horizon_end + 1))
    n_years = len(projection_years)

    # Get baseline
    baseline_2024 = df_migration[df_migration["year"] == 2024][
        "nd_intl_migration"
    ].values
    if len(baseline_2024) > 0:
        baseline = float(baseline_2024[0])
    else:
        baseline = float(df_migration["nd_intl_migration"].iloc[-1])

    # Get trend parameters with uncertainty
    median_trend = estimates.get("quantile", {}).get("median_trend", 200)
    q10_trend = estimates.get("quantile", {}).get("q10_trend", 0)
    q90_trend = estimates.get("quantile", {}).get("q90_trend", 300)

    # Estimate standard error of trend from quantile range
    # IQR corresponds to roughly 1.35 std devs for normal
    if q90_trend and q10_trend:
        trend_std = (q90_trend - q10_trend) / (2 * 1.28)  # 80% range -> ~1.28 z
    else:
        trend_std = abs(median_trend) * 0.5  # Default to 50% CV

    # ARIMA standard errors (if available)
    arima_forecasts = estimates.get("arima", {}).get("forecasts", [])
    arima_ses = [f.get("se", 1000) for f in arima_forecasts]

    print(f"  Trend mean: {median_trend:.2f}, std: {trend_std:.2f}")

    duration_bundle = load_duration_bundle(result)
    wave_registry = None
    active_waves: list = []
    wave_baseline = None
    if duration_bundle:
        wave_registry, wave_baseline = build_nd_wave_registry(df_migration)
        active_waves = wave_registry.active_waves()
        if active_waves:
            wave_registry.annotate_survival(duration_bundle.predictor, horizon=n_years)
            LOGGER.info(
                "Wave adjustment active: %s wave(s) detected (baseline=%.1f).",
                len(active_waves),
                wave_baseline,
            )

    # Storage for simulation results
    simulations = np.zeros((n_draws, n_years))
    wave_adjustments = np.zeros((n_draws, n_years))
    wave_rng = np.random.default_rng(42)

    for draw in range(n_draws):
        # Draw trend parameter
        trend_draw = np.random.normal(median_trend, trend_std)

        # Draw random walk innovation variance scale
        innovation_scale = np.random.uniform(0.8, 1.2)

        # Simulate path
        current = baseline
        for t in range(n_years):
            # Trend component
            if t < len(arima_ses):
                # Use ARIMA forecast uncertainty for near term
                se = arima_ses[t] * innovation_scale
                point = (
                    arima_forecasts[t]["point"]
                    if t < len(arima_forecasts)
                    else current + trend_draw
                )
                current = np.random.normal(point, se)
            else:
                # Random walk with drift for longer horizon
                innovation_std = (
                    trend_std * np.sqrt(t - len(arima_ses) + 1) * innovation_scale
                )
                current = current + trend_draw + np.random.normal(0, innovation_std)

            # Enforce non-negativity
            simulations[draw, t] = max(0, current)

        if active_waves:
            draw_adjustment = np.zeros(n_years)
            for wave in active_waves:
                draw_adjustment += simulate_wave_contributions(
                    wave,
                    duration_bundle.predictor,
                    duration_bundle.lifecycle_stats,
                    horizon=n_years,
                    rng=wave_rng,
                )
            wave_adjustments[draw, :] = draw_adjustment
            simulations[draw, :] = np.maximum(0, simulations[draw, :] + draw_adjustment)

    # Compute percentiles
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    percentile_values = {}
    for p in percentiles:
        percentile_values[f"p{p}"] = np.percentile(simulations, p, axis=0).tolist()

    # Summary statistics by year
    mc_summary = []
    for t, year in enumerate(projection_years):
        year_sims = simulations[:, t]
        mc_summary.append(
            {
                "year": year,
                "mean": float(np.mean(year_sims)),
                "std": float(np.std(year_sims)),
                "min": float(np.min(year_sims)),
                "max": float(np.max(year_sims)),
                "p5": float(np.percentile(year_sims, 5)),
                "p25": float(np.percentile(year_sims, 25)),
                "p50": float(np.percentile(year_sims, 50)),
                "p75": float(np.percentile(year_sims, 75)),
                "p95": float(np.percentile(year_sims, 95)),
            }
        )

    print(
        f"  2030 median: {mc_summary[5]['p50']:.0f} (95% CI: [{mc_summary[5]['p5']:.0f}, {mc_summary[5]['p95']:.0f}])"
    )
    print(
        f"  2045 median: {mc_summary[-1]['p50']:.0f} (95% CI: [{mc_summary[-1]['p5']:.0f}, {mc_summary[-1]['p95']:.0f}])"
    )

    result.add_decision(
        decision_id="D004",
        category="uncertainty_quantification",
        decision=f"Ran {n_draws}-draw Monte Carlo simulation",
        rationale="Propagate parameter uncertainty through projections",
        alternatives=["Analytical prediction intervals", "Bootstrapping"],
        evidence="Trend uncertainty from quantile regression range",
    )

    wave_summary = None
    if active_waves:
        wave_summary = {
            "active_waves": len(active_waves),
            "baseline": wave_baseline,
            "mean_adjustment_2030": float(np.mean(wave_adjustments[:, 5]))
            if n_years > 5
            else None,
            "mean_adjustment_2045": float(np.mean(wave_adjustments[:, -1]))
            if n_years > 0
            else None,
        }
        result.add_decision(
            decision_id="D004A",
            category="scenario_integration",
            decision="Applied hazard-based wave persistence adjustments",
            rationale="Map duration model outputs into scenario Monte Carlo paths",
            alternatives=["No wave adjustments", "Deterministic wave overlays"],
            evidence=f"Active waves detected: {wave_summary['active_waves']}",
        )

    mc_results = {
        "n_draws": n_draws,
        "projection_years": projection_years,
        "parameters": {
            "trend_mean": median_trend,
            "trend_std": trend_std,
            "baseline_2024": baseline,
        },
        "wave_adjustment": wave_summary,
        "percentiles": percentile_values,
        "summary_by_year": mc_summary,
        "simulations_shape": list(simulations.shape),
    }

    return mc_results, simulations


def compute_confidence_intervals(
    scenarios: dict, mc_results: dict, result: ModuleResult
) -> dict:
    """
    Compute prediction intervals and fan chart data.

    Combines scenario projections with Monte Carlo uncertainty.
    """
    print("\n--- Computing Confidence Intervals ---")

    ci_data = {
        "projection_years": mc_results["projection_years"],
        "intervals": {},
    }

    # Extract MC percentiles
    summary = mc_results["summary_by_year"]
    [s["year"] for s in summary]

    # 50% CI (25th-75th)
    ci_data["intervals"]["ci_50"] = {
        "lower": [s["p25"] for s in summary],
        "upper": [s["p75"] for s in summary],
    }

    # 80% CI (10th-90th)
    percentiles = mc_results["percentiles"]
    if "p10" in percentiles:
        ci_data["intervals"]["ci_80"] = {
            "lower": percentiles["p10"],
            "upper": percentiles["p90"],
        }
    else:
        # Approximate from summary
        ci_data["intervals"]["ci_80"] = {
            "lower": [s["p5"] * 1.2 for s in summary],  # Rough approximation
            "upper": [s["p95"] * 0.85 for s in summary],
        }

    # 95% CI (5th-95th)
    ci_data["intervals"]["ci_95"] = {
        "lower": [s["p5"] for s in summary],
        "upper": [s["p95"] for s in summary],
    }

    # Median (point forecast)
    ci_data["median"] = [s["p50"] for s in summary]
    ci_data["mean"] = [s["mean"] for s in summary]

    # Add scenario paths for comparison
    ci_data["scenarios"] = {}
    for scenario_name, scenario in scenarios.items():
        ci_data["scenarios"][scenario_name] = [
            p["value"] for p in scenario["projections"]
        ]

    print("  Computed 50%, 80%, and 95% prediction intervals")
    print(f"  Added {len(scenarios)} scenario paths for comparison")

    return ci_data


def create_combined_forecasts_df(
    scenarios: dict, mc_results: dict, ci_data: dict
) -> pd.DataFrame:
    """Create DataFrame with combined forecasts for parquet output."""
    years = mc_results["projection_years"]
    summary = mc_results["summary_by_year"]

    rows = []
    for i, year in enumerate(years):
        row = {
            "year": year,
            "mc_mean": summary[i]["mean"],
            "mc_median": summary[i]["p50"],
            "mc_std": summary[i]["std"],
            "ci_50_lower": summary[i]["p25"],
            "ci_50_upper": summary[i]["p75"],
            "ci_95_lower": summary[i]["p5"],
            "ci_95_upper": summary[i]["p95"],
        }
        # Add scenarios
        for scenario_name, scenario in scenarios.items():
            row[f"scenario_{scenario_name}"] = scenario["projections"][i]["value"]
        rows.append(row)

    return pd.DataFrame(rows)


def create_scenario_projections_df(scenarios: dict) -> pd.DataFrame:
    """Create DataFrame with scenario projections for parquet output."""
    rows = []
    for scenario_name, scenario in scenarios.items():
        for proj in scenario["projections"]:
            rows.append(
                {
                    "scenario": scenario_name,
                    "scenario_name": scenario["name"],
                    "year": proj["year"],
                    "value": proj["value"],
                }
            )
    return pd.DataFrame(rows)


def create_monte_carlo_df(
    simulations: np.ndarray, projection_years: list
) -> pd.DataFrame:
    """Create DataFrame with Monte Carlo simulation results."""
    n_draws, n_years = simulations.shape

    # For efficiency, store percentiles rather than all draws
    rows = []
    for t, year in enumerate(projection_years):
        year_sims = simulations[:, t]
        row = {
            "year": year,
            "mean": np.mean(year_sims),
            "std": np.std(year_sims),
            "min": np.min(year_sims),
            "p5": np.percentile(year_sims, 5),
            "p10": np.percentile(year_sims, 10),
            "p25": np.percentile(year_sims, 25),
            "p50": np.percentile(year_sims, 50),
            "p75": np.percentile(year_sims, 75),
            "p90": np.percentile(year_sims, 90),
            "p95": np.percentile(year_sims, 95),
            "max": np.max(year_sims),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def plot_fan_chart(
    df_migration: pd.DataFrame,
    ci_data: dict,
    mc_results: dict,
    result: ModuleResult,
):
    """Create fan chart visualization with confidence bands."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Historical data
    hist_years = df_migration["year"].values
    hist_values = df_migration["nd_intl_migration"].values

    ax.plot(
        hist_years,
        hist_values,
        "o-",
        color=COLORS["primary"],
        linewidth=2,
        markersize=6,
        label="Historical",
        zorder=10,
    )

    # Projection years
    proj_years = ci_data["projection_years"]

    # 95% CI band
    ax.fill_between(
        proj_years,
        ci_data["intervals"]["ci_95"]["lower"],
        ci_data["intervals"]["ci_95"]["upper"],
        alpha=0.15,
        color=COLORS["primary"],
        label="95% CI",
    )

    # 80% CI band (if available)
    if "ci_80" in ci_data["intervals"]:
        ax.fill_between(
            proj_years,
            ci_data["intervals"]["ci_80"]["lower"],
            ci_data["intervals"]["ci_80"]["upper"],
            alpha=0.25,
            color=COLORS["primary"],
            label="80% CI",
        )

    # 50% CI band
    ax.fill_between(
        proj_years,
        ci_data["intervals"]["ci_50"]["lower"],
        ci_data["intervals"]["ci_50"]["upper"],
        alpha=0.4,
        color=COLORS["primary"],
        label="50% CI",
    )

    # Median projection
    ax.plot(
        proj_years,
        ci_data["median"],
        "-",
        color=COLORS["primary"],
        linewidth=2,
        label="Median Projection",
    )

    # Connect historical to projection
    ax.plot(
        [hist_years[-1], proj_years[0]],
        [hist_values[-1], ci_data["median"][0]],
        "--",
        color=COLORS["neutral"],
        linewidth=1,
    )

    # Vertical line at base year
    ax.axvline(2024, color=COLORS["neutral"], linestyle="--", linewidth=1, alpha=0.5)
    ax.text(
        2024.2, ax.get_ylim()[1] * 0.95, "2024", fontsize=9, color=COLORS["neutral"]
    )

    # COVID annotation
    covid_idx = np.where(hist_years == 2020)[0]
    if len(covid_idx) > 0:
        ax.annotate(
            "COVID-19",
            xy=(2020, hist_values[covid_idx[0]]),
            xytext=(2018, hist_values[covid_idx[0]] + 1500),
            arrowprops={"arrowstyle": "->", "color": COLORS["secondary"]},
            fontsize=9,
            color=COLORS["secondary"],
        )

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("International Migration to ND", fontsize=12)
    ax.set_xlim(2009, 2046)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add text box with key statistics
    textstr = f"2030 Median: {ci_data['median'][5]:,.0f}\n2045 Median: {ci_data['median'][-1]:,.0f}"
    props = {"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5}
    ax.text(
        0.98,
        0.02,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=props,
    )

    save_figure(
        fig,
        str(FIGURES_DIR / "module_9_fan_chart"),
        "International Migration to North Dakota: Fan Chart Projection (2025-2045)",
        "Monte Carlo simulation with 1,000 draws | Census Bureau Components of Change",
    )


def plot_scenario_comparison(
    df_migration: pd.DataFrame,
    scenarios: dict,
    ci_data: dict,
    result: ModuleResult,
):
    """Plot comparison of different scenarios."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Historical data
    hist_years = df_migration["year"].values
    hist_values = df_migration["nd_intl_migration"].values

    ax.plot(
        hist_years,
        hist_values,
        "o-",
        color="black",
        linewidth=2,
        markersize=6,
        label="Historical",
        zorder=10,
    )

    # Plot each scenario
    scenario_styles = {
        "cbo_full": {"linestyle": "-", "linewidth": 2.5},
        "moderate": {"linestyle": "-", "linewidth": 2},
        "zero": {"linestyle": "--", "linewidth": 2},
        "pre_2020_trend": {"linestyle": "-.", "linewidth": 2},
    }

    for scenario_name, scenario in scenarios.items():
        proj_years = [p["year"] for p in scenario["projections"]]
        proj_values = [p["value"] for p in scenario["projections"]]

        style = scenario_styles.get(scenario_name, {"linestyle": "-", "linewidth": 1.5})
        color = SCENARIO_COLORS.get(scenario_name, COLORS["neutral"])

        ax.plot(
            proj_years,
            proj_values,
            color=color,
            label=scenario["name"],
            **style,
        )

        # Connect to historical
        ax.plot(
            [hist_years[-1], proj_years[0]],
            [hist_values[-1], proj_values[0]],
            color=color,
            linestyle=":",
            linewidth=1,
            alpha=0.5,
        )

    # Add Monte Carlo median for reference
    proj_years = ci_data["projection_years"]
    ax.plot(
        proj_years,
        ci_data["median"],
        color=COLORS["neutral"],
        linewidth=1.5,
        linestyle=":",
        label="MC Median",
    )

    # Vertical line at base year
    ax.axvline(2024, color=COLORS["neutral"], linestyle="--", linewidth=1, alpha=0.5)

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("International Migration to ND", fontsize=12)
    ax.set_xlim(2009, 2046)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add 2045 endpoint values
    ax.get_ylim()[1]
    for scenario_name, scenario in scenarios.items():
        final_val = scenario["projections"][-1]["value"]
        color = SCENARIO_COLORS.get(scenario_name, COLORS["neutral"])
        ax.text(
            2045.5,
            final_val,
            f"{final_val:,.0f}",
            fontsize=8,
            color=color,
            va="center",
        )

    save_figure(
        fig,
        str(FIGURES_DIR / "module_9_scenario_comparison"),
        "International Migration to North Dakota: Scenario Comparison (2025-2045)",
        "Multiple projection scenarios | Census Bureau Components of Change",
    )


def run_analysis() -> ModuleResult:
    """Main analysis function for Module 9."""
    result = ModuleResult(
        module_id="9",
        analysis_name="scenario_modeling",
    )

    print("Loading migration data...")
    df_migration = load_migration_data(result)

    print("\nLoading previous module results...")
    all_results = load_previous_results(result)

    # Extract estimates from previous modules
    estimates = extract_model_estimates(all_results, result)

    # Calculate model weights
    weights = calculate_model_weights(estimates, result)

    # Record parameters
    result.parameters = {
        "base_year": 2024,
        "projection_horizon": "2025-2045",
        "n_projection_years": 21,
        "monte_carlo_draws": 1000,
        "scenarios": ["CBO Full", "Moderate", "Zero", "Pre-2020 Trend"],
        "confidence_intervals": ["50%", "80%", "95%"],
        "model_weights": weights,
        "input_modules": list(all_results.keys()),
    }

    # Generate scenarios
    scenarios = generate_scenarios(df_migration, estimates, result)

    # Run Monte Carlo simulation
    mc_results, simulations = monte_carlo_simulation(
        df_migration, estimates, result, n_draws=1000
    )

    # Compute prediction intervals
    ci_data = compute_confidence_intervals(scenarios, mc_results, result)

    # Create output DataFrames
    print("\n--- Creating Output DataFrames ---")

    df_combined = create_combined_forecasts_df(scenarios, mc_results, ci_data)
    df_scenarios = create_scenario_projections_df(scenarios)
    df_monte_carlo = create_monte_carlo_df(simulations, mc_results["projection_years"])

    # Save DataFrames to parquet
    combined_path = RESULTS_DIR / "module_9_combined_forecasts.parquet"
    df_combined.to_parquet(combined_path, index=False)
    print(f"  Combined forecasts saved: {combined_path}")

    scenarios_path = RESULTS_DIR / "module_9_scenario_projections.parquet"
    df_scenarios.to_parquet(scenarios_path, index=False)
    print(f"  Scenario projections saved: {scenarios_path}")

    mc_path = RESULTS_DIR / "module_9_monte_carlo.parquet"
    df_monte_carlo.to_parquet(mc_path, index=False)
    print(f"  Monte Carlo results saved: {mc_path}")

    # Generate visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    plot_fan_chart(df_migration, ci_data, mc_results, result)
    plot_scenario_comparison(df_migration, scenarios, ci_data, result)

    # Compile results
    result.results = {
        "model_averaging": {
            "aic_weights": weights.get("aic_based", {}),
            "r2_weights": weights.get("r2_based", {}),
        },
        "scenarios": {
            name: {
                "description": s["description"],
                "final_2045_value": s["projections"][-1]["value"],
                "assumptions": s["assumptions"],
            }
            for name, s in scenarios.items()
        },
        "monte_carlo": {
            "n_draws": mc_results["n_draws"],
            "median_2030": mc_results["summary_by_year"][5]["p50"],
            "median_2045": mc_results["summary_by_year"][-1]["p50"],
            "ci_95_2045": [
                mc_results["summary_by_year"][-1]["p5"],
                mc_results["summary_by_year"][-1]["p95"],
            ],
            "wave_adjustment": mc_results.get("wave_adjustment"),
        },
        "confidence_intervals": {
            "ci_50": {
                "2030": [
                    ci_data["intervals"]["ci_50"]["lower"][5],
                    ci_data["intervals"]["ci_50"]["upper"][5],
                ],
                "2045": [
                    ci_data["intervals"]["ci_50"]["lower"][-1],
                    ci_data["intervals"]["ci_50"]["upper"][-1],
                ],
            },
            "ci_95": {
                "2030": [
                    ci_data["intervals"]["ci_95"]["lower"][5],
                    ci_data["intervals"]["ci_95"]["upper"][5],
                ],
                "2045": [
                    ci_data["intervals"]["ci_95"]["lower"][-1],
                    ci_data["intervals"]["ci_95"]["upper"][-1],
                ],
            },
        },
    }

    # Diagnostics
    result.diagnostics = {
        "input_modules_loaded": len(all_results),
        "estimates_extracted": {k: bool(v) for k, v in estimates.items()},
        "historical_years": len(df_migration),
        "projection_years": len(mc_results["projection_years"]),
        "wave_adjustment_applied": bool(mc_results.get("wave_adjustment")),
        "mc_convergence": {
            "mean_2045": mc_results["summary_by_year"][-1]["mean"],
            "std_2045": mc_results["summary_by_year"][-1]["std"],
            "cv_2045": mc_results["summary_by_year"][-1]["std"]
            / mc_results["summary_by_year"][-1]["mean"]
            if mc_results["summary_by_year"][-1]["mean"] > 0
            else None,
        },
    }

    # Next steps
    result.next_steps = [
        "Integrate scenario projections with cohort-component model",
        "Sensitivity analysis on key parameters",
        "Compare with official CBO and Census projections",
        "Update scenarios as new policy information becomes available",
        "Create interactive dashboard for scenario exploration",
    ]

    return result


def main():
    """Main entry point."""
    print("=" * 70)
    print("Module 9: Scenario Modeling Agent")
    print(f"Started: {datetime.now(UTC).isoformat()}")
    print("=" * 70)

    try:
        result = run_analysis()
        output_file = result.save("module_9_scenario_modeling.json")

        print("\n" + "=" * 70)
        print("Analysis completed successfully!")
        print("=" * 70)

        print(f"\nMain output: {output_file}")

        print("\nKey Results:")
        print(f"  Scenarios generated: {len(result.results['scenarios'])}")
        for name, scenario in result.results["scenarios"].items():
            print(f"    - {name}: {scenario['final_2045_value']:,.0f} (2045)")

        print("\nMonte Carlo Simulation:")
        print(f"  2030 Median: {result.results['monte_carlo']['median_2030']:,.0f}")
        print(f"  2045 Median: {result.results['monte_carlo']['median_2045']:,.0f}")
        ci_95 = result.results["monte_carlo"]["ci_95_2045"]
        print(f"  2045 95% CI: [{ci_95[0]:,.0f}, {ci_95[1]:,.0f}]")

        if result.warnings:
            print("\nWarnings:")
            for w in result.warnings:
                print(f"  - {w}")

        print(f"\nDecisions logged: {len(result.decisions)}")
        for d in result.decisions:
            print(f"  [{d['decision_id']}] {d['decision']}")

        print("\nOutput files generated:")
        print("  Parquet files:")
        print("    - module_9_combined_forecasts.parquet")
        print("    - module_9_scenario_projections.parquet")
        print("    - module_9_monte_carlo.parquet")
        print("  Figures:")
        print("    - module_9_fan_chart.png/pdf")
        print("    - module_9_scenario_comparison.png/pdf")
        print("  JSON:")
        print("    - module_9_scenario_modeling.json")

        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
