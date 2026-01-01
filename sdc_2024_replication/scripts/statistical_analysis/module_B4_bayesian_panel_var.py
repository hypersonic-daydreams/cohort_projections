#!/usr/bin/env python3
"""
Module B4: Bayesian and Panel VAR Extensions
=============================================

Implements Bayesian VAR with Minnesota prior and Panel VAR methods
to address small-n limitations in North Dakota migration analysis,
per ADR-020 Phase B4.

Components:
1. Minnesota/Litterman prior construction for VAR shrinkage
2. Bayesian VAR estimation (PyMC MCMC with conjugate fallback)
3. Panel VAR with entity/time fixed effects using 50-state data
4. Model comparison: Classical vs Bayesian vs Panel

Key Motivation:
Short time series (n=15) limits reliability of classical VAR.
Bayesian methods with informative priors stabilize estimation
and provide better uncertainty quantification.

Usage:
    uv run python module_B4_bayesian_panel_var.py
"""

import json
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Suppress warnings during execution
warnings.filterwarnings("ignore")

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # cohort_projections/
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "immigration" / "analysis"
RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"

# Ensure output directories exist
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Import B4 module components
from module_B4_bayesian_panel import (  # noqa: E402
    construct_minnesota_prior,
    MinnesotaPrior,
    estimate_bayesian_var,
    PYMC_AVAILABLE,
    estimate_panel_var,
    PanelVARResult,
    compare_var_models,
    ModelComparisonResult,
)
from module_B4_bayesian_panel.minnesota_prior import (  # noqa: E402
    estimate_ar1_variances,
    summarize_prior,
)

# Colorblind-safe palette
COLORS = {
    "primary": "#0072B2",  # Blue
    "secondary": "#D55E00",  # Vermillion/Orange
    "tertiary": "#009E73",  # Teal/Green
    "quaternary": "#CC79A7",  # Pink
    "highlight": "#F0E442",  # Yellow
    "neutral": "#999999",  # Gray
    "ci_fill": "#0072B2",  # Blue with alpha=0.2
}


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
        self.next_steps: list[str] = []

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
            "next_steps": self.next_steps,
        }

    def save(self, filename: str) -> Path:
        """Save results to JSON file."""
        output_path = RESULTS_DIR / filename
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        print(f"Results saved to: {output_path}")
        return output_path


def load_nd_data() -> pd.DataFrame:
    """Load North Dakota migration summary data."""
    filepath = DATA_DIR / "nd_migration_summary.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"ND data not found: {filepath}")
    df = pd.read_csv(filepath)
    print(
        f"Loaded ND data: {len(df)} observations ({df['year'].min()}-{df['year'].max()})"
    )
    return df


def load_panel_data() -> pd.DataFrame:
    """Load 50-state panel data for panel VAR analysis."""
    filepath = DATA_DIR / "combined_components_of_change.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"Panel data not found: {filepath}")
    df = pd.read_csv(filepath)

    # Filter to states only (exclude territories if present)
    if "state" in df.columns:
        territories = ["Puerto Rico", "District of Columbia"]
        df = df[~df["state"].isin(territories)]

    print(
        f"Loaded panel data: {len(df['state'].unique())} states, {len(df['year'].unique())} years"
    )
    return df


def plot_prior_specification(
    prior: MinnesotaPrior,
    output_path: Path,
) -> None:
    """Create visualization of Minnesota prior specification."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Prior mean structure
    ax1 = axes[0]

    im = ax1.imshow(prior.prior_mean, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
    ax1.set_xlabel("Equation (Dependent Variable)")
    ax1.set_ylabel("Coefficient Index")
    ax1.set_title("Prior Mean Matrix\n(Random walk: own lag-1 = 1)")
    plt.colorbar(im, ax=ax1, label="Prior Mean")

    # Right: Prior standard deviation
    ax2 = axes[1]
    prior_std = np.sqrt(prior.prior_var)
    im2 = ax2.imshow(prior_std, cmap="viridis", aspect="auto")
    ax2.set_xlabel("Equation (Dependent Variable)")
    ax2.set_ylabel("Coefficient Index")
    ax2.set_title("Prior Standard Deviation\n(Shrinkage intensity)")
    plt.colorbar(im2, ax=ax2, label="Prior SD")

    plt.tight_layout()
    fig.suptitle(
        f"Minnesota Prior Specification (lambda1={prior.hyperparameters['lambda1']}, "
        f"lambda2={prior.hyperparameters['lambda2']})",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_coefficient_comparison(
    comparison: ModelComparisonResult,
    var_cols: list[str],
    output_path: Path,
) -> None:
    """Create visualization comparing classical and Bayesian coefficients."""
    fig, axes = plt.subplots(1, len(var_cols), figsize=(6 * len(var_cols), 5))
    if len(var_cols) == 1:
        axes = [axes]

    for i, var in enumerate(var_cols):
        ax = axes[i]

        if var not in comparison.coefficient_comparison:
            continue

        coef_comp = comparison.coefficient_comparison[var]

        coef_names = []
        classical_vals = []
        bayesian_vals = []

        for coef_name in coef_comp:
            if coef_name == "summary":
                continue
            coef_names.append(coef_name)
            classical_vals.append(coef_comp[coef_name]["classical"])
            bayesian_vals.append(coef_comp[coef_name]["bayesian"])

        x = np.arange(len(coef_names))
        width = 0.35

        ax.bar(
            x - width / 2,
            classical_vals,
            width,
            label="Classical",
            color=COLORS["primary"],
            alpha=0.7,
        )
        ax.bar(
            x + width / 2,
            bayesian_vals,
            width,
            label="Bayesian",
            color=COLORS["secondary"],
            alpha=0.7,
        )

        ax.set_xlabel("Coefficient")
        ax.set_ylabel("Estimate")
        ax.set_title(f"Equation: {var}")
        ax.set_xticks(x)
        ax.set_xticklabels(coef_names, rotation=45, ha="right")
        ax.legend()
        ax.axhline(y=0, color=COLORS["neutral"], linestyle="--", linewidth=0.8)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.suptitle(
        "Coefficient Comparison: Classical vs Bayesian VAR",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_uncertainty_comparison(
    comparison: ModelComparisonResult,
    var_cols: list[str],
    output_path: Path,
) -> None:
    """Create visualization comparing uncertainty quantification."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Gather all SE ratios
    ratios = []
    labels = []

    for var in var_cols:
        if var not in comparison.uncertainty_comparison:
            continue
        for coef_name in comparison.uncertainty_comparison[var]:
            if coef_name == "summary":
                continue
            ratio = comparison.uncertainty_comparison[var][coef_name]["ratio"]
            ratios.append(ratio)
            labels.append(f"{var}:{coef_name}")

    if len(ratios) > 0:
        colors = [COLORS["tertiary"] if r < 1 else COLORS["quaternary"] for r in ratios]

        ax.barh(range(len(ratios)), ratios, color=colors, alpha=0.7)
        ax.set_yticks(range(len(ratios)))
        ax.set_yticklabels(labels)
        ax.axvline(
            x=1,
            color=COLORS["neutral"],
            linestyle="--",
            linewidth=2,
            label="Equal uncertainty",
        )
        ax.set_xlabel("Bayesian SD / Classical SE")
        ax.set_title(
            "Uncertainty Comparison\n(< 1: Bayesian more confident, > 1: Classical more confident)"
        )
        ax.legend()
        ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_panel_var_effects(
    panel_result: PanelVARResult,
    output_path: Path,
) -> None:
    """Create visualization of panel VAR entity effects."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Entity fixed effects distribution
    ax1 = axes[0]
    entity_effects = panel_result.entity_effects
    if "reference" in entity_effects:
        ref = entity_effects.pop("reference")
        entity_effects[ref] = 0.0  # Reference category has effect = 0

    entities = list(entity_effects.keys())
    effects = [entity_effects[e] for e in entities]

    # Sort by effect size
    sorted_idx = np.argsort(effects)
    entities = [entities[i] for i in sorted_idx]
    effects = [effects[i] for i in sorted_idx]

    # Highlight ND
    colors = [
        COLORS["secondary"] if "North Dakota" in e else COLORS["primary"]
        for e in entities
    ]

    ax1.barh(range(len(entities)), effects, color=colors, alpha=0.7)
    ax1.set_yticks(range(len(entities)))
    ax1.set_yticklabels(entities, fontsize=8)
    ax1.axvline(x=0, color=COLORS["neutral"], linestyle="--")
    ax1.set_xlabel("Fixed Effect")
    ax1.set_title("State Fixed Effects\n(North Dakota highlighted)")
    ax1.grid(True, alpha=0.3, axis="x")

    # Right: ND-specific coefficients (if available)
    ax2 = axes[1]
    if panel_result.nd_coefficients is not None:
        nd_coefs = panel_result.nd_coefficients

        # Get main coefficients
        main_coefs = panel_result.coefficients
        coef_names = list(main_coefs.keys())

        main_vals = [main_coefs[c] for c in coef_names]
        nd_total = []
        for c in coef_names:
            if "total_ND_effect" in nd_coefs and c in nd_coefs["total_ND_effect"]:
                nd_total.append(nd_coefs["total_ND_effect"][c])
            else:
                nd_total.append(main_vals[coef_names.index(c)])

        x = np.arange(len(coef_names))
        width = 0.35

        ax2.bar(
            x - width / 2,
            main_vals,
            width,
            label="Average (all states)",
            color=COLORS["primary"],
            alpha=0.7,
        )
        ax2.bar(
            x + width / 2,
            nd_total,
            width,
            label="ND-specific",
            color=COLORS["secondary"],
            alpha=0.7,
        )

        ax2.set_xlabel("Coefficient")
        ax2.set_ylabel("Estimate")
        ax2.set_title("AR Coefficients: Average vs ND-Specific")
        ax2.set_xticks(x)
        ax2.set_xticklabels(coef_names)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")
    else:
        ax2.text(
            0.5,
            0.5,
            "ND-specific coefficients\nnot available",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )
        ax2.set_title("ND-Specific Coefficients")

    plt.tight_layout()
    fig.suptitle(
        "Panel VAR: Entity Effects and ND-Specific Dynamics",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


def run_analysis() -> ModuleResult:
    """
    Main analysis function implementing B4 Bayesian/Panel VAR analysis.

    Returns:
        ModuleResult object with all findings
    """
    result = ModuleResult(
        module_id="B4",
        analysis_name="bayesian_panel_var",
    )

    # Record PyMC availability
    result.diagnostics["pymc_available"] = PYMC_AVAILABLE
    if not PYMC_AVAILABLE:
        result.warnings.append(
            "PyMC not installed. Using conjugate analytical posterior (faster but approximate). "
            "Install with: uv pip install cohort_projections[bayesian]"
        )

    # Load data
    print("\n" + "=" * 60)
    print("Loading Data")
    print("=" * 60)

    nd_data = load_nd_data()
    result.input_files.append("nd_migration_summary.csv")

    panel_data = load_panel_data()
    result.input_files.append("combined_components_of_change.csv")

    # Record parameters
    result.parameters = {
        "nd_observations": len(nd_data),
        "nd_time_period": f"{nd_data['year'].min()}-{nd_data['year'].max()}",
        "panel_states": len(panel_data["state"].unique()),
        "panel_years": len(panel_data["year"].unique()),
        "n_lags": 1,
        "minnesota_prior": {
            "lambda1": 0.1,
            "lambda2": 0.5,
            "lambda3": 1.0,
        },
    }

    # =========================================================================
    # 1. MINNESOTA PRIOR CONSTRUCTION
    # =========================================================================
    print("\n" + "=" * 60)
    print("1. Minnesota Prior Construction")
    print("=" * 60)

    var_cols = ["nd_intl_migration", "us_intl_migration"]

    # Estimate AR(1) variances for scaling
    sigma_sq = estimate_ar1_variances(nd_data, var_cols)
    print(f"  AR(1) residual variances: {dict(zip(var_cols, sigma_sq))}")

    # Construct prior
    prior = construct_minnesota_prior(
        n_vars=2,
        n_lags=1,
        sigma_estimates=sigma_sq,
        lambda1=0.1,
        lambda2=0.5,
        lambda3=1.0,
        include_constant=True,
        variable_names=var_cols,
    )

    prior_summary = summarize_prior(prior)
    result.results["minnesota_prior"] = prior_summary

    print("\nPrior specification:")
    print(
        f"  Hyperparameters: lambda1={prior.hyperparameters['lambda1']}, "
        f"lambda2={prior.hyperparameters['lambda2']}, lambda3={prior.hyperparameters['lambda3']}"
    )
    print(f"  {prior_summary['shrinkage_intensity']['interpretation']}")

    # Plot prior
    plot_prior_specification(prior, FIGURES_DIR / "module_B4_minnesota_prior.png")

    # =========================================================================
    # 2. BAYESIAN VAR ESTIMATION
    # =========================================================================
    print("\n" + "=" * 60)
    print("2. Bayesian VAR Estimation")
    print("=" * 60)

    # Estimate Bayesian VAR (uses conjugate fallback if PyMC unavailable)
    bvar_result = estimate_bayesian_var(
        data=nd_data,
        var_cols=var_cols,
        n_lags=1,
        lambda1=0.1,
        lambda2=0.5,
        lambda3=1.0,
        use_pymc=PYMC_AVAILABLE,
        n_samples=1000,  # Reduced for faster execution
        n_tune=500,
        n_chains=2,
    )

    result.results["bayesian_var"] = bvar_result.to_dict()

    print(f"\nEstimation method: {bvar_result.method}")
    print(f"Observations used: {bvar_result.n_obs}")

    print("\nPosterior mean coefficients:")
    for var in var_cols:
        print(f"\n  Equation: {var}")
        for coef, val in bvar_result.coefficients[var].items():
            se = bvar_result.coef_std[var][coef]
            ci = bvar_result.credible_intervals[var][coef]
            print(
                f"    {coef}: {val:.4f} (SD={se:.4f}, 90% CI=[{ci[0]:.4f}, {ci[1]:.4f}])"
            )

    if bvar_result.diagnostics:
        print("\nDiagnostics:")
        for key, val in bvar_result.diagnostics.items():
            if not isinstance(val, dict):
                print(f"  {key}: {val}")

    # =========================================================================
    # 3. PANEL VAR ESTIMATION
    # =========================================================================
    print("\n" + "=" * 60)
    print("3. Panel VAR Estimation (50-State Data)")
    print("=" * 60)

    # Panel VAR with two-way fixed effects
    panel_result = estimate_panel_var(
        df=panel_data,
        entity_col="state",
        time_col="year",
        target_var="intl_migration",
        n_lags=1,
        method="panel_fe",
        focal_entity="North Dakota",
        nd_interaction=True,
    )

    result.results["panel_var"] = panel_result.to_dict()

    print(f"\nPanel VAR method: {panel_result.method}")
    print(f"Entities: {panel_result.n_entities}, Periods: {panel_result.n_periods}")
    print(f"Total observations: {panel_result.n_obs}")
    print(f"R-squared: {panel_result.r_squared:.4f}")

    print("\nPooled AR coefficients:")
    for coef, val in panel_result.coefficients.items():
        se = panel_result.coef_std.get(coef, np.nan)
        print(f"  {coef}: {val:.4f} (SE={se:.4f})")

    if panel_result.nd_coefficients:
        print("\nND-specific effects:")
        for key, val in panel_result.nd_coefficients.items():
            if isinstance(val, (int, float)):
                print(f"  {key}: {val:.4f}")

    if "nd_interaction_f_test" in panel_result.diagnostics:
        f_test = panel_result.diagnostics["nd_interaction_f_test"]
        if "f_statistic" in f_test:
            print("\nND interaction F-test:")
            print(f"  F-statistic: {f_test['f_statistic']:.2f}")
            print(f"  p-value: {f_test['p_value']:.4f}")
            print(f"  Significant at 5%: {f_test['significant_at_05']}")

    # Plot panel VAR effects
    plot_panel_var_effects(
        panel_result, FIGURES_DIR / "module_B4_panel_var_effects.png"
    )

    # Mean group estimator for comparison
    print("\nMean Group Estimator:")
    mg_result = estimate_panel_var(
        df=panel_data,
        entity_col="state",
        time_col="year",
        target_var="intl_migration",
        n_lags=1,
        method="mean_group",
        focal_entity="North Dakota",
    )

    print(f"  Mean AR(1) coefficient: {mg_result.coefficients.get('L1', 'N/A')}")
    if mg_result.nd_coefficients and "ND_L1" in mg_result.nd_coefficients:
        print(f"  ND AR(1) coefficient: {mg_result.nd_coefficients['ND_L1']:.4f}")
        if "deviation_from_mean" in mg_result.nd_coefficients:
            dev = mg_result.nd_coefficients["deviation_from_mean"].get("L1", 0)
            print(f"  ND deviation from mean: {dev:.4f}")

    result.results["panel_var_mean_group"] = mg_result.to_dict()

    # =========================================================================
    # 4. MODEL COMPARISON
    # =========================================================================
    print("\n" + "=" * 60)
    print("4. Model Comparison: Classical vs Bayesian")
    print("=" * 60)

    comparison = compare_var_models(
        data=nd_data,
        var_cols=var_cols,
        bayesian_result=bvar_result,
        n_lags=1,
        panel_result=panel_result,
    )

    result.results["model_comparison"] = comparison.to_dict()

    print("\nCoefficient comparison:")
    if "summary" in comparison.coefficient_comparison:
        summary = comparison.coefficient_comparison["summary"]
        print(f"  Mean absolute difference: {summary['mean_absolute_difference']:.4f}")
        print(f"  Sign agreement rate: {summary['sign_agreement_rate']*100:.1f}%")

    print("\nUncertainty comparison:")
    if "summary" in comparison.uncertainty_comparison:
        uncert = comparison.uncertainty_comparison["summary"]
        print(f"  Mean SE ratio (Bayesian/Classical): {uncert['mean_se_ratio']:.2f}")
        print(
            f"  Bayesian tighter in {uncert['bayesian_tighter_pct']:.0f}% of coefficients"
        )

    print("\nForecast comparison:")
    if "rmse_improvement_pct" in comparison.forecast_comparison:
        print(
            f"  RMSE improvement (Bayesian): {comparison.forecast_comparison['rmse_improvement_pct']:.1f}%"
        )
    if (
        "classical" in comparison.forecast_comparison
        and "rmse" in comparison.forecast_comparison["classical"]
    ):
        print(
            f"  Classical RMSE: {comparison.forecast_comparison['classical']['rmse']:.2f}"
        )
    if (
        "bayesian" in comparison.forecast_comparison
        and "rmse" in comparison.forecast_comparison["bayesian"]
    ):
        print(
            f"  Bayesian RMSE: {comparison.forecast_comparison['bayesian']['rmse']:.2f}"
        )

    print("\n" + "-" * 40)
    print("RECOMMENDATION:")
    print("-" * 40)
    rec = comparison.recommendation
    print(f"  {rec['recommendation']}")
    print(f"  {rec['summary']}")
    print("\n  Reasons:")
    for reason in rec["reasons"]:
        print(f"    - {reason}")

    # Create comparison figures
    plot_coefficient_comparison(
        comparison, var_cols, FIGURES_DIR / "module_B4_coefficient_comparison.png"
    )
    plot_uncertainty_comparison(
        comparison, var_cols, FIGURES_DIR / "module_B4_uncertainty_comparison.png"
    )

    # =========================================================================
    # 5. SUMMARY AND DIAGNOSTICS
    # =========================================================================
    print("\n" + "=" * 60)
    print("5. Summary")
    print("=" * 60)

    result.diagnostics.update(
        {
            "bayesian_method_used": bvar_result.method,
            "panel_method_used": panel_result.method,
            "nd_in_panel": panel_result.diagnostics.get("nd_in_sample", False),
            "figures_generated": [
                "module_B4_minnesota_prior.png",
                "module_B4_panel_var_effects.png",
                "module_B4_coefficient_comparison.png",
                "module_B4_uncertainty_comparison.png",
            ],
        }
    )

    # Warnings
    if comparison.recommendation["recommendation"] == "CLASSICAL_PREFERRED":
        result.warnings.append(
            "Bayesian VAR did not improve forecast accuracy. "
            "Minnesota prior may be too informative for this application."
        )

    if not panel_result.diagnostics.get("nd_in_sample", True):
        result.warnings.append("North Dakota not found in panel data.")

    # Next steps
    result.next_steps = [
        "B5: Synthetic control methodology using panel data",
        "B6: Create unit tests for B4 module functions",
        "Consider adjusting Minnesota prior hyperparameters based on comparison",
        "Document Bayesian VAR value assessment in journal article supplement",
    ]

    return result


def main():
    """Main entry point."""
    print("=" * 60)
    print("Module B4: Bayesian and Panel VAR Extensions")
    print(f"Started: {datetime.now(UTC).isoformat()}")
    print("=" * 60)

    try:
        result = run_analysis()
        output_file = result.save("module_B4_bayesian_panel_var.json")

        print("\n" + "=" * 60)
        print("Analysis completed successfully!")
        print(f"Output: {output_file}")

        if result.warnings:
            print(f"\nWarnings ({len(result.warnings)}):")
            for w in result.warnings:
                print(f"  - {w}")

        if result.next_steps:
            print("\nNext Steps:")
            for s in result.next_steps:
                print(f"  - {s}")

        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
