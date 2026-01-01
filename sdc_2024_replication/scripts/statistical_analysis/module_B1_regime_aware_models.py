#!/usr/bin/env python3
"""
Module B1: Regime-Aware Statistical Models
==========================================

Implements regime-aware statistical analysis for the extended time series
(2000-2024) per ADR-020 Option C (Hybrid Approach).

Components:
1. Vintage dummy variables for level shift detection
2. Piecewise trend estimation with regime-specific slopes
3. COVID-19 intervention modeling
4. Heteroskedasticity-robust inference
5. Sensitivity analysis suite

Usage:
    uv run python module_B1_regime_aware_models.py
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # cohort_projections/
RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"
ADR_REPORTS_DIR = (
    PROJECT_ROOT / "docs" / "adr" / "020-reports" / "chatgpt_review_package"
)

# Ensure output directories exist
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Import regime-aware module components
from module_regime_aware import (  # noqa: E402
    create_vintage_dummies,
    estimate_piecewise_trend,
    estimate_covid_effect,
    calculate_counterfactual_2020,
    estimate_regime_variances,
    estimate_with_robust_se,
    run_sensitivity_suite,
    create_sensitivity_table,
)
from module_regime_aware.piecewise_trends import estimate_simple_trend  # noqa: E402


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


def load_extended_data() -> pd.DataFrame:
    """Load the extended n=25 dataset from Phase A artifacts."""
    filepath = ADR_REPORTS_DIR / "agent2_nd_migration_data.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"Extended data file not found: {filepath}")

    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} observations from {filepath.name}")
    return df


def plot_piecewise_trends(
    df: pd.DataFrame,
    piecewise_result,
    simple_result: dict,
    output_path: Path,
) -> None:
    """Create visualization of piecewise vs simple trends."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Time series with regime shading
    ax1 = axes[0]
    colors = {"2009": "#e74c3c", "2020": "#3498db", "2024": "#2ecc71"}

    for vintage in df["vintage"].unique():
        subset = df[df["vintage"] == vintage]
        ax1.scatter(
            subset["year"],
            subset["intl_migration"],
            c=colors.get(str(vintage), "gray"),
            s=80,
            label=f"Vintage {vintage}",
            zorder=5,
        )

    # Add vertical lines at regime boundaries
    ax1.axvline(
        x=2010, color="gray", linestyle="--", alpha=0.5, label="Vintage boundary"
    )
    ax1.axvline(x=2020, color="gray", linestyle="--", alpha=0.5)

    ax1.set_xlabel("Year")
    ax1.set_ylabel("International Migration")
    ax1.set_title("North Dakota International Migration by Vintage")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Slope comparison
    ax2 = axes[1]
    slopes = piecewise_result.slopes
    slope_se = piecewise_result.slope_se

    regimes = list(slopes.keys())
    slope_vals = [slopes[r] for r in regimes]
    se_vals = [slope_se[r] for r in regimes]

    ax2.bar(
        regimes,
        slope_vals,
        yerr=[1.96 * se for se in se_vals],
        capsize=5,
        color=["#e74c3c", "#3498db", "#2ecc71"],
        alpha=0.7,
    )

    # Add simple trend line
    ax2.axhline(
        y=simple_result["slope"],
        color="black",
        linestyle="--",
        label=f"Simple trend: {simple_result['slope']:.1f}",
    )

    ax2.set_xlabel("Regime")
    ax2.set_ylabel("Trend Slope (migrants/year)")
    ax2.set_title("Regime-Specific Trend Slopes\n(with 95% CI)")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_covid_intervention(
    df: pd.DataFrame,
    covid_result,
    counterfactual: dict,
    output_path: Path,
) -> None:
    """Create visualization of COVID intervention effect."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot actual data
    ax.plot(
        df["year"],
        df["intl_migration"],
        "o-",
        color="#3498db",
        markersize=8,
        label="Actual",
    )

    # Highlight 2020
    covid_year = df[df["year"] == 2020]
    if not covid_year.empty:
        ax.scatter(
            covid_year["year"],
            covid_year["intl_migration"],
            s=200,
            color="#e74c3c",
            zorder=10,
            label="COVID year (2020)",
        )

    # Add counterfactual point
    if "counterfactual_2020" in counterfactual:
        ax.scatter(
            2020,
            counterfactual["counterfactual_2020"],
            s=200,
            marker="x",
            color="#2ecc71",
            linewidths=3,
            zorder=10,
            label=f"Counterfactual: {counterfactual['counterfactual_2020']:.0f}",
        )

        # Draw arrow showing impact
        ax.annotate(
            "",
            xy=(2020, covid_year["intl_migration"].values[0]),
            xytext=(2020, counterfactual["counterfactual_2020"]),
            arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=2),
        )

    ax.set_xlabel("Year")
    ax.set_ylabel("International Migration")
    ax.set_title(
        "COVID-19 Impact on North Dakota International Migration\n"
        f"Effect: {covid_result.covid_effect:.0f} (p={covid_result.covid_effect_pvalue:.3f})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_robust_se_comparison(
    se_comparison,
    output_path: Path,
) -> None:
    """Create visualization comparing standard error estimates."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract SE for trend coefficient
    methods = ["OLS", "HC0", "HC1", "HC2", "HC3", "HAC"]
    se_data = se_comparison.to_dict()["se_comparison"]

    # Get SE for 't' (trend)
    if "t" in se_data["OLS"]:
        se_vals = [se_data[m].get("t", np.nan) for m in methods]
    else:
        # Use first non-const coefficient
        first_var = [k for k in se_data["OLS"].keys() if k != "const"][0]
        se_vals = [se_data[m].get(first_var, np.nan) for m in methods]

    colors = ["#95a5a6", "#3498db", "#3498db", "#3498db", "#e74c3c", "#2ecc71"]
    bars = ax.bar(methods, se_vals, color=colors, alpha=0.7)

    ax.set_xlabel("Standard Error Estimator")
    ax.set_ylabel("Standard Error of Trend Coefficient")
    ax.set_title(
        "Comparison of Standard Error Estimators\n"
        "(OLS may underestimate; HC3/HAC are robust)"
    )
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar, val in zip(bars, se_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_sensitivity_summary(
    sensitivity_result,
    output_path: Path,
) -> None:
    """Create visualization of sensitivity analysis results."""
    fig, ax = plt.subplots(figsize=(12, 6))

    specs = sensitivity_result.specifications
    names = list(specs.keys())
    slopes = [specs[n].trend_slope for n in names]
    ses = [specs[n].trend_se for n in names]
    significant = [specs[n].trend_pvalue < 0.05 for n in names]

    # Color by significance
    colors = ["#2ecc71" if s else "#e74c3c" for s in significant]

    y_pos = range(len(names))
    ax.barh(
        y_pos,
        slopes,
        xerr=[1.96 * se for se in ses],
        capsize=3,
        color=colors,
        alpha=0.7,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([specs[n].description for n in names])
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)

    ax.set_xlabel("Trend Slope (migrants/year)")
    ax.set_title(
        "Sensitivity Analysis: Trend Estimates Across Specifications\n"
        "(Green = significant at 5%, Red = not significant)"
    )
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def run_analysis() -> ModuleResult:
    """
    Main analysis function implementing all B1 components.

    Returns:
        ModuleResult object with all findings
    """
    result = ModuleResult(
        module_id="B1",
        analysis_name="regime_aware_models",
    )

    # Load data
    df = load_extended_data()
    result.input_files.append("agent2_nd_migration_data.csv")

    # Record parameters
    result.parameters = {
        "data_source": "Phase A extended series (2000-2024)",
        "n_observations": len(df),
        "y_variable": "intl_migration",
        "cov_type": "HAC",
        "maxlags": 2,
        "vintage_boundaries": [2010, 2020],
    }

    print("\n" + "=" * 60)
    print("1. VINTAGE DUMMIES ANALYSIS")
    print("=" * 60)

    # Add vintage dummies
    df_dummies = create_vintage_dummies(df, "year")
    print("Created vintage dummy variables")
    print(f"  vintage_2010s: {df_dummies['vintage_2010s'].sum()} observations")
    print(f"  vintage_2020s: {df_dummies['vintage_2020s'].sum()} observations")

    result.results["vintage_dummies"] = {
        "n_2000s": int((df_dummies["vintage_code"] == 2009).sum()),
        "n_2010s": int((df_dummies["vintage_code"] == 2020).sum()),
        "n_2020s": int((df_dummies["vintage_code"] == 2024).sum()),
    }

    print("\n" + "=" * 60)
    print("2. PIECEWISE TREND ANALYSIS")
    print("=" * 60)

    # Estimate piecewise trends
    piecewise_result = estimate_piecewise_trend(df, "intl_migration", "year")
    simple_result = estimate_simple_trend(df, "intl_migration", "year")

    print("\nSimple trend (pooled):")
    print(f"  Slope: {simple_result['slope']:.1f} migrants/year")
    print(f"  SE: {simple_result['slope_se']:.1f}")
    print(f"  p-value: {simple_result['slope_pvalue']:.4f}")

    print("\nPiecewise trends:")
    for regime, slope in piecewise_result.slopes.items():
        se = piecewise_result.slope_se[regime]
        print(f"  {regime}: {slope:.1f} (SE: {se:.1f})")

    if piecewise_result.slope_equality_test:
        test = piecewise_result.slope_equality_test
        print("\nSlope equality test:")
        print(f"  F-statistic: {test['f_statistic']:.2f}")
        print(f"  p-value: {test['p_value']:.4f}")
        print(f"  Reject equality: {test['reject_equality_at_05']}")

    result.results["piecewise_trends"] = piecewise_result.to_dict()
    result.results["simple_trend"] = simple_result

    # Create piecewise trends figure
    plot_piecewise_trends(
        df,
        piecewise_result,
        simple_result,
        FIGURES_DIR / "module_B1_piecewise_trends.png",
    )

    print("\n" + "=" * 60)
    print("3. COVID INTERVENTION ANALYSIS")
    print("=" * 60)

    # Estimate COVID effect
    covid_result = estimate_covid_effect(
        df, "intl_migration", "year", intervention_type="pulse", include_trend=True
    )

    print("\nCOVID pulse intervention effect:")
    print(f"  Effect: {covid_result.covid_effect:.0f} migrants")
    print(f"  SE: {covid_result.covid_effect_se:.0f}")
    print(f"  p-value: {covid_result.covid_effect_pvalue:.4f}")
    print(f"  AIC with COVID: {covid_result.aic_with:.1f}")
    print(f"  AIC without COVID: {covid_result.aic_without:.1f}")
    print(f"  Preferred model: {covid_result.preferred_model}")

    # Calculate counterfactual
    counterfactual = calculate_counterfactual_2020(df, "intl_migration", "year")
    if "counterfactual_2020" in counterfactual:
        print("\nCounterfactual analysis (based on 2015-2019 trend):")
        print(f"  Counterfactual 2020: {counterfactual['counterfactual_2020']:.0f}")
        print(f"  Actual 2020: {counterfactual['actual_2020']:.0f}")
        print(
            f"  COVID impact: {counterfactual['covid_impact']:.0f} ({counterfactual['covid_impact_pct']:.1f}%)"
        )

    result.results["covid_intervention"] = covid_result.to_dict()
    result.results["covid_counterfactual"] = counterfactual

    # Create COVID figure
    plot_covid_intervention(
        df,
        covid_result,
        counterfactual,
        FIGURES_DIR / "module_B1_covid_intervention.png",
    )

    print("\n" + "=" * 60)
    print("4. HETEROSKEDASTICITY ANALYSIS")
    print("=" * 60)

    # Estimate regime variances
    variance_result = estimate_regime_variances(df, "intl_migration", "vintage")

    print("\nRegime-specific variances:")
    for regime, var in variance_result.variances.items():
        print(f"  Vintage {regime}: {var:.0f}")
    print(f"\nVariance ratio (max/min): {variance_result.variance_ratio:.1f}")
    print(
        f"Levene test: stat={variance_result.levene_statistic:.2f}, p={variance_result.levene_pvalue:.4f}"
    )
    print(f"Heteroskedastic at 5%: {variance_result.heteroskedastic}")

    result.results["regime_variances"] = variance_result.to_dict()

    # Compare robust SE estimators
    df_with_t = df.copy()
    df_with_t["t"] = df_with_t["year"] - df_with_t["year"].min()
    se_comparison = estimate_with_robust_se(df_with_t, "intl_migration", ["t"])

    print("\nStandard error comparison for trend coefficient:")
    se_dict = se_comparison.to_dict()["se_comparison"]
    for method in ["OLS", "HC3", "HAC"]:
        print(f"  {method}: {se_dict[method].get('t', 'N/A'):.1f}")

    result.results["robust_se_comparison"] = se_comparison.to_dict()

    # Create SE comparison figure
    plot_robust_se_comparison(
        se_comparison, FIGURES_DIR / "module_B1_robust_se_comparison.png"
    )

    print("\n" + "=" * 60)
    print("5. SENSITIVITY ANALYSIS")
    print("=" * 60)

    # Run sensitivity suite
    sensitivity_result = run_sensitivity_suite(df, "intl_migration", "year")

    print(
        f"\nSensitivity analysis across {len(sensitivity_result.specifications)} specifications:"
    )
    sensitivity_table = create_sensitivity_table(sensitivity_result)
    print(sensitivity_table.to_string(index=False))

    print("\nComparison summary:")
    summary = sensitivity_result.comparison_summary
    print(
        f"  Slope range: [{summary['slope_range']['min']:.1f}, {summary['slope_range']['max']:.1f}]"
    )
    print(f"  Sign consistent: {summary['sign_consistent']}")
    print(f"  Significance consistent: {summary['significance_consistent']}")

    print(f"\nRobustness assessment: {sensitivity_result.robustness_assessment}")

    result.results["sensitivity_analysis"] = sensitivity_result.to_dict()

    # Create sensitivity figure
    plot_sensitivity_summary(
        sensitivity_result, FIGURES_DIR / "module_B1_sensitivity_summary.png"
    )

    # Save sensitivity table as CSV
    sensitivity_table.to_csv(
        RESULTS_DIR / "module_B1_sensitivity_summary.csv", index=False
    )
    print(f"Saved: {RESULTS_DIR / 'module_B1_sensitivity_summary.csv'}")

    # Diagnostics
    result.diagnostics = {
        "data_years": f"{df['year'].min()}-{df['year'].max()}",
        "negative_value_year": 2003,
        "negative_value": int(df[df["year"] == 2003]["intl_migration"].values[0]),
        "covid_year_value": int(df[df["year"] == 2020]["intl_migration"].values[0]),
        "max_value_year": int(df.loc[df["intl_migration"].idxmax(), "year"]),
        "max_value": int(df["intl_migration"].max()),
    }

    # Warnings
    if variance_result.heteroskedastic:
        result.warnings.append(
            f"Significant heteroskedasticity detected (Levene p={variance_result.levene_pvalue:.4f}). "
            "Using HAC/HC3 standard errors is recommended."
        )

    if df[df["year"] == 2003]["intl_migration"].values[0] < 0:
        result.warnings.append(
            "2003 has negative international migration (-545). "
            "Log transformations will not work; analysis uses levels."
        )

    # Next steps
    result.next_steps = [
        "B2: Apply regime-aware specifications to 50-state panel",
        "B3: Incorporate sensitivity results into journal article",
        "B4: Consider Bayesian VAR with Minnesota prior for small-n inference",
        "B6: Create unit tests for regime-aware module functions",
    ]

    return result


def main():
    """Main entry point."""
    print("=" * 60)
    print("Module B1: Regime-Aware Statistical Models")
    print(f"Started: {datetime.now(UTC).isoformat()}")
    print("=" * 60)

    try:
        result = run_analysis()
        output_file = result.save("module_B1_regime_aware_models.json")
        print("\n" + "=" * 60)
        print("Analysis completed successfully!")
        print(f"Output: {output_file}")

        if result.warnings:
            print("\nWarnings:")
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
