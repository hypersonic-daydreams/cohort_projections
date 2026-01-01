"""
Sensitivity Analysis Suite
==========================

Runs coordinated sensitivity analyses across multiple data specifications
as recommended by external review (ChatGPT 5.2 Pro).

Per ADR-020 Option C, sensitivity analyses compare:
1. n=15 baseline (2010-2024)
2. n=25 with vintage controls (2000-2024)
3. Excluding 2020 (COVID shock)
4. Post-methodology-change only (2010-2024)
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

from .vintage_dummies import create_vintage_dummies


@dataclass
class SpecificationResult:
    """Results from a single sensitivity specification."""

    name: str
    description: str
    n_obs: int
    mean: float
    std: float
    min_val: float
    max_val: float
    trend_slope: float
    trend_se: float
    trend_pvalue: float
    r_squared: float
    years: list[int]
    vintage_controls: bool

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "n_obs": self.n_obs,
            "summary_stats": {
                "mean": self.mean,
                "std": self.std,
                "min": self.min_val,
                "max": self.max_val,
            },
            "trend": {
                "slope": self.trend_slope,
                "se": self.trend_se,
                "p_value": self.trend_pvalue,
                "significant_at_05": self.trend_pvalue < 0.05,
            },
            "r_squared": self.r_squared,
            "year_range": f"{min(self.years)}-{max(self.years)}",
            "vintage_controls": self.vintage_controls,
        }


@dataclass
class SensitivityResult:
    """Container for complete sensitivity analysis results."""

    specifications: dict[str, SpecificationResult]
    comparison_summary: dict = field(default_factory=dict)
    robustness_assessment: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "specifications": {k: v.to_dict() for k, v in self.specifications.items()},
            "comparison_summary": self.comparison_summary,
            "robustness_assessment": self.robustness_assessment,
        }


def _estimate_trend(
    df: pd.DataFrame,
    y_col: str,
    year_col: str = "year",
    vintage_controls: bool = False,
    cov_type: str = "HAC",
    maxlags: int = 2,
) -> dict:
    """Estimate linear trend with optional vintage controls."""
    df = df.copy()
    min_year = df[year_col].min()
    df["t"] = df[year_col] - min_year

    if vintage_controls:
        df = create_vintage_dummies(df, year_col)
        X_cols = ["t", "vintage_2010s", "vintage_2020s"]
        X = sm.add_constant(df[X_cols])
    else:
        X = sm.add_constant(df[["t"]])

    y = df[y_col]

    if cov_type == "HAC":
        model = OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    else:
        model = OLS(y, X).fit(cov_type=cov_type)

    return {
        "slope": float(model.params["t"]),
        "se": float(model.bse["t"]),
        "pvalue": float(model.pvalues["t"]),
        "r_squared": float(model.rsquared),
    }


# Standard specification definitions
STANDARD_SPECIFICATIONS = {
    "n15_baseline": {
        "filter": lambda d: d[d["year"] >= 2010],
        "description": "Primary window (2010-2024, n=15)",
        "vintage_controls": False,
    },
    "n25_with_vintage": {
        "filter": lambda d: d,
        "description": "Extended with vintage dummies (2000-2024, n=25)",
        "vintage_controls": True,
    },
    "n24_excl_2020": {
        "filter": lambda d: d[d["year"] != 2020],
        "description": "Excluding COVID year (n=24)",
        "vintage_controls": True,
    },
    "n14_excl_2020_post2010": {
        "filter": lambda d: d[(d["year"] >= 2010) & (d["year"] != 2020)],
        "description": "2010-2024 excluding COVID (n=14)",
        "vintage_controls": False,
    },
    "n10_vintage_2009": {
        "filter": lambda d: d[d["year"] < 2010],
        "description": "Vintage 2009 only (2000-2009, n=10)",
        "vintage_controls": False,
    },
    "n10_vintage_2020": {
        "filter": lambda d: d[(d["year"] >= 2010) & (d["year"] < 2020)],
        "description": "Vintage 2020 only (2010-2019, n=10)",
        "vintage_controls": False,
    },
}


def run_sensitivity_suite(
    df: pd.DataFrame,
    y_col: str = "intl_migration",
    year_col: str = "year",
    specifications: Optional[dict] = None,
    cov_type: str = "HAC",
    maxlags: int = 2,
) -> SensitivityResult:
    """
    Run complete sensitivity analysis suite.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset (should include 2000-2024)
    y_col : str
        Name of the dependent variable column
    year_col : str
        Name of the year column
    specifications : dict, optional
        Custom specifications. If None, uses STANDARD_SPECIFICATIONS
    cov_type : str
        Covariance type for standard errors
    maxlags : int
        Maximum lags for HAC standard errors

    Returns
    -------
    SensitivityResult
        Complete sensitivity analysis results
    """
    if specifications is None:
        specifications = STANDARD_SPECIFICATIONS

    results = {}

    for spec_name, spec in specifications.items():
        # Apply filter
        df_spec = spec["filter"](df.copy())

        if len(df_spec) < 3:
            # Skip specifications with too few observations
            continue

        # Estimate trend
        trend_result = _estimate_trend(
            df_spec,
            y_col,
            year_col,
            vintage_controls=spec.get("vintage_controls", False),
            cov_type=cov_type,
            maxlags=maxlags,
        )

        # Compile specification result
        results[spec_name] = SpecificationResult(
            name=spec_name,
            description=spec["description"],
            n_obs=len(df_spec),
            mean=float(df_spec[y_col].mean()),
            std=float(df_spec[y_col].std()),
            min_val=float(df_spec[y_col].min()),
            max_val=float(df_spec[y_col].max()),
            trend_slope=trend_result["slope"],
            trend_se=trend_result["se"],
            trend_pvalue=trend_result["pvalue"],
            r_squared=trend_result["r_squared"],
            years=sorted(df_spec[year_col].tolist()),
            vintage_controls=spec.get("vintage_controls", False),
        )

    # Generate comparison summary
    comparison = _generate_comparison_summary(results)

    # Generate robustness assessment
    assessment = _assess_robustness(results)

    return SensitivityResult(
        specifications=results,
        comparison_summary=comparison,
        robustness_assessment=assessment,
    )


def _generate_comparison_summary(results: dict[str, SpecificationResult]) -> dict:
    """Generate comparison summary across specifications."""
    if not results:
        return {}

    slopes = {k: v.trend_slope for k, v in results.items()}
    pvalues = {k: v.trend_pvalue for k, v in results.items()}

    # Check sign consistency
    signs = [np.sign(s) for s in slopes.values() if not np.isnan(s)]
    sign_consistent = len(set(signs)) <= 1 if signs else False

    # Check significance consistency
    sig_at_05 = {k: p < 0.05 for k, p in pvalues.items()}
    all_significant = all(sig_at_05.values())
    none_significant = not any(sig_at_05.values())

    return {
        "slope_range": {
            "min": float(min(slopes.values())),
            "max": float(max(slopes.values())),
            "range": float(max(slopes.values()) - min(slopes.values())),
        },
        "sign_consistent": sign_consistent,
        "significance_consistent": all_significant or none_significant,
        "all_significant_at_05": all_significant,
        "none_significant_at_05": none_significant,
        "specification_count": len(results),
    }


def _assess_robustness(results: dict[str, SpecificationResult]) -> str:
    """Generate qualitative robustness assessment."""
    if not results:
        return "Insufficient specifications for assessment"

    slopes = [v.trend_slope for v in results.values() if not np.isnan(v.trend_slope)]
    pvalues = [v.trend_pvalue for v in results.values()]

    if not slopes:
        return "No valid trend estimates"

    # Check criteria
    signs = [np.sign(s) for s in slopes]
    sign_consistent = len(set(signs)) <= 1

    # Coefficient of variation for slopes
    if np.mean(slopes) != 0:
        cv = np.std(slopes) / abs(np.mean(slopes))
    else:
        cv = np.inf

    sig_at_05 = [p < 0.05 for p in pvalues]
    sig_consistent = all(sig_at_05) or not any(sig_at_05)

    # Generate assessment
    if sign_consistent and cv < 0.5 and sig_consistent:
        return "ROBUST: Results are consistent across specifications. Sign, magnitude, and significance are stable."
    elif sign_consistent and cv < 1.0:
        return "MODERATELY ROBUST: Sign is consistent but magnitude varies somewhat. Interpret with caution."
    elif sign_consistent:
        return "WEAKLY ROBUST: Sign is consistent but magnitude varies substantially. Results should be interpreted cautiously."
    else:
        return "NOT ROBUST: Results differ in sign across specifications. Conclusions are sensitive to specification choice."


def create_sensitivity_table(result: SensitivityResult) -> pd.DataFrame:
    """
    Create a formatted sensitivity analysis table.

    Parameters
    ----------
    result : SensitivityResult
        Results from run_sensitivity_suite

    Returns
    -------
    pd.DataFrame
        Formatted table for display or export
    """
    rows = []
    for name, spec in result.specifications.items():
        rows.append(
            {
                "Specification": name,
                "Description": spec.description,
                "N": spec.n_obs,
                "Mean": f"{spec.mean:.0f}",
                "Trend Slope": f"{spec.trend_slope:.1f}",
                "SE": f"{spec.trend_se:.1f}",
                "p-value": f"{spec.trend_pvalue:.3f}",
                "Sig.": "*" if spec.trend_pvalue < 0.05 else "",
                "R²": f"{spec.r_squared:.3f}",
            }
        )

    return pd.DataFrame(rows)
