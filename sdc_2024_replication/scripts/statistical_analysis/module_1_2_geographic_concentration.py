#!/usr/bin/env python3
"""
Module 1.2: Geographic Concentration Analysis
=============================================

Analyzes geographic concentration of immigration using:
1. Herfindahl-Hirschman Index (HHI) by origin country for ND
2. Location Quotients for ND vs US by origin country
3. Concentration trends over time (2009-2023)
4. Identification of top origin countries for ND

Author: Claude Code (Module 1.2)
Date: 2024-12-28
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # cohort_projections/
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "immigration" / "analysis"
RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"

# Ensure output directories exist
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Plot settings
plt.style.use("seaborn-v0_8-whitegrid")
FIGURE_DPI = 300


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
        self.decisions: list[str] = []
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
            "decisions": self.decisions,
            "warnings": self.warnings,
            "next_steps": self.next_steps,
        }

    def save(self, filename: str) -> Path:
        """Save results to JSON file."""
        output_path = RESULTS_DIR / filename
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=_json_serializer)
        print(f"Results saved to: {output_path}")
        return output_path


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for numpy types."""
    if isinstance(obj, np.integer | np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    return str(obj)


def load_data(filename: str) -> pd.DataFrame:
    """Load data file from analysis directory."""
    filepath = DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    if filepath.suffix == ".csv":
        return pd.read_csv(filepath)
    elif filepath.suffix == ".parquet":
        return pd.read_parquet(filepath)
    else:
        raise ValueError(f"Unsupported file type: {filepath.suffix}")


def calculate_hhi(shares: pd.Series) -> float:
    """
    Calculate Herfindahl-Hirschman Index.

    HHI = sum(share_i^2) * 10000

    Interpretation:
    - < 1500: Unconcentrated
    - 1500-2500: Moderately concentrated
    - > 2500: Highly concentrated

    Args:
        shares: Series of market shares (should sum to 1)

    Returns:
        HHI value (0-10000 scale)
    """
    # Normalize shares to sum to 1
    normalized = shares / shares.sum()
    # HHI on 10000 scale (multiply squared shares by 10000)
    return float((normalized**2).sum() * 10000)


def interpret_hhi(hhi: float) -> str:
    """Interpret HHI value."""
    if hhi < 1500:
        return "unconcentrated"
    elif hhi < 2500:
        return "moderately concentrated"
    else:
        return "highly concentrated"


def calculate_location_quotient(
    nd_country: float, nd_total: float, us_country: float, us_total: float
) -> float:
    """
    Calculate Location Quotient.

    LQ = (ND_country / ND_total) / (US_country / US_total)

    Interpretation:
    - LQ > 1: ND has disproportionately high share from that country
    - LQ < 1: ND has disproportionately low share from that country
    - LQ = 1: ND matches national average
    """
    if us_total == 0 or us_country == 0 or nd_total == 0:
        return np.nan

    nd_share = nd_country / nd_total
    us_share = us_country / us_total

    return nd_share / us_share


def compute_descriptive_stats(data: pd.Series, name: str) -> dict:
    """Compute SPSS-style descriptive statistics."""
    clean_data = data.dropna()

    if len(clean_data) == 0:
        return {"error": "No valid data"}

    stats_dict = {
        "variable": name,
        "n": int(len(clean_data)),
        "missing": int(len(data) - len(clean_data)),
        "mean": float(clean_data.mean()),
        "std_error": float(clean_data.std() / np.sqrt(len(clean_data))),
        "std_deviation": float(clean_data.std()),
        "variance": float(clean_data.var()),
        "minimum": float(clean_data.min()),
        "maximum": float(clean_data.max()),
        "range": float(clean_data.max() - clean_data.min()),
        "percentile_25": float(clean_data.quantile(0.25)),
        "median": float(clean_data.median()),
        "percentile_75": float(clean_data.quantile(0.75)),
        "iqr": float(clean_data.quantile(0.75) - clean_data.quantile(0.25)),
        "skewness": float(clean_data.skew()) if len(clean_data) > 2 else np.nan,
        "kurtosis": float(clean_data.kurtosis()) if len(clean_data) > 3 else np.nan,
    }

    # Add 95% CI for mean
    if len(clean_data) > 1:
        ci = stats.t.interval(
            0.95,
            len(clean_data) - 1,
            loc=clean_data.mean(),
            scale=stats.sem(clean_data),
        )
        stats_dict["ci_95_lower"] = float(ci[0])
        stats_dict["ci_95_upper"] = float(ci[1])

    return stats_dict


def run_hhi_analysis(result: ModuleResult) -> dict:
    """
    Calculate HHI for ND immigration by origin country.

    Uses DHS LPR data for FY2023.
    """
    print("\n--- HHI Analysis ---")

    # Load DHS LPR data
    df = load_data("dhs_lpr_by_state_country.parquet")
    result.input_files.append("dhs_lpr_by_state_country.parquet")

    # Filter to ND and countries only (exclude regions)
    nd_lpr = df[(df["state"] == "North Dakota") & (~df["is_region"])].copy()

    result.decisions.append(
        f"Filtered to ND countries only (is_region=False): {len(nd_lpr)} countries"
    )

    # Remove zero/NaN counts
    nd_lpr = nd_lpr[nd_lpr["lpr_count"] > 0]
    result.decisions.append(
        f"Removed countries with zero LPR count: {len(nd_lpr)} countries remaining"
    )

    # Calculate shares
    total_lpr = nd_lpr["lpr_count"].sum()
    nd_lpr["share"] = nd_lpr["lpr_count"] / total_lpr

    # Calculate HHI
    hhi = calculate_hhi(nd_lpr["lpr_count"])
    interpretation = interpret_hhi(hhi)

    # Get top contributors to HHI
    nd_lpr["hhi_contribution"] = (nd_lpr["share"] ** 2) * 10000
    top_contributors = nd_lpr.nlargest(10, "hhi_contribution")[
        ["region_country_of_birth", "lpr_count", "share", "hhi_contribution"]
    ].to_dict("records")

    # Regional breakdown
    df_regions = df[(df["state"] == "North Dakota") & (df["is_region"])].copy()
    df_regions = df_regions[
        ~df_regions["region_country_of_birth"].isin(["Total", "Unknown"])
    ]
    df_regions = df_regions[df_regions["lpr_count"] > 0]

    region_total = df_regions["lpr_count"].sum()
    df_regions["share"] = df_regions["lpr_count"] / region_total

    regional_hhi = calculate_hhi(df_regions["lpr_count"])

    regional_breakdown = (
        df_regions[["region_country_of_birth", "lpr_count", "share"]]
        .sort_values("lpr_count", ascending=False)
        .to_dict("records")
    )

    hhi_results = {
        "fiscal_year": 2023,
        "total_lpr_count": int(total_lpr),
        "n_countries": len(nd_lpr),
        "hhi_value": hhi,
        "hhi_interpretation": interpretation,
        "hhi_thresholds": {
            "unconcentrated": "<1500",
            "moderately_concentrated": "1500-2500",
            "highly_concentrated": ">2500",
        },
        "top_10_hhi_contributors": top_contributors,
        "regional_breakdown": {
            "hhi_value": regional_hhi,
            "hhi_interpretation": interpret_hhi(regional_hhi),
            "regions": regional_breakdown,
        },
        "descriptive_stats": compute_descriptive_stats(
            nd_lpr["share"], "country_share"
        ),
    }

    print(f"  Total ND LPR (FY2023): {total_lpr:,.0f}")
    print(f"  Number of origin countries: {len(nd_lpr)}")
    print(f"  HHI (country-level): {hhi:.2f} ({interpretation})")
    print(f"  HHI (region-level): {regional_hhi:.2f} ({interpret_hhi(regional_hhi)})")

    return hhi_results


def run_location_quotient_analysis(result: ModuleResult) -> dict:
    """
    Calculate Location Quotients for ND vs US by origin country.

    Uses ACS foreign-born data.
    """
    print("\n--- Location Quotient Analysis ---")

    # Load ND share data (already has ND and national totals)
    df = load_data("acs_foreign_born_nd_share.parquet")
    result.input_files.append("acs_foreign_born_nd_share.parquet")

    # Focus on 2023 data at country level
    df_2023 = df[(df["year"] == 2023) & (df["level"] == "country")].copy()

    result.decisions.append(
        f"Filtered to 2023 country-level data: {len(df_2023)} countries"
    )

    # Get ND and national totals
    df_total = df[(df["year"] == 2023) & (df["level"] == "total")]
    nd_total = df_total["nd_foreign_born"].sum()

    # Load full ACS data to get national totals
    acs_df = load_data("acs_foreign_born_by_state_origin.parquet")
    result.input_files.append("acs_foreign_born_by_state_origin.parquet")

    # Calculate national total foreign-born (sum all states' total)
    national_2023 = acs_df[(acs_df["year"] == 2023) & (acs_df["level"] == "total")][
        "foreign_born_pop"
    ].sum()

    # Get national totals by country
    national_by_country = (
        acs_df[(acs_df["year"] == 2023) & (acs_df["level"] == "country")]
        .groupby("country")["foreign_born_pop"]
        .sum()
        .reset_index()
    )
    national_by_country.rename(columns={"foreign_born_pop": "us_total"}, inplace=True)

    # Merge ND data with national data
    df_2023 = df_2023.merge(national_by_country, on="country", how="left")

    # Calculate LQ
    df_2023["nd_share"] = df_2023["nd_foreign_born"] / nd_total
    df_2023["us_share"] = df_2023["us_total"] / national_2023
    df_2023["location_quotient"] = df_2023["nd_share"] / df_2023["us_share"]

    # Handle infinities and NaN
    df_2023.loc[~np.isfinite(df_2023["location_quotient"]), "location_quotient"] = (
        np.nan
    )

    # Filter out zeros and NaN for analysis
    valid_lq = df_2023[
        (df_2023["nd_foreign_born"] > 0)
        & (df_2023["us_total"] > 0)
        & (df_2023["location_quotient"].notna())
    ].copy()

    result.decisions.append(
        f"Countries with valid LQ data: {len(valid_lq)} (excluded {len(df_2023) - len(valid_lq)} with zero/missing values)"
    )

    # Top countries by LQ (overrepresented in ND)
    top_lq = valid_lq.nlargest(20, "location_quotient")[
        [
            "country",
            "nd_foreign_born",
            "us_total",
            "nd_share",
            "us_share",
            "location_quotient",
        ]
    ].to_dict("records")

    # Bottom countries by LQ (underrepresented in ND)
    bottom_lq = valid_lq.nsmallest(10, "location_quotient")[
        [
            "country",
            "nd_foreign_born",
            "us_total",
            "nd_share",
            "us_share",
            "location_quotient",
        ]
    ].to_dict("records")

    # Countries with LQ > 1 (overrepresented)
    overrepresented = valid_lq[valid_lq["location_quotient"] > 1]
    underrepresented = valid_lq[valid_lq["location_quotient"] < 1]

    # By region analysis
    df_region = df[(df["year"] == 2023) & (df["level"] == "region")].copy()

    # Get national totals by region
    national_by_region = (
        acs_df[(acs_df["year"] == 2023) & (acs_df["level"] == "region")]
        .groupby("region")["foreign_born_pop"]
        .sum()
        .reset_index()
    )
    national_by_region.rename(columns={"foreign_born_pop": "us_total"}, inplace=True)

    df_region = df_region.merge(national_by_region, on="region", how="left")
    df_region["nd_share"] = df_region["nd_foreign_born"] / nd_total
    df_region["us_share"] = df_region["us_total"] / national_2023
    df_region["location_quotient"] = df_region["nd_share"] / df_region["us_share"]

    regional_lq = (
        df_region[
            [
                "region",
                "nd_foreign_born",
                "us_total",
                "nd_share",
                "us_share",
                "location_quotient",
            ]
        ]
        .dropna()
        .sort_values("location_quotient", ascending=False)
        .to_dict("records")
    )

    lq_results = {
        "year": 2023,
        "nd_total_foreign_born": int(nd_total),
        "national_total_foreign_born": int(national_2023),
        "nd_share_of_national": float(nd_total / national_2023),
        "n_countries_analyzed": len(valid_lq),
        "descriptive_stats": compute_descriptive_stats(
            valid_lq["location_quotient"], "location_quotient"
        ),
        "top_20_lq_countries": top_lq,
        "bottom_10_lq_countries": bottom_lq,
        "overrepresentation_summary": {
            "n_countries_lq_gt_1": len(overrepresented),
            "n_countries_lq_lt_1": len(underrepresented),
            "pct_overrepresented": float(len(overrepresented) / len(valid_lq) * 100),
        },
        "regional_location_quotients": regional_lq,
    }

    # Store for figure generation
    lq_results["_data_for_figures"] = {"valid_lq_df": valid_lq.to_dict("records")}

    print(f"  ND total foreign-born (2023): {nd_total:,.0f}")
    print(f"  National total foreign-born (2023): {national_2023:,.0f}")
    print(f"  ND share of national: {nd_total/national_2023:.4%}")
    print(f"  Countries analyzed: {len(valid_lq)}")
    print(
        f"  Countries with LQ > 1: {len(overrepresented)} ({len(overrepresented)/len(valid_lq)*100:.1f}%)"
    )

    return lq_results


def run_concentration_trends(result: ModuleResult) -> dict:
    """
    Analyze concentration trends over time (2009-2023).
    """
    print("\n--- Concentration Trends Analysis ---")

    # Load ACS data
    acs_df = load_data("acs_foreign_born_by_state_origin.parquet")

    # Filter to ND data at country level
    nd_df = acs_df[
        (acs_df["state_name"] == "North Dakota") & (acs_df["level"] == "country")
    ].copy()

    # Calculate HHI by year
    hhi_by_year = []

    for year in sorted(nd_df["year"].unique()):
        year_data = nd_df[nd_df["year"] == year]
        year_data = year_data[year_data["foreign_born_pop"] > 0]

        if len(year_data) > 0:
            hhi = calculate_hhi(year_data["foreign_born_pop"])
            total = year_data["foreign_born_pop"].sum()
            n_countries = len(year_data)

            hhi_by_year.append(
                {
                    "year": int(year),
                    "hhi": hhi,
                    "interpretation": interpret_hhi(hhi),
                    "total_foreign_born": int(total),
                    "n_countries": n_countries,
                }
            )

    hhi_df = pd.DataFrame(hhi_by_year)

    # Time trend regression
    if len(hhi_df) > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            hhi_df["year"], hhi_df["hhi"]
        )

        trend_analysis = {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_value**2),
            "p_value": float(p_value),
            "std_error": float(std_err),
            "interpretation": "increasing" if slope > 0 else "decreasing",
            "significant": bool(p_value < 0.05),
        }

        # 95% CI for slope
        n = len(hhi_df)
        t_crit = stats.t.ppf(0.975, n - 2)
        slope_ci = (slope - t_crit * std_err, slope + t_crit * std_err)
        trend_analysis["slope_ci_95"] = [float(slope_ci[0]), float(slope_ci[1])]
    else:
        trend_analysis = {"error": "Insufficient data for trend analysis"}

    # Regional HHI trends
    nd_region = acs_df[
        (acs_df["state_name"] == "North Dakota") & (acs_df["level"] == "region")
    ].copy()

    region_hhi_by_year = []
    for year in sorted(nd_region["year"].unique()):
        year_data = nd_region[nd_region["year"] == year]
        year_data = year_data[year_data["foreign_born_pop"] > 0]

        if len(year_data) > 0:
            hhi = calculate_hhi(year_data["foreign_born_pop"])
            region_hhi_by_year.append(
                {"year": int(year), "hhi": hhi, "interpretation": interpret_hhi(hhi)}
            )

    trends_results = {
        "time_period": f"{hhi_df['year'].min()}-{hhi_df['year'].max()}",
        "n_years": len(hhi_df),
        "country_level_hhi_by_year": hhi_by_year,
        "region_level_hhi_by_year": region_hhi_by_year,
        "trend_analysis": trend_analysis,
        "descriptive_stats": compute_descriptive_stats(hhi_df["hhi"], "hhi_over_time"),
        "change_summary": {
            "start_hhi": float(hhi_df.iloc[0]["hhi"]),
            "end_hhi": float(hhi_df.iloc[-1]["hhi"]),
            "absolute_change": float(hhi_df.iloc[-1]["hhi"] - hhi_df.iloc[0]["hhi"]),
            "percent_change": float(
                (hhi_df.iloc[-1]["hhi"] - hhi_df.iloc[0]["hhi"])
                / hhi_df.iloc[0]["hhi"]
                * 100
            ),
        },
    }

    print(f"  Time period: {trends_results['time_period']}")
    print(f"  HHI start: {trends_results['change_summary']['start_hhi']:.2f}")
    print(f"  HHI end: {trends_results['change_summary']['end_hhi']:.2f}")
    print(
        f"  Change: {trends_results['change_summary']['absolute_change']:.2f} ({trends_results['change_summary']['percent_change']:.1f}%)"
    )
    if "slope" in trend_analysis:
        print(
            f"  Trend: {trend_analysis['interpretation']} (p={trend_analysis['p_value']:.4f})"
        )

    return trends_results


def run_top_origins_analysis(result: ModuleResult) -> dict:
    """
    Identify top origin countries for ND with highest LQs.
    """
    print("\n--- Top Origins Analysis ---")

    # Load ND share data
    df = load_data("acs_foreign_born_nd_share.parquet")

    # Get 2023 country-level data
    df_2023 = df[(df["year"] == 2023) & (df["level"] == "country")].copy()

    # Load full ACS data for national totals
    acs_df = load_data("acs_foreign_born_by_state_origin.parquet")

    # Get ND total
    nd_total = df[(df["year"] == 2023) & (df["level"] == "total")][
        "nd_foreign_born"
    ].sum()

    # Get national totals
    national_2023 = acs_df[(acs_df["year"] == 2023) & (acs_df["level"] == "total")][
        "foreign_born_pop"
    ].sum()

    national_by_country = (
        acs_df[(acs_df["year"] == 2023) & (acs_df["level"] == "country")]
        .groupby("country")["foreign_born_pop"]
        .sum()
        .reset_index()
    )
    national_by_country.rename(columns={"foreign_born_pop": "us_total"}, inplace=True)

    # Merge
    df_2023 = df_2023.merge(national_by_country, on="country", how="left")

    # Calculate metrics
    df_2023["nd_share"] = df_2023["nd_foreign_born"] / nd_total
    df_2023["us_share"] = df_2023["us_total"] / national_2023
    df_2023["location_quotient"] = df_2023["nd_share"] / df_2023["us_share"]

    # Filter valid data
    valid = df_2023[
        (df_2023["nd_foreign_born"] > 0)
        & (df_2023["us_total"] > 0)
        & (df_2023["location_quotient"].notna())
        & np.isfinite(df_2023["location_quotient"])
    ].copy()

    # Top by absolute numbers
    top_absolute = valid.nlargest(15, "nd_foreign_born")[
        ["country", "nd_foreign_born", "nd_share", "location_quotient"]
    ].to_dict("records")

    # Top by LQ (minimum population threshold)
    min_pop = 100  # Require at least 100 foreign-born for meaningful LQ
    result.decisions.append(
        f"Applied minimum population threshold of {min_pop} for top LQ analysis"
    )

    significant_pop = valid[valid["nd_foreign_born"] >= min_pop]
    top_lq = significant_pop.nlargest(15, "location_quotient")[
        ["country", "nd_foreign_born", "nd_share", "location_quotient"]
    ].to_dict("records")

    # Combined ranking (balanced score)
    valid["rank_pop"] = valid["nd_foreign_born"].rank(ascending=False)
    valid["rank_lq"] = valid["location_quotient"].rank(ascending=False)
    valid["combined_score"] = (valid["rank_pop"] + valid["rank_lq"]) / 2

    top_combined = valid.nsmallest(15, "combined_score")[
        [
            "country",
            "nd_foreign_born",
            "nd_share",
            "location_quotient",
            "combined_score",
        ]
    ].to_dict("records")

    # Calculate correlations
    corr_pop_lq = valid["nd_foreign_born"].corr(valid["location_quotient"])
    corr_pop_lq_spearman = stats.spearmanr(
        valid["nd_foreign_born"], valid["location_quotient"]
    )

    origins_results = {
        "year": 2023,
        "n_countries_total": len(valid),
        "n_countries_above_threshold": len(significant_pop),
        "population_threshold": min_pop,
        "top_15_by_population": top_absolute,
        "top_15_by_lq": top_lq,
        "top_15_combined_rank": top_combined,
        "correlation_analysis": {
            "pearson_r": float(corr_pop_lq),
            "spearman_rho": float(corr_pop_lq_spearman.statistic),
            "spearman_p_value": float(corr_pop_lq_spearman.pvalue),
            "interpretation": "weak"
            if abs(corr_pop_lq) < 0.3
            else ("moderate" if abs(corr_pop_lq) < 0.7 else "strong"),
        },
    }

    print(f"  Total countries: {len(valid)}")
    print(f"  Countries above {min_pop} threshold: {len(significant_pop)}")
    print(
        f"  Top country by population: {top_absolute[0]['country']} ({top_absolute[0]['nd_foreign_born']:,.0f})"
    )
    print(
        f"  Top country by LQ: {top_lq[0]['country']} (LQ={top_lq[0]['location_quotient']:.2f})"
    )
    print(f"  Correlation (pop vs LQ): r={corr_pop_lq:.3f}")

    return origins_results


def create_figures(
    hhi_results: dict, lq_results: dict, trends_results: dict, origins_results: dict
):
    """Generate all required figures."""
    print("\n--- Generating Figures ---")

    # Figure 1: HHI by origin regions (heatmap-style)
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    regions = hhi_results["regional_breakdown"]["regions"]
    region_names = [r["region_country_of_birth"] for r in regions]
    region_shares = [r["share"] * 100 for r in regions]

    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(regions)))
    bars = ax1.barh(region_names[::-1], region_shares[::-1], color=colors[::-1])

    ax1.set_xlabel("Share of ND LPR (%)", fontsize=11)
    ax1.set_title(
        f"North Dakota LPR by World Region (FY2023)\nHHI = {hhi_results['regional_breakdown']['hhi_value']:.0f} ({hhi_results['regional_breakdown']['hhi_interpretation']})",
        fontsize=12,
    )

    # Add value labels
    for bar, share in zip(bars, region_shares[::-1], strict=False):
        ax1.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{share:.1f}%",
            va="center",
            fontsize=9,
        )

    ax1.set_xlim(0, max(region_shares) * 1.15)
    plt.tight_layout()

    for fmt in ["png", "pdf"]:
        fig1.savefig(
            FIGURES_DIR / f"module_1_2_concentration_heatmap.{fmt}", dpi=FIGURE_DPI
        )
    plt.close(fig1)
    print("  Saved: module_1_2_concentration_heatmap.png/pdf")

    # Figure 2: Top 20 countries by Location Quotient
    fig2, ax2 = plt.subplots(figsize=(12, 8))

    top_lq = lq_results["top_20_lq_countries"]
    countries = [r["country"][:25] for r in top_lq]  # Truncate long names
    lq_values = [r["location_quotient"] for r in top_lq]
    pop_values = [r["nd_foreign_born"] for r in top_lq]

    # Color by population
    norm_pop = np.array(pop_values) / max(pop_values)
    colors = plt.cm.YlOrRd(norm_pop)

    bars = ax2.barh(countries[::-1], lq_values[::-1], color=colors[::-1])

    ax2.axvline(
        x=1, color="black", linestyle="--", linewidth=1, label="US Average (LQ=1)"
    )
    ax2.set_xlabel("Location Quotient", fontsize=11)
    ax2.set_title(
        "Top 20 Countries by Location Quotient in North Dakota (2023)\nColor intensity = population size",
        fontsize=12,
    )
    ax2.legend(loc="lower right")

    # Add value labels
    for bar, lq in zip(bars, lq_values[::-1], strict=False):
        ax2.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{lq:.2f}",
            va="center",
            fontsize=8,
        )

    plt.tight_layout()

    for fmt in ["png", "pdf"]:
        fig2.savefig(FIGURES_DIR / f"module_1_2_lq_bar_chart.{fmt}", dpi=FIGURE_DPI)
    plt.close(fig2)
    print("  Saved: module_1_2_lq_bar_chart.png/pdf")

    # Figure 3: HHI over time
    fig3, ax3 = plt.subplots(figsize=(10, 6))

    hhi_data = trends_results["country_level_hhi_by_year"]
    years = [d["year"] for d in hhi_data]
    hhi_values = [d["hhi"] for d in hhi_data]

    ax3.plot(
        years, hhi_values, "b-o", linewidth=2, markersize=6, label="Country-level HHI"
    )

    # Add regional HHI
    region_hhi_data = trends_results["region_level_hhi_by_year"]
    region_years = [d["year"] for d in region_hhi_data]
    region_hhi = [d["hhi"] for d in region_hhi_data]
    ax3.plot(
        region_years,
        region_hhi,
        "g--s",
        linewidth=1.5,
        markersize=5,
        alpha=0.7,
        label="Region-level HHI",
    )

    # Add trend line if significant
    if "slope" in trends_results["trend_analysis"]:
        trend = trends_results["trend_analysis"]
        x_line = np.array([min(years), max(years)])
        y_line = trend["intercept"] + trend["slope"] * x_line
        ax3.plot(
            x_line,
            y_line,
            "r--",
            linewidth=1.5,
            alpha=0.5,
            label=f'Trend (slope={trend["slope"]:.1f}/yr, p={trend["p_value"]:.3f})',
        )

    # HHI threshold lines
    ax3.axhline(y=1500, color="orange", linestyle=":", linewidth=1, alpha=0.7)
    ax3.axhline(y=2500, color="red", linestyle=":", linewidth=1, alpha=0.7)
    ax3.text(max(years), 1550, "Moderate (1500)", fontsize=8, color="orange")
    ax3.text(max(years), 2550, "High (2500)", fontsize=8, color="red")

    ax3.set_xlabel("Year", fontsize=11)
    ax3.set_ylabel("Herfindahl-Hirschman Index", fontsize=11)
    ax3.set_title(
        "Immigration Concentration Trends in North Dakota (2009-2023)", fontsize=12
    )
    ax3.legend(loc="best")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    for fmt in ["png", "pdf"]:
        fig3.savefig(
            FIGURES_DIR / f"module_1_2_concentration_trends.{fmt}", dpi=FIGURE_DPI
        )
    plt.close(fig3)
    print("  Saved: module_1_2_concentration_trends.png/pdf")

    # Figure 4: ND's top origin countries
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Top by population
    top_pop = origins_results["top_15_by_population"]
    countries_pop = [r["country"][:20] for r in top_pop]
    pop_values = [r["nd_foreign_born"] for r in top_pop]

    ax4a.barh(countries_pop[::-1], pop_values[::-1], color="steelblue")
    ax4a.set_xlabel("Foreign-Born Population", fontsize=11)
    ax4a.set_title("Top 15 Origin Countries by Population", fontsize=11)

    for _i, (bar, v) in enumerate(zip(ax4a.patches, pop_values[::-1], strict=False)):
        ax4a.text(
            bar.get_width() + 50,
            bar.get_y() + bar.get_height() / 2,
            f"{v:,.0f}",
            va="center",
            fontsize=8,
        )

    # Right: Top by LQ (with min population threshold)
    top_lq_thresh = origins_results["top_15_by_lq"]
    countries_lq = [r["country"][:20] for r in top_lq_thresh]
    lq_vals = [r["location_quotient"] for r in top_lq_thresh]

    ax4b.barh(countries_lq[::-1], lq_vals[::-1], color="forestgreen")
    ax4b.axvline(x=1, color="black", linestyle="--", linewidth=1)
    ax4b.set_xlabel("Location Quotient", fontsize=11)
    ax4b.set_title(
        f"Top 15 by LQ (min pop: {origins_results['population_threshold']})",
        fontsize=11,
    )

    for bar, v in zip(ax4b.patches, lq_vals[::-1], strict=False):
        ax4b.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.2f}",
            va="center",
            fontsize=8,
        )

    fig4.suptitle("North Dakota Top Origin Countries (2023)", fontsize=13, y=1.02)
    plt.tight_layout()

    for fmt in ["png", "pdf"]:
        fig4.savefig(
            FIGURES_DIR / f"module_1_2_top_origins_nd.{fmt}",
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )
    plt.close(fig4)
    print("  Saved: module_1_2_top_origins_nd.png/pdf")


def run_analysis() -> tuple[ModuleResult, ModuleResult]:
    """
    Main analysis function.

    Returns:
        Tuple of (HHI ModuleResult, LQ ModuleResult)
    """
    # Initialize result containers
    hhi_result = ModuleResult(
        module_id="1.2",
        analysis_name="hhi_concentration_analysis",
    )

    lq_result = ModuleResult(
        module_id="1.2",
        analysis_name="location_quotient_analysis",
    )

    # Record parameters
    hhi_result.parameters = {
        "hhi_scale": "0-10000",
        "thresholds": {
            "unconcentrated": "<1500",
            "moderately_concentrated": "1500-2500",
            "highly_concentrated": ">2500",
        },
        "data_source": "DHS LPR data FY2023",
    }

    lq_result.parameters = {
        "reference_geography": "United States",
        "target_geography": "North Dakota",
        "interpretation": "LQ > 1 indicates overrepresentation relative to national average",
        "data_source": "ACS foreign-born data 2009-2023",
    }

    # Run analyses
    hhi_results = run_hhi_analysis(hhi_result)
    lq_results = run_location_quotient_analysis(lq_result)
    trends_results = run_concentration_trends(lq_result)
    origins_results = run_top_origins_analysis(lq_result)

    # Store results
    hhi_result.results = hhi_results
    lq_result.results = {
        "location_quotients": {
            k: v for k, v in lq_results.items() if not k.startswith("_")
        },
        "concentration_trends": trends_results,
        "top_origins": origins_results,
    }

    # Create figures
    create_figures(hhi_results, lq_results, trends_results, origins_results)

    # Add diagnostics
    hhi_result.diagnostics = {
        "data_quality": {
            "total_countries_in_source": hhi_results["n_countries"],
            "countries_with_valid_data": hhi_results["n_countries"],
        }
    }

    lq_result.diagnostics = {
        "data_quality": {
            "years_available": trends_results["n_years"],
            "countries_analyzed": lq_results["n_countries_analyzed"],
            "countries_with_valid_lq": origins_results["n_countries_total"],
        },
        "trend_significance": trends_results["trend_analysis"],
    }

    # Add next steps
    hhi_result.next_steps = [
        "Compare HHI with other states to contextualize ND's concentration",
        "Analyze sub-national (county) concentration patterns",
    ]

    lq_result.next_steps = [
        "Module 1.3: Time series analysis of immigration trends",
        "Investigate drivers of high LQ for specific countries",
        "Cross-reference with refugee resettlement data",
    ]

    return hhi_result, lq_result


def main():
    """Main entry point."""
    print("=" * 60)
    print("Module 1.2: Geographic Concentration Analysis")
    print(f"Started: {datetime.now(UTC).isoformat()}")
    print("=" * 60)

    try:
        hhi_result, lq_result = run_analysis()

        # Save results
        hhi_output = hhi_result.save("module_1_2_hhi_analysis.json")
        lq_output = lq_result.save("module_1_2_location_quotients.json")

        print("\n" + "=" * 60)
        print("Analysis completed successfully!")
        print("=" * 60)
        print("\nOutput files:")
        print(f"  - {hhi_output}")
        print(f"  - {lq_output}")
        print(f"\nFigures saved to: {FIGURES_DIR}")

        if hhi_result.warnings or lq_result.warnings:
            print("\nWarnings:")
            for w in hhi_result.warnings + lq_result.warnings:
                print(f"  - {w}")

        print("\nDecisions made:")
        for d in hhi_result.decisions + lq_result.decisions:
            print(f"  - {d}")

        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
