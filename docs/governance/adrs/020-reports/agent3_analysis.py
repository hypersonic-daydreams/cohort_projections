# mypy: ignore-errors
#!/usr/bin/env python3
"""
Agent 3: Cross-Vintage Comparability Assessment
ADR-019 Investigation

This script performs comprehensive analysis to determine whether the three Census Bureau
PEP vintages (2009, 2020, 2024) measure the same underlying construct for North Dakota
international migration.

.. deprecated:: 2026-01-01
    This is a **legacy Phase A research script** from the ADR-019/020 investigation.
    It was used for one-time exploratory analysis and is retained for reproducibility
    and audit purposes only. This script is NOT production code and should NOT be
    modified or extended.

    The analysis outputs from this script have been incorporated into the final
    ADR-020 decision. For current methodology, see:
    - sdc_2024_replication/scripts/statistical_analysis/module_B1_regime_aware_models.py
    - sdc_2024_replication/scripts/statistical_analysis/module_B2_multistate_placebo.py

Status: DEPRECATED / ARCHIVED
Linting: Exempted from strict linting (see pyproject.toml per-file-ignores)
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# Configuration
DATA_DIR = Path("/home/nhaarstad/workspace/demography/cohort_projections/data")
OUTPUT_DIR = Path("/home/nhaarstad/workspace/demography/cohort_projections/docs/adr/020-reports")

# State FIPS codes
STATE_FIPS = {
    "ND": 38,  # North Dakota
    "SD": 46,  # South Dakota
    "MT": 30,  # Montana
    "WY": 56,  # Wyoming
    "US": 0,  # United States total
}

# Vintage periods
VINTAGE_PERIODS = {
    "2009": (2000, 2009),
    "2020": (2010, 2019),
    "2024": (2020, 2024),
}


def load_pep_data() -> pd.DataFrame:
    """Load PEP state migration components data."""
    filepath = DATA_DIR / "processed/immigration/state_migration_components_2000_2024.csv"
    df = pd.read_csv(filepath)
    return df


def load_acs_foreign_born() -> pd.DataFrame:
    """Load ACS B05006 foreign-born data and extract total foreign-born by state."""
    acs_dir = DATA_DIR / "raw/immigration/census_foreign_born"
    all_years_file = acs_dir / "b05006_states_all_years.csv"

    if all_years_file.exists():
        df = pd.read_csv(all_years_file)
        # B05006_001E is total foreign-born population
        # Extract just what we need
        cols_to_keep = ["B05006_001E", "B05006_001M", "state", "year", "NAME"]
        available_cols = [c for c in cols_to_keep if c in df.columns]
        df = df[available_cols].copy()
        df.rename(
            columns={"B05006_001E": "foreign_born", "B05006_001M": "foreign_born_moe"}, inplace=True
        )
        df["state"] = pd.to_numeric(df["state"], errors="coerce")
        return df
    return pd.DataFrame()


def filter_state_data(df: pd.DataFrame, state_fips: int) -> pd.DataFrame:
    """Filter data for a specific state."""
    # Handle different column names for state identifier
    if "STATE" in df.columns:
        return df[df["STATE"] == state_fips].copy()
    elif "state" in df.columns:
        return df[df["state"] == state_fips].copy()
    return df


def get_us_data(df: pd.DataFrame) -> pd.DataFrame:
    """Get US-level data (SUMLEV=10)."""
    return df[df["SUMLEV"] == 10].copy()


def calculate_correlation_with_ci(x: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> dict:
    """Calculate Pearson correlation with confidence interval using Fisher's z-transform."""
    # Remove any NaN pairs
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]

    n = len(x_clean)
    if n < 3:
        return {
            "correlation": np.nan,
            "p_value": np.nan,
            "lower": np.nan,
            "upper": np.nan,
            "n": n,
        }

    r, p_value = stats.pearsonr(x_clean, y_clean)

    # Fisher z-transform for confidence interval
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)

    z_lower = z - z_crit * se
    z_upper = z + z_crit * se

    r_lower = np.tanh(z_lower)
    r_upper = np.tanh(z_upper)

    return {
        "correlation": r,
        "p_value": p_value,
        "lower": r_lower,
        "upper": r_upper,
        "n": n,
    }


def analyze_nd_us_correlation_by_vintage(pep_df: pd.DataFrame) -> pd.DataFrame:
    """Correlate ND international migration with US total by vintage period."""
    results = []

    # Get ND data (STATE=38) and US data (SUMLEV=10)
    nd_data = filter_state_data(pep_df[pep_df["SUMLEV"] == 40], STATE_FIPS["ND"])
    us_data = get_us_data(pep_df)

    for vintage, (start_year, end_year) in VINTAGE_PERIODS.items():
        # Filter by vintage
        nd_vintage = nd_data[nd_data["vintage"] == int(vintage)]
        us_vintage = us_data[us_data["vintage"] == int(vintage)]

        # Merge on year
        merged = pd.merge(nd_vintage, us_vintage, on="year", suffixes=("_nd", "_us"))
        merged = merged[(merged["year"] >= start_year) & (merged["year"] <= end_year)]

        if len(merged) < 3:
            continue

        x = merged["INTERNATIONALMIG_nd"].values
        y = merged["INTERNATIONALMIG_us"].values

        corr_result = calculate_correlation_with_ci(x, y)

        # Calculate ND share of US
        nd_share_mean = (x / y).mean() * 100
        nd_share_std = (x / y).std() * 100

        results.append(
            {
                "indicator": "ND_intl_mig vs US_intl_mig",
                "period": f"{vintage} ({start_year}-{end_year})",
                "n": corr_result["n"],
                "correlation": corr_result["correlation"],
                "correlation_95_lower": corr_result["lower"],
                "correlation_95_upper": corr_result["upper"],
                "p_value": corr_result["p_value"],
                "nd_share_mean_pct": nd_share_mean,
                "nd_share_std_pct": nd_share_std,
                "interpretation": interpret_correlation(corr_result["correlation"]),
            }
        )

    return pd.DataFrame(results)


def interpret_correlation(r: float) -> str:
    """Interpret correlation coefficient magnitude."""
    if np.isnan(r):
        return "insufficient data"
    r_abs = abs(r)
    if r_abs >= 0.9:
        return "very strong"
    elif r_abs >= 0.7:
        return "strong"
    elif r_abs >= 0.5:
        return "moderate"
    elif r_abs >= 0.3:
        return "weak"
    else:
        return "negligible"


def analyze_events_by_vintage(pep_df: pd.DataFrame) -> dict:
    """Analyze response to known events by vintage period."""
    nd_data = filter_state_data(pep_df[pep_df["SUMLEV"] == 40], STATE_FIPS["ND"])
    us_data = get_us_data(pep_df)

    results = {
        "financial_crisis_2008": {},
        "bakken_boom": {},
        "policy_changes_2017": {},
    }

    # Financial Crisis Analysis (2007-2010)
    # Look at year-over-year changes
    crisis_years = [2007, 2008, 2009, 2010]
    nd_crisis = nd_data[nd_data["year"].isin(crisis_years)].sort_values("year")
    us_crisis = us_data[us_data["year"].isin(crisis_years)].sort_values("year")

    if len(nd_crisis) > 0:
        nd_crisis_vals = nd_crisis.groupby("year")["INTERNATIONALMIG"].first()
        us_crisis_vals = us_crisis.groupby("year")["INTERNATIONALMIG"].first()

        results["financial_crisis_2008"] = {
            "nd_values": nd_crisis_vals.to_dict(),
            "us_values": us_crisis_vals.to_dict(),
            "nd_pct_change_2007_2009": (
                (nd_crisis_vals.get(2009, np.nan) - nd_crisis_vals.get(2007, np.nan))
                / nd_crisis_vals.get(2007, np.nan)
                * 100
            )
            if 2007 in nd_crisis_vals and 2009 in nd_crisis_vals
            else np.nan,
            "us_pct_change_2007_2009": (
                (us_crisis_vals.get(2009, np.nan) - us_crisis_vals.get(2007, np.nan))
                / us_crisis_vals.get(2007, np.nan)
                * 100
            )
            if 2007 in us_crisis_vals and 2009 in us_crisis_vals
            else np.nan,
        }

    # Bakken Boom Analysis (2011-2015)
    bakken_years = list(range(2010, 2016))
    nd_bakken = nd_data[
        (nd_data["year"].isin(bakken_years)) & (nd_data["vintage"] == 2020)
    ].sort_values("year")

    if len(nd_bakken) > 0:
        nd_bakken_vals = nd_bakken.groupby("year")["INTERNATIONALMIG"].first()
        nd_domestic = nd_bakken.groupby("year")["DOMESTICMIG"].first()

        results["bakken_boom"] = {
            "nd_international_values": nd_bakken_vals.to_dict(),
            "nd_domestic_values": nd_domestic.to_dict(),
            "nd_intl_growth_2010_2015": (
                (nd_bakken_vals.get(2015, np.nan) - nd_bakken_vals.get(2010, np.nan))
                / nd_bakken_vals.get(2010, np.nan)
                * 100
            )
            if 2010 in nd_bakken_vals and 2015 in nd_bakken_vals
            else np.nan,
            "domestic_to_intl_ratio_mean": (nd_domestic / nd_bakken_vals).mean()
            if len(nd_bakken_vals) > 0
            else np.nan,
        }

    # Policy Changes Analysis (2017-2020)
    policy_years = list(range(2016, 2021))
    nd_policy_2020 = nd_data[
        (nd_data["year"].isin(policy_years)) & (nd_data["vintage"] == 2020)
    ].sort_values("year")
    nd_policy_2024 = nd_data[
        (nd_data["year"].isin(policy_years)) & (nd_data["vintage"] == 2024)
    ].sort_values("year")
    us_policy = us_data[us_data["year"].isin(policy_years)].sort_values("year")

    # Get values from appropriate vintage
    nd_vals = {}
    us_vals = {}
    for year in policy_years:
        if year <= 2019:
            row = nd_policy_2020[nd_policy_2020["year"] == year]
        else:
            row = nd_policy_2024[nd_policy_2024["year"] == year]
        if len(row) > 0:
            nd_vals[year] = row["INTERNATIONALMIG"].values[0]

        us_row = us_policy[us_policy["year"] == year]
        if len(us_row) > 0:
            us_vals[year] = us_row["INTERNATIONALMIG"].values[0]

    if nd_vals:
        results["policy_changes_2017"] = {
            "nd_values": nd_vals,
            "us_values": us_vals,
            "nd_pct_change_2016_2019": (
                (nd_vals.get(2019, np.nan) - nd_vals.get(2016, np.nan))
                / nd_vals.get(2016, np.nan)
                * 100
            )
            if 2016 in nd_vals and 2019 in nd_vals
            else np.nan,
        }

    return results


def compare_with_acs_data(pep_df: pd.DataFrame, acs_df: pd.DataFrame) -> pd.DataFrame:
    """Compare PEP international migration with ACS foreign-born changes."""
    if acs_df.empty:
        return pd.DataFrame()

    # Get ND data
    nd_pep = filter_state_data(pep_df[pep_df["SUMLEV"] == 40], STATE_FIPS["ND"])
    nd_acs = filter_state_data(acs_df, STATE_FIPS["ND"])

    if nd_acs.empty:
        return pd.DataFrame()

    # Calculate year-over-year changes in foreign-born
    nd_acs = nd_acs.sort_values("year")
    nd_acs["foreign_born_change"] = nd_acs["foreign_born"].diff()

    # Merge with PEP data
    results = []
    for _, pep_row in nd_pep.iterrows():
        year = pep_row["year"]
        acs_row = nd_acs[nd_acs["year"] == year]

        if len(acs_row) > 0:
            results.append(
                {
                    "year": year,
                    "pep_intl_migration": pep_row["INTERNATIONALMIG"],
                    "pep_vintage": pep_row["vintage"],
                    "acs_foreign_born": acs_row["foreign_born"].values[0],
                    "acs_foreign_born_change": acs_row["foreign_born_change"].values[0]
                    if not pd.isna(acs_row["foreign_born_change"].values[0])
                    else np.nan,
                }
            )

    df = pd.DataFrame(results)

    # Add source agreement indicator
    if len(df) > 0 and "acs_foreign_born_change" in df.columns:
        # Agreement if both show same direction
        df["source_agreement"] = "NA"
        for idx, row in df.iterrows():
            if pd.notna(row["pep_intl_migration"]) and pd.notna(row["acs_foreign_born_change"]):
                pep_sign = np.sign(row["pep_intl_migration"])
                acs_sign = np.sign(row["acs_foreign_born_change"])
                if pep_sign == acs_sign:
                    df.loc[idx, "source_agreement"] = "agree"
                else:
                    df.loc[idx, "source_agreement"] = "disagree"

    return df


def analyze_state_comparison(pep_df: pd.DataFrame) -> pd.DataFrame:
    """Compare ND with similar small states (SD, MT, WY)."""
    comparison_states = ["ND", "SD", "MT", "WY"]
    results = []

    for state in comparison_states:
        state_fips = STATE_FIPS[state]
        state_data = filter_state_data(pep_df[pep_df["SUMLEV"] == 40], state_fips)

        if state_data.empty:
            continue

        for vintage in [2009, 2020]:
            vintage_data = state_data[state_data["vintage"] == vintage]

            if vintage_data.empty:
                continue

            # Get transition year value (last year of previous vintage)
            if vintage == 2009:
                transition_year = 2009
            else:
                transition_year = 2019

            trans_row = vintage_data[vintage_data["year"] == transition_year]

            # Calculate metrics
            mean_intl = vintage_data["INTERNATIONALMIG"].mean()
            std_intl = vintage_data["INTERNATIONALMIG"].std()
            cv = std_intl / mean_intl if mean_intl != 0 else np.nan

            results.append(
                {
                    "state": state,
                    "metric": "mean_intl_mig",
                    f"vintage_{vintage}_value": mean_intl,
                    "cv": cv,
                    "n_years": len(vintage_data),
                }
            )

    # Pivot to compare across vintages
    df = pd.DataFrame(results)
    if df.empty:
        return df

    # Group by state and compare 2009 vs 2020 vintages
    comparison = []
    for state in comparison_states:
        state_df = df[df["state"] == state]
        v2009 = state_df[state_df["vintage_2009_value"].notna()]["vintage_2009_value"].values
        v2020 = state_df[state_df["vintage_2020_value"].notna()]["vintage_2020_value"].values

        if len(v2009) > 0 and len(v2020) > 0:
            v2009_val = v2009[0] if len(v2009) > 0 else np.nan
            v2020_val = v2020[0] if len(v2020) > 0 else np.nan

            if v2009_val and v2020_val:
                change = (v2020_val - v2009_val) / abs(v2009_val) * 100
            else:
                change = np.nan

            comparison.append(
                {
                    "state": state,
                    "metric": "mean_intl_mig",
                    "vintage_2009_value": v2009_val,
                    "vintage_2020_value": v2020_val,
                    "transition_change_pct": change,
                }
            )

    comparison_df = pd.DataFrame(comparison)

    # Determine if similar to ND
    if len(comparison_df) > 0 and "ND" in comparison_df["state"].values:
        nd_change = comparison_df[comparison_df["state"] == "ND"]["transition_change_pct"].values[0]
        comparison_df["similar_to_nd"] = comparison_df["transition_change_pct"].apply(
            lambda x: abs(x - nd_change) < 50 if pd.notna(x) and pd.notna(nd_change) else False
        )

    return comparison_df


def analyze_nd_share_stability(pep_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze stability of ND's share of US international migration across vintages."""
    nd_data = filter_state_data(pep_df[pep_df["SUMLEV"] == 40], STATE_FIPS["ND"])
    us_data = get_us_data(pep_df)

    results = []

    for vintage in [2009, 2020, 2024]:
        nd_v = nd_data[nd_data["vintage"] == vintage].copy()
        us_v = us_data[us_data["vintage"] == vintage].copy()

        merged = pd.merge(nd_v, us_v, on="year", suffixes=("_nd", "_us"))

        if len(merged) > 0:
            merged["nd_share"] = merged["INTERNATIONALMIG_nd"] / merged["INTERNATIONALMIG_us"] * 100

            for _, row in merged.iterrows():
                results.append(
                    {
                        "year": row["year"],
                        "vintage": vintage,
                        "nd_intl_mig": row["INTERNATIONALMIG_nd"],
                        "us_intl_mig": row["INTERNATIONALMIG_us"],
                        "nd_share_pct": row["nd_share"],
                    }
                )

    return pd.DataFrame(results)


def run_coherence_checks(pep_df: pd.DataFrame) -> dict:
    """Run internal consistency checks on the data."""
    nd_data = filter_state_data(pep_df[pep_df["SUMLEV"] == 40], STATE_FIPS["ND"])

    results = {
        "netmig_consistency": {},
        "intl_domestic_relationship": {},
        "vintage_overlap_comparison": {},
    }

    # Check 1: NETMIG = INTERNATIONALMIG + DOMESTICMIG
    if all(col in nd_data.columns for col in ["NETMIG", "INTERNATIONALMIG", "DOMESTICMIG"]):
        nd_data["calculated_netmig"] = nd_data["INTERNATIONALMIG"] + nd_data["DOMESTICMIG"]
        nd_data["residual"] = nd_data["NETMIG"] - nd_data["calculated_netmig"]

        results["netmig_consistency"] = {
            "mean_residual": nd_data["residual"].mean(),
            "max_residual": nd_data["residual"].abs().max(),
            "all_consistent": (nd_data["residual"].abs() < 1).all(),
        }

    # Check 2: Relationship between international and domestic migration
    for vintage in [2009, 2020, 2024]:
        v_data = nd_data[nd_data["vintage"] == vintage]
        if len(v_data) > 2:
            corr_result = calculate_correlation_with_ci(
                v_data["INTERNATIONALMIG"].values, v_data["DOMESTICMIG"].values
            )
            results["intl_domestic_relationship"][str(vintage)] = {
                "correlation": corr_result["correlation"],
                "p_value": corr_result["p_value"],
                "n": corr_result["n"],
            }

    # Check 3: Compare overlapping years between vintages (if any)
    # 2009 vintage: 2000-2009, 2020 vintage: 2010-2019
    # No direct overlap, but check transition consistency
    v2009_last = nd_data[(nd_data["vintage"] == 2009) & (nd_data["year"] == 2009)]
    v2020_first = nd_data[(nd_data["vintage"] == 2020) & (nd_data["year"] == 2010)]

    if len(v2009_last) > 0 and len(v2020_first) > 0:
        results["vintage_overlap_comparison"]["2009_to_2020_transition"] = {
            "last_2009_vintage_value": v2009_last["INTERNATIONALMIG"].values[0],
            "first_2020_vintage_value": v2020_first["INTERNATIONALMIG"].values[0],
            "year_gap": 1,
            "absolute_change": v2020_first["INTERNATIONALMIG"].values[0]
            - v2009_last["INTERNATIONALMIG"].values[0],
        }

    v2020_last = nd_data[(nd_data["vintage"] == 2020) & (nd_data["year"] == 2019)]
    v2024_first = nd_data[(nd_data["vintage"] == 2024) & (nd_data["year"] == 2020)]

    if len(v2020_last) > 0 and len(v2024_first) > 0:
        results["vintage_overlap_comparison"]["2020_to_2024_transition"] = {
            "last_2020_vintage_value": v2020_last["INTERNATIONALMIG"].values[0],
            "first_2024_vintage_value": v2024_first["INTERNATIONALMIG"].values[0],
            "year_gap": 1,
            "absolute_change": v2024_first["INTERNATIONALMIG"].values[0]
            - v2020_last["INTERNATIONALMIG"].values[0],
        }

    return results


def analyze_detailed_state_patterns(pep_df: pd.DataFrame) -> pd.DataFrame:
    """Detailed analysis of state patterns at vintage transitions."""
    states_to_analyze = ["ND", "SD", "MT", "WY"]
    results = []

    for state in states_to_analyze:
        state_fips = STATE_FIPS[state]
        state_data = filter_state_data(pep_df[pep_df["SUMLEV"] == 40], state_fips)
        us_data = get_us_data(pep_df)

        for vintage in [2009, 2020, 2024]:
            v_data = state_data[state_data["vintage"] == vintage]
            v_us = us_data[us_data["vintage"] == vintage]

            if len(v_data) == 0:
                continue

            merged = pd.merge(v_data, v_us, on="year", suffixes=("_state", "_us"))

            if len(merged) > 0:
                merged["state_share"] = (
                    merged["INTERNATIONALMIG_state"] / merged["INTERNATIONALMIG_us"] * 100
                )

                results.append(
                    {
                        "state": state,
                        "vintage": vintage,
                        "n_years": len(v_data),
                        "mean_intl_mig": v_data["INTERNATIONALMIG"].mean(),
                        "std_intl_mig": v_data["INTERNATIONALMIG"].std(),
                        "min_intl_mig": v_data["INTERNATIONALMIG"].min(),
                        "max_intl_mig": v_data["INTERNATIONALMIG"].max(),
                        "mean_state_share_pct": merged["state_share"].mean(),
                        "std_state_share_pct": merged["state_share"].std(),
                    }
                )

    return pd.DataFrame(results)


def create_findings_summary(
    corr_results: pd.DataFrame,
    event_analysis: dict,
    coherence_results: dict,
    state_comparison: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> dict:
    """Create machine-readable findings summary."""

    findings = {
        "agent_id": 3,
        "report_date": "2026-01-01",
        "title": "Cross-Vintage Comparability Assessment",
        "findings": [],
        "overall_assessment": {},
    }

    # Finding 1: Correlation stability
    if len(corr_results) > 0:
        corr_stable = True
        corr_values = corr_results["correlation"].dropna().values
        if len(corr_values) > 1:
            corr_range = max(corr_values) - min(corr_values)
            corr_stable = corr_range < 0.3  # Consider stable if range < 0.3

        findings["findings"].append(
            {
                "id": "F3.1",
                "title": "ND-US Correlation Stability",
                "conclusion": "stable" if corr_stable else "unstable",
                "evidence": {
                    "correlation_values": corr_results["correlation"].tolist()
                    if len(corr_results) > 0
                    else [],
                    "correlation_range": float(corr_range) if len(corr_values) > 1 else None,
                },
                "confidence": "medium",
            }
        )

    # Finding 2: Event response coherence
    event_coherent = True
    if event_analysis.get("bakken_boom", {}).get("nd_intl_growth_2010_2015"):
        bakken_growth = event_analysis["bakken_boom"]["nd_intl_growth_2010_2015"]
        # Expect positive growth during Bakken boom
        event_coherent = bakken_growth > 0

    findings["findings"].append(
        {
            "id": "F3.2",
            "title": "Response to Known Events",
            "conclusion": "coherent" if event_coherent else "incoherent",
            "evidence": event_analysis,
            "confidence": "medium",
        }
    )

    # Finding 3: State comparison
    nd_unique = True
    if len(state_comparison) > 0 and "similar_to_nd" in state_comparison.columns:
        similar_states = state_comparison[state_comparison["similar_to_nd"] == True]
        nd_unique = len(similar_states) <= 1  # Only ND itself

    findings["findings"].append(
        {
            "id": "F3.3",
            "title": "Cross-State Pattern Comparison",
            "conclusion": "nd_unique" if nd_unique else "pattern_shared",
            "evidence": {
                "n_similar_states": len(state_comparison[state_comparison["similar_to_nd"] == True])
                if "similar_to_nd" in state_comparison.columns
                else 0,
            },
            "confidence": "low",
        }
    )

    # Finding 4: Internal coherence
    coherence_ok = True
    if coherence_results.get("netmig_consistency", {}).get("all_consistent") is not None:
        coherence_ok = coherence_results["netmig_consistency"]["all_consistent"]

    findings["findings"].append(
        {
            "id": "F3.4",
            "title": "Internal Consistency",
            "conclusion": "consistent" if coherence_ok else "inconsistent",
            "evidence": coherence_results,
            "confidence": "high",
        }
    )

    # Overall assessment
    n_positive = sum(
        [
            corr_stable if len(corr_results) > 0 else False,
            event_coherent,
            not nd_unique,  # Shared patterns suggest measurement consistency
            coherence_ok,
        ]
    )

    if n_positive >= 3:
        overall = "likely_comparable"
    elif n_positive >= 2:
        overall = "possibly_comparable"
    else:
        overall = "likely_not_comparable"

    findings["overall_assessment"] = {
        "conclusion": overall,
        "n_positive_indicators": n_positive,
        "n_total_indicators": 4,
        "recommendation": "Proceed with caution"
        if overall in ["likely_comparable", "possibly_comparable"]
        else "Do not extend",
        "confidence_level": "Medium",
        "key_uncertainties": [
            "Limited sample size within vintage periods",
            "ACS data comparison limited to stock vs flow measures",
            "No direct validation against administrative immigration data",
        ],
    }

    return findings


def create_sources_json() -> dict:
    """Create sources and references JSON."""
    return {
        "agent_id": 3,
        "report_date": "2026-01-01",
        "primary_sources": [
            {
                "name": "Census Bureau PEP State Migration Components",
                "path": str(
                    DATA_DIR / "processed/immigration/state_migration_components_2000_2024.csv"
                ),
                "coverage": "2000-2024",
                "vintages": ["2009", "2020", "2024"],
            },
        ],
        "secondary_sources": [
            {
                "name": "ACS Table B05006 Foreign-Born by Place of Birth",
                "path": str(DATA_DIR / "raw/immigration/census_foreign_born/"),
                "coverage": "2009-2023",
                "notes": "Stock measure (total foreign-born), not flow measure",
            },
            {
                "name": "DHS Legal Permanent Resident Data",
                "path": str(DATA_DIR / "raw/immigration/dhs_lpr/"),
                "coverage": "2007-2023",
                "notes": "Files exist but not processed in this analysis",
            },
        ],
        "methodology_references": [
            {
                "title": "Fisher z-transformation for correlation confidence intervals",
                "citation": "Fisher, R. A. (1921). On the 'probable error' of a coefficient of correlation deduced from a small sample.",
            },
        ],
    }


def main():
    """Main analysis function."""
    print("Agent 3: Cross-Vintage Comparability Assessment")
    print("=" * 60)

    # Load data
    print("\n1. Loading data...")
    pep_df = load_pep_data()
    print(f"   PEP data: {len(pep_df)} rows")

    acs_df = load_acs_foreign_born()
    print(f"   ACS data: {len(acs_df)} rows")

    # Task 1: Correlation with external indicators
    print("\n2. Analyzing ND-US correlations by vintage...")
    corr_results = analyze_nd_us_correlation_by_vintage(pep_df)
    print(corr_results.to_string())

    # Task 2: Event analysis
    print("\n3. Analyzing response to known events...")
    event_analysis = analyze_events_by_vintage(pep_df)
    for event, data in event_analysis.items():
        print(f"   {event}: {list(data.keys())}")

    # Task 3: ACS comparison
    print("\n4. Comparing with ACS foreign-born data...")
    validation_df = compare_with_acs_data(pep_df, acs_df)
    if len(validation_df) > 0:
        print(f"   {len(validation_df)} years with both PEP and ACS data")
    else:
        print("   No ACS comparison possible")

    # Task 4: State comparison
    print("\n5. Comparing patterns across similar states...")
    state_comparison = analyze_state_comparison(pep_df)
    print(state_comparison.to_string() if len(state_comparison) > 0 else "   No comparison data")

    detailed_state = analyze_detailed_state_patterns(pep_df)

    # Task 5: Coherence checks
    print("\n6. Running coherence checks...")
    coherence_results = run_coherence_checks(pep_df)
    print(f"   NETMIG consistency: {coherence_results.get('netmig_consistency', {})}")

    # ND share stability
    print("\n7. Analyzing ND share stability...")
    nd_share_df = analyze_nd_share_stability(pep_df)

    # Save outputs
    print("\n8. Saving outputs...")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save correlation results
    corr_output = corr_results.copy()
    corr_output.to_csv(OUTPUT_DIR / "agent3_external_correlations.csv", index=False)
    print("   Saved: agent3_external_correlations.csv")

    # Save state comparison
    if len(state_comparison) > 0:
        state_comparison.to_csv(OUTPUT_DIR / "agent3_state_comparison.csv", index=False)
        print("   Saved: agent3_state_comparison.csv")

    # Save validation data
    if len(validation_df) > 0:
        validation_df.to_csv(OUTPUT_DIR / "agent3_validation_data.csv", index=False)
        print("   Saved: agent3_validation_data.csv")

    # Save coherence checks
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        elif pd.isna(obj):
            return None
        return obj

    coherence_clean = convert_numpy(coherence_results)
    with open(OUTPUT_DIR / "agent3_coherence_checks.json", "w") as f:
        json.dump(coherence_clean, f, indent=2)
    print("   Saved: agent3_coherence_checks.json")

    # Create and save findings summary
    findings = create_findings_summary(
        corr_results, event_analysis, coherence_results, state_comparison, validation_df
    )
    findings_clean = convert_numpy(findings)
    with open(OUTPUT_DIR / "agent3_findings_summary.json", "w") as f:
        json.dump(findings_clean, f, indent=2)
    print("   Saved: agent3_findings_summary.json")

    # Save sources
    sources = create_sources_json()
    with open(OUTPUT_DIR / "agent3_sources.json", "w") as f:
        json.dump(sources, f, indent=2)
    print("   Saved: agent3_sources.json")

    # Save detailed state patterns
    detailed_state.to_csv(OUTPUT_DIR / "agent3_detailed_state_patterns.csv", index=False)
    print("   Saved: agent3_detailed_state_patterns.csv")

    # Save ND share data
    nd_share_df.to_csv(OUTPUT_DIR / "agent3_nd_share_analysis.csv", index=False)
    print("   Saved: agent3_nd_share_analysis.csv")

    print("\n" + "=" * 60)
    print("Analysis complete!")

    return {
        "corr_results": corr_results,
        "event_analysis": event_analysis,
        "validation_df": validation_df,
        "state_comparison": state_comparison,
        "coherence_results": coherence_results,
        "findings": findings,
        "detailed_state": detailed_state,
        "nd_share_df": nd_share_df,
    }


if __name__ == "__main__":
    results = main()
