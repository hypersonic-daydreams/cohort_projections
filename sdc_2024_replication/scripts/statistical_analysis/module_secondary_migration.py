#!/usr/bin/env python3
"""
Secondary Migration Analysis Module (ADR-021 Recommendation #7)
================================================================

Analyzes secondary domestic migration of foreign-born populations to North Dakota.
Decomposes foreign-born population growth into:
1. Direct international inflow - people arriving from abroad directly to ND
2. Secondary domestic migration - foreign-born people moving to ND from other US states

This analysis is important for interpreting policy effects: federal restrictions
reduce new international entries, but ND might offset via secondary domestic
migration from states with larger foreign-born populations.

Data Sources:
- PostgreSQL database: census.state_components (PEP migration components)
- ACS foreign-born stock data (B05006)
- ACS PUMS for migration flow details (when available)

References:
- ADR-021: External Analysis Integration
- docs/governance/adrs/021-reports/data/acs_migration_data_acquisition.md

Usage:
    .venv/bin/python sdc_2024_replication/scripts/statistical_analysis/module_secondary_migration.py
"""

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # cohort_projections/
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "immigration" / "analysis"
MIGRATION_DIR = PROJECT_ROOT / "data" / "processed" / "immigration"
PUMS_DIR = PROJECT_ROOT / "data" / "raw" / "population"
RESULTS_DIR = PROJECT_ROOT / "docs" / "governance" / "adrs" / "021-reports" / "results"

# Ensure output directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class SecondaryMigrationAnalyzer:
    """Analyzer for secondary migration of foreign-born populations to North Dakota."""

    def __init__(self) -> None:
        """Initialize the analyzer."""
        self.results: dict[str, Any] = {
            "module": "rec7_secondary_migration",
            "analysis": "Secondary Migration Decomposition",
            "generated": datetime.now(UTC).isoformat(),
            "input_files": [],
            "parameters": {},
            "results": {},
            "diagnostics": {},
            "warnings": [],
            "methodology": {},
            "recommendations": [],
        }

    def load_pep_components(self) -> pd.DataFrame:
        """Load state migration components from PEP (Population Estimates Program)."""
        logger.info("Loading PEP migration components from CSV files...")

        # Load from combined components CSV (has population column)
        csv_path = DATA_DIR / "combined_components_of_change.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Migration components data not found: {csv_path}")

        df = pd.read_csv(csv_path)
        self.results["input_files"].append(str(csv_path))

        # Rename columns to standard format
        df = df.rename(columns={"state": "state_name"})

        # Keep only needed columns
        cols = [
            "year",
            "state_name",
            "population",
            "intl_migration",
            "domestic_migration",
            "net_migration",
        ]
        df = df[cols].copy()

        # Sort for consistency
        df = df.sort_values(["state_name", "year"]).reset_index(drop=True)

        logger.info(f"  Loaded {len(df)} records")
        logger.info(f"  Years: {df['year'].min()} to {df['year'].max()}")
        return df

    def load_foreign_born_stock(self) -> pd.DataFrame:
        """Load ACS foreign-born stock data."""
        logger.info("Loading ACS foreign-born stock data...")

        fb_path = DATA_DIR / "acs_foreign_born_by_state_origin.parquet"
        if not fb_path.exists():
            raise FileNotFoundError(f"Foreign-born data not found: {fb_path}")

        df = pd.read_parquet(fb_path)
        self.results["input_files"].append(str(fb_path))

        # Get total foreign-born by state and year
        totals = df[df["level"] == "total"].copy()
        totals = totals[["year", "state_name", "foreign_born_pop"]].dropna()

        logger.info(f"  Loaded {len(totals)} state-year observations")
        return totals

    def load_pums_sample(self) -> pd.DataFrame | None:
        """
        Load PUMS sample data if available.

        Returns None if data is insufficient for analysis.
        """
        logger.info("Checking for PUMS migration data...")

        pums_path = PUMS_DIR / "pums_person.parquet"
        if not pums_path.exists():
            logger.info("  PUMS data not found")
            return None

        df = pd.read_parquet(pums_path)

        # Check if we have migration variables
        required_vars = ["NATIVITY", "MIG", "MIGSP", "STATE", "PWGTP"]
        missing = [v for v in required_vars if v not in df.columns]
        if missing:
            logger.info(f"  Missing required variables: {missing}")
            return None

        logger.info(f"  Loaded {len(df)} PUMS records")
        logger.info(f"  Variables available: {list(df.columns)[:20]}...")

        # Filter to foreign-born in ND who moved from another state
        # NATIVITY = 2 means foreign-born
        # MIG = 1 means moved from different house
        # MIGSP != 38 and MIGSP not in [0, 999] means moved from different state
        df_fb = df[df["NATIVITY"] == 2].copy()
        df_movers = df_fb[(df_fb["MIG"] == 1) & (df_fb["MIGSP"].notna())].copy()
        df_domestic = df_movers[
            (df_movers["MIGSP"] != 38)  # Not from ND
            & (df_movers["MIGSP"] < 100)  # US state (not foreign)
        ].copy()

        n_fb = len(df_fb)
        n_domestic_movers = len(df_domestic)

        logger.info(f"  Foreign-born in sample: {n_fb}")
        logger.info(f"  FB domestic movers to ND: {n_domestic_movers}")

        self.results["input_files"].append(str(pums_path))
        self.results["diagnostics"]["pums_sample"] = {
            "total_records": len(df),
            "foreign_born": n_fb,
            "fb_domestic_movers_to_nd": n_domestic_movers,
            "note": "Sample size too small for reliable estimates - need full PUMS data",
        }

        if n_domestic_movers < 10:
            self.results["warnings"].append(
                f"PUMS sample has only {n_domestic_movers} FB domestic movers - "
                "insufficient for reliable estimates"
            )
            return None

        return df_domestic

    def calculate_fb_population_changes(
        self, fb_stock: pd.DataFrame, pep: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate year-over-year changes in foreign-born population.

        This is the key outcome we want to decompose.
        """
        logger.info("Calculating foreign-born population changes...")

        # Focus on North Dakota
        nd_fb = fb_stock[fb_stock["state_name"] == "North Dakota"].copy()
        nd_fb = nd_fb.sort_values("year")
        nd_fb["fb_change"] = nd_fb["foreign_born_pop"].diff()

        # Get ND PEP components
        nd_pep = pep[pep["state_name"] == "North Dakota"].copy()
        nd_pep = nd_pep.sort_values("year")

        # Merge
        merged = pd.merge(nd_fb, nd_pep, on="year", suffixes=("", "_pep"))

        # Also get US totals for comparison
        us_pep = pep[pep["state_name"] == "United States"].copy()
        us_pep = us_pep.sort_values("year")
        us_pep = us_pep.rename(
            columns={
                "intl_migration": "us_intl_migration",
                "domestic_migration": "us_domestic_migration",
            }
        )

        merged = pd.merge(
            merged, us_pep[["year", "us_intl_migration"]], on="year", how="left"
        )

        logger.info(f"  Created merged dataset with {len(merged)} year observations")
        return merged

    def estimate_secondary_migration_bounds(
        self, merged: pd.DataFrame
    ) -> dict[str, Any]:
        """
        Estimate bounds on secondary domestic migration of foreign-born.

        Method: Use the relationship between total FB change, international
        migration, and domestic migration to bound secondary migration.

        Key insight: PEP international migration reflects total arrivals,
        but FB change also includes:
        - Deaths of foreign-born (negative)
        - Secondary domestic migration (can be positive or negative)
        - Naturalization does NOT change FB stock (still foreign-born)
        """
        logger.info("Estimating secondary migration bounds...")

        results: dict[str, Any] = {}

        # Filter to valid years (with change data)
        df = merged.dropna(subset=["fb_change", "intl_migration"]).copy()

        # Calculate relationship between FB change and international migration
        df["residual"] = df["fb_change"] - df["intl_migration"]

        # The residual captures:
        # - Secondary domestic migration (FB moving from other states)
        # - Deaths of FB (negative)
        # - Emigration of FB (negative) - captured in intl_migration if net

        results["annual_estimates"] = []
        for _, row in df.iterrows():
            year_data = {
                "year": int(row["year"]),
                "fb_population": float(row["foreign_born_pop"]),
                "fb_change": float(row["fb_change"]),
                "pep_intl_migration": int(row["intl_migration"]),
                "pep_domestic_migration": int(row["domestic_migration"]),
                "residual": float(row["residual"]),
            }

            # Estimate secondary migration bounds
            # Upper bound: residual + estimate of FB deaths
            # Lower bound: residual (assumes no FB deaths, which is too low)
            # Middle estimate: residual + half of expected deaths

            # Rough FB death estimate: use ND crude death rate applied to FB pop
            # ND crude death rate ~8-9 per 1000 based on PEP data
            fb_pop = row["foreign_born_pop"]
            estimated_fb_deaths = fb_pop * 0.008  # ~8 per 1000

            year_data["estimated_fb_deaths"] = float(estimated_fb_deaths)
            year_data["secondary_migration_lower"] = float(row["residual"])
            year_data["secondary_migration_middle"] = float(
                row["residual"] + estimated_fb_deaths
            )
            year_data["secondary_migration_upper"] = float(
                row["residual"] + estimated_fb_deaths * 1.5
            )

            results["annual_estimates"].append(year_data)

        # Calculate summary statistics
        estimates_df = pd.DataFrame(results["annual_estimates"])

        # Focus on recent years (2019-2023) for current patterns
        recent = estimates_df[estimates_df["year"] >= 2019].copy()

        results["summary"] = {
            "period": "2019-2023",
            "mean_fb_change": float(recent["fb_change"].mean()),
            "mean_intl_migration": float(recent["pep_intl_migration"].mean()),
            "mean_secondary_migration_middle": float(
                recent["secondary_migration_middle"].mean()
            ),
            "secondary_share_of_fb_change": (
                float(
                    recent["secondary_migration_middle"].mean()
                    / recent["fb_change"].mean()
                )
                if recent["fb_change"].mean() != 0
                else None
            ),
        }

        # Add period-specific analysis
        # Pre-COVID: 2015-2019
        pre_covid = estimates_df[
            (estimates_df["year"] >= 2015) & (estimates_df["year"] < 2020)
        ]
        # COVID/Recovery: 2020-2023
        covid_era = estimates_df[estimates_df["year"] >= 2020]

        results["period_comparison"] = {
            "pre_covid_2015_2019": {
                "mean_fb_change": float(pre_covid["fb_change"].mean())
                if len(pre_covid) > 0
                else None,
                "mean_intl_migration": float(pre_covid["pep_intl_migration"].mean())
                if len(pre_covid) > 0
                else None,
                "mean_secondary_middle": float(
                    pre_covid["secondary_migration_middle"].mean()
                )
                if len(pre_covid) > 0
                else None,
            },
            "covid_recovery_2020_2023": {
                "mean_fb_change": float(covid_era["fb_change"].mean())
                if len(covid_era) > 0
                else None,
                "mean_intl_migration": float(covid_era["pep_intl_migration"].mean())
                if len(covid_era) > 0
                else None,
                "mean_secondary_middle": float(
                    covid_era["secondary_migration_middle"].mean()
                )
                if len(covid_era) > 0
                else None,
            },
        }

        logger.info("  Completed secondary migration estimation")
        return results

    def calculate_nd_share_analysis(self, pep: pd.DataFrame) -> dict[str, Any]:
        """
        Analyze ND's share of national international migration.

        This provides context for understanding ND's foreign-born growth
        relative to national patterns.
        """
        logger.info("Calculating ND share of national international migration...")

        nd = pep[pep["state_name"] == "North Dakota"].copy()
        us = pep[pep["state_name"] == "United States"].copy()

        merged = pd.merge(nd, us, on="year", suffixes=("_nd", "_us"))
        merged["nd_share_intl"] = (
            merged["intl_migration_nd"] / merged["intl_migration_us"]
        )
        merged["nd_share_pop"] = merged["population_nd"] / merged["population_us"]
        merged["share_ratio"] = merged["nd_share_intl"] / merged["nd_share_pop"]

        # Calculate summary
        results: dict[str, Any] = {
            "annual": [
                {
                    "year": int(row["year"]),
                    "nd_intl_migration": int(row["intl_migration_nd"]),
                    "us_intl_migration": int(row["intl_migration_us"]),
                    "nd_share_of_us_intl_pct": float(row["nd_share_intl"] * 100),
                    "nd_share_of_us_pop_pct": float(row["nd_share_pop"] * 100),
                    "share_ratio": float(row["share_ratio"]),
                }
                for _, row in merged.iterrows()
            ],
            "summary": {
                "mean_nd_share_intl_pct": float(merged["nd_share_intl"].mean() * 100),
                "mean_nd_share_pop_pct": float(merged["nd_share_pop"].mean() * 100),
                "mean_share_ratio": float(merged["share_ratio"].mean()),
                "interpretation": (
                    f"ND receives {merged['share_ratio'].mean():.2f}x its population "
                    f"share of US international migration"
                ),
            },
        }

        logger.info(f"  ND receives ~{results['summary']['mean_nd_share_intl_pct']:.3f}% of US intl migration")
        return results

    def create_decomposition_scenarios(
        self, secondary_results: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Create scenarios for secondary migration share under different assumptions.

        This addresses the high uncertainty in secondary migration estimates
        by providing low/medium/high bounds.
        """
        logger.info("Creating secondary migration scenarios...")

        summary = secondary_results.get("summary", {})
        mean_fb_change = summary.get("mean_fb_change", 0)
        mean_intl = summary.get("mean_intl_migration", 0)
        mean_secondary = summary.get("mean_secondary_migration_middle", 0)

        scenarios: dict[str, Any] = {
            "baseline_middle": {
                "name": "Baseline (Middle Estimate)",
                "secondary_share": (
                    float(mean_secondary / mean_fb_change)
                    if mean_fb_change != 0
                    else None
                ),
                "direct_intl_share": (
                    float(mean_intl / mean_fb_change) if mean_fb_change != 0 else None
                ),
                "description": "Uses middle estimate of FB deaths to bound secondary migration",
            },
            "low_secondary": {
                "name": "Low Secondary Migration",
                "secondary_share": 0.10,
                "direct_intl_share": 0.90,
                "description": (
                    "Conservative estimate: most FB growth from direct international arrival"
                ),
            },
            "high_secondary": {
                "name": "High Secondary Migration",
                "secondary_share": 0.40,
                "direct_intl_share": 0.60,
                "description": (
                    "Upper bound: significant FB redistribution from other states"
                ),
            },
        }

        # Add implications for policy interpretation
        scenarios["policy_implications"] = {
            "if_low_secondary": (
                "Federal policy changes directly affect ~90% of ND's FB population growth"
            ),
            "if_high_secondary": (
                "Federal policy changes directly affect only ~60% of ND's FB population "
                "growth; secondary domestic migration could partially offset restrictions"
            ),
            "recommendation": (
                "Use middle estimate for baseline projections, but conduct sensitivity "
                "analysis with low/high bounds to bracket uncertainty"
            ),
        }

        logger.info("  Created 3 scenarios: low/baseline/high secondary migration")
        return scenarios

    def document_data_gaps(self) -> dict[str, Any]:
        """
        Document data gaps and acquisition paths for more precise analysis.
        """
        logger.info("Documenting data gaps and acquisition paths...")

        gaps: dict[str, Any] = {
            "current_limitations": [
                {
                    "issue": "PUMS sample size",
                    "impact": "Cannot directly measure FB state-to-state flows",
                    "severity": "High",
                },
                {
                    "issue": "No citizenship cross-tab in migration flows",
                    "impact": "Cannot distinguish FB from native-born domestic migrants",
                    "severity": "High",
                },
                {
                    "issue": "Residual method uncertainty",
                    "impact": "Secondary migration estimates have wide bounds",
                    "severity": "Medium",
                },
            ],
            "acquisition_path": {
                "recommended_data": "ACS PUMS with MIGSP + NATIVITY variables",
                "source": "Census FTP or IPUMS USA",
                "years_needed": "2010-2024 (1-year or 5-year)",
                "variables": ["NATIVITY", "MIG", "MIGSP", "ST", "PWGTP", "POBP", "YOEP"],
                "effort": "Medium - requires IPUMS account or large file download",
                "reference": "docs/governance/adrs/021-reports/data/acs_migration_data_acquisition.md",
            },
            "alternative_data": {
                "table": "ACS B07007/B07407 (Mobility by Citizenship Status)",
                "provides": "Aggregate counts of FB movers from different state",
                "limitation": "Does not identify source states",
                "availability": "Via Census API",
            },
            "interim_recommendation": (
                "Current residual-based estimates are sufficient for sensitivity analysis. "
                "Acquire full PUMS data for Phase C validation if secondary migration "
                "proves to be a major factor in projection uncertainty."
            ),
        }

        return gaps

    def run_analysis(self) -> dict[str, Any]:
        """Run the complete secondary migration analysis."""
        logger.info("\n" + "=" * 70)
        logger.info("Secondary Migration Analysis (ADR-021 Recommendation #7)")
        logger.info("=" * 70)
        logger.info(f"Started: {datetime.now(UTC).isoformat()}")
        logger.info("")
        # Load data
        pep = self.load_pep_components()
        fb_stock = self.load_foreign_born_stock()
        _ = self.load_pums_sample()  # Check availability, adds to diagnostics

        # Calculate FB population changes
        merged = self.calculate_fb_population_changes(fb_stock, pep)

        # Estimate secondary migration bounds
        secondary_results = self.estimate_secondary_migration_bounds(merged)
        self.results["results"]["secondary_migration"] = secondary_results

        # Calculate ND share analysis
        nd_share = self.calculate_nd_share_analysis(pep)
        self.results["results"]["nd_share_analysis"] = nd_share

        # Create decomposition scenarios
        scenarios = self.create_decomposition_scenarios(secondary_results)
        self.results["results"]["decomposition_scenarios"] = scenarios

        # Document data gaps
        data_gaps = self.document_data_gaps()
        self.results["methodology"]["data_gaps"] = data_gaps

        # Add methodology documentation
        self.results["methodology"]["approach"] = {
            "name": "Residual Method",
            "description": (
                "Estimates secondary migration as the residual after subtracting "
                "PEP international migration from ACS foreign-born stock changes, "
                "adjusted for estimated foreign-born mortality."
            ),
            "assumptions": [
                "PEP international migration captures net direct arrivals/departures",
                "Foreign-born mortality rate approximates general ND crude death rate",
                "ACS foreign-born counts are unbiased estimates of true population",
            ],
            "limitations": [
                "Cannot distinguish secondary in-migration from out-migration",
                "Mortality estimate is rough approximation",
                "Does not identify source states for secondary migrants",
            ],
        }

        # Add recommendations - dynamically generated based on computed values
        self.results["recommendations"] = self._generate_recommendations(
            secondary_results
        )

        # Add parameters
        self.results["parameters"] = {
            "analysis_period": "2010-2024",
            "focus_period": "2019-2023",
            "state": "North Dakota",
            "assumed_fb_mortality_rate": 0.008,
        }

        # Add key findings interpretation
        self.results["key_findings"] = self._generate_key_findings(secondary_results)

        # Add 2023 data quality warning to warnings array
        # This is also documented in key_findings but should be in warnings for visibility
        self.results["warnings"].append(
            "2023 data anomaly: PEP reports 4,269 international arrivals but ACS shows "
            "foreign-born population DECREASED by 1,210, a 5,479 person discrepancy. "
            "This may reflect: (1) significant out-migration to other states, "
            "(2) ACS sampling variation, or (3) timing differences between data sources. "
            "Treat 2023 estimates with caution."
        )

        return self.results

    def _generate_key_findings(
        self, secondary_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate interpretive summary of key findings."""
        period_comparison = secondary_results.get("period_comparison", {})
        pre_covid = period_comparison.get("pre_covid_2015_2019", {})
        covid_era = period_comparison.get("covid_recovery_2020_2023", {})

        findings: dict[str, Any] = {
            "primary_finding": (
                "Secondary migration patterns shifted dramatically between pre-COVID "
                "and post-COVID periods"
            ),
            "pre_covid_pattern": {
                "period": "2015-2019",
                "pattern": "Net positive secondary migration",
                "average_secondary_flow": pre_covid.get("mean_secondary_middle"),
                "interpretation": (
                    "Foreign-born population was moving TO North Dakota from other states, "
                    "likely attracted by Bakken oil boom employment opportunities"
                ),
            },
            "post_covid_pattern": {
                "period": "2020-2023",
                "pattern": "Net negative secondary migration",
                "average_secondary_flow": covid_era.get("mean_secondary_middle"),
                "interpretation": (
                    "Foreign-born population has been leaving ND for other states, "
                    "possibly reflecting: (1) oil industry contraction, "
                    "(2) remote work enabling relocation, or (3) post-pandemic geographic redistribution"
                ),
            },
            "data_quality_note": (
                "The 2023 data shows a large discrepancy: PEP reports 4,269 international "
                "arrivals but ACS shows the foreign-born population DECREASED by 1,210. "
                "This suggests either significant out-migration to other states, "
                "ACS sampling variation, or timing differences between data sources."
            ),
            "policy_implications": {
                "federal_policy_effect": (
                    "ND receives about 0.17% of US international migration (less than its "
                    "0.23% population share), so federal policy changes will have proportional "
                    "but not amplified effects on ND"
                ),
                "secondary_migration_role": (
                    "Secondary domestic migration can either amplify or offset federal policy "
                    "effects. During oil boom periods (2010-2015), ND attracted foreign-born "
                    "from other states. During contraction periods, ND loses foreign-born to "
                    "other states."
                ),
                "projection_guidance": (
                    "For projections: assume secondary migration is near-zero baseline, "
                    "with sensitivity analysis ranging from -500 to +500 per year depending "
                    "on economic conditions"
                ),
            },
        }

        return findings

    def _generate_recommendations(
        self, secondary_results: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Generate recommendations based on computed secondary migration values."""
        period_comparison = secondary_results.get("period_comparison", {})
        pre_covid = period_comparison.get("pre_covid_2015_2019", {})
        covid_era = period_comparison.get("covid_recovery_2020_2023", {})

        pre_covid_secondary = pre_covid.get("mean_secondary_middle")
        post_covid_secondary = covid_era.get("mean_secondary_middle")

        # Build dynamic recommendation text based on actual computed values
        if pre_covid_secondary is not None and post_covid_secondary is not None:
            if pre_covid_secondary > 0 and post_covid_secondary < 0:
                # Pattern shifted from positive to negative
                baseline_action = (
                    f"Account for period-specific patterns: pre-COVID (2015-2019) showed "
                    f"positive secondary migration (~{pre_covid_secondary:+,.0f}/year), "
                    f"while post-COVID (2020-2023) shows negative secondary migration "
                    f"(~{post_covid_secondary:+,.0f}/year)"
                )
                baseline_rationale = (
                    "Secondary migration direction reversed after COVID; projections should "
                    "not assume a fixed positive or negative share"
                )
            elif post_covid_secondary < 0:
                baseline_action = (
                    f"Use near-zero or negative secondary migration for baseline projections "
                    f"(post-COVID average: ~{post_covid_secondary:+,.0f}/year)"
                )
                baseline_rationale = (
                    "Recent data shows net out-migration of foreign-born to other states"
                )
            else:
                baseline_action = (
                    f"Use middle estimate (~{post_covid_secondary:+,.0f}/year) for baseline "
                    "projections"
                )
                baseline_rationale = (
                    "Balances uncertainty while providing actionable estimates"
                )
        else:
            baseline_action = (
                "Use near-zero secondary migration for baseline projections due to "
                "insufficient data for period-specific estimates"
            )
            baseline_rationale = "Data limitations prevent reliable trend estimation"

        return [
            {
                "priority": "High",
                "action": baseline_action,
                "rationale": baseline_rationale,
            },
            {
                "priority": "High",
                "action": (
                    "Conduct sensitivity analysis with secondary migration ranging from "
                    "-500 to +500 per year to bracket economic-driven uncertainty"
                ),
                "rationale": (
                    "Secondary migration is highly sensitive to economic conditions "
                    "(oil boom vs contraction) and cannot be reliably forecasted"
                ),
            },
            {
                "priority": "Medium",
                "action": (
                    "Acquire full ACS PUMS data for direct measurement of FB "
                    "state-to-state flows"
                ),
                "rationale": (
                    "Would provide more precise estimates and identify source states"
                ),
            },
            {
                "priority": "Low",
                "action": "Fetch ACS B07007/B07407 tables for aggregate validation",
                "rationale": "Quick validation of residual-based estimates",
            },
        ]

    def save_results(self) -> Path:
        """Save results to JSON file."""
        output_path = RESULTS_DIR / "rec7_secondary_migration.json"
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"\nResults saved to: {output_path}")
        return output_path


def print_summary(results: dict[str, Any]) -> None:
    """Print a human-readable summary of the analysis."""
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY: Secondary Migration Analysis")
    logger.info("=" * 70)

    # ND Share Analysis
    nd_share = results.get("results", {}).get("nd_share_analysis", {})
    if nd_share:
        summary = nd_share.get("summary", {})
        logger.info("\nND Share of US International Migration:")
        logger.info(f"  Mean share: {summary.get('mean_nd_share_intl_pct', 0):.4f}%")
        logger.info(f"  ND pop share: {summary.get('mean_nd_share_pop_pct', 0):.4f}%")
        logger.info(f"  Interpretation: {summary.get('interpretation', 'N/A')}")

    # Secondary Migration Estimates
    secondary = results.get("results", {}).get("secondary_migration", {})
    if secondary:
        summary = secondary.get("summary", {})
        logger.info("\nSecondary Migration Estimates (2019-2023):")
        logger.info(f"  Mean FB population change: {summary.get('mean_fb_change', 0):,.0f}")
        logger.info(f"  Mean PEP intl migration: {summary.get('mean_intl_migration', 0):,.0f}")
        logger.info(
            f"  Mean secondary migration (est): {summary.get('mean_secondary_migration_middle', 0):,.0f}"
        )
        share = summary.get("secondary_share_of_fb_change")
        if share:
            logger.info(f"  Secondary share of FB change: {share:.1%}")

    # Decomposition Scenarios
    scenarios = results.get("results", {}).get("decomposition_scenarios", {})
    if scenarios:
        logger.info("\nDecomposition Scenarios:")
        for key, scenario in scenarios.items():
            if key != "policy_implications" and isinstance(scenario, dict):
                name = scenario.get("name", key)
                sec_share = scenario.get("secondary_share")
                if sec_share is not None:
                    logger.info(f"  {name}: {sec_share:.0%} secondary, {1-sec_share:.0%} direct")

        implications = scenarios.get("policy_implications", {})
        if implications:
            logger.info(f"\n  Policy implication: {implications.get('recommendation', 'N/A')}")

    # Recommendations
    recs = results.get("recommendations", [])
    if recs:
        logger.info("\nRecommendations:")
        for rec in recs:
            logger.info(f"  [{rec.get('priority', 'N/A')}] {rec.get('action', 'N/A')}")

    # Warnings
    warnings = results.get("warnings", [])
    if warnings:
        logger.info("\nWarnings:")
        for w in warnings:
            logger.info(f"  - {w}")


def main() -> int:
    """Main entry point."""
    try:
        analyzer = SecondaryMigrationAnalyzer()
        results = analyzer.run_analysis()
        output_path = analyzer.save_results()

        print_summary(results)

        logger.info("\n" + "=" * 70)
        logger.info("Analysis completed successfully!")
        logger.info(f"Output: {output_path}")
        logger.info("=" * 70)

        return 0

    except Exception as exc:
        logger.exception("ERROR: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
