#!/usr/bin/env python3
"""
Module 10: Two-Component Estimand Framework (ADR-021 Recommendation #1)
=========================================================================

Implements the Y_t = Y_t^dur + Y_t^temp decomposition in the projection pipeline,
separating immigration into durable-status and temporary/precarious components.

Key Features:
1. Status-specific classification:
   - Durable (Y_t^dur): Refugee arrivals + regularized parole
   - Temporary (Y_t^temp): Non-regularized parole

2. Integration with prior wave outputs:
   - Rec #2: Status durability and regularization probability (50.3%)
   - Rec #3: LSSND capacity multiplier (67.2%)
   - Rec #4: Policy regime framework
   - Rec #6: Policy scenario projections

3. Cohort-based survival modeling:
   - Tracks arrival cohorts by status over time
   - Applies status-specific survival functions
   - Models regularization transition from temporary to durable

4. Long-horizon projections (to 2045):
   - Status-tracked forecasts by scenario
   - Uncertainty quantification via Monte Carlo
   - PEP continuity validation

Components:
- Y_t^dur = refugees_surviving(t) + parolees_regularized(t)
- Y_t^temp = parolees_not_regularized(t) * survival_probability(t)

Data Sources:
- census.state_components: PEP international migration
- rpc.refugee_arrivals: USRAP refugee arrivals
- Rec #2/3/4/6 JSON outputs: Wave 1-3 parameters

Usage:
    uv run python module_10_two_component_estimand.py

References:
- ADR-021 Recommendation #1: Clarify estimand as Y_t = Y_t^dur + Y_t^temp
- ADR-021 Phase B Wave 4: Implementation specification
"""

from __future__ import annotations

import json
import sys
import traceback
import warnings
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import db_config

# Import regime framework from Rec #4
from statistical_analysis.module_regime_framework import (
    PolicyRegime,
    get_regime,
    get_regime_params,
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # cohort_projections/
RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"
ADR_RESULTS_DIR = (
    PROJECT_ROOT / "docs" / "governance" / "adrs" / "021-reports" / "results"
)
ADR_FIGURES_DIR = (
    PROJECT_ROOT / "docs" / "governance" / "adrs" / "021-reports" / "figures"
)

# Ensure output directories exist
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
ADR_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ADR_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# COLOR PALETTE (colorblind-safe)
# =============================================================================

COLORS = {
    "durable": "#0072B2",  # Blue - durable/permanent
    "temporary": "#D55E00",  # Vermillion/Orange - temporary/precarious
    "total": "#333333",  # Dark gray - aggregate
    "refugee": "#009E73",  # Teal - refugee component
    "regularized": "#56B4E9",  # Light blue - regularized parole
    "parole": "#E69F00",  # Yellow/orange - non-regularized parole
    "historical": "#000000",  # Black - historical data
    "projection": "#999999",  # Gray - projection uncertainty
}


# =============================================================================
# ESTIMAND COMPONENT DEFINITIONS
# =============================================================================


class EstimandComponent(Enum):
    """
    Components of the two-component estimand decomposition.

    Y_t = Y_t^dur + Y_t^temp where:
    - Y_t^dur: Durable-status immigrants (high long-run presence)
    - Y_t^temp: Temporary-status immigrants (precarious, cliff risk)
    """

    DURABLE = "durable"
    TEMPORARY = "temporary"


class StatusCategory(Enum):
    """
    Legal status categories for arrival classification.

    Maps to estimand components:
    - REFUGEE -> DURABLE (direct path to LPR)
    - PAROLE_REGULARIZED -> DURABLE (after regularization)
    - PAROLE_NON_REGULARIZED -> TEMPORARY (faces cliff hazard)
    - OTHER -> DURABLE (mostly LPR family/employment)
    """

    REFUGEE = "refugee"
    PAROLE_REGULARIZED = "parole_regularized"
    PAROLE_NON_REGULARIZED = "parole_non_regularized"
    OTHER = "other"


@dataclass(frozen=True)
class SurvivalParameters:
    """
    Status-specific survival parameters from Rec #2.

    Attributes:
        status: StatusCategory enum
        survival_1yr: 1-year survival probability
        survival_5yr: 5-year survival probability
        survival_10yr: 10-year survival probability
        cliff_start: Year when cliff hazard begins (None if no cliff)
        cliff_end: Year when cliff hazard ends (None if no cliff)
    """

    status: StatusCategory
    survival_1yr: float
    survival_5yr: float
    survival_10yr: float
    cliff_start: float | None = None
    cliff_end: float | None = None

    def get_survival_at(self, duration: float) -> float:
        """
        Get survival probability at a specific duration.

        Uses interpolation between known survival points.
        """
        if duration <= 0:
            return 1.0
        if duration <= 1:
            return 1.0 - (1.0 - self.survival_1yr) * duration
        if duration <= 5:
            progress = (duration - 1) / 4
            return self.survival_1yr - (self.survival_1yr - self.survival_5yr) * progress
        if duration <= 10:
            progress = (duration - 5) / 5
            return self.survival_5yr - (self.survival_5yr - self.survival_10yr) * progress
        # Beyond 10 years: extrapolate with slow decay
        annual_decay = (self.survival_5yr - self.survival_10yr) / 5
        return max(0.1, self.survival_10yr - annual_decay * (duration - 10))


# =============================================================================
# DATA CLASSES FOR TWO-COMPONENT ESTIMAND
# =============================================================================


@dataclass
class ArrivalCohort:
    """
    Represents a single arrival cohort tracked over time.

    Tracks original arrivals by status category and their survival
    through the observation period and into projections.

    Attributes:
        arrival_year: Year of arrival (fiscal/calendar)
        refugee_arrivals: Initial refugee arrivals
        parole_arrivals: Initial parole arrivals
        other_arrivals: Initial other category arrivals
        regime: Policy regime at time of arrival
    """

    arrival_year: int
    refugee_arrivals: float
    parole_arrivals: float
    other_arrivals: float
    regime: PolicyRegime

    def get_total_arrivals(self) -> float:
        """Get total arrivals in cohort."""
        return self.refugee_arrivals + self.parole_arrivals + self.other_arrivals

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "arrival_year": self.arrival_year,
            "refugee_arrivals": self.refugee_arrivals,
            "parole_arrivals": self.parole_arrivals,
            "other_arrivals": self.other_arrivals,
            "total_arrivals": self.get_total_arrivals(),
            "regime": self.regime.value,
        }


@dataclass
class CohortSurvivalState:
    """
    State of a cohort at a specific observation year.

    Tracks remaining population by status category after applying
    survival functions and regularization transitions.

    Attributes:
        cohort: The original ArrivalCohort
        observation_year: Year of observation
        refugee_surviving: Refugees still present
        parole_regularized: Parolees who regularized (now durable)
        parole_temporary: Parolees who did not regularize (temporary)
        other_surviving: Other category still present
    """

    cohort: ArrivalCohort
    observation_year: int
    refugee_surviving: float
    parole_regularized: float
    parole_temporary: float
    other_surviving: float

    @property
    def duration(self) -> int:
        """Duration in years since arrival."""
        return self.observation_year - self.cohort.arrival_year

    @property
    def durable_component(self) -> float:
        """Y_t^dur: Sum of durable-status population."""
        return self.refugee_surviving + self.parole_regularized + self.other_surviving

    @property
    def temporary_component(self) -> float:
        """Y_t^temp: Sum of temporary-status population."""
        return self.parole_temporary

    @property
    def total(self) -> float:
        """Y_t: Total surviving population."""
        return self.durable_component + self.temporary_component

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "arrival_year": self.cohort.arrival_year,
            "observation_year": self.observation_year,
            "duration": self.duration,
            "refugee_surviving": self.refugee_surviving,
            "parole_regularized": self.parole_regularized,
            "parole_temporary": self.parole_temporary,
            "other_surviving": self.other_surviving,
            "durable_component": self.durable_component,
            "temporary_component": self.temporary_component,
            "total": self.total,
        }


@dataclass
class TwoComponentEstimate:
    """
    Two-component estimand estimate for a single year.

    Y_t = Y_t^dur + Y_t^temp

    Aggregates all cohorts present in the observation year.

    Attributes:
        year: Observation year
        y_durable: Durable component (Y_t^dur)
        y_temporary: Temporary component (Y_t^temp)
        y_total: Total (Y_t = Y_t^dur + Y_t^temp)
        pep_total: Census PEP net international migration (for validation)
        cohort_states: Individual cohort survival states
    """

    year: int
    y_durable: float
    y_temporary: float
    y_total: float
    pep_total: float | None = None
    cohort_states: list[CohortSurvivalState] = field(default_factory=list)

    @property
    def durable_share(self) -> float:
        """Share of total that is durable."""
        if self.y_total <= 0:
            return 0.0
        return self.y_durable / self.y_total

    @property
    def temporary_share(self) -> float:
        """Share of total that is temporary."""
        if self.y_total <= 0:
            return 0.0
        return self.y_temporary / self.y_total

    @property
    def pep_residual(self) -> float | None:
        """Residual: PEP total - estimated total (for validation)."""
        if self.pep_total is None:
            return None
        return self.pep_total - self.y_total

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "year": self.year,
            "y_durable": self.y_durable,
            "y_temporary": self.y_temporary,
            "y_total": self.y_total,
            "durable_share": self.durable_share,
            "temporary_share": self.temporary_share,
            "pep_total": self.pep_total,
            "pep_residual": self.pep_residual,
        }


@dataclass
class ProjectionTimeSeries:
    """
    Time series of two-component estimates for projection.

    Contains historical and projected estimates with uncertainty bounds.

    Attributes:
        estimates: List of TwoComponentEstimate objects
        scenario_name: Name of projection scenario (if applicable)
        base_year: Last historical year (start of projection)
        horizon_year: End of projection horizon
        uncertainty: Dict with p10, p50, p90 bounds
    """

    estimates: list[TwoComponentEstimate]
    scenario_name: str | None = None
    base_year: int = 2024
    horizon_year: int = 2045
    uncertainty: dict[str, list[float]] = field(default_factory=dict)

    def get_years(self) -> list[int]:
        """Get list of years in time series."""
        return [e.year for e in self.estimates]

    def get_durable_series(self) -> list[float]:
        """Get durable component time series."""
        return [e.y_durable for e in self.estimates]

    def get_temporary_series(self) -> list[float]:
        """Get temporary component time series."""
        return [e.y_temporary for e in self.estimates]

    def get_total_series(self) -> list[float]:
        """Get total time series."""
        return [e.y_total for e in self.estimates]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "scenario_name": self.scenario_name,
            "base_year": self.base_year,
            "horizon_year": self.horizon_year,
            "years": self.get_years(),
            "y_durable": self.get_durable_series(),
            "y_temporary": self.get_temporary_series(),
            "y_total": self.get_total_series(),
            "uncertainty": self.uncertainty,
            "estimates": [e.to_dict() for e in self.estimates],
        }


# =============================================================================
# MODULE RESULT CONTAINER
# =============================================================================


@dataclass
class ModuleResult:
    """Standard result container for all modules."""

    module_id: str
    analysis_name: str
    input_files: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)

    def add_decision(
        self,
        decision_id: str,
        category: str,
        decision: str,
        rationale: str,
        alternatives: list[str] | None = None,
        evidence: str | None = None,
        reversible: bool = True,
    ) -> None:
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

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
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
# LOAD WAVE 1-3 PARAMETERS
# =============================================================================


def load_rec2_parameters(result: ModuleResult) -> dict[str, Any]:
    """
    Load status durability parameters from Rec #2.

    Returns regularization probability and survival curves.
    """
    rec2_path = ADR_RESULTS_DIR / "rec2_status_durability.json"

    if not rec2_path.exists():
        rec2_path = RESULTS_DIR / "module_8b_status_durability.json"

    params: dict[str, Any] = {
        # Default values if file not found
        "regularization_probability": 0.503,
        "regularization_lower": 0.299,
        "regularization_upper": 0.725,
        "refugee_survival_5yr": 0.954,
        "parole_survival_5yr": 0.341,
        "parole_cliff_start": 2.0,
        "parole_cliff_end": 4.0,
        "other_survival_5yr": 0.90,
    }

    if rec2_path.exists():
        with open(rec2_path) as f:
            rec2_data = json.load(f)
        result.input_files.append(str(rec2_path))

        # Extract regularization parameters
        regularization = rec2_data.get("results", {}).get("regularization", {})
        uncertainty = regularization.get("uncertainty", {})

        # Extract survival curves
        wave_integration = rec2_data.get("results", {}).get("wave_integration", {})
        status_curves = wave_integration.get("status_survival_curves", {})
        cliff_params = wave_integration.get("cliff_parameters", {})

        params.update({
            "regularization_probability": uncertainty.get("central_estimate", 0.503),
            "regularization_lower": uncertainty.get("lower_bound", 0.299),
            "regularization_upper": uncertainty.get("upper_bound", 0.725),
            "refugee_survival_5yr": status_curves.get("refugee", {}).get(
                "survival_5yr", 0.954
            ),
            "parole_survival_5yr": status_curves.get("parole", {}).get(
                "survival_5yr", 0.341
            ),
            "parole_cliff_start": cliff_params.get("cliff_start_year", 2.0),
            "parole_cliff_end": cliff_params.get("cliff_end_year", 4.0),
            "other_survival_5yr": status_curves.get("other", {}).get(
                "survival_5yr", 0.90
            ),
        })

        print(f"  Loaded Rec #2: regularization={params['regularization_probability']:.1%}")
    else:
        result.warnings.append("Rec #2 results not found; using default parameters")
        print("  WARNING: Rec #2 results not found; using defaults")

    return params


def load_rec3_parameters(result: ModuleResult) -> dict[str, Any]:
    """
    Load LSSND capacity multiplier from Rec #3.
    """
    rec3_path = ADR_RESULTS_DIR / "rec3_lssnd_synthetic_control.json"

    if not rec3_path.exists():
        rec3_path = RESULTS_DIR / "module_7b_lssnd_synthetic_control.json"

    params: dict[str, Any] = {
        "capacity_multiplier": 0.672,
        "capacity_lower": 0.585,
        "capacity_upper": 0.649,
        "nd_share": 0.0074,
    }

    if rec3_path.exists():
        with open(rec3_path) as f:
            rec3_data = json.load(f)
        result.input_files.append(str(rec3_path))

        capacity_param = rec3_data.get("results", {}).get("capacity_parameter", {})
        uncertainty = capacity_param.get("uncertainty", {})
        ns_estimate = rec3_data.get("results", {}).get("national_share_estimate", {})
        cf_share = ns_estimate.get("counterfactual_share", {})

        params.update({
            "capacity_multiplier": capacity_param.get("value", 0.672),
            "capacity_lower": uncertainty.get("capacity_range_min", 0.585),
            "capacity_upper": uncertainty.get("capacity_range_max", 0.649),
            "nd_share": cf_share.get("value", 0.0074),
        })

        print(f"  Loaded Rec #3: capacity={params['capacity_multiplier']:.1%}")
    else:
        result.warnings.append("Rec #3 results not found; using default parameters")
        print("  WARNING: Rec #3 results not found; using defaults")

    return params


def load_rec6_projections(result: ModuleResult) -> dict[str, Any] | None:
    """
    Load policy scenario projections from Rec #6.
    """
    rec6_path = ADR_RESULTS_DIR / "rec6_policy_scenarios.json"

    if not rec6_path.exists():
        rec6_path = RESULTS_DIR / "module_9b_policy_scenarios.json"

    if rec6_path.exists():
        with open(rec6_path) as f:
            rec6_data = json.load(f)
        result.input_files.append(str(rec6_path))
        print("  Loaded Rec #6 policy scenario projections")
        return rec6_data.get("results", {})
    else:
        result.warnings.append("Rec #6 results not found; projections will use defaults")
        print("  WARNING: Rec #6 results not found")
        return None


# =============================================================================
# DATA LOADING FROM POSTGRESQL
# =============================================================================


def load_pep_migration(result: ModuleResult) -> pd.DataFrame:
    """
    Load Census PEP international migration for North Dakota.
    """
    conn = db_config.get_db_connection()
    try:
        query = """
        SELECT
            year,
            intl_migration
        FROM census.state_components
        WHERE state_name = 'North Dakota'
          AND intl_migration IS NOT NULL
        ORDER BY year
        """
        df = pd.read_sql(query, conn)
        result.input_files.append("census.state_components (PostgreSQL)")
        print(f"  Loaded PEP migration: {len(df)} years ({df['year'].min()}-{df['year'].max()})")
        return df
    finally:
        conn.close()


def load_refugee_arrivals(result: ModuleResult) -> pd.DataFrame:
    """
    Load ND refugee arrivals by fiscal year.
    """
    conn = db_config.get_db_connection()
    try:
        query = """
        SELECT
            fiscal_year as year,
            SUM(arrivals) as refugee_arrivals
        FROM rpc.refugee_arrivals
        WHERE destination_state = 'North Dakota'
        GROUP BY fiscal_year
        ORDER BY fiscal_year
        """
        df = pd.read_sql(query, conn)
        result.input_files.append("rpc.refugee_arrivals (PostgreSQL)")
        print(f"  Loaded refugee arrivals: {len(df)} years")
        return df
    finally:
        conn.close()


def load_national_refugee_totals(result: ModuleResult) -> pd.DataFrame:
    """
    Load national refugee arrival totals by fiscal year.
    """
    conn = db_config.get_db_connection()
    try:
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
        result.input_files.append("rpc.refugee_arrivals (national, PostgreSQL)")
        print(f"  Loaded national refugee totals: {len(df)} years (FY2002-2020)")
    finally:
        conn.close()

    official_post_2020 = pd.DataFrame(
        {
            "year": [2021, 2022, 2023, 2024],
            "national_arrivals": [11411, 25519, 60014, 100034],
        }
    )

    df = pd.concat([df, official_post_2020], ignore_index=True)
    df = df.sort_values("year").reset_index(drop=True)
    result.input_files.append("RPC/DHS official national totals (FY2021-2024)")
    print("  Added official FY2021-2024 national totals")
    print(f"  Total: {len(df)} years (FY2002-2024)")
    return df


# =============================================================================
# STATUS CLASSIFICATION PIPELINE
# =============================================================================


def classify_arrivals_by_status(
    df_pep: pd.DataFrame,
    df_refugee: pd.DataFrame,
    result: ModuleResult,
    non_humanitarian_share: float = 0.05,
) -> list[ArrivalCohort]:
    """
    Classify historical arrivals by status category.

    Uses the residual method from Rec #2:
    - Refugee arrivals: Direct from USRAP data
    - Parole proxy: PEP total - Refugee - Non-humanitarian
    - Other: Estimated as share of total (LPR family, employment)

    Args:
        df_pep: PEP international migration data for ND
        df_refugee: Refugee arrivals data for ND
        result: ModuleResult for logging
        non_humanitarian_share: Estimated share of non-humanitarian migration

    Returns:
        List of ArrivalCohort objects
    """
    print("\n" + "=" * 60)
    print("CLASSIFYING ARRIVALS BY STATUS")
    print("=" * 60)

    # Merge PEP and refugee data
    df = df_pep.merge(df_refugee, on="year", how="outer")
    df = df.sort_values("year").reset_index(drop=True)

    # Fill missing
    df["refugee_arrivals"] = df["refugee_arrivals"].fillna(0)
    df["intl_migration"] = df["intl_migration"].fillna(0)

    cohorts: list[ArrivalCohort] = []

    for _, row in df.iterrows():
        year = int(row["year"])

        # Skip years outside regime-defined period
        if year < 2010 or year > 2024:
            continue

        total = max(0, float(row["intl_migration"]))
        refugee = max(0, float(row["refugee_arrivals"]))

        # Get regime for year
        try:
            regime = get_regime(year)
        except ValueError:
            continue

        # Estimate non-humanitarian (LPR family, employment)
        other = max(0, total * non_humanitarian_share)

        # Parole proxy = residual
        parole = max(0, total - refugee - other)

        cohort = ArrivalCohort(
            arrival_year=year,
            refugee_arrivals=refugee,
            parole_arrivals=parole,
            other_arrivals=other,
            regime=regime,
        )
        cohorts.append(cohort)

    # Log summary by regime
    print("\nStatus classification by regime:")
    for regime in PolicyRegime:
        regime_cohorts = [c for c in cohorts if c.regime == regime]
        if regime_cohorts:
            total_refugee = sum(c.refugee_arrivals for c in regime_cohorts)
            total_parole = sum(c.parole_arrivals for c in regime_cohorts)
            total_other = sum(c.other_arrivals for c in regime_cohorts)
            total_all = total_refugee + total_parole + total_other

            refugee_share = (total_refugee / total_all * 100) if total_all > 0 else 0
            parole_share = (total_parole / total_all * 100) if total_all > 0 else 0

            years = [c.arrival_year for c in regime_cohorts]
            print(f"\n  {regime.value.upper()} ({min(years)}-{max(years)}):")
            print(f"    Total migration: {total_all:,.0f}")
            print(f"    Refugee: {total_refugee:,.0f} ({refugee_share:.1f}%)")
            print(f"    Parole proxy: {total_parole:,.0f} ({parole_share:.1f}%)")
            print(f"    Other: {total_other:,.0f}")

    result.add_decision(
        decision_id="TC001",
        category="methodology",
        decision=f"Status classification using residual method: Parole = PEP - Refugee - {non_humanitarian_share*100:.0f}% other",
        rationale="Direct parole arrival counts unavailable; residual method provides proxy using PEP total minus known components",
        alternatives=["ACS entry cohort data", "DHS I-94 records", "USCIS statistics"],
        evidence=f"Classified {len(cohorts)} cohorts from {cohorts[0].arrival_year}-{cohorts[-1].arrival_year}",
    )

    return cohorts


# =============================================================================
# BUILD SURVIVAL PARAMETER OBJECTS
# =============================================================================


def build_survival_parameters(
    rec2_params: dict[str, Any],
) -> dict[StatusCategory, SurvivalParameters]:
    """
    Build SurvivalParameters objects from Rec #2 parameters.
    """
    return {
        StatusCategory.REFUGEE: SurvivalParameters(
            status=StatusCategory.REFUGEE,
            survival_1yr=0.98,
            survival_5yr=rec2_params["refugee_survival_5yr"],
            survival_10yr=0.92,
            cliff_start=None,
            cliff_end=None,
        ),
        StatusCategory.PAROLE_REGULARIZED: SurvivalParameters(
            status=StatusCategory.PAROLE_REGULARIZED,
            survival_1yr=0.97,
            survival_5yr=0.90,  # Similar to refugee after regularization
            survival_10yr=0.85,
            cliff_start=None,
            cliff_end=None,
        ),
        StatusCategory.PAROLE_NON_REGULARIZED: SurvivalParameters(
            status=StatusCategory.PAROLE_NON_REGULARIZED,
            survival_1yr=0.90,
            survival_5yr=rec2_params["parole_survival_5yr"],
            survival_10yr=0.15,
            cliff_start=rec2_params["parole_cliff_start"],
            cliff_end=rec2_params["parole_cliff_end"],
        ),
        StatusCategory.OTHER: SurvivalParameters(
            status=StatusCategory.OTHER,
            survival_1yr=0.95,
            survival_5yr=rec2_params["other_survival_5yr"],
            survival_10yr=0.80,
            cliff_start=None,
            cliff_end=None,
        ),
    }


# =============================================================================
# COHORT SURVIVAL AND REGULARIZATION MODEL
# =============================================================================


def apply_cohort_survival(
    cohort: ArrivalCohort,
    observation_year: int,
    survival_params: dict[StatusCategory, SurvivalParameters],
    regularization_prob: float,
) -> CohortSurvivalState:
    """
    Apply survival and regularization model to a single cohort.

    Models:
    1. Refugee survival: High retention, no cliff
    2. Parole regularization: At cliff_start, regularization_prob convert to durable
    3. Non-regularized parole: High hazard during cliff, low survival after
    4. Other: Intermediate retention

    Args:
        cohort: The arrival cohort
        observation_year: Year to observe survival
        survival_params: Status-specific survival parameters
        regularization_prob: Probability of parole regularization

    Returns:
        CohortSurvivalState with surviving population by status
    """
    duration = observation_year - cohort.arrival_year

    if duration < 0:
        # Cohort hasn't arrived yet
        return CohortSurvivalState(
            cohort=cohort,
            observation_year=observation_year,
            refugee_surviving=0.0,
            parole_regularized=0.0,
            parole_temporary=0.0,
            other_surviving=0.0,
        )

    # Refugee survival
    refugee_survival = survival_params[StatusCategory.REFUGEE].get_survival_at(duration)
    refugee_surviving = cohort.refugee_arrivals * refugee_survival

    # Parole regularization and survival
    parole_cliff_start = survival_params[StatusCategory.PAROLE_NON_REGULARIZED].cliff_start or 2.0
    parole_cliff_end = survival_params[StatusCategory.PAROLE_NON_REGULARIZED].cliff_end or 4.0

    if duration < parole_cliff_start:
        # Pre-cliff: all parole is temporary, high survival
        temp_survival = survival_params[StatusCategory.PAROLE_NON_REGULARIZED].get_survival_at(duration)
        parole_regularized = 0.0
        parole_temporary = cohort.parole_arrivals * temp_survival
    elif duration <= parole_cliff_end:
        # During cliff: regularization begins
        cliff_progress = (duration - parole_cliff_start) / (parole_cliff_end - parole_cliff_start)
        # Linear regularization over cliff period
        effective_reg_prob = regularization_prob * cliff_progress

        # Regularized parolees switch to durable survival
        reg_survival = survival_params[StatusCategory.PAROLE_REGULARIZED].get_survival_at(
            duration - parole_cliff_start
        )
        parole_regularized = cohort.parole_arrivals * effective_reg_prob * reg_survival

        # Non-regularized face cliff hazard
        temp_survival = survival_params[StatusCategory.PAROLE_NON_REGULARIZED].get_survival_at(duration)
        parole_temporary = cohort.parole_arrivals * (1 - effective_reg_prob) * temp_survival
    else:
        # Post-cliff: regularization complete
        reg_survival = survival_params[StatusCategory.PAROLE_REGULARIZED].get_survival_at(
            duration - parole_cliff_end
        )
        parole_regularized = cohort.parole_arrivals * regularization_prob * reg_survival

        # Non-regularized mostly departed
        temp_survival = survival_params[StatusCategory.PAROLE_NON_REGULARIZED].get_survival_at(duration)
        parole_temporary = cohort.parole_arrivals * (1 - regularization_prob) * temp_survival

    # Other category survival
    other_survival = survival_params[StatusCategory.OTHER].get_survival_at(duration)
    other_surviving = cohort.other_arrivals * other_survival

    return CohortSurvivalState(
        cohort=cohort,
        observation_year=observation_year,
        refugee_surviving=refugee_surviving,
        parole_regularized=parole_regularized,
        parole_temporary=parole_temporary,
        other_surviving=other_surviving,
    )


def compute_two_component_estimate(
    cohorts: list[ArrivalCohort],
    observation_year: int,
    survival_params: dict[StatusCategory, SurvivalParameters],
    regularization_prob: float,
    pep_total: float | None = None,
) -> TwoComponentEstimate:
    """
    Compute two-component estimate for a single year.

    Aggregates surviving population from all cohorts present in the observation year.

    Args:
        cohorts: All arrival cohorts
        observation_year: Year to compute estimate
        survival_params: Status-specific survival parameters
        regularization_prob: Parole regularization probability
        pep_total: Optional PEP total for validation

    Returns:
        TwoComponentEstimate for the observation year
    """
    cohort_states: list[CohortSurvivalState] = []
    total_durable = 0.0
    total_temporary = 0.0

    for cohort in cohorts:
        if cohort.arrival_year <= observation_year:
            state = apply_cohort_survival(
                cohort=cohort,
                observation_year=observation_year,
                survival_params=survival_params,
                regularization_prob=regularization_prob,
            )
            cohort_states.append(state)
            total_durable += state.durable_component
            total_temporary += state.temporary_component

    return TwoComponentEstimate(
        year=observation_year,
        y_durable=total_durable,
        y_temporary=total_temporary,
        y_total=total_durable + total_temporary,
        pep_total=pep_total,
        cohort_states=cohort_states,
    )


# =============================================================================
# HISTORICAL DECOMPOSITION
# =============================================================================


def compute_historical_decomposition(
    cohorts: list[ArrivalCohort],
    df_pep: pd.DataFrame,
    survival_params: dict[StatusCategory, SurvivalParameters],
    regularization_prob: float,
    result: ModuleResult,
) -> list[TwoComponentEstimate]:
    """
    Compute two-component decomposition for historical period.

    Args:
        cohorts: Classified arrival cohorts
        df_pep: PEP migration data for validation
        survival_params: Status-specific survival parameters
        regularization_prob: Parole regularization probability
        result: ModuleResult for logging

    Returns:
        List of TwoComponentEstimate for each historical year
    """
    print("\n" + "=" * 60)
    print("COMPUTING HISTORICAL DECOMPOSITION")
    print("=" * 60)

    # Create PEP lookup
    pep_lookup = dict(zip(df_pep["year"], df_pep["intl_migration"]))

    estimates: list[TwoComponentEstimate] = []

    for year in range(2010, 2025):
        pep_total = pep_lookup.get(year)
        estimate = compute_two_component_estimate(
            cohorts=cohorts,
            observation_year=year,
            survival_params=survival_params,
            regularization_prob=regularization_prob,
            pep_total=pep_total,
        )
        estimates.append(estimate)

    # Print summary by regime
    print("\nHistorical decomposition by regime:")
    for regime in PolicyRegime:
        params = get_regime_params(regime)
        regime_estimates = [
            e for e in estimates
            if params.start_year <= e.year <= params.end_year
        ]

        if regime_estimates:
            mean_durable = np.mean([e.y_durable for e in regime_estimates])
            mean_temporary = np.mean([e.y_temporary for e in regime_estimates])
            mean_total = np.mean([e.y_total for e in regime_estimates])
            mean_durable_share = np.mean([e.durable_share for e in regime_estimates])

            print(f"\n  {regime.value.upper()} ({params.start_year}-{params.end_year}):")
            print(f"    Mean durable (Y^dur): {mean_durable:,.1f}")
            print(f"    Mean temporary (Y^temp): {mean_temporary:,.1f}")
            print(f"    Mean total: {mean_total:,.1f}")
            print(f"    Durable share: {mean_durable_share*100:.1f}%")

    result.add_decision(
        decision_id="TC002",
        category="methodology",
        decision=f"Historical decomposition using regularization probability = {regularization_prob:.1%}",
        rationale="Central estimate from Rec #2 scenario analysis",
        evidence=f"Computed for {len(estimates)} years (2010-2024)",
    )

    return estimates


# =============================================================================
# PROJECTION ENGINE
# =============================================================================


def project_two_component_estimand(
    cohorts: list[ArrivalCohort],
    survival_params: dict[StatusCategory, SurvivalParameters],
    regularization_prob: float,
    rec3_params: dict[str, Any],
    rec6_projections: dict[str, Any] | None,
    result: ModuleResult,
    base_year: int = 2024,
    horizon_year: int = 2045,
    n_simulations: int = 500,
) -> dict[str, ProjectionTimeSeries]:
    """
    Generate two-component projections for all scenarios.

    Integrates with Rec #6 policy scenarios and adds status tracking.

    Args:
        cohorts: Historical arrival cohorts
        survival_params: Status-specific survival parameters
        regularization_prob: Central regularization probability
        rec3_params: Capacity parameters from Rec #3
        rec6_projections: Scenario projections from Rec #6
        result: ModuleResult for logging
        base_year: Last historical year
        horizon_year: End of projection horizon
        n_simulations: Monte Carlo simulation count

    Returns:
        Dict mapping scenario names to ProjectionTimeSeries
    """
    print("\n" + "=" * 60)
    print("GENERATING TWO-COMPONENT PROJECTIONS")
    print("=" * 60)

    projections: dict[str, ProjectionTimeSeries] = {}
    rng = np.random.default_rng(42)

    # Define scenarios based on Rec #6 or defaults
    if rec6_projections and "scenarios" in rec6_projections:
        scenario_names = list(rec6_projections["scenarios"].keys())
    else:
        scenario_names = ["status_quo", "durable_growth", "restriction"]

    # Scenario parameter variations
    scenario_params: dict[str, dict[str, Any]] = {
        "status_quo": {
            "refugee_growth": 0.02,
            "parole_continuation": True,
            "parole_growth": -0.05,
            "regularization": regularization_prob,
            "capacity_recovery_rate": 0.05,
        },
        "durable_growth": {
            "refugee_growth": 0.05,
            "parole_continuation": True,
            "parole_growth": 0.00,
            "regularization": 0.725,  # High regularization
            "capacity_recovery_rate": 0.10,
        },
        "parole_cliff": {
            "refugee_growth": 0.02,
            "parole_continuation": False,
            "parole_growth": -0.30,  # Rapid decline
            "regularization": 0.299,  # Low regularization
            "capacity_recovery_rate": 0.03,
        },
        "restriction": {
            "refugee_growth": -0.05,
            "parole_continuation": False,
            "parole_growth": -0.20,
            "regularization": 0.299,
            "capacity_recovery_rate": 0.00,
        },
        "welcome_corps": {
            "refugee_growth": 0.03,
            "parole_continuation": True,
            "parole_growth": 0.00,
            "regularization": regularization_prob,
            "capacity_recovery_rate": 0.08,
            "private_sponsorship_growth": 0.20,
        },
    }

    for scenario_name in scenario_names:
        print(f"\n  Projecting scenario: {scenario_name}")

        params = scenario_params.get(scenario_name, scenario_params["status_quo"])
        capacity_mult = rec3_params["capacity_multiplier"]

        # Storage for Monte Carlo
        proj_years = list(range(base_year + 1, horizon_year + 1))
        n_years = len(proj_years)
        durable_sims = np.zeros((n_simulations, n_years))
        temp_sims = np.zeros((n_simulations, n_years))

        for sim in range(n_simulations):
            # Draw stochastic parameters
            refugee_var = rng.uniform(0.85, 1.15)
            parole_var = rng.uniform(0.80, 1.20)
            reg_var = rng.normal(1.0, 0.1)
            capacity_var = rng.uniform(0.95, 1.05)

            # Get last historical values from Volatility regime as base
            vol_cohorts = [c for c in cohorts if c.regime == PolicyRegime.VOLATILITY]
            if vol_cohorts:
                base_refugee = float(np.mean([c.refugee_arrivals for c in vol_cohorts]))
                base_parole = float(np.mean([c.parole_arrivals for c in vol_cohorts]))
                base_other = float(np.mean([c.other_arrivals for c in vol_cohorts]))
            else:
                base_refugee = 200.0
                base_parole = 2500.0
                base_other = 150.0

            # Build future cohorts
            future_cohorts = list(cohorts)  # Copy historical

            for t, year in enumerate(proj_years):
                years_from_base = year - base_year

                # Capacity recovery
                current_capacity = min(
                    1.0,
                    capacity_mult + params["capacity_recovery_rate"] * years_from_base
                ) * capacity_var

                # Project new arrivals
                refugee_new = (
                    base_refugee *
                    (1 + params["refugee_growth"]) ** years_from_base *
                    current_capacity *
                    refugee_var
                )

                if params["parole_continuation"]:
                    parole_new = (
                        base_parole *
                        (1 + params["parole_growth"]) ** years_from_base *
                        parole_var
                    )
                else:
                    # Parole winds down
                    parole_new = max(
                        0,
                        base_parole * max(0, 1 - years_from_base / 4) * parole_var
                    )

                other_new = base_other * current_capacity

                # Add private sponsorship if applicable
                if params.get("private_sponsorship_growth"):
                    ps_base = 30
                    ps_growth = params["private_sponsorship_growth"]
                    private_sponsorship = ps_base * (1 + ps_growth) ** years_from_base
                    refugee_new += private_sponsorship

                # Create future cohort
                future_cohort = ArrivalCohort(
                    arrival_year=year,
                    refugee_arrivals=refugee_new,
                    parole_arrivals=parole_new,
                    other_arrivals=other_new,
                    regime=PolicyRegime.VOLATILITY,  # Future is uncertain
                )
                future_cohorts.append(future_cohort)

                # Compute two-component estimate
                effective_reg = params["regularization"] * reg_var
                effective_reg = max(0.1, min(0.9, effective_reg))  # Bound

                estimate = compute_two_component_estimate(
                    cohorts=future_cohorts,
                    observation_year=year,
                    survival_params=survival_params,
                    regularization_prob=effective_reg,
                )

                durable_sims[sim, t] = estimate.y_durable
                temp_sims[sim, t] = estimate.y_temporary

        # Compute percentiles
        total_sims = durable_sims + temp_sims

        estimates: list[TwoComponentEstimate] = []
        for t, year in enumerate(proj_years):
            estimates.append(TwoComponentEstimate(
                year=year,
                y_durable=float(np.median(durable_sims[:, t])),
                y_temporary=float(np.median(temp_sims[:, t])),
                y_total=float(np.median(total_sims[:, t])),
            ))

        uncertainty = {
            "durable_p10": np.percentile(durable_sims, 10, axis=0).tolist(),
            "durable_p50": np.percentile(durable_sims, 50, axis=0).tolist(),
            "durable_p90": np.percentile(durable_sims, 90, axis=0).tolist(),
            "temporary_p10": np.percentile(temp_sims, 10, axis=0).tolist(),
            "temporary_p50": np.percentile(temp_sims, 50, axis=0).tolist(),
            "temporary_p90": np.percentile(temp_sims, 90, axis=0).tolist(),
            "total_p10": np.percentile(total_sims, 10, axis=0).tolist(),
            "total_p50": np.percentile(total_sims, 50, axis=0).tolist(),
            "total_p90": np.percentile(total_sims, 90, axis=0).tolist(),
        }

        projection = ProjectionTimeSeries(
            estimates=estimates,
            scenario_name=scenario_name,
            base_year=base_year,
            horizon_year=horizon_year,
            uncertainty=uncertainty,
        )
        projections[scenario_name] = projection

        # Print summary
        final_est = estimates[-1]
        print(f"    2045: Total={final_est.y_total:,.0f}, Durable={final_est.y_durable:,.0f}, Temporary={final_est.y_temporary:,.0f}")

    result.add_decision(
        decision_id="TC003",
        category="methodology",
        decision=f"Monte Carlo projection with {n_simulations} simulations per scenario",
        rationale="Propagate parameter uncertainty through projection horizon",
        evidence=f"Generated {len(projections)} scenario projections to {horizon_year}",
    )

    return projections


# =============================================================================
# VALIDATION AGAINST EMPIRICAL TARGETS
# =============================================================================


def validate_against_empirical_targets(
    historical_estimates: list[TwoComponentEstimate],
    cohorts: list[ArrivalCohort],
    result: ModuleResult,
) -> dict[str, Any]:
    """
    Validate decomposition against empirical composition targets from ADR-021.

    Targets from Rec #2 and Rec #4:
    - Expansion (2010-2016): 92.4% refugee share
    - Restriction (2017-2020): ~100% refugee
    - Volatility (2021-2024): 6.7% refugee share, 88.3% parole

    Args:
        historical_estimates: Computed historical decomposition
        cohorts: Classified arrival cohorts
        result: ModuleResult for logging

    Returns:
        Validation results dictionary
    """
    print("\n" + "=" * 60)
    print("VALIDATING AGAINST EMPIRICAL TARGETS")
    print("=" * 60)

    validation_results: dict[str, Any] = {
        "regime_validation": {},
        "pep_continuity": {},
        "overall_assessment": None,
    }

    # Empirical targets from Rec #2/Rec #4
    # Using a dataclass-like structure for type safety
    @dataclass
    class ValidationTarget:
        refugee_share: float
        tolerance: float
        description: str

    targets: dict[PolicyRegime, ValidationTarget] = {
        PolicyRegime.EXPANSION: ValidationTarget(
            refugee_share=0.924,
            tolerance=0.10,
            description="Expansion (2010-2016): 92.4% refugee share",
        ),
        PolicyRegime.RESTRICTION: ValidationTarget(
            refugee_share=1.00,  # Near 100%
            tolerance=0.15,  # Higher tolerance due to data quirks
            description="Restriction (2017-2020): ~100% refugee",
        ),
        PolicyRegime.VOLATILITY: ValidationTarget(
            refugee_share=0.067,
            tolerance=0.05,
            description="Volatility (2021-2024): 6.7% refugee share",
        ),
    }

    all_passed = True

    for regime, target in targets.items():
        regime_params = get_regime_params(regime)

        # Get cohorts for regime
        regime_cohorts = [c for c in cohorts if c.regime == regime]

        if not regime_cohorts:
            continue

        # Calculate actual composition from arrivals
        total_refugee = sum(c.refugee_arrivals for c in regime_cohorts)
        total_all = sum(c.get_total_arrivals() for c in regime_cohorts)
        actual_refugee_share = total_refugee / total_all if total_all > 0 else 0.0

        # Check against target
        diff = abs(actual_refugee_share - target.refugee_share)
        passed = diff <= target.tolerance

        if not passed:
            all_passed = False

        validation_results["regime_validation"][regime.value] = {
            "target_refugee_share": target.refugee_share,
            "actual_refugee_share": actual_refugee_share,
            "difference": diff,
            "tolerance": target.tolerance,
            "passed": passed,
            "description": target.description,
            "year_range": f"{regime_params.start_year}-{regime_params.end_year}",
        }

        status = "PASS" if passed else "FAIL"
        print(f"\n  {regime.value.upper()} ({regime_params.start_year}-{regime_params.end_year}) - {status}")
        print(f"    Target refugee share: {target.refugee_share*100:.1f}%")
        print(f"    Actual refugee share: {actual_refugee_share*100:.1f}%")
        print(f"    Difference: {diff*100:.1f}% (tolerance: {target.tolerance*100:.0f}%)")

    # PEP continuity check
    print("\n  PEP CONTINUITY CHECK:")
    continuity_results: list[dict[str, Any]] = []

    for estimate in historical_estimates:
        if estimate.pep_total is not None and estimate.pep_residual is not None:
            residual = estimate.pep_residual
            relative_residual = residual / estimate.pep_total if estimate.pep_total != 0 else 0.0

            continuity_results.append({
                "year": estimate.year,
                "estimated_total": estimate.y_total,
                "pep_total": estimate.pep_total,
                "residual": residual,
                "relative_residual": relative_residual,
            })

    if continuity_results:
        mean_residual = np.mean([r["relative_residual"] for r in continuity_results])
        std_residual = np.std([r["relative_residual"] for r in continuity_results])

        validation_results["pep_continuity"] = {
            "mean_relative_residual": mean_residual,
            "std_relative_residual": std_residual,
            "note": "Positive residual = underestimation (expected for flow vs stock)",
        }

        print(f"    Mean relative residual: {mean_residual*100:.1f}%")
        print(f"    Std relative residual: {std_residual*100:.1f}%")
        print("    Note: Decomposition models surviving stock, PEP measures flow")

    validation_results["overall_assessment"] = "PASS" if all_passed else "PARTIAL"

    result.add_decision(
        decision_id="TC004",
        category="validation",
        decision=f"Empirical validation: {validation_results['overall_assessment']}",
        rationale="Checked decomposition against Rec #2/Rec #4 composition targets",
        evidence=f"Validated against {len(targets)} regime targets",
    )

    return validation_results


# =============================================================================
# VISUALIZATION
# =============================================================================


def plot_historical_decomposition(
    estimates: list[TwoComponentEstimate],
    df_pep: pd.DataFrame,
    result: ModuleResult,
    save_path: Path | None = None,
) -> None:
    """Plot historical two-component decomposition."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Stacked area plot of decomposition
    ax1 = axes[0, 0]
    years = [e.year for e in estimates]
    durable = [e.y_durable for e in estimates]
    temporary = [e.y_temporary for e in estimates]

    ax1.fill_between(years, 0, durable, color=COLORS["durable"], alpha=0.7, label="Durable (Y^dur)")
    ax1.fill_between(years, durable, [d + t for d, t in zip(durable, temporary)],
                     color=COLORS["temporary"], alpha=0.7, label="Temporary (Y^temp)")

    # Add regime shading
    ax1.axvspan(2010, 2016, color="#0072B2", alpha=0.1, label="_Expansion")
    ax1.axvspan(2017, 2020, color="#E31A1C", alpha=0.1, label="_Restriction")
    ax1.axvspan(2021, 2024, color="#D55E00", alpha=0.1, label="_Volatility")

    ax1.set_xlabel("Year", fontsize=11)
    ax1.set_ylabel("Population", fontsize=11)
    ax1.set_title("Historical Decomposition: Y_t = Y_t^dur + Y_t^temp", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Component shares over time
    ax2 = axes[0, 1]
    durable_share = [e.durable_share * 100 for e in estimates]
    temporary_share = [e.temporary_share * 100 for e in estimates]

    ax2.plot(years, durable_share, "s-", color=COLORS["durable"], linewidth=2, label="Durable share")
    ax2.plot(years, temporary_share, "o-", color=COLORS["temporary"], linewidth=2, label="Temporary share")

    ax2.axhline(50, color="#999999", linestyle="--", alpha=0.5)
    ax2.axvline(2017, color="#E31A1C", linestyle="--", alpha=0.5)
    ax2.axvline(2021, color="#D55E00", linestyle="--", alpha=0.5)

    ax2.set_xlabel("Year", fontsize=11)
    ax2.set_ylabel("Share (%)", fontsize=11)
    ax2.set_title("Component Shares by Year", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Comparison with PEP
    ax3 = axes[1, 0]
    pep_lookup = dict(zip(df_pep["year"], df_pep["intl_migration"]))
    estimated_total = [e.y_total for e in estimates]
    pep_values = [pep_lookup.get(y, np.nan) for y in years]

    ax3.plot(years, pep_values, "o-", color=COLORS["historical"], linewidth=2, label="PEP Total (flow)")
    ax3.plot(years, estimated_total, "s--", color=COLORS["total"], linewidth=2, label="Estimated Total (stock)")

    ax3.set_xlabel("Year", fontsize=11)
    ax3.set_ylabel("Migration", fontsize=11)
    ax3.set_title("PEP vs Estimated Total", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Regime-specific composition bars
    ax4 = axes[1, 1]

    regime_data = []
    for regime in PolicyRegime:
        params = get_regime_params(regime)
        regime_ests = [e for e in estimates if params.start_year <= e.year <= params.end_year]
        if regime_ests:
            mean_durable = np.mean([e.durable_share for e in regime_ests])
            mean_temporary = np.mean([e.temporary_share for e in regime_ests])
            regime_data.append({
                "regime": regime.value,
                "durable": mean_durable * 100,
                "temporary": mean_temporary * 100,
            })

    if regime_data:
        x = np.arange(len(regime_data))
        width = 0.35

        bars1 = ax4.bar(x - width/2, [d["durable"] for d in regime_data], width,
                        label="Durable", color=COLORS["durable"])
        bars2 = ax4.bar(x + width/2, [d["temporary"] for d in regime_data], width,
                        label="Temporary", color=COLORS["temporary"])

        ax4.bar_label(bars1, fmt="%.0f%%", fontsize=9)
        ax4.bar_label(bars2, fmt="%.0f%%", fontsize=9)

        ax4.set_xticks(x)
        ax4.set_xticklabels([d["regime"].capitalize() for d in regime_data])
        ax4.set_ylabel("Share (%)", fontsize=11)
        ax4.set_title("Mean Component Shares by Regime", fontsize=12, fontweight="bold")
        ax4.legend(fontsize=9)
        ax4.set_ylim(0, 110)
        ax4.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Module 10: Two-Component Estimand Framework (ADR-021 Rec #1)",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout(rect=(0, 0, 1, 0.96))

    if save_path is None:
        save_path = FIGURES_DIR / "module_10_historical_decomposition"

    for ext in [".png", ".pdf"]:
        fig.savefig(str(save_path) + ext, dpi=300, bbox_inches="tight")

    adr_path = ADR_FIGURES_DIR / "rec1_historical_decomposition"
    for ext in [".png", ".pdf"]:
        fig.savefig(str(adr_path) + ext, dpi=300, bbox_inches="tight")

    plt.close(fig)
    print(f"\nFigure saved: {save_path}.png/.pdf")


def plot_projection_comparison(
    historical_estimates: list[TwoComponentEstimate],
    projections: dict[str, ProjectionTimeSeries],
    result: ModuleResult,
    save_path: Path | None = None,
) -> None:
    """Plot projection comparison across scenarios."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    scenario_colors = {
        "status_quo": "#999999",
        "durable_growth": "#0072B2",
        "parole_cliff": "#D55E00",
        "restriction": "#E31A1C",
        "welcome_corps": "#009E73",
    }

    # Panel 1: Total projection comparison
    ax1 = axes[0, 0]

    # Historical
    hist_years = [e.year for e in historical_estimates]
    hist_total = [e.y_total for e in historical_estimates]
    ax1.plot(hist_years, hist_total, "o-", color=COLORS["historical"], linewidth=2, label="Historical")

    # Projections
    for scenario_name, projection in projections.items():
        color = scenario_colors.get(scenario_name, "#999999")
        years = projection.get_years()
        total = projection.get_total_series()

        ax1.plot(years, total, "-", color=color, linewidth=2, label=scenario_name.replace("_", " ").title())

        # Uncertainty for status_quo
        if scenario_name == "status_quo" and projection.uncertainty:
            ax1.fill_between(
                years,
                projection.uncertainty.get("total_p10", total),
                projection.uncertainty.get("total_p90", total),
                color=color, alpha=0.15
            )

    ax1.axvline(2024, color="#999999", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Year", fontsize=11)
    ax1.set_ylabel("Total Population", fontsize=11)
    ax1.set_title("Total Projection by Scenario", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Durable component projection
    ax2 = axes[0, 1]

    ax2.plot(hist_years, [e.y_durable for e in historical_estimates],
             "o-", color=COLORS["historical"], linewidth=2, label="Historical")

    for scenario_name, projection in projections.items():
        color = scenario_colors.get(scenario_name, "#999999")
        ax2.plot(projection.get_years(), projection.get_durable_series(),
                 "-", color=color, linewidth=2, label=scenario_name.replace("_", " ").title())

    ax2.axvline(2024, color="#999999", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Year", fontsize=11)
    ax2.set_ylabel("Durable Population (Y^dur)", fontsize=11)
    ax2.set_title("Durable Component Projection", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9, loc="upper left")
    ax2.grid(True, alpha=0.3)

    # Panel 3: Temporary component projection
    ax3 = axes[1, 0]

    ax3.plot(hist_years, [e.y_temporary for e in historical_estimates],
             "o-", color=COLORS["historical"], linewidth=2, label="Historical")

    for scenario_name, projection in projections.items():
        color = scenario_colors.get(scenario_name, "#999999")
        ax3.plot(projection.get_years(), projection.get_temporary_series(),
                 "-", color=color, linewidth=2, label=scenario_name.replace("_", " ").title())

    ax3.axvline(2024, color="#999999", linestyle="--", alpha=0.5)
    ax3.set_xlabel("Year", fontsize=11)
    ax3.set_ylabel("Temporary Population (Y^temp)", fontsize=11)
    ax3.set_title("Temporary Component Projection", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=9, loc="upper left")
    ax3.grid(True, alpha=0.3)

    # Panel 4: 2045 summary bar chart
    ax4 = axes[1, 1]

    scenario_names = []
    values_2045_durable = []
    values_2045_temp = []
    colors_list = []

    for scenario_name, projection in projections.items():
        final_est = projection.estimates[-1]
        scenario_names.append(scenario_name.replace("_", " ").title())
        values_2045_durable.append(final_est.y_durable)
        values_2045_temp.append(final_est.y_temporary)
        colors_list.append(scenario_colors.get(scenario_name, "#999999"))

    x = np.arange(len(scenario_names))
    width = 0.35

    bars1 = ax4.bar(x - width/2, values_2045_durable, width,
                    label="Durable", color=COLORS["durable"], alpha=0.8)
    bars2 = ax4.bar(x + width/2, values_2045_temp, width,
                    label="Temporary", color=COLORS["temporary"], alpha=0.8)

    ax4.bar_label(bars1, fmt="{:,.0f}", fontsize=8, rotation=90, padding=3)
    ax4.bar_label(bars2, fmt="{:,.0f}", fontsize=8, rotation=90, padding=3)

    ax4.set_xticks(x)
    ax4.set_xticklabels(scenario_names, rotation=15, ha="right")
    ax4.set_ylabel("Population (2045)", fontsize=11)
    ax4.set_title("2045 Projection by Component", fontsize=12, fontweight="bold")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Two-Component Estimand Projections (2025-2045)",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout(rect=(0, 0, 1, 0.96))

    if save_path is None:
        save_path = FIGURES_DIR / "module_10_projection_comparison"

    for ext in [".png", ".pdf"]:
        fig.savefig(str(save_path) + ext, dpi=300, bbox_inches="tight")

    adr_path = ADR_FIGURES_DIR / "rec1_projection_comparison"
    for ext in [".png", ".pdf"]:
        fig.savefig(str(adr_path) + ext, dpi=300, bbox_inches="tight")

    plt.close(fig)
    print(f"Figure saved: {save_path}.png/.pdf")


# =============================================================================
# MAIN ANALYSIS
# =============================================================================


def run_analysis() -> ModuleResult:
    """Main analysis function for Module 10: Two-Component Estimand."""
    result = ModuleResult(
        module_id="10",
        analysis_name="two_component_estimand",
    )

    print("=" * 70)
    print("Module 10: Two-Component Estimand Framework")
    print("ADR-021 Recommendation #1: Y_t = Y_t^dur + Y_t^temp")
    print(f"Started: {datetime.now(UTC).isoformat()}")
    print("=" * 70)

    # =========================================================================
    # 1. Load Wave 1-3 Parameters
    # =========================================================================
    print("\n[1/8] Loading Wave 1-3 parameters...")

    rec2_params = load_rec2_parameters(result)
    rec3_params = load_rec3_parameters(result)
    rec6_projections = load_rec6_projections(result)

    # =========================================================================
    # 2. Load Data from PostgreSQL
    # =========================================================================
    print("\n[2/8] Loading data from PostgreSQL...")

    df_pep = load_pep_migration(result)
    df_refugee = load_refugee_arrivals(result)
    # Load national totals for input_files tracking (used in future extensions)
    _ = load_national_refugee_totals(result)

    # =========================================================================
    # 3. Classify Arrivals by Status
    # =========================================================================
    print("\n[3/8] Classifying arrivals by status...")

    cohorts = classify_arrivals_by_status(df_pep, df_refugee, result)

    # =========================================================================
    # 4. Build Survival Parameters
    # =========================================================================
    print("\n[4/8] Building survival parameters...")

    survival_params = build_survival_parameters(rec2_params)
    regularization_prob = rec2_params["regularization_probability"]

    print(f"  Regularization probability: {regularization_prob:.1%}")
    print(f"  Refugee 5-year survival: {survival_params[StatusCategory.REFUGEE].survival_5yr:.1%}")
    print(f"  Parole cliff: years {survival_params[StatusCategory.PAROLE_NON_REGULARIZED].cliff_start}-{survival_params[StatusCategory.PAROLE_NON_REGULARIZED].cliff_end}")

    # =========================================================================
    # 5. Compute Historical Decomposition
    # =========================================================================
    print("\n[5/8] Computing historical decomposition...")

    historical_estimates = compute_historical_decomposition(
        cohorts=cohorts,
        df_pep=df_pep,
        survival_params=survival_params,
        regularization_prob=regularization_prob,
        result=result,
    )

    # =========================================================================
    # 6. Validate Against Empirical Targets
    # =========================================================================
    print("\n[6/8] Validating against empirical targets...")

    validation_results = validate_against_empirical_targets(
        historical_estimates=historical_estimates,
        cohorts=cohorts,
        result=result,
    )

    # =========================================================================
    # 7. Generate Projections
    # =========================================================================
    print("\n[7/8] Generating two-component projections...")

    projections = project_two_component_estimand(
        cohorts=cohorts,
        survival_params=survival_params,
        regularization_prob=regularization_prob,
        rec3_params=rec3_params,
        rec6_projections=rec6_projections,
        result=result,
    )

    # =========================================================================
    # 8. Generate Visualizations
    # =========================================================================
    print("\n[8/8] Generating visualizations...")

    plot_historical_decomposition(historical_estimates, df_pep, result)
    plot_projection_comparison(historical_estimates, projections, result)

    # =========================================================================
    # Compile Results
    # =========================================================================
    result.parameters = {
        "estimand_formula": "Y_t = Y_t^dur + Y_t^temp",
        "components": {
            "Y_dur": "Durable = Refugee + Regularized Parole + Other",
            "Y_temp": "Temporary = Non-regularized Parole",
        },
        "wave_inputs": {
            "rec2_regularization_probability": rec2_params["regularization_probability"],
            "rec2_refugee_survival_5yr": rec2_params["refugee_survival_5yr"],
            "rec2_parole_survival_5yr": rec2_params["parole_survival_5yr"],
            "rec3_capacity_multiplier": rec3_params["capacity_multiplier"],
            "rec3_nd_share": rec3_params["nd_share"],
        },
        "survival_parameters": {
            status.value: {
                "survival_1yr": params.survival_1yr,
                "survival_5yr": params.survival_5yr,
                "survival_10yr": params.survival_10yr,
                "cliff": (params.cliff_start, params.cliff_end) if params.cliff_start else None,
            }
            for status, params in survival_params.items()
        },
        "projection_settings": {
            "base_year": 2024,
            "horizon_year": 2045,
            "n_simulations": 500,
        },
    }

    result.results = {
        "historical_decomposition": {
            "summary": {
                "n_years": len(historical_estimates),
                "total_cohorts": len(cohorts),
                "mean_durable": np.mean([e.y_durable for e in historical_estimates]),
                "mean_temporary": np.mean([e.y_temporary for e in historical_estimates]),
                "mean_durable_share": np.mean([e.durable_share for e in historical_estimates]),
            },
            "by_year": [e.to_dict() for e in historical_estimates],
        },
        "validation": validation_results,
        "projections": {
            name: proj.to_dict() for name, proj in projections.items()
        },
        "projection_summary_2045": {
            name: {
                "total": proj.estimates[-1].y_total,
                "durable": proj.estimates[-1].y_durable,
                "temporary": proj.estimates[-1].y_temporary,
                "durable_share": proj.estimates[-1].durable_share,
            }
            for name, proj in projections.items()
        },
    }

    result.diagnostics = {
        "cohort_summary": {
            "n_cohorts": len(cohorts),
            "year_range": f"{cohorts[0].arrival_year}-{cohorts[-1].arrival_year}",
            "total_arrivals": sum(c.get_total_arrivals() for c in cohorts),
        },
        "validation_status": validation_results["overall_assessment"],
        "scenarios_projected": list(projections.keys()),
    }

    result.next_steps = [
        "Integrate two-component estimates with cohort-component population model",
        "Develop age-specific decomposition (young/working-age/senior)",
        "Connect to workforce and fiscal impact modules",
        "Monitor 2025+ data for validation of projections",
        "Extend to multi-state decomposition for comparative analysis",
    ]

    return result


def main() -> int:
    """Main entry point."""
    try:
        result = run_analysis()

        # Save results
        output_file = result.save("module_10_two_component_estimand.json")
        adr_output = result.save(
            "rec1_two_component_estimand.json",
            output_dir=ADR_RESULTS_DIR,
        )

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("=" * 70)

        print(f"\nMain output: {output_file}")
        print(f"ADR output: {adr_output}")

        # Summary
        print("\n" + "-" * 70)
        print("KEY RESULTS")
        print("-" * 70)

        print("\n1. ESTIMAND DECOMPOSITION:")
        print("   Y_t = Y_t^dur + Y_t^temp")
        print("   Y^dur = Refugee + Regularized Parole + Other (high retention)")
        print("   Y^temp = Non-regularized Parole (cliff risk)")

        if "historical_decomposition" in result.results:
            summary = result.results["historical_decomposition"]["summary"]
            print("\n2. HISTORICAL MEAN (2010-2024):")
            print(f"   Durable (Y^dur): {summary['mean_durable']:,.0f}")
            print(f"   Temporary (Y^temp): {summary['mean_temporary']:,.0f}")
            print(f"   Durable share: {summary['mean_durable_share']*100:.1f}%")

        if "projection_summary_2045" in result.results:
            print("\n3. 2045 PROJECTIONS:")
            for scenario, summary in result.results["projection_summary_2045"].items():
                print(f"   {scenario}: Total={summary['total']:,.0f} (Durable={summary['durable']:,.0f}, Temp={summary['temporary']:,.0f})")

        if "validation" in result.results:
            print(f"\n4. VALIDATION: {result.results['validation']['overall_assessment']}")

        if result.warnings:
            print("\nWarnings:")
            for w in result.warnings:
                print(f"  - {w}")

        print(f"\nDecisions logged: {len(result.decisions)}")
        for d in result.decisions:
            print(f"  [{d['decision_id']}] {d['decision'][:60]}...")

        print("\nFigures generated:")
        print("  - module_10_historical_decomposition.png/pdf")
        print("  - module_10_projection_comparison.png/pdf")
        print("  - rec1_historical_decomposition.png/pdf (ADR)")
        print("  - rec1_projection_comparison.png/pdf (ADR)")

        return 0

    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
