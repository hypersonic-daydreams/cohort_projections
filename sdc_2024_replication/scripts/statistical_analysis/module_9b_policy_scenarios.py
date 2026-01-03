#!/usr/bin/env python3
"""
Module 9b: Policy-Lever Scenario Framework (ADR-021 Recommendation #6)
=======================================================================

Redesigns the scenario framework from generic growth-rate scenarios to
mechanism-based policy scenarios that leverage outputs from Wave 1 and Wave 2.

Key Features:
1. Explicit policy levers:
   - Refugee ceiling trajectory (Presidential Determinations)
   - Parole program continuation vs termination
   - Regularization probability (from Rec #2)
   - ND reception capacity parameter (from Rec #3)

2. Named policy scenarios:
   - "Durable-Growth": High ceiling + full capacity recovery + high regularization
   - "Parole-Cliff": Near-term high (parole), then attrition years 2-4
   - "Restriction": Low ceiling + capacity drag + low regularization
   - "Welcome-Corps": Private sponsorship growth scenario
   - "Status-Quo": Current policy trajectory continuation

3. Integrations:
   - Rec #2: Status-specific survival curves and regularization probability
   - Rec #3: LSSND capacity multiplier (67.2%)
   - Rec #4: PolicyRegime framework for period classification

Usage:
    uv run python module_9b_policy_scenarios.py

References:
- ADR-021 Phase B Wave 3
- Rec #2: module_8b_status_durability.py
- Rec #3: rec3_lssnd_synthetic_control.json
- Rec #4: module_regime_framework.py
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

SCENARIO_COLORS = {
    "durable_growth": "#0072B2",  # Blue - optimistic
    "parole_cliff": "#D55E00",  # Vermillion - cliff scenario
    "restriction": "#E31A1C",  # Red - restrictive
    "welcome_corps": "#009E73",  # Teal - new pathway
    "status_quo": "#999999",  # Gray - baseline
    "historical": "#000000",  # Black - historical
}

COMPONENT_COLORS = {
    "durable": "#0072B2",  # Blue
    "temporary": "#D55E00",  # Orange
    "total": "#999999",  # Gray
}


# =============================================================================
# DATA CLASSES FOR POLICY SCENARIOS
# =============================================================================


class ScenarioType(Enum):
    """Enumeration of named policy scenarios."""

    DURABLE_GROWTH = "durable_growth"
    PAROLE_CLIFF = "parole_cliff"
    RESTRICTION = "restriction"
    WELCOME_CORPS = "welcome_corps"
    STATUS_QUO = "status_quo"


@dataclass(frozen=True)
class RefugeeCeilingTrajectory:
    """
    Refugee ceiling trajectory over projection horizon.

    Attributes:
        years: List of fiscal years
        ceilings: List of ceiling values for each year
        description: Human-readable description
    """

    years: tuple[int, ...]
    ceilings: tuple[int, ...]
    description: str

    def get_ceiling(self, year: int) -> int:
        """Get ceiling for a specific year, using last value for years beyond range."""
        if year in self.years:
            idx = self.years.index(year)
            return self.ceilings[idx]
        elif year > max(self.years):
            return self.ceilings[-1]  # Use last ceiling
        elif year < min(self.years):
            return self.ceilings[0]  # Use first ceiling
        else:
            # Interpolate for missing years
            for i, y in enumerate(self.years):
                if y > year:
                    return self.ceilings[i - 1]
            return self.ceilings[-1]


@dataclass(frozen=True)
class CapacityTrajectory:
    """
    Local resettlement capacity trajectory (from Rec #3).

    Models recovery from LSSND closure shock.

    Attributes:
        base_multiplier: Initial capacity multiplier (0.672 from Rec #3)
        target_multiplier: Target capacity after recovery
        recovery_years: Years to reach target capacity
        recovery_path: "linear", "exponential", or "step"
    """

    base_multiplier: float
    target_multiplier: float
    recovery_years: int
    recovery_path: str

    def get_multiplier(self, years_from_base: int) -> float:
        """Get capacity multiplier at t years from base year."""
        if years_from_base <= 0:
            return self.base_multiplier
        if years_from_base >= self.recovery_years:
            return self.target_multiplier

        progress = years_from_base / self.recovery_years

        if self.recovery_path == "linear":
            return self.base_multiplier + progress * (
                self.target_multiplier - self.base_multiplier
            )
        elif self.recovery_path == "exponential":
            # Exponential approach to target
            return self.target_multiplier - (
                self.target_multiplier - self.base_multiplier
            ) * np.exp(-3 * progress)
        else:  # step
            return self.target_multiplier if progress >= 0.5 else self.base_multiplier


@dataclass(frozen=True)
class StatusDurabilityParams:
    """
    Status durability parameters (from Rec #2).

    Attributes:
        regularization_probability: Probability parole -> permanent status
        refugee_survival_5yr: 5-year survival probability for refugees
        parole_survival_5yr: 5-year survival probability for parolees
        parole_cliff_years: Years when parole cliff occurs (start, end)
    """

    regularization_probability: float
    refugee_survival_5yr: float
    parole_survival_5yr: float
    parole_cliff_years: tuple[float, float]


@dataclass
class PolicyScenario:
    """
    Complete policy scenario definition.

    Combines all policy levers into a coherent scenario for projection.

    Attributes:
        name: Scenario identifier (ScenarioType enum value)
        display_name: Human-readable name
        description: Detailed description
        refugee_ceiling: RefugeeCeilingTrajectory
        parole_continuation: Whether parole programs continue
        parole_annual_estimate: Estimated annual parole arrivals if continuing
        capacity: CapacityTrajectory
        durability: StatusDurabilityParams
        welcome_corps_growth: Optional private sponsorship growth rate
    """

    name: ScenarioType
    display_name: str
    description: str
    refugee_ceiling: RefugeeCeilingTrajectory
    parole_continuation: bool
    parole_annual_estimate: int
    capacity: CapacityTrajectory
    durability: StatusDurabilityParams
    welcome_corps_growth: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name.value,
            "display_name": self.display_name,
            "description": self.description,
            "refugee_ceiling": {
                "years": list(self.refugee_ceiling.years),
                "ceilings": list(self.refugee_ceiling.ceilings),
                "description": self.refugee_ceiling.description,
            },
            "parole_continuation": self.parole_continuation,
            "parole_annual_estimate": self.parole_annual_estimate,
            "capacity": {
                "base_multiplier": self.capacity.base_multiplier,
                "target_multiplier": self.capacity.target_multiplier,
                "recovery_years": self.capacity.recovery_years,
                "recovery_path": self.capacity.recovery_path,
            },
            "durability": {
                "regularization_probability": self.durability.regularization_probability,
                "refugee_survival_5yr": self.durability.refugee_survival_5yr,
                "parole_survival_5yr": self.durability.parole_survival_5yr,
                "parole_cliff_years": list(self.durability.parole_cliff_years),
            },
            "welcome_corps_growth": self.welcome_corps_growth,
        }


@dataclass
class ScenarioProjection:
    """
    Projection results for a single scenario.

    Attributes:
        scenario: The PolicyScenario used
        years: Projection years
        durable_component: Refugee + regularized parole (high retention)
        temporary_component: Non-regularized parole (low retention)
        total_projection: Sum of durable + temporary
        uncertainty_bounds: Dict with p10, p50, p90 for total projection
        decomposition: Year-by-year breakdown by source
    """

    scenario: PolicyScenario
    years: list[int]
    durable_component: list[float]
    temporary_component: list[float]
    total_projection: list[float]
    uncertainty_bounds: dict[str, list[float]]
    decomposition: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "scenario_name": self.scenario.name.value,
            "scenario_display_name": self.scenario.display_name,
            "years": self.years,
            "durable_component": self.durable_component,
            "temporary_component": self.temporary_component,
            "total_projection": self.total_projection,
            "uncertainty_bounds": self.uncertainty_bounds,
            "decomposition": self.decomposition,
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
# LOAD WAVE 1-2 RESULTS
# =============================================================================


def load_rec3_capacity_parameter(result: ModuleResult) -> float:
    """
    Load capacity multiplier from Rec #3 synthetic control analysis.

    Returns the LSSND capacity multiplier (approximately 0.672).
    """
    rec3_path = ADR_RESULTS_DIR / "rec3_lssnd_synthetic_control.json"

    if not rec3_path.exists():
        # Fall back to local results directory
        rec3_path = RESULTS_DIR / "module_7b_lssnd_synthetic_control.json"

    if rec3_path.exists():
        with open(rec3_path) as f:
            rec3_data = json.load(f)
        result.input_files.append(str(rec3_path))

        # Extract capacity parameter
        capacity_param = rec3_data.get("results", {}).get("capacity_parameter", {})
        capacity_multiplier = capacity_param.get("value", 0.672)

        print(f"  Loaded Rec #3 capacity multiplier: {capacity_multiplier:.3f}")
        return float(capacity_multiplier)
    else:
        result.warnings.append(
            "Rec #3 results not found; using default capacity multiplier 0.672"
        )
        print("  WARNING: Rec #3 results not found; using default 0.672")
        return 0.672


def load_rec2_durability_params(result: ModuleResult) -> dict[str, Any]:
    """
    Load status durability parameters from Rec #2 analysis.

    Returns regularization probability and survival parameters.
    """
    rec2_path = ADR_RESULTS_DIR / "rec2_status_durability.json"

    if not rec2_path.exists():
        rec2_path = RESULTS_DIR / "module_8b_status_durability.json"

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

        params = {
            "regularization_central": uncertainty.get("central_estimate", 0.503),
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
        }

        print(f"  Loaded Rec #2 regularization: {params['regularization_central']:.1%}")
        print(f"  Refugee 5-year survival: {params['refugee_survival_5yr']:.1%}")
        print(f"  Parole 5-year survival: {params['parole_survival_5yr']:.1%}")

        return params
    else:
        result.warnings.append(
            "Rec #2 results not found; using default durability params"
        )
        print("  WARNING: Rec #2 results not found; using defaults")
        return {
            "regularization_central": 0.503,
            "regularization_lower": 0.299,
            "regularization_upper": 0.725,
            "refugee_survival_5yr": 0.954,
            "parole_survival_5yr": 0.341,
            "parole_cliff_start": 2.0,
            "parole_cliff_end": 4.0,
        }


def load_historical_data(result: ModuleResult) -> pd.DataFrame:
    """
    Load historical ND international migration data.

    Returns DataFrame with year and nd_intl_migration columns.
    """
    conn = db_config.get_db_connection()
    try:
        query = """
        SELECT
            year,
            intl_migration as nd_intl_migration
        FROM census.state_components
        WHERE state_name = 'North Dakota'
          AND intl_migration IS NOT NULL
        ORDER BY year
        """
        df = pd.read_sql(query, conn)
        result.input_files.append("census.state_components (PostgreSQL)")
        print(f"  Loaded historical data: {len(df)} years ({df['year'].min()}-{df['year'].max()})")
        return df
    finally:
        conn.close()


def load_nd_refugee_arrivals(result: ModuleResult) -> pd.DataFrame:
    """Load ND refugee arrivals by fiscal year."""
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
    """Load national refugee arrival totals by fiscal year."""
    conn = db_config.get_db_connection()
    try:
        query = """
        SELECT
            fiscal_year as year,
            SUM(arrivals) as national_arrivals
        FROM rpc.refugee_arrivals
        GROUP BY fiscal_year
        ORDER BY fiscal_year
        """
        df = pd.read_sql(query, conn)
        result.input_files.append("rpc.refugee_arrivals (national, PostgreSQL)")
        print(f"  Loaded national refugee totals: {len(df)} years")
        return df
    finally:
        conn.close()


# =============================================================================
# BUILD POLICY SCENARIOS
# =============================================================================


def build_policy_scenarios(
    capacity_multiplier: float,
    durability_params: dict[str, Any],
    result: ModuleResult,
) -> dict[ScenarioType, PolicyScenario]:
    """
    Build the five named policy scenarios with parameters from Wave 1-2.

    Scenarios:
    1. Durable-Growth: Optimistic - high ceilings, capacity recovery, high regularization
    2. Parole-Cliff: Near-term parole surge, then attrition
    3. Restriction: Low ceilings, capacity drag, low regularization
    4. Welcome-Corps: Private sponsorship growth
    5. Status-Quo: Current trajectory continuation
    """
    print("\n" + "=" * 60)
    print("BUILDING POLICY SCENARIOS")
    print("=" * 60)

    scenarios: dict[ScenarioType, PolicyScenario] = {}

    # =========================================================================
    # Scenario 1: DURABLE-GROWTH
    # =========================================================================
    # High refugee ceilings, full capacity recovery, high regularization
    durable_growth = PolicyScenario(
        name=ScenarioType.DURABLE_GROWTH,
        display_name="Durable Growth",
        description=(
            "Optimistic scenario: Refugee ceilings rise to 150,000 by FY2030, "
            "ND resettlement capacity fully recovers over 5 years (Global Refuge expansion), "
            "and 72.5% of parolees regularize to permanent status."
        ),
        refugee_ceiling=RefugeeCeilingTrajectory(
            years=tuple(range(2025, 2031)) + (2035, 2040, 2045),
            ceilings=(125000, 130000, 140000, 145000, 150000, 150000, 150000, 150000, 150000),
            description="Rising ceilings to 150K by FY2030, sustained",
        ),
        parole_continuation=True,
        parole_annual_estimate=2000,  # Continued parole programs
        capacity=CapacityTrajectory(
            base_multiplier=capacity_multiplier,
            target_multiplier=1.0,  # Full recovery
            recovery_years=5,
            recovery_path="exponential",
        ),
        durability=StatusDurabilityParams(
            regularization_probability=durability_params["regularization_upper"],
            refugee_survival_5yr=durability_params["refugee_survival_5yr"],
            parole_survival_5yr=0.65,  # Higher due to regularization
            parole_cliff_years=(
                durability_params["parole_cliff_start"],
                durability_params["parole_cliff_end"],
            ),
        ),
        welcome_corps_growth=0.15,  # 15% annual growth in private sponsorship
    )
    scenarios[ScenarioType.DURABLE_GROWTH] = durable_growth
    print(f"\n  {durable_growth.display_name}:")
    print("    Ceiling trajectory: rises to 150,000 by 2030")
    print(f"    Capacity recovery: {capacity_multiplier:.1%} -> 100% over 5 years")
    print(f"    Regularization: {durability_params['regularization_upper']:.1%}")

    # =========================================================================
    # Scenario 2: PAROLE-CLIFF
    # =========================================================================
    # Near-term high parole arrivals, then cliff at years 2-4
    parole_cliff = PolicyScenario(
        name=ScenarioType.PAROLE_CLIFF,
        display_name="Parole Cliff",
        description=(
            "Parole-heavy scenario: Current parole programs (U4U, OAW) wind down, "
            "new parole limited. Near-term high arrivals followed by departure cliff "
            "at years 2-4 as temporary status expires without regularization pathway."
        ),
        refugee_ceiling=RefugeeCeilingTrajectory(
            years=tuple(range(2025, 2031)) + (2035, 2040, 2045),
            ceilings=(50000, 50000, 60000, 70000, 80000, 85000, 90000, 95000, 100000),
            description="Gradual recovery from reduced ceilings",
        ),
        parole_continuation=False,  # Programs wind down
        parole_annual_estimate=500,  # Minimal new parole after 2025
        capacity=CapacityTrajectory(
            base_multiplier=capacity_multiplier,
            target_multiplier=0.85,  # Partial recovery only
            recovery_years=8,
            recovery_path="linear",
        ),
        durability=StatusDurabilityParams(
            regularization_probability=durability_params["regularization_lower"],
            refugee_survival_5yr=durability_params["refugee_survival_5yr"],
            parole_survival_5yr=durability_params["parole_survival_5yr"],  # Low - cliff effect
            parole_cliff_years=(
                durability_params["parole_cliff_start"],
                durability_params["parole_cliff_end"],
            ),
        ),
    )
    scenarios[ScenarioType.PAROLE_CLIFF] = parole_cliff
    print(f"\n  {parole_cliff.display_name}:")
    print("    Ceiling trajectory: starts at 50,000, gradual recovery")
    print("    Parole programs: wind down after 2025")
    print(f"    Regularization: {durability_params['regularization_lower']:.1%}")

    # =========================================================================
    # Scenario 3: RESTRICTION
    # =========================================================================
    # Sustained low ceilings, capacity drag, low regularization
    restriction = PolicyScenario(
        name=ScenarioType.RESTRICTION,
        display_name="Restriction",
        description=(
            "Restrictive scenario: Refugee ceilings remain at 2017-2020 levels (15-30K), "
            "ND capacity remains at LSSND-closure level (67%), parole programs terminated, "
            "minimal regularization pathway (30%)."
        ),
        refugee_ceiling=RefugeeCeilingTrajectory(
            years=tuple(range(2025, 2031)) + (2035, 2040, 2045),
            ceilings=(15000, 20000, 25000, 25000, 30000, 30000, 30000, 30000, 30000),
            description="Low ceilings sustained at Restriction-era levels",
        ),
        parole_continuation=False,
        parole_annual_estimate=200,  # Minimal exceptional cases
        capacity=CapacityTrajectory(
            base_multiplier=capacity_multiplier,
            target_multiplier=capacity_multiplier,  # No recovery
            recovery_years=20,  # Effectively no change
            recovery_path="linear",
        ),
        durability=StatusDurabilityParams(
            regularization_probability=durability_params["regularization_lower"],
            refugee_survival_5yr=durability_params["refugee_survival_5yr"],
            parole_survival_5yr=0.25,  # Very low due to termination
            parole_cliff_years=(
                durability_params["parole_cliff_start"],
                durability_params["parole_cliff_end"],
            ),
        ),
    )
    scenarios[ScenarioType.RESTRICTION] = restriction
    print(f"\n  {restriction.display_name}:")
    print("    Ceiling trajectory: 15,000-30,000 sustained")
    print(f"    Capacity: stuck at {capacity_multiplier:.1%}")
    print("    Parole: terminated")

    # =========================================================================
    # Scenario 4: WELCOME-CORPS
    # =========================================================================
    # Private sponsorship growth as emerging channel
    welcome_corps = PolicyScenario(
        name=ScenarioType.WELCOME_CORPS,
        display_name="Welcome Corps Growth",
        description=(
            "Private sponsorship scenario: Moderate traditional ceilings (100K), "
            "but Welcome Corps private sponsorship grows 20% annually, adding "
            "supplemental resettlement capacity outside traditional agencies."
        ),
        refugee_ceiling=RefugeeCeilingTrajectory(
            years=tuple(range(2025, 2031)) + (2035, 2040, 2045),
            ceilings=(100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000),
            description="Stable moderate ceilings with private sponsorship growth",
        ),
        parole_continuation=True,
        parole_annual_estimate=1500,
        capacity=CapacityTrajectory(
            base_multiplier=capacity_multiplier,
            target_multiplier=1.1,  # Exceeds baseline due to private sponsors
            recovery_years=7,
            recovery_path="exponential",
        ),
        durability=StatusDurabilityParams(
            regularization_probability=durability_params["regularization_central"],
            refugee_survival_5yr=durability_params["refugee_survival_5yr"],
            parole_survival_5yr=0.55,  # Moderate
            parole_cliff_years=(
                durability_params["parole_cliff_start"],
                durability_params["parole_cliff_end"],
            ),
        ),
        welcome_corps_growth=0.20,  # 20% annual growth
    )
    scenarios[ScenarioType.WELCOME_CORPS] = welcome_corps
    print(f"\n  {welcome_corps.display_name}:")
    print("    Ceiling: stable at 100,000")
    print("    Welcome Corps growth: 20% annually")
    print("    Capacity: exceeds baseline (1.1x) due to private sponsors")

    # =========================================================================
    # Scenario 5: STATUS-QUO
    # =========================================================================
    # Current policy trajectory continuation
    status_quo = PolicyScenario(
        name=ScenarioType.STATUS_QUO,
        display_name="Status Quo",
        description=(
            "Baseline scenario: Current policy trajectory continues - "
            "125K ceilings with partial achievement (~60%), existing parole programs "
            "continue with uncertainty, partial capacity recovery, central regularization estimate."
        ),
        refugee_ceiling=RefugeeCeilingTrajectory(
            years=tuple(range(2025, 2031)) + (2035, 2040, 2045),
            ceilings=(125000, 125000, 125000, 125000, 125000, 125000, 125000, 125000, 125000),
            description="Ceilings maintained at 125K (actual arrivals ~60%)",
        ),
        parole_continuation=True,
        parole_annual_estimate=2500,  # Current parole levels
        capacity=CapacityTrajectory(
            base_multiplier=capacity_multiplier,
            target_multiplier=0.90,  # Partial recovery
            recovery_years=6,
            recovery_path="linear",
        ),
        durability=StatusDurabilityParams(
            regularization_probability=durability_params["regularization_central"],
            refugee_survival_5yr=durability_params["refugee_survival_5yr"],
            parole_survival_5yr=0.50,  # Central estimate
            parole_cliff_years=(
                durability_params["parole_cliff_start"],
                durability_params["parole_cliff_end"],
            ),
        ),
        welcome_corps_growth=0.10,  # Modest 10% growth
    )
    scenarios[ScenarioType.STATUS_QUO] = status_quo
    print(f"\n  {status_quo.display_name}:")
    print("    Ceiling: 125,000 (60% achieved)")
    print(f"    Capacity recovery: {capacity_multiplier:.1%} -> 90% over 6 years")
    print(f"    Regularization: {durability_params['regularization_central']:.1%}")

    result.add_decision(
        decision_id="PS001",
        category="scenario_design",
        decision="Built 5 named policy scenarios with mechanism-based parameters",
        rationale="Replace generic growth scenarios with explicit policy levers",
        alternatives=[
            "Single central scenario with uncertainty",
            "Monte Carlo over all parameters",
        ],
        evidence=f"Capacity multiplier={capacity_multiplier:.3f}, Regularization range=[{durability_params['regularization_lower']:.1%}, {durability_params['regularization_upper']:.1%}]",
    )

    return scenarios


# =============================================================================
# ND SHARE ESTIMATION
# =============================================================================


def estimate_nd_share(
    df_refugee: pd.DataFrame,
    df_national: pd.DataFrame,
    result: ModuleResult,
) -> float:
    """
    Estimate ND's historical share of national refugee arrivals.

    Uses pre-LSSND closure (2010-2019) average share as baseline.
    """
    # Merge ND and national
    df_merged = df_refugee.merge(df_national, on="year", how="inner")

    # Calculate share by year
    df_merged["nd_share"] = df_merged["refugee_arrivals"] / df_merged["national_arrivals"]

    # Pre-LSSND (2010-2019) average
    pre_lssnd = df_merged[(df_merged["year"] >= 2010) & (df_merged["year"] <= 2019)]

    if len(pre_lssnd) > 0:
        nd_share = float(pre_lssnd["nd_share"].mean())
        print(f"  ND share of national refugees (2010-2019): {nd_share:.4f} ({nd_share*100:.2f}%)")
    else:
        nd_share = 0.0074  # Default from Rec #3
        result.warnings.append("Could not calculate ND share; using default 0.74%")

    result.add_decision(
        decision_id="PS002",
        category="parameters",
        decision=f"Using ND share = {nd_share:.4f} for refugee allocation",
        rationale="Historical share during stable period (2010-2019) before LSSND closure",
        evidence=f"Based on {len(pre_lssnd)} years of data",
    )

    return nd_share


# =============================================================================
# PROJECTION ENGINE
# =============================================================================


def project_scenario(
    scenario: PolicyScenario,
    nd_share: float,
    base_year: int = 2024,
    horizon_end: int = 2045,
    n_simulations: int = 1000,
    result: ModuleResult | None = None,
) -> ScenarioProjection:
    """
    Generate projection for a single policy scenario.

    Uses Monte Carlo simulation for uncertainty quantification.

    Args:
        scenario: PolicyScenario to project
        nd_share: ND's share of national refugee arrivals
        base_year: Starting year for projection
        horizon_end: End year for projection
        n_simulations: Number of Monte Carlo draws
        result: Optional ModuleResult for logging

    Returns:
        ScenarioProjection with point estimates and uncertainty bounds
    """
    rng = np.random.default_rng(42)

    proj_years = list(range(base_year + 1, horizon_end + 1))
    n_years = len(proj_years)

    # Storage for Monte Carlo simulation
    durable_sims = np.zeros((n_simulations, n_years))
    temporary_sims = np.zeros((n_simulations, n_years))

    # Ceiling achievement rate (ceilings are aspirational)
    ceiling_achievement = 0.60  # Typical ~60% of ceiling achieved

    for sim in range(n_simulations):
        # Draw stochastic parameters
        ceiling_variation = rng.uniform(0.85, 1.15)
        share_variation = rng.uniform(0.90, 1.10)
        parole_variation = rng.uniform(0.80, 1.20)

        # Track cumulative cohorts for survival modeling
        parole_cohorts: list[tuple[int, float]] = []  # (arrival_year, amount)

        for t, year in enumerate(proj_years):
            years_from_base = year - base_year

            # Get capacity multiplier for this year
            cap_mult = scenario.capacity.get_multiplier(years_from_base)

            # --- Refugee Component ---
            ceiling = scenario.refugee_ceiling.get_ceiling(year)
            national_arrivals = ceiling * ceiling_achievement * ceiling_variation
            nd_refugee = national_arrivals * nd_share * cap_mult * share_variation

            # --- Parole Component ---
            if scenario.parole_continuation:
                nd_parole_new = scenario.parole_annual_estimate * parole_variation
            else:
                # Parole winds down
                wind_down_factor = max(0, 1 - (years_from_base / 3))
                nd_parole_new = scenario.parole_annual_estimate * wind_down_factor * parole_variation

            parole_cohorts.append((year, nd_parole_new))

            # --- Welcome Corps Component ---
            welcome_corps = 0.0
            if scenario.welcome_corps_growth is not None:
                # Start with ~50 ND arrivals, grow at specified rate
                base_wc = 50
                welcome_corps = base_wc * (1 + scenario.welcome_corps_growth) ** years_from_base

            # --- Apply Survival/Durability ---
            # Durable: refugees + regularized parole survivors + welcome corps
            durable = nd_refugee * scenario.durability.refugee_survival_5yr
            durable += welcome_corps * scenario.durability.refugee_survival_5yr

            # Parole survivors after cliff
            for cohort_year, cohort_amount in parole_cohorts:
                cohort_age = year - cohort_year
                cliff_start, cliff_end = scenario.durability.parole_cliff_years

                if cohort_age <= cliff_start:
                    # Pre-cliff: full survival
                    survival_factor = 1.0
                elif cohort_age <= cliff_end:
                    # During cliff
                    cliff_progress = (cohort_age - cliff_start) / (cliff_end - cliff_start)
                    reg_prob = scenario.durability.regularization_probability
                    # Regularized survive, non-regularized depart
                    survival_factor = reg_prob + (1 - reg_prob) * (1 - cliff_progress)
                else:
                    # Post-cliff
                    reg_prob = scenario.durability.regularization_probability
                    survival_factor = reg_prob * scenario.durability.refugee_survival_5yr + \
                                      (1 - reg_prob) * 0.10  # Non-regularized mostly departed

                # Add surviving parole to durable or temporary
                if cohort_age > cliff_end:
                    # Post-cliff survivors are mostly regularized = durable
                    durable += cohort_amount * survival_factor * reg_prob
                    temporary_sims[sim, t] += cohort_amount * survival_factor * (1 - reg_prob)
                else:
                    temporary_sims[sim, t] += cohort_amount * survival_factor

            durable_sims[sim, t] = durable

    # Compute summary statistics
    total_sims = durable_sims + temporary_sims

    durable_point = np.median(durable_sims, axis=0).tolist()
    temporary_point = np.median(temporary_sims, axis=0).tolist()
    total_point = np.median(total_sims, axis=0).tolist()

    uncertainty_bounds = {
        "p10": np.percentile(total_sims, 10, axis=0).tolist(),
        "p25": np.percentile(total_sims, 25, axis=0).tolist(),
        "p50": np.percentile(total_sims, 50, axis=0).tolist(),
        "p75": np.percentile(total_sims, 75, axis=0).tolist(),
        "p90": np.percentile(total_sims, 90, axis=0).tolist(),
    }

    # Create decomposition
    decomposition = []
    for t, year in enumerate(proj_years):
        decomposition.append({
            "year": year,
            "durable": durable_point[t],
            "temporary": temporary_point[t],
            "total": total_point[t],
            "uncertainty_width": uncertainty_bounds["p90"][t] - uncertainty_bounds["p10"][t],
        })

    return ScenarioProjection(
        scenario=scenario,
        years=proj_years,
        durable_component=durable_point,
        temporary_component=temporary_point,
        total_projection=total_point,
        uncertainty_bounds=uncertainty_bounds,
        decomposition=decomposition,
    )


def run_all_projections(
    scenarios: dict[ScenarioType, PolicyScenario],
    nd_share: float,
    result: ModuleResult,
    n_simulations: int = 1000,
) -> dict[ScenarioType, ScenarioProjection]:
    """
    Run projections for all scenarios.
    """
    print("\n" + "=" * 60)
    print("RUNNING SCENARIO PROJECTIONS")
    print("=" * 60)

    projections: dict[ScenarioType, ScenarioProjection] = {}

    for scenario_type, scenario in scenarios.items():
        print(f"\n  Projecting: {scenario.display_name}...")
        projection = project_scenario(
            scenario=scenario,
            nd_share=nd_share,
            n_simulations=n_simulations,
            result=result,
        )
        projections[scenario_type] = projection

        # Print summary
        total_2030 = projection.total_projection[5]  # 2030 is index 5 (2025, 2026, ..., 2030)
        total_2045 = projection.total_projection[-1]
        print(f"    2030: {total_2030:,.0f} (durable: {projection.durable_component[5]:,.0f})")
        print(f"    2045: {total_2045:,.0f} (durable: {projection.durable_component[-1]:,.0f})")

    result.add_decision(
        decision_id="PS003",
        category="methodology",
        decision=f"Ran Monte Carlo projections with {n_simulations} simulations per scenario",
        rationale="Propagate parameter uncertainty through projections",
        alternatives=["Deterministic projections", "Bootstrapping"],
        evidence="Random seed fixed at 42 for reproducibility",
    )

    return projections


# =============================================================================
# VISUALIZATION
# =============================================================================


def plot_scenario_comparison(
    df_historical: pd.DataFrame,
    projections: dict[ScenarioType, ScenarioProjection],
    result: ModuleResult,
    save_path: Path | None = None,
) -> None:
    """Plot comparison of all scenarios with historical data."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Historical data
    hist_years = df_historical["year"].values
    hist_values = df_historical["nd_intl_migration"].values

    ax.plot(
        hist_years,
        hist_values,
        "o-",
        color=SCENARIO_COLORS["historical"],
        linewidth=2,
        markersize=6,
        label="Historical",
        zorder=10,
    )

    # Plot each scenario
    for scenario_type, projection in projections.items():
        color = SCENARIO_COLORS.get(scenario_type.value, "#999999")
        years = projection.years
        total = projection.total_projection

        ax.plot(
            years,
            total,
            "-",
            color=color,
            linewidth=2.5,
            label=projection.scenario.display_name,
        )

        # Add confidence band for status quo
        if scenario_type == ScenarioType.STATUS_QUO:
            ax.fill_between(
                years,
                projection.uncertainty_bounds["p10"],
                projection.uncertainty_bounds["p90"],
                color=color,
                alpha=0.15,
                label="Status Quo 80% CI",
            )

        # Connect to historical
        ax.plot(
            [hist_years[-1], years[0]],
            [hist_values[-1], total[0]],
            ":",
            color=color,
            linewidth=1,
            alpha=0.5,
        )

    # Annotations
    ax.axvline(2024, color="#999999", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(2024.2, ax.get_ylim()[1] * 0.95, "Base Year", fontsize=9, color="#999999")

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("ND International Migration", fontsize=12)
    ax.set_xlim(2009, 2046)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Title
    ax.set_title(
        "Policy Scenario Comparison: ND International Migration (2025-2045)",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()

    if save_path is None:
        save_path = FIGURES_DIR / "module_9b_scenario_comparison"

    for ext in [".png", ".pdf"]:
        fig.savefig(str(save_path) + ext, dpi=300, bbox_inches="tight")

    # Also save to ADR directory
    adr_path = ADR_FIGURES_DIR / "rec6_scenario_comparison"
    for ext in [".png", ".pdf"]:
        fig.savefig(str(adr_path) + ext, dpi=300, bbox_inches="tight")

    plt.close(fig)
    print(f"\nFigure saved: {save_path}.png/.pdf")


def plot_durable_temporary_decomposition(
    projections: dict[ScenarioType, ScenarioProjection],
    result: ModuleResult,
    save_path: Path | None = None,
) -> None:
    """Plot durable vs temporary component decomposition for key scenarios."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    scenarios_to_plot = [
        ScenarioType.DURABLE_GROWTH,
        ScenarioType.PAROLE_CLIFF,
        ScenarioType.STATUS_QUO,
        ScenarioType.RESTRICTION,
    ]

    for ax_idx, scenario_type in enumerate(scenarios_to_plot):
        ax = axes[ax_idx // 2, ax_idx % 2]

        if scenario_type not in projections:
            continue

        projection = projections[scenario_type]
        years = projection.years
        durable = projection.durable_component
        temporary = projection.temporary_component

        # Stacked area
        ax.fill_between(
            years,
            durable,
            color=COMPONENT_COLORS["durable"],
            alpha=0.7,
            label="Durable (Refugee + Regularized)",
        )
        ax.fill_between(
            years,
            durable,
            np.array(durable) + np.array(temporary),
            color=COMPONENT_COLORS["temporary"],
            alpha=0.7,
            label="Temporary (Non-regularized Parole)",
        )

        ax.set_title(projection.scenario.display_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Year", fontsize=10)
        ax.set_ylabel("Migration", fontsize=10)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Durable vs Temporary Components by Scenario",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.96))

    if save_path is None:
        save_path = FIGURES_DIR / "module_9b_component_decomposition"

    for ext in [".png", ".pdf"]:
        fig.savefig(str(save_path) + ext, dpi=300, bbox_inches="tight")

    adr_path = ADR_FIGURES_DIR / "rec6_component_decomposition"
    for ext in [".png", ".pdf"]:
        fig.savefig(str(adr_path) + ext, dpi=300, bbox_inches="tight")

    plt.close(fig)
    print(f"Figure saved: {save_path}.png/.pdf")


def plot_policy_lever_sensitivity(
    projections: dict[ScenarioType, ScenarioProjection],
    result: ModuleResult,
    save_path: Path | None = None,
) -> None:
    """Plot sensitivity of 2045 outcomes to policy levers."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Extract 2045 values for each scenario
    scenario_names = []
    values_2045 = []
    durable_2045 = []
    colors = []

    for scenario_type in [
        ScenarioType.RESTRICTION,
        ScenarioType.PAROLE_CLIFF,
        ScenarioType.STATUS_QUO,
        ScenarioType.WELCOME_CORPS,
        ScenarioType.DURABLE_GROWTH,
    ]:
        if scenario_type in projections:
            projection = projections[scenario_type]
            scenario_names.append(projection.scenario.display_name)
            values_2045.append(projection.total_projection[-1])
            durable_2045.append(projection.durable_component[-1])
            colors.append(SCENARIO_COLORS.get(scenario_type.value, "#999999"))

    x = np.arange(len(scenario_names))
    width = 0.35

    bars_total = ax.bar(x - width / 2, values_2045, width, label="Total 2045", color=colors, alpha=0.8)
    bars_durable = ax.bar(x + width / 2, durable_2045, width, label="Durable 2045", color=colors, alpha=0.4, hatch="//")

    ax.bar_label(bars_total, fmt="{:,.0f}", fontsize=9)
    ax.bar_label(bars_durable, fmt="{:,.0f}", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, rotation=15, ha="right")
    ax.set_ylabel("Projected Migration (2045)", fontsize=12)
    ax.set_title("2045 Projections by Policy Scenario", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path is None:
        save_path = FIGURES_DIR / "module_9b_policy_sensitivity"

    for ext in [".png", ".pdf"]:
        fig.savefig(str(save_path) + ext, dpi=300, bbox_inches="tight")

    adr_path = ADR_FIGURES_DIR / "rec6_policy_sensitivity"
    for ext in [".png", ".pdf"]:
        fig.savefig(str(adr_path) + ext, dpi=300, bbox_inches="tight")

    plt.close(fig)
    print(f"Figure saved: {save_path}.png/.pdf")


# =============================================================================
# MAIN ANALYSIS
# =============================================================================


def run_analysis() -> ModuleResult:
    """Main analysis function for Module 9b Policy Scenarios."""
    result = ModuleResult(
        module_id="9b",
        analysis_name="policy_scenarios",
    )

    print("=" * 70)
    print("Module 9b: Policy-Lever Scenario Framework")
    print("ADR-021 Recommendation #6")
    print(f"Started: {datetime.now(UTC).isoformat()}")
    print("=" * 70)

    # =========================================================================
    # 1. Load Wave 1-2 Results
    # =========================================================================
    print("\n[1/6] Loading Wave 1-2 results...")

    capacity_multiplier = load_rec3_capacity_parameter(result)
    durability_params = load_rec2_durability_params(result)

    # =========================================================================
    # 2. Load Historical Data
    # =========================================================================
    print("\n[2/6] Loading historical data...")

    df_historical = load_historical_data(result)
    df_refugee = load_nd_refugee_arrivals(result)
    df_national = load_national_refugee_totals(result)

    # =========================================================================
    # 3. Estimate ND Share
    # =========================================================================
    print("\n[3/6] Estimating ND share...")

    nd_share = estimate_nd_share(df_refugee, df_national, result)

    # =========================================================================
    # 4. Build Policy Scenarios
    # =========================================================================
    print("\n[4/6] Building policy scenarios...")

    scenarios = build_policy_scenarios(capacity_multiplier, durability_params, result)

    # =========================================================================
    # 5. Run Projections
    # =========================================================================
    print("\n[5/6] Running projections...")

    projections = run_all_projections(scenarios, nd_share, result, n_simulations=1000)

    # =========================================================================
    # 6. Generate Visualizations
    # =========================================================================
    print("\n[6/6] Generating visualizations...")

    plot_scenario_comparison(df_historical, projections, result)
    plot_durable_temporary_decomposition(projections, result)
    plot_policy_lever_sensitivity(projections, result)

    # =========================================================================
    # Compile Results
    # =========================================================================
    result.parameters = {
        "wave_1_2_inputs": {
            "rec3_capacity_multiplier": capacity_multiplier,
            "rec2_regularization_central": durability_params["regularization_central"],
            "rec2_regularization_range": [
                durability_params["regularization_lower"],
                durability_params["regularization_upper"],
            ],
            "rec2_refugee_survival_5yr": durability_params["refugee_survival_5yr"],
            "rec2_parole_survival_5yr": durability_params["parole_survival_5yr"],
        },
        "nd_share": nd_share,
        "base_year": 2024,
        "projection_horizon": "2025-2045",
        "n_simulations": 1000,
        "scenarios": [s.value for s in ScenarioType],
    }

    result.results = {
        "scenarios": {
            scenario_type.value: scenario.to_dict()
            for scenario_type, scenario in scenarios.items()
        },
        "projections": {
            scenario_type.value: projection.to_dict()
            for scenario_type, projection in projections.items()
        },
        "summary_2045": {
            scenario_type.value: {
                "total": projection.total_projection[-1],
                "durable": projection.durable_component[-1],
                "temporary": projection.temporary_component[-1],
                "uncertainty_80": [
                    projection.uncertainty_bounds["p10"][-1],
                    projection.uncertainty_bounds["p90"][-1],
                ],
            }
            for scenario_type, projection in projections.items()
        },
        "policy_lever_mapping": {
            "refugee_ceiling": "Affects durable component through ND share allocation",
            "parole_continuation": "Affects temporary component size and cliff timing",
            "regularization_probability": "Converts temporary to durable over time",
            "capacity_multiplier": "Scales all arrivals based on local infrastructure",
            "welcome_corps_growth": "Adds supplemental durable pathway",
        },
    }

    result.diagnostics = {
        "historical_years": len(df_historical),
        "projection_years": 21,
        "scenarios_built": len(scenarios),
        "projections_completed": len(projections),
    }

    result.next_steps = [
        "Integrate projections with cohort-component model for age structure",
        "Develop interactive scenario dashboard for stakeholder exploration",
        "Monitor policy developments and update scenario parameters",
        "Validate Welcome Corps assumptions as program matures",
        "Connect to workforce and fiscal impact modules",
    ]

    return result


def main() -> int:
    """Main entry point."""
    try:
        result = run_analysis()

        # Save results
        output_file = result.save("module_9b_policy_scenarios.json")
        adr_output = result.save(
            "rec6_policy_scenarios.json",
            output_dir=ADR_RESULTS_DIR,
        )

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("=" * 70)

        print(f"\nMain output: {output_file}")
        print(f"ADR output: {adr_output}")

        # Summary
        print("\n" + "-" * 70)
        print("KEY RESULTS: 2045 PROJECTIONS BY SCENARIO")
        print("-" * 70)

        if "summary_2045" in result.results:
            for scenario_name, summary in result.results["summary_2045"].items():
                print(f"\n  {scenario_name.upper()}:")
                print(f"    Total: {summary['total']:,.0f}")
                print(f"    Durable: {summary['durable']:,.0f}")
                print(f"    Temporary: {summary['temporary']:,.0f}")
                print(f"    80% CI: [{summary['uncertainty_80'][0]:,.0f}, {summary['uncertainty_80'][1]:,.0f}]")

        if result.warnings:
            print("\nWarnings:")
            for w in result.warnings:
                print(f"  - {w}")

        print(f"\nDecisions logged: {len(result.decisions)}")
        for d in result.decisions:
            print(f"  [{d['decision_id']}] {d['decision'][:60]}...")

        print("\nFigures generated:")
        print("  - module_9b_scenario_comparison.png/pdf")
        print("  - module_9b_component_decomposition.png/pdf")
        print("  - module_9b_policy_sensitivity.png/pdf")
        print("  - rec6_scenario_comparison.png/pdf (ADR)")
        print("  - rec6_component_decomposition.png/pdf (ADR)")
        print("  - rec6_policy_sensitivity.png/pdf (ADR)")

        return 0

    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
