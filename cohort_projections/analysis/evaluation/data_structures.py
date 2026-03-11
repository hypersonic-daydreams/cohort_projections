"""Canonical data structures for the evaluation framework.

These dataclasses define the standard schemas referenced throughout the
Evaluation Blueprint.  They are designed to be easily convertible to/from
pandas DataFrames and serializable to Parquet/CSV.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Run identity (attached to every evaluation artefact)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RunIdentity:
    """Metadata uniquely identifying a single projection run."""

    run_id: str
    model_name: str
    model_family: str
    projection_origin_year: int
    training_window: str  # e.g. "2000-2020"
    data_vintage: str  # e.g. "PEP Vintage 2025"
    parameter_set: str  # JSON-serialised dict or profile path
    notes: str = ""


# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------
@dataclass
class ExperimentRegistryEntry:
    """One row in the experiment registry (append-only ledger)."""

    run_id: str
    timestamp: dt.datetime
    model_name: str
    model_family: str
    baseline_flag: bool
    projection_origin_year: int
    training_start_year: int
    training_end_year: int
    data_vintage: str
    parameter_json: str  # JSON string of parameter dict
    scenario_name: str
    notes: str = ""
    git_commit: str = ""
    git_dirty: bool = False


# ---------------------------------------------------------------------------
# Projection results (tidy format)
# ---------------------------------------------------------------------------
@dataclass
class ProjectionResultRecord:
    """One row in the projection results table."""

    run_id: str
    geography: str  # FIPS code or "state"
    geography_type: str  # "state", "county", "county_group"
    year: int
    horizon: int  # years since projection origin
    sex: str  # "male", "female", "total"
    age_group: str  # e.g. "0-4", "5-9", ..., "85+", "total"
    target: str  # "population", "births", "deaths", "net_migration"
    projected_value: float
    actual_value: float
    base_value: float  # value at projection origin


# ---------------------------------------------------------------------------
# Component table
# ---------------------------------------------------------------------------
@dataclass
class ComponentRecord:
    """One row in the demographic component table."""

    run_id: str
    geography: str
    year: int
    horizon: int
    component: str  # "births", "deaths", "net_migration", "in_migration", "out_migration"
    projected_component_value: float
    actual_component_value: float


# ---------------------------------------------------------------------------
# Diagnostics table
# ---------------------------------------------------------------------------
@dataclass
class DiagnosticRecord:
    """One row in the diagnostics table."""

    run_id: str
    metric_name: str  # e.g. "mae", "mape", "jsd"
    metric_group: str  # "accuracy", "bias", "realism", "stability"
    geography: str
    geography_group: str  # "state", "bakken", "rural", "urban_college", "reservation"
    target: str
    horizon: int | None = None
    value: float = 0.0
    comparison_run_id: str = ""
    notes: str = ""


# ---------------------------------------------------------------------------
# Model scorecard
# ---------------------------------------------------------------------------
@dataclass
class ScorecardEntry:
    """Multi-axis scorecard summarising one run."""

    run_id: str
    model_name: str
    near_term_accuracy: float  # composite, lower is better
    long_term_accuracy: float
    bias_calibration: float  # closer to 0 is better
    age_structure_realism: float  # 0-1 (1 = perfect)
    robustness_stability: float  # 0-1 (1 = most stable)
    interpretability: float  # 0-1 (partially qualitative)
    composite_score: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
