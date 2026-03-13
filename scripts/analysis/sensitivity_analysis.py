#!/usr/bin/env python3
"""Walk-forward sensitivity analysis for registered projection methods.

Created: 2026-03-03
Author: Claude Code (automated)
Modified: 2026-03-04 — Refactored to use METHOD_DISPATCH registry

Purpose
-------
Systematically perturb key parameters of registered population projection
methods and measure the impact on projection accuracy.  For each parameter
perturbation, run a full projection from origin year 2015 (and optionally 2020)
through to the 2024 validation point, computing state-level percentage error
and county-level MAPE.  This reveals which parameters most influence the
projection outcome.

Parameters Tested
-----------------
- Migration rates: +/-10%, +/-25%, +/-50% scaling
- Fertility rates: +/-10%, +/-25% scaling
- Survival rates: +/-1%, +/-5% perturbation (applied to death probability q)
- SDC Bakken dampening: 0.4, 0.5, 0.6 (baseline), 0.7, 0.8
- 2026 mortality improvement rate: 0.0, 0.0025, 0.005 (baseline), 0.0075, 0.01
- 2026 convergence schedule: all-recent, all-medium, all-longterm vs baseline

Method
------
1. Load all shared data (population snapshots, migration rates, survival,
   fertility) from the walk-forward validation module.
2. For each method x parameter x perturbation level:
   a. Apply the perturbation to the relevant input data.
   b. Run the projection from origin 2015 through 2024.
   c. Compute state-level percentage error and county-level MAPE at 2024.
3. Aggregate results into tornado-diagram format (ranked by impact magnitude).
4. Generate interactive HTML report with Plotly tornado diagrams, spider
   charts, and parameter sweep line charts.

Inputs
------
Same data files as walk_forward_validation.py (via module import):
- Census/PEP population snapshots (2000-2024)
- Residual migration rates (5 periods)
- CDC ND 2020 life table survival rates
- 2018-2022 blended annual ASFRs

Outputs
-------
- data/analysis/walk_forward/sensitivity_results.csv — per-perturbation results
- data/analysis/walk_forward/sensitivity_tornado.csv — ranked by impact magnitude
- data/analysis/walk_forward/sensitivity_report.html — interactive Plotly report

Usage
-----
    python scripts/analysis/sensitivity_analysis.py
    python scripts/analysis/sensitivity_analysis.py --methods sdc_2024 m2026
    python scripts/analysis/sensitivity_analysis.py --methods m2026 m2026r1
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — allow import of walk_forward_validation as a sibling script
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import walk_forward_validation as wfv  # noqa: E402
from walk_forward_validation import METHOD_DISPATCH  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = PROJECT_ROOT / "data" / "analysis" / "walk_forward"

# Default methods for backward compatibility (used when --methods is not specified)
DEFAULT_METHODS = ["sdc_2024", "m2026"]

# Origin years and validation target for sensitivity analysis
SENSITIVITY_ORIGINS = [2015, 2020]
VALIDATION_YEAR = 2024

# Baseline parameter values
BASELINE_SDC_DAMPENING = wfv.SDC_2024_CONFIG["sdc_bakken_dampening"]  # 0.6
BASELINE_MORTALITY_IMPROVEMENT = wfv.MORTALITY_IMPROVEMENT_RATE  # 0.005


def method_display_label(method_key: str) -> str:
    """Derive a human-readable display label from a method registry key.

    Examples:
        "sdc_2024" -> "SDC 2024"
        "m2026"    -> "M2026"
        "m2026r1"  -> "M2026 R1"
    """
    import re

    # Split on underscores first
    parts = method_key.split("_")
    result_parts: list[str] = []
    for part in parts:
        # Insert space before revision suffix (e.g., "m2026r1" -> "m2026 r1")
        sub_parts = re.sub(r"(r\d+)$", r" \1", part).split()
        result_parts.extend(p.upper() for p in sub_parts)
    return " ".join(result_parts)


# Which perturbation parameters apply to which method types.
# "shared" parameters apply to all methods; method-specific params apply only
# to methods whose is_annual flag matches the expected type.
SHARED_PARAMS = ["migration_rate", "fertility_rate", "survival_rate"]

# Map perturbation parameter -> predicate on METHOD_DISPATCH entry.
# If the predicate returns True, the perturbation applies to that method.
METHOD_PARAM_APPLICABILITY: dict[str, Callable[[dict[str, object]], bool]] = {
    "migration_rate": lambda _dispatch: True,
    "fertility_rate": lambda _dispatch: True,
    "survival_rate": lambda _dispatch: True,
    "sdc_bakken_dampening": lambda dispatch: not bool(dispatch["is_annual"]),
    "mortality_improvement": lambda dispatch: bool(dispatch["is_annual"]),
    "convergence_schedule": lambda dispatch: bool(dispatch["is_annual"]),
}


# ---------------------------------------------------------------------------
# Perturbation Definitions
# ---------------------------------------------------------------------------

PERTURBATIONS: dict[str, list[dict]] = {
    "migration_rate": [
        {"label": "-50%", "factor": 0.50},
        {"label": "-25%", "factor": 0.75},
        {"label": "-10%", "factor": 0.90},
        {"label": "baseline", "factor": 1.00},
        {"label": "+10%", "factor": 1.10},
        {"label": "+25%", "factor": 1.25},
        {"label": "+50%", "factor": 1.50},
    ],
    "fertility_rate": [
        {"label": "-25%", "factor": 0.75},
        {"label": "-10%", "factor": 0.90},
        {"label": "baseline", "factor": 1.00},
        {"label": "+10%", "factor": 1.10},
        {"label": "+25%", "factor": 1.25},
    ],
    "survival_rate": [
        # Perturbation applied to death probability q = 1 - S
        # +5% q means more deaths (lower survival), -5% q means fewer deaths
        {"label": "-5% q (fewer deaths)", "delta_q_pct": -5.0},
        {"label": "-1% q (fewer deaths)", "delta_q_pct": -1.0},
        {"label": "baseline", "delta_q_pct": 0.0},
        {"label": "+1% q (more deaths)", "delta_q_pct": +1.0},
        {"label": "+5% q (more deaths)", "delta_q_pct": +5.0},
    ],
    "sdc_bakken_dampening": [
        {"label": "0.4", "value": 0.4},
        {"label": "0.5", "value": 0.5},
        {"label": "0.6 (baseline)", "value": 0.6},
        {"label": "0.7", "value": 0.7},
        {"label": "0.8", "value": 0.8},
    ],
    "mortality_improvement": [
        {"label": "0.0%", "value": 0.0},
        {"label": "0.25%", "value": 0.0025},
        {"label": "0.5% (baseline)", "value": 0.005},
        {"label": "0.75%", "value": 0.0075},
        {"label": "1.0%", "value": 0.01},
    ],
    "convergence_schedule": [
        {"label": "all_recent", "window": "recent"},
        {"label": "baseline (blended)", "window": "baseline"},
        {"label": "all_medium", "window": "medium"},
        {"label": "all_longterm", "window": "longterm"},
    ],
}


class SensitivityTask(TypedDict):
    """One sensitivity scenario queued for sequential or parallel execution."""

    method: str
    param_name: str
    level: dict[str, object]
    origin_year: int
    base_pop: pd.DataFrame
    survival: dict[tuple[str, str], float]
    fertility: dict[str, float]
    mig_raw: pd.DataFrame
    actual_county_totals: dict[str, float]
    actual_state_total: float
    method_config: dict[str, object] | None


# ---------------------------------------------------------------------------
# Helper: perturb survival rates
# ---------------------------------------------------------------------------


def perturb_survival(
    base_survival: dict[tuple[str, str], float],
    delta_q_pct: float,
) -> dict[tuple[str, str], float]:
    """Perturb survival rates by adjusting death probability q by delta_q_pct%.

    For a survival rate S, q = 1 - S.
    Perturbed q_new = q * (1 + delta_q_pct / 100).
    Perturbed S_new = 1 - q_new, clamped to [0, 1].
    """
    perturbed: dict[tuple[str, str], float] = {}
    for key, surv in base_survival.items():
        q = 1.0 - surv
        q_new = q * (1.0 + delta_q_pct / 100.0)
        perturbed[key] = max(0.0, min(1.0, 1.0 - q_new))
    return perturbed


def perturb_fertility(
    base_fertility: dict[str, float],
    factor: float,
) -> dict[str, float]:
    """Scale all fertility rates by a multiplicative factor."""
    return {k: v * factor for k, v in base_fertility.items()}


def perturb_migration_rates(
    mig_raw: pd.DataFrame,
    factor: float,
) -> pd.DataFrame:
    """Scale all migration rates by a multiplicative factor."""
    df = mig_raw.copy()
    df["migration_rate"] = df["migration_rate"] * factor
    return df


# ---------------------------------------------------------------------------
# Override convergence windows to use a single window for all steps
# ---------------------------------------------------------------------------


def make_flat_convergence_windows(
    windows: dict[str, pd.DataFrame],
    target_window: str,
) -> dict[str, pd.DataFrame]:
    """Return convergence windows where all three levels are the same.

    This simulates using all-recent, all-medium, or all-longterm rates
    throughout the projection, eliminating the convergence schedule.
    """
    target = windows[target_window].copy()
    return {
        "recent": target.copy(),
        "medium": target.copy(),
        "longterm": target.copy(),
    }


# ---------------------------------------------------------------------------
# Core: run one projection scenario and compute error metrics
# ---------------------------------------------------------------------------


def _prepare_sdc_rates_with_dampening(
    mig_raw: pd.DataFrame,
    origin_year: int,
    dampening: float,
) -> pd.DataFrame:
    """Prepare SDC migration rates with a custom Bakken dampening factor.

    Replicates prepare_sdc_rates but with a configurable dampening value.
    """
    periods = wfv.AVAILABLE_PERIODS[origin_year]
    mask = mig_raw.apply(
        lambda r: (r["period_start"], r["period_end"]) in periods, axis=1
    )
    df = mig_raw[mask].copy()

    # Convert annualized to 5-year
    df["rate_5yr"] = (1 + df["migration_rate"]) ** 5 - 1

    # Simple average across periods
    avg = (
        df.groupby(["county_fips", "age_group", "sex"], as_index=False)["rate_5yr"]
        .mean()
        .rename(columns={"rate_5yr": "migration_rate_5yr"})
    )

    # Bakken dampening with custom factor
    bakken_mask = avg["county_fips"].isin(wfv.SDC_2024_CONFIG["bakken_fips"])
    avg.loc[bakken_mask, "migration_rate_5yr"] *= dampening

    return avg


def run_scenario(
    method: str,
    origin_year: int,
    base_pop: pd.DataFrame,
    survival: dict[tuple[str, str], float],
    fertility: dict[str, float],
    mig_raw: pd.DataFrame,
    actual_county_totals: dict[str, float],
    actual_state_total: float,
    *,
    sdc_dampening: float = BASELINE_SDC_DAMPENING,
    mortality_improvement: float = BASELINE_MORTALITY_IMPROVEMENT,
    convergence_override: str | None = None,
) -> dict:
    """Run a single projection scenario and return error metrics.

    Uses METHOD_DISPATCH to look up prepare/project callables for the given
    method, rather than hardcoding method-specific if/elif branches.

    Args:
        method: A key in METHOD_DISPATCH (e.g. 'sdc_2024', 'm2026', 'm2026r1').
        origin_year: Origin year of the projection.
        base_pop: Base population DataFrame.
        survival: 5-year survival rates.
        fertility: Annual ASFRs.
        mig_raw: Raw migration rates DataFrame (potentially perturbed).
        actual_county_totals: Actual county populations at validation year.
        actual_state_total: Actual state total at validation year.
        sdc_dampening: Bakken dampening factor (SDC-type methods).
        mortality_improvement: Annual mortality improvement rate (annual methods).
        convergence_override: If set, use this window for all convergence
            steps ('recent', 'medium', or 'longterm'). None = baseline schedule.

    Returns:
        dict with keys: state_pct_error, county_mape, projected_state, etc.
    """
    if method not in METHOD_DISPATCH:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Available: {list(METHOD_DISPATCH.keys())}"
        )

    dispatch = METHOD_DISPATCH[method]
    is_annual = dispatch["is_annual"]
    cfg = dispatch["config"]  # type: ignore[index]
    counties = sorted(base_pop["county_fips"].unique())
    n_years = VALIDATION_YEAR - origin_year

    projected_totals: dict[str, float] = {}

    if not is_annual:
        # SDC-type (5-year step) methods: use custom dampening preparation
        sdc_mig = _prepare_sdc_rates_with_dampening(mig_raw, origin_year, sdc_dampening)
        n_steps = (n_years + wfv.STEP - 1) // wfv.STEP

        for fips in counties:
            proj = dispatch["project"](  # type: ignore[operator]
                base_pop, survival, fertility, sdc_mig, fips, n_steps, origin_year, cfg
            )
            # Interpolate to get validation year value
            proj_annual = wfv.interpolate_county_annual(proj, origin_year, VALIDATION_YEAR)
            projected_totals[fips] = proj_annual.get(VALIDATION_YEAR, 0.0)

    else:
        # Annual convergence methods (m2026, m2026r1, etc.)
        windows = dispatch["prepare"](mig_raw, origin_year, cfg)  # type: ignore[operator]

        if convergence_override is not None:
            windows = make_flat_convergence_windows(windows, convergence_override)

        # Temporarily override mortality improvement rate
        original_rate = wfv.MORTALITY_IMPROVEMENT_RATE
        wfv.MORTALITY_IMPROVEMENT_RATE = mortality_improvement

        for fips in counties:
            proj = dispatch["project"](  # type: ignore[operator]
                base_pop, survival, fertility, windows, fips, n_years, origin_year, cfg
            )
            projected_totals[fips] = proj.get(VALIDATION_YEAR, 0.0)

        # Restore original rate
        wfv.MORTALITY_IMPROVEMENT_RATE = original_rate

    # Compute state total
    projected_state = sum(projected_totals.values())

    # State-level error
    state_error = projected_state - actual_state_total
    state_pct_error = (state_error / actual_state_total * 100) if actual_state_total > 0 else 0.0

    # County-level MAPE
    abs_pct_errors = []
    for fips in counties:
        proj_pop = projected_totals.get(fips, 0.0)
        actual_pop = actual_county_totals.get(fips, 0.0)
        if actual_pop > 0:
            abs_pct_errors.append(abs((proj_pop - actual_pop) / actual_pop * 100))
    county_mape = float(np.mean(abs_pct_errors)) if abs_pct_errors else 0.0

    return {
        "state_pct_error": round(state_pct_error, 4),
        "county_mape": round(county_mape, 4),
        "projected_state": round(projected_state, 0),
        "actual_state": round(actual_state_total, 0),
    }


def _run_single_sensitivity_task(
    *,
    method: str,
    param_name: str,
    level: dict[str, object],
    origin_year: int,
    base_pop: pd.DataFrame,
    survival: dict[tuple[str, str], float],
    fertility: dict[str, float],
    mig_raw: pd.DataFrame,
    actual_county_totals: dict[str, float],
    actual_state_total: float,
    method_config: dict[str, object] | None = None,
) -> dict[str, object]:
    """Run one sensitivity scenario and return its result record."""
    original_config: object | None = None
    if method_config is not None:
        original_config = METHOD_DISPATCH[method]["config"]  # type: ignore[index]
        METHOD_DISPATCH[method]["config"] = method_config  # type: ignore[index]

    try:
        perturbed_mig = mig_raw
        perturbed_survival = survival
        perturbed_fertility = fertility
        sdc_damp = BASELINE_SDC_DAMPENING
        mort_imp = BASELINE_MORTALITY_IMPROVEMENT
        conv_override = None

        if param_name == "migration_rate":
            perturbed_mig = perturb_migration_rates(
                mig_raw, cast(float, level["factor"])
            )
        elif param_name == "fertility_rate":
            perturbed_fertility = perturb_fertility(
                fertility, cast(float, level["factor"])
            )
        elif param_name == "survival_rate":
            perturbed_survival = perturb_survival(
                survival, cast(float, level["delta_q_pct"])
            )
        elif param_name == "sdc_bakken_dampening":
            sdc_damp = cast(float, level["value"])
        elif param_name == "mortality_improvement":
            mort_imp = cast(float, level["value"])
        elif (
            param_name == "convergence_schedule"
            and str(level["window"]) != "baseline"
        ):
            conv_override = str(level["window"])

        result = run_scenario(
            method=method,
            origin_year=origin_year,
            base_pop=base_pop,
            survival=perturbed_survival,
            fertility=perturbed_fertility,
            mig_raw=perturbed_mig,
            actual_county_totals=actual_county_totals,
            actual_state_total=actual_state_total,
            sdc_dampening=sdc_damp,
            mortality_improvement=mort_imp,
            convergence_override=conv_override,
        )

        return {
            "method": method,
            "parameter": param_name,
            "perturbation_level": str(level["label"]),
            "origin_year": origin_year,
            "state_pct_error": result["state_pct_error"],
            "county_mape": result["county_mape"],
            "projected_state": result["projected_state"],
            "actual_state": result["actual_state"],
        }
    finally:
        if original_config is not None:
            METHOD_DISPATCH[method]["config"] = original_config  # type: ignore[index]


# ---------------------------------------------------------------------------
# Main sensitivity analysis runner
# ---------------------------------------------------------------------------


def run_sensitivity_analysis(
    snapshots: dict[int, pd.DataFrame],
    mig_raw: pd.DataFrame,
    survival: dict[tuple[str, str], float],
    fertility: dict[str, float],
    methods: list[str] | None = None,
    workers: int = 1,
) -> pd.DataFrame:
    """Run the full sensitivity analysis across all parameters and methods.

    Args:
        snapshots: Population snapshots by year.
        mig_raw: Raw migration rates DataFrame.
        survival: 5-year survival rates.
        fertility: Annual ASFRs.
        methods: List of method keys from METHOD_DISPATCH. Defaults to
            DEFAULT_METHODS for backward compatibility.
        workers: Number of parallel workers. ``1`` runs sequentially.
            ``0`` auto-detects ``min(total_scenarios, cpu_count)``.

    Returns DataFrame with columns:
    [method, parameter, perturbation_level, origin_year, state_pct_error, county_mape]
    """
    if methods is None:
        methods = list(DEFAULT_METHODS)

    # Pre-compute actual county totals and state totals for each origin's validation
    actual_data: dict[int, tuple[dict[str, float], float]] = {}
    for origin_year in SENSITIVITY_ORIGINS:
        actual_df = snapshots[VALIDATION_YEAR]
        county_tots = (
            actual_df.groupby("county_fips")["population"].sum().to_dict()
        )
        state_total = sum(county_tots.values())
        actual_data[origin_year] = (county_tots, state_total)

    method_configs: dict[str, dict[str, object]] = {
        method: METHOD_DISPATCH[method]["config"]  # type: ignore[index]
        for method in methods
    }
    tasks: list[SensitivityTask] = []

    for method in methods:
        dispatch = METHOD_DISPATCH[method]
        print(f"\n  Method: {method}")
        for param_name, levels in PERTURBATIONS.items():
            # Skip parameters that don't apply to this method type
            applicability = METHOD_PARAM_APPLICABILITY.get(param_name)
            if applicability is not None and not applicability(dispatch):
                continue
            print(f"    Parameter: {param_name} ({len(levels)} levels)")
            for level in levels:
                for origin_year in SENSITIVITY_ORIGINS:
                    county_tots, state_total = actual_data[origin_year]
                    tasks.append(
                        {
                            "method": method,
                            "param_name": param_name,
                            "level": level,
                            "origin_year": origin_year,
                            "base_pop": snapshots[origin_year],
                            "survival": survival,
                            "fertility": fertility,
                            "mig_raw": mig_raw,
                            "actual_county_totals": county_tots,
                            "actual_state_total": state_total,
                            "method_config": method_configs[method],
                        }
                    )

    total_scenarios = len(tasks)
    print(f"\n  Total scenarios to run: {total_scenarios}")
    if total_scenarios == 0:
        return pd.DataFrame(
            columns=[
                "method",
                "parameter",
                "perturbation_level",
                "origin_year",
                "state_pct_error",
                "county_mape",
                "projected_state",
                "actual_state",
            ]
        )

    if workers == 0:
        effective_workers = min(total_scenarios, os.cpu_count() or 1)
    else:
        effective_workers = max(1, min(workers, total_scenarios))

    records: list[dict[str, object]] = []
    scenario_count = 0

    if effective_workers <= 1:
        for task in tasks:
            records.append(_run_single_sensitivity_task(**task))
            scenario_count += 1
            if scenario_count % 10 == 0 or scenario_count == total_scenarios:
                print(
                    f"      Progress: {scenario_count}/{total_scenarios} "
                    f"({100 * scenario_count / total_scenarios:.0f}%)"
                )
    else:
        print(
            f"  Running {total_scenarios} sensitivity scenarios with "
            f"{effective_workers} workers..."
        )
        futures: dict[int, Future[dict[str, object]]] = {}
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            for idx, task in enumerate(tasks):
                futures[idx] = executor.submit(_run_single_sensitivity_task, **task)

        for idx, task in enumerate(tasks):
            fut = futures[idx]
            try:
                record = fut.result()
            except Exception as exc:
                print(
                    "      WARNING: Worker for sensitivity task "
                    f"{idx + 1}/{total_scenarios} failed ({exc!r}), "
                    "retrying sequentially..."
                )
                record = _run_single_sensitivity_task(**task)
            records.append(record)
            scenario_count += 1
            if scenario_count % 10 == 0 or scenario_count == total_scenarios:
                print(
                    f"      Progress: {scenario_count}/{total_scenarios} "
                    f"({100 * scenario_count / total_scenarios:.0f}%)"
                )

    print(f"\n  Completed {scenario_count} scenarios.")
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Tornado diagram data
# ---------------------------------------------------------------------------


def compute_tornado_data(results: pd.DataFrame) -> pd.DataFrame:
    """Compute tornado diagram data: for each method x parameter, find the
    high/low perturbation impact and rank by total swing magnitude.

    Averages across origin years for each perturbation level, then computes
    the range (max - min) of state_pct_error to measure total influence.

    Returns DataFrame sorted by impact magnitude (descending).
    """
    records: list[dict] = []

    for method in results["method"].unique():
        method_df = results[results["method"] == method]

        for param in method_df["parameter"].unique():
            param_df = method_df[method_df["parameter"] == param]

            # Average across origin years for each perturbation level
            avg_by_level = (
                param_df.groupby("perturbation_level")
                .agg(
                    state_pct_error=("state_pct_error", "mean"),
                    county_mape=("county_mape", "mean"),
                )
                .reset_index()
            )

            # Find baseline error
            baseline_rows = avg_by_level[
                avg_by_level["perturbation_level"].str.contains("baseline", case=False)
            ]
            if len(baseline_rows) > 0:
                baseline_state_error = baseline_rows["state_pct_error"].iloc[0]
                baseline_county_mape = baseline_rows["county_mape"].iloc[0]
            else:
                baseline_state_error = avg_by_level["state_pct_error"].mean()
                baseline_county_mape = avg_by_level["county_mape"].mean()

            # Find max and min deviations from baseline
            deviations = avg_by_level["state_pct_error"] - baseline_state_error
            mape_deviations = avg_by_level["county_mape"] - baseline_county_mape

            min_dev_idx = deviations.idxmin()
            max_dev_idx = deviations.idxmax()

            min_dev = deviations.loc[min_dev_idx]
            max_dev = deviations.loc[max_dev_idx]

            min_label = avg_by_level.loc[min_dev_idx, "perturbation_level"]
            max_label = avg_by_level.loc[max_dev_idx, "perturbation_level"]

            swing = max_dev - min_dev

            # Same for MAPE
            mape_min_dev_idx = mape_deviations.idxmin()
            mape_max_dev_idx = mape_deviations.idxmax()
            mape_min_dev = mape_deviations.loc[mape_min_dev_idx]
            mape_max_dev = mape_deviations.loc[mape_max_dev_idx]
            mape_swing = mape_max_dev - mape_min_dev

            records.append(
                {
                    "method": method,
                    "parameter": param,
                    "baseline_state_error": round(baseline_state_error, 4),
                    "baseline_county_mape": round(baseline_county_mape, 4),
                    "low_deviation": round(min_dev, 4),
                    "high_deviation": round(max_dev, 4),
                    "low_label": min_label,
                    "high_label": max_label,
                    "swing_state_error": round(swing, 4),
                    "mape_low_deviation": round(mape_min_dev, 4),
                    "mape_high_deviation": round(mape_max_dev, 4),
                    "mape_swing": round(mape_swing, 4),
                }
            )

    tornado_df = pd.DataFrame(records)
    # Sort by swing magnitude descending within each method
    tornado_df = tornado_df.sort_values(
        ["method", "swing_state_error"], ascending=[True, False]
    ).reset_index(drop=True)

    return tornado_df


# ---------------------------------------------------------------------------
# HTML Report Generation
# ---------------------------------------------------------------------------


def build_html_report(
    results: pd.DataFrame,
    tornado: pd.DataFrame,
) -> str:
    """Build an interactive HTML report with Plotly charts.

    Sections:
    1. Tornado diagrams (one per method) — state % error sensitivity
    2. Tornado diagrams (one per method) — county MAPE sensitivity
    3. Spider/radar charts (one per method) — normalized parameter impact
    4. Parameter sweep line charts — error vs parameter value
    5. Summary table
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Discover methods present in the results
    methods_in_results = list(results["method"].unique())

    # Color palette for an arbitrary number of methods
    _color_palette = [
        "#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336",
        "#00BCD4", "#795548", "#607D8B", "#E91E63", "#3F51B5",
    ]
    method_colors = {
        m: _color_palette[i % len(_color_palette)]
        for i, m in enumerate(methods_in_results)
    }

    # Color scheme
    colors = {
        **method_colors,
        "low": "#4CAF50",
        "high": "#F44336",
        "baseline": "#9E9E9E",
        "bg": "#FAFAFA",
    }

    # Human-readable labels derived from method keys
    method_labels = {m: method_display_label(m) for m in methods_in_results}

    # Collect all figure HTML snippets with section metadata
    sections: list[dict] = []

    # ------------------------------------------------------------------
    # 1. Tornado Diagrams — State % Error
    # ------------------------------------------------------------------
    for method in methods_in_results:
        m_tornado = tornado[tornado["method"] == method].sort_values(
            "swing_state_error", ascending=True
        )
        if len(m_tornado) == 0:
            continue

        fig = go.Figure()

        # Low deviations (left bars)
        fig.add_trace(go.Bar(
            y=m_tornado["parameter"],
            x=m_tornado["low_deviation"],
            orientation="h",
            name="Low perturbation",
            marker_color=colors["low"],
            text=m_tornado["low_label"],
            textposition="inside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Perturbation: %{text}<br>"
                "State error shift: %{x:+.2f}pp<br>"
                "<extra></extra>"
            ),
        ))

        # High deviations (right bars)
        fig.add_trace(go.Bar(
            y=m_tornado["parameter"],
            x=m_tornado["high_deviation"],
            orientation="h",
            name="High perturbation",
            marker_color=colors["high"],
            text=m_tornado["high_label"],
            textposition="inside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Perturbation: %{text}<br>"
                "State error shift: %{x:+.2f}pp<br>"
                "<extra></extra>"
            ),
        ))

        baseline_err = m_tornado["baseline_state_error"].iloc[0]
        fig.update_layout(
            title=(
                f"Sensitivity Tornado — {method_labels[method]}<br>"
                f"<sub>Deviation from baseline state error ({baseline_err:+.2f}%); "
                f"origin average of "
                f"{', '.join(str(y) for y in SENSITIVITY_ORIGINS)}</sub>"
            ),
            xaxis_title="Deviation in State % Error (pp)",
            yaxis_title="Parameter",
            barmode="overlay",
            height=350 + len(m_tornado) * 50,
            template="plotly_white",
            legend={
                "orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "center", "x": 0.5
            },
        )

        # Add vertical baseline line at zero
        fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)

        sections.append({
            "section": "tornado_state",
            "method": method,
            "html": fig.to_html(full_html=False, include_plotlyjs=False),
        })

    # ------------------------------------------------------------------
    # 2. Tornado Diagrams — County MAPE
    # ------------------------------------------------------------------
    for method in methods_in_results:
        m_tornado = tornado[tornado["method"] == method].sort_values(
            "mape_swing", ascending=True
        )
        if len(m_tornado) == 0:
            continue

        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=m_tornado["parameter"],
            x=m_tornado["mape_low_deviation"],
            orientation="h",
            name="Low perturbation",
            marker_color=colors["low"],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "County MAPE shift: %{x:+.2f}pp<br>"
                "<extra></extra>"
            ),
        ))

        fig.add_trace(go.Bar(
            y=m_tornado["parameter"],
            x=m_tornado["mape_high_deviation"],
            orientation="h",
            name="High perturbation",
            marker_color=colors["high"],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "County MAPE shift: %{x:+.2f}pp<br>"
                "<extra></extra>"
            ),
        ))

        baseline_mape = m_tornado["baseline_county_mape"].iloc[0]
        fig.update_layout(
            title=(
                f"Sensitivity Tornado (County MAPE) — {method_labels[method]}<br>"
                f"<sub>Deviation from baseline county MAPE "
                f"({baseline_mape:.2f}%)</sub>"
            ),
            xaxis_title="Deviation in County MAPE (pp)",
            yaxis_title="Parameter",
            barmode="overlay",
            height=350 + len(m_tornado) * 50,
            template="plotly_white",
            legend={
                "orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "center", "x": 0.5
            },
        )

        fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)

        sections.append({
            "section": "tornado_mape",
            "method": method,
            "html": fig.to_html(full_html=False, include_plotlyjs=False),
        })

    # ------------------------------------------------------------------
    # 3. Spider / Radar Charts
    # ------------------------------------------------------------------
    for method in methods_in_results:
        m_tornado = tornado[tornado["method"] == method]
        if len(m_tornado) == 0:
            continue

        # Normalize swings to [0, 1] range for radar chart
        max_swing = m_tornado["swing_state_error"].max()
        if max_swing == 0:
            max_swing = 1.0

        categories = m_tornado["parameter"].tolist()
        state_values = (m_tornado["swing_state_error"] / max_swing).tolist()

        max_mape_swing = m_tornado["mape_swing"].max()
        if max_mape_swing == 0:
            max_mape_swing = 1.0
        mape_values = (m_tornado["mape_swing"] / max_mape_swing).tolist()

        # Close the polygon
        categories_closed = categories + [categories[0]]
        state_closed = state_values + [state_values[0]]
        mape_closed = mape_values + [mape_values[0]]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=state_closed,
            theta=categories_closed,
            fill="toself",
            name="State % Error Swing",
            line_color=colors[method],
            opacity=0.6,
            hovertemplate=(
                "<b>%{theta}</b><br>"
                "Normalized swing: %{r:.2f}<br>"
                "<extra></extra>"
            ),
        ))

        fig.add_trace(go.Scatterpolar(
            r=mape_closed,
            theta=categories_closed,
            fill="toself",
            name="County MAPE Swing",
            line_color="#9C27B0",
            opacity=0.4,
            hovertemplate=(
                "<b>%{theta}</b><br>"
                "Normalized MAPE swing: %{r:.2f}<br>"
                "<extra></extra>"
            ),
        ))

        fig.update_layout(
            title=(
                f"Parameter Sensitivity Radar — {method_labels[method]}<br>"
                f"<sub>Normalized swing magnitude "
                f"(1.0 = most influential)</sub>"
            ),
            polar={
                "radialaxis": {"visible": True, "range": [0, 1.05]},
            },
            height=500,
            template="plotly_white",
            legend={
                "orientation": "h", "yanchor": "bottom", "y": -0.15, "xanchor": "center", "x": 0.5
            },
        )

        sections.append({
            "section": "radar",
            "method": method,
            "html": fig.to_html(full_html=False, include_plotlyjs=False),
        })

    # ------------------------------------------------------------------
    # 4. Parameter Sweep Line Charts
    # ------------------------------------------------------------------
    # Define the ordered x-axis for each parameter
    param_x_order = {
        "migration_rate": [
            "-50%", "-25%", "-10%", "baseline", "+10%", "+25%", "+50%",
        ],
        "fertility_rate": ["-25%", "-10%", "baseline", "+10%", "+25%"],
        "survival_rate": [
            "-5% q (fewer deaths)", "-1% q (fewer deaths)", "baseline",
            "+1% q (more deaths)", "+5% q (more deaths)",
        ],
        "sdc_bakken_dampening": [
            "0.4", "0.5", "0.6 (baseline)", "0.7", "0.8",
        ],
        "mortality_improvement": [
            "0.0%", "0.25%", "0.5% (baseline)", "0.75%", "1.0%",
        ],
        "convergence_schedule": [
            "all_recent", "baseline (blended)", "all_medium", "all_longterm",
        ],
    }

    # Dynamically classify parameters as shared (multiple methods) or
    # method-specific based on the actual results data.
    all_params = list(results["parameter"].unique())
    param_method_counts: dict[str, list[str]] = {}
    for param in all_params:
        param_methods = list(
            results[results["parameter"] == param]["method"].unique()
        )
        param_method_counts[param] = param_methods

    shared_params = [p for p, ms in param_method_counts.items() if len(ms) > 1]
    specific_params = [p for p, ms in param_method_counts.items() if len(ms) == 1]

    # Shared parameters: overlay all methods that have data for this param
    for param in shared_params:
        x_order = param_x_order.get(param, [])
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("State % Error", "County MAPE"),
            horizontal_spacing=0.12,
        )

        for method in param_method_counts[param]:
            param_df = results[
                (results["method"] == method) & (results["parameter"] == param)
            ]
            if len(param_df) == 0:
                continue

            avg = (
                param_df.groupby("perturbation_level")
                .agg(
                    state_pct_error=("state_pct_error", "mean"),
                    county_mape=("county_mape", "mean"),
                )
                .reset_index()
            )

            # Sort by defined x-axis order
            avg["sort_order"] = avg["perturbation_level"].map(
                {v: i for i, v in enumerate(x_order)}
            )
            avg = avg.sort_values("sort_order").dropna(subset=["sort_order"])

            fig.add_trace(
                go.Scatter(
                    x=avg["perturbation_level"],
                    y=avg["state_pct_error"],
                    mode="lines+markers",
                    name=method_labels[method],
                    line_color=colors[method],
                    legendgroup=method,
                    hovertemplate=(
                        f"<b>{method_labels[method]}</b><br>"
                        "Level: %{x}<br>"
                        "State error: %{y:.2f}%<br>"
                        "<extra></extra>"
                    ),
                ),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=avg["perturbation_level"],
                    y=avg["county_mape"],
                    mode="lines+markers",
                    name=method_labels[method],
                    line_color=colors[method],
                    legendgroup=method,
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{method_labels[method]}</b><br>"
                        "Level: %{x}<br>"
                        "County MAPE: %{y:.2f}%<br>"
                        "<extra></extra>"
                    ),
                ),
                row=1, col=2,
            )

        param_label = param.replace("_", " ").title()
        origin_str = ", ".join(str(y) for y in SENSITIVITY_ORIGINS)
        fig.update_layout(
            title=(
                f"Parameter Sweep: {param_label}<br>"
                f"<sub>Origin average ({origin_str}), "
                f"validated at {VALIDATION_YEAR}</sub>"
            ),
            height=400,
            template="plotly_white",
            legend={
                "orientation": "h", "yanchor": "bottom", "y": 1.08, "xanchor": "center", "x": 0.5
            },
        )
        fig.update_xaxes(title_text="Perturbation Level", row=1, col=1)
        fig.update_xaxes(title_text="Perturbation Level", row=1, col=2)
        fig.update_yaxes(title_text="State % Error", row=1, col=1)
        fig.update_yaxes(title_text="County MAPE (%)", row=1, col=2)

        sections.append({
            "section": "sweep_shared",
            "param": param,
            "html": fig.to_html(full_html=False, include_plotlyjs=False),
        })

    # Method-specific parameters (only one method has data for these)
    for param in specific_params:
        method = param_method_counts[param][0]  # single method
        x_order = param_x_order.get(param, [])
        param_df = results[
            (results["method"] == method) & (results["parameter"] == param)
        ]
        if len(param_df) == 0:
            continue

        avg = (
            param_df.groupby("perturbation_level")
            .agg(
                state_pct_error=("state_pct_error", "mean"),
                county_mape=("county_mape", "mean"),
            )
            .reset_index()
        )

        avg["sort_order"] = avg["perturbation_level"].map(
            {v: i for i, v in enumerate(x_order)}
        )
        avg = avg.sort_values("sort_order").dropna(subset=["sort_order"])

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("State % Error", "County MAPE"),
            horizontal_spacing=0.12,
        )

        fig.add_trace(
            go.Scatter(
                x=avg["perturbation_level"],
                y=avg["state_pct_error"],
                mode="lines+markers",
                name="State Error",
                line_color=colors[method],
                hovertemplate=(
                    "Level: %{x}<br>"
                    "State error: %{y:.2f}%<br>"
                    "<extra></extra>"
                ),
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=avg["perturbation_level"],
                y=avg["county_mape"],
                mode="lines+markers",
                name="County MAPE",
                line_color="#9C27B0",
                hovertemplate=(
                    "Level: %{x}<br>"
                    "County MAPE: %{y:.2f}%<br>"
                    "<extra></extra>"
                ),
            ),
            row=1, col=2,
        )

        param_label = param.replace("_", " ").title()
        origin_str = ", ".join(str(y) for y in SENSITIVITY_ORIGINS)
        fig.update_layout(
            title=(
                f"Parameter Sweep: {param_label} "
                f"({method_labels[method]} only)<br>"
                f"<sub>Origin average ({origin_str}), "
                f"validated at {VALIDATION_YEAR}</sub>"
            ),
            height=400,
            template="plotly_white",
            legend={
                "orientation": "h", "yanchor": "bottom", "y": 1.08,
                "xanchor": "center", "x": 0.5,
            },
        )
        fig.update_xaxes(title_text="Perturbation Level", row=1, col=1)
        fig.update_xaxes(title_text="Perturbation Level", row=1, col=2)
        fig.update_yaxes(title_text="State % Error", row=1, col=1)
        fig.update_yaxes(title_text="County MAPE (%)", row=1, col=2)

        sections.append({
            "section": "sweep_specific",
            "method": method,
            "param": param,
            "html": fig.to_html(full_html=False, include_plotlyjs=False),
        })

    # ------------------------------------------------------------------
    # 5. Summary Table
    # ------------------------------------------------------------------
    summary_fig = go.Figure()

    # Dynamically allocate vertical space for each method's table
    n_methods = len(methods_in_results)
    table_height_each = 0.9 / max(n_methods, 1)  # leave 10% gap
    gap = 0.05
    table_y_ranges = {}
    for i, m in enumerate(reversed(methods_in_results)):
        y_bottom = i * (table_height_each + gap)
        y_top = y_bottom + table_height_each
        table_y_ranges[m] = [y_bottom, min(y_top, 1.0)]

    for method in methods_in_results:
        m_tornado = tornado[tornado["method"] == method].sort_values(
            "swing_state_error", ascending=False
        )
        if len(m_tornado) == 0:
            continue

        summary_fig.add_trace(go.Table(
            header={
                "values": [
                    "Parameter", "Baseline State Error",
                    "Low Deviation", "High Deviation", "Swing (pp)",
                    "Low Label", "High Label",
                ],
                "fill_color": colors[method],
                "font_color": "white",
                "align": "left",
            },
            cells={
                "values": [
                    m_tornado["parameter"],
                    m_tornado["baseline_state_error"].apply(
                        lambda x: f"{x:+.2f}%"
                    ),
                    m_tornado["low_deviation"].apply(lambda x: f"{x:+.2f}pp"),
                    m_tornado["high_deviation"].apply(lambda x: f"{x:+.2f}pp"),
                    m_tornado["swing_state_error"].apply(lambda x: f"{x:.2f}pp"),
                    m_tornado["low_label"],
                    m_tornado["high_label"],
                ],
                "align": "left",
            },
            domain={
                "x": [0, 1],
                "y": table_y_ranges[method],
            },
        ))

    # Build annotations dynamically for each method's table label
    table_annotations = []
    for m in methods_in_results:
        y_range = table_y_ranges.get(m, [0, 1])
        table_annotations.append(
            {
                "text": method_labels[m],
                "x": 0.02,
                "y": y_range[1] + 0.01,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 14, "color": colors[m]},
            }
        )

    summary_fig.update_layout(
        title=(
            "Sensitivity Analysis Summary Table<br>"
            "<sub>Parameters ranked by state error swing magnitude</sub>"
        ),
        height=max(400, 300 * n_methods),
        template="plotly_white",
        annotations=table_annotations,
    )

    summary_html = summary_fig.to_html(full_html=False, include_plotlyjs=False)

    # ------------------------------------------------------------------
    # Assemble final HTML
    # ------------------------------------------------------------------
    plotly_cdn = "https://cdn.plot.ly/plotly-2.35.2.min.js"
    origin_str = ", ".join(str(y) for y in SENSITIVITY_ORIGINS)

    html_parts: list[str] = []
    html_parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Sensitivity Analysis Report</title>
    <script src="{plotly_cdn}"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
                         Roboto, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: {colors['bg']};
            color: #333;
        }}
        h1 {{
            border-bottom: 3px solid #333;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
            margin-top: 40px;
        }}
        h3 {{
            color: #777;
        }}
        .chart-container {{
            background: white;
            border-radius: 8px;
            padding: 15px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metadata {{
            background: #E3F2FD;
            border-radius: 8px;
            padding: 15px;
            margin: 20px 0;
            font-size: 0.9em;
        }}
        .metadata dt {{
            font-weight: bold;
            display: inline;
        }}
        .metadata dd {{
            display: inline;
            margin-left: 5px;
            margin-right: 20px;
        }}
    </style>
</head>
<body>
    <h1>Sensitivity Analysis Report</h1>
    <p>North Dakota Population Projections: {', '.join(method_labels[m] for m in methods_in_results)}</p>

    <div class="metadata">
        <dl>
            <dt>Origin years:</dt><dd>{origin_str}</dd>
            <dt>Validation year:</dt><dd>{VALIDATION_YEAR}</dd>
            <dt>Methods:</dt>
            <dd>{', '.join(method_labels[m] for m in methods_in_results)}</dd>
            <dt>Generated:</dt>
            <dd>{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</dd>
        </dl>
    </div>
""")

    # Section 1: Tornado — State Error
    html_parts.append("""
    <h2>1. Tornado Diagrams &mdash; State Percentage Error</h2>
    <p>Horizontal bars show the deviation in state-level percentage error
    from baseline when each parameter is perturbed to its extreme low and
    high values. Parameters are sorted by total swing (high minus low), so
    the most influential parameters appear at the bottom.</p>
""")
    for s in sections:
        if s["section"] == "tornado_state":
            label = method_labels[s["method"]]
            html_parts.append(f"""
    <h3>{label}</h3>
    <div class="chart-container">{s['html']}</div>
""")

    # Section 2: Tornado — County MAPE
    html_parts.append("""
    <h2>2. Tornado Diagrams &mdash; County MAPE</h2>
    <p>Same approach but measuring county-level Mean Absolute Percentage
    Error.</p>
""")
    for s in sections:
        if s["section"] == "tornado_mape":
            label = method_labels[s["method"]]
            html_parts.append(f"""
    <h3>{label}</h3>
    <div class="chart-container">{s['html']}</div>
""")

    # Section 3: Radar
    html_parts.append("""
    <h2>3. Radar Charts &mdash; Normalized Sensitivity</h2>
    <p>Radar plots show the relative influence of each parameter, normalized
    so the most influential parameter has value 1.0. This makes it easy to
    compare across parameters of different units.</p>
""")
    for s in sections:
        if s["section"] == "radar":
            html_parts.append(
                f"""    <div class="chart-container">{s['html']}</div>\n"""
            )

    # Section 4: Sweeps
    html_parts.append("""
    <h2>4. Parameter Sweep Charts</h2>
    <p>Line charts showing how state percentage error and county MAPE
    change as each parameter is swept through its range. For shared
    parameters (migration, fertility, survival), both methods are overlaid
    for direct comparison.</p>
""")
    for s in sections:
        if s["section"] in ("sweep_shared", "sweep_specific"):
            html_parts.append(
                f"""    <div class="chart-container">{s['html']}</div>\n"""
            )

    # Section 5: Summary table
    html_parts.append(f"""
    <h2>5. Summary Table</h2>
    <div class="chart-container">{summary_html}</div>
""")

    # Section 6: Methodology notes
    html_parts.append(f"""
    <h2>6. Methodology Notes</h2>
    <ul>
        <li><strong>Migration rate perturbation:</strong> All county-age-sex
            migration rates are multiplicatively scaled by the stated
            factor.</li>
        <li><strong>Fertility rate perturbation:</strong> All age-specific
            fertility rates (ASFRs) are multiplicatively scaled.</li>
        <li><strong>Survival rate perturbation:</strong> The death probability
            q = 1 - S is scaled by the stated percentage. This avoids pushing
            survival rates beyond 1.0 while maintaining proportional
            perturbation near the boundary.</li>
        <li><strong>SDC Bakken dampening:</strong> The flat post-averaging
            dampening factor applied to migration rates in Bakken oil
            counties (baseline = 0.6).</li>
        <li><strong>Mortality improvement rate:</strong> Annual compound
            reduction in death probability for the M2026 method
            (baseline = 0.5%/year).</li>
        <li><strong>Convergence schedule:</strong> The M2026 method normally
            ramps from recent to long-term migration rates over the
            projection horizon. These tests override the schedule to use a
            single window throughout.</li>
        <li><strong>Metrics:</strong> State % error = (projected - actual) /
            actual * 100. County MAPE = mean of |county % errors| across all
            53 counties.</li>
        <li><strong>Origin averaging:</strong> Tornado and sweep charts
            average results across origin years {origin_str} to reduce
            origin-specific noise.</li>
    </ul>
</body>
</html>
""")

    return "".join(html_parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _output_path(base_name: str, label: str | None) -> Path:
    """Build output file path, optionally prefixed with a run label."""
    if label:
        return OUTPUT_DIR / f"{label}_{base_name}"
    return OUTPUT_DIR / base_name


def main() -> None:
    """Run sensitivity analysis and produce outputs."""
    import argparse

    parser = argparse.ArgumentParser(description="Sensitivity analysis")
    parser.add_argument(
        "--run-label",
        type=str,
        default=None,
        help="Optional prefix for output filenames (e.g., 'phase1')",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
        help=(
            f"Methods to run sensitivity analysis for "
            f"(default: {DEFAULT_METHODS}). "
            f"Available: {list(METHOD_DISPATCH.keys())}"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of parallel workers for sensitivity scenarios. "
            "0 = auto-detect min(total_scenarios, cpu_count). "
            "1 = sequential (default)."
        ),
    )
    args = parser.parse_args()
    label = args.run_label
    methods: list[str] = args.methods

    # Validate method names against registry
    for m in methods:
        if m not in METHOD_DISPATCH:
            parser.error(
                f"Unknown method '{m}'. "
                f"Available: {list(METHOD_DISPATCH.keys())}"
            )

    method_names = ", ".join(method_display_label(m) for m in methods)
    print("=" * 80)
    print(f"Sensitivity Analysis: {method_names}")
    print("=" * 80)
    print(f"  Methods: {methods}")
    if label:
        print(f"  Run label: {label}")
    print(f"  Workers: {args.workers}")

    # 1. Load data (reusing walk-forward validation infrastructure)
    print("\nLoading population snapshots...")
    snapshots = wfv.load_all_snapshots()

    print("\nLoading migration rates...")
    mig_raw = wfv.load_migration_rates_raw()
    periods = sorted(
        mig_raw[["period_start", "period_end"]]
        .drop_duplicates()
        .values.tolist()
    )
    print(f"  Periods: {len(periods)} — {periods}")

    print("\nLoading survival and fertility rates...")
    survival = wfv.load_survival_rates()
    fertility = wfv.load_fertility_rates()
    print(f"  Survival rates: {len(survival)} (age_group, sex) pairs")
    print(f"  Fertility rates: {len(fertility)} age groups")

    # 2. Run sensitivity analysis
    print("\n" + "-" * 80)
    print("Running sensitivity analysis...")
    print("-" * 80)
    results = run_sensitivity_analysis(
        snapshots, mig_raw, survival, fertility, methods=methods, workers=args.workers
    )

    # 3. Compute tornado data
    print("\nComputing tornado diagram data...")
    tornado_data = compute_tornado_data(results)

    # 4. Write output CSVs
    print("\nWriting output files...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_path = _output_path("sensitivity_results.csv", label)
    results.to_csv(results_path, index=False)
    print(
        f"  {results_path.relative_to(PROJECT_ROOT)} ({len(results)} rows)"
    )

    tornado_path = _output_path("sensitivity_tornado.csv", label)
    tornado_data.to_csv(tornado_path, index=False)
    print(
        f"  {tornado_path.relative_to(PROJECT_ROOT)} "
        f"({len(tornado_data)} rows)"
    )

    # 5. Build HTML report
    print("\nBuilding interactive HTML report...")
    html = build_html_report(results, tornado_data)
    html_path = _output_path("sensitivity_report.html", label)
    html_path.write_text(html, encoding="utf-8")
    print(f"  {html_path.relative_to(PROJECT_ROOT)} ({len(html):,} bytes)")

    # 6. Print summary
    print("\n" + "=" * 80)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("=" * 80)

    for method in methods:
        method_label = method_display_label(method)
        m_tornado = tornado_data[tornado_data["method"] == method].sort_values(
            "swing_state_error", ascending=False
        )
        if len(m_tornado) == 0:
            continue

        print(f"\n  {method_label} — Parameters ranked by state error swing:")
        print(
            f"  {'Parameter':<30} {'Swing (pp)':>10} "
            f"{'Low':>12} {'High':>12}"
        )
        print("  " + "-" * 66)
        for _, row in m_tornado.iterrows():
            print(
                f"  {row['parameter']:<30} "
                f"{row['swing_state_error']:>10.2f} "
                f"{row['low_deviation']:>+11.2f} "
                f"{row['high_deviation']:>+11.2f}"
            )

    print("\n" + "=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
