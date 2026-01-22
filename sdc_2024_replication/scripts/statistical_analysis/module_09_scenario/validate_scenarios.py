#!/usr/bin/env python3
"""
Validate scenario arithmetic for Module 9 projections.

Checks:
- CBO (Jan 2026) scales CBO net immigration by ND share.
- Pre-2020 Trend is anchored to 2019 with the 2010--2019 slope.
- CV diagnostics for historical series and 2045 Monte Carlo distribution.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
ANALYSIS_DIR = (
    PROJECT_ROOT / "sdc_2024_replication" / "scripts" / "statistical_analysis"
)
RESULTS_DIR = ANALYSIS_DIR / "results"
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "immigration" / "analysis"

SCENARIO_META_PATH = RESULTS_DIR / "module_9_scenario_modeling.json"
SCENARIO_PROJ_PATH = RESULTS_DIR / "module_9_scenario_projections.parquet"
COMBINED_PATH = RESULTS_DIR / "module_9_combined_forecasts.parquet"
MIGRATION_PATH = DATA_DIR / "nd_migration_summary.csv"

TOL_ABS = 1e-2
TOL_REL = 1e-6


def load_json(path: Path) -> dict:
    """Load JSON content from disk."""
    with path.open() as handle:
        return json.load(handle)


def is_close(actual: float, expected: float) -> bool:
    """Check numeric closeness with configured tolerances."""
    return np.isclose(actual, expected, rtol=TOL_REL, atol=TOL_ABS)


def validate_series(
    label: str, years: Iterable[int], actual: Iterable[float], expected: Iterable[float]
) -> list[str]:
    """Return mismatch messages for a scenario series."""
    issues = []
    for year, actual_val, expected_val in zip(years, actual, expected, strict=False):
        if not is_close(actual_val, expected_val):
            issues.append(
                f"{label} mismatch {year}: actual={actual_val:.2f}, expected={expected_val:.2f}"
            )
    return issues


def validate_cbo_full(
    projections: pd.DataFrame, assumptions: dict
) -> list[str]:
    """Validate CBO (Jan 2026) scenario arithmetic."""
    issues = []
    workbook = assumptions.get("cbo_workbook")
    if not workbook:
        return ["CBO scenario missing `cbo_workbook` in assumptions."]
    workbook_path = PROJECT_ROOT / workbook
    if not workbook_path.exists():
        return [f"CBO workbook missing: {workbook_path}"]

    df = pd.read_excel(workbook_path, sheet_name="Figure 7", header=None)
    mask = df.iloc[:, 0].apply(lambda x: isinstance(x, (int, float)) and 1900 <= x <= 2100) & df.iloc[:, 1].notna()
    series = df.loc[mask, [0, 1]].copy()
    series.columns = ["year", "value"]
    series["year"] = series["year"].astype(int)
    max_value = float(series["value"].max())
    if max_value < 50:
        series["net_immigration_persons"] = series["value"].astype(float) * 1_000_000.0
    else:
        series["net_immigration_persons"] = series["value"].astype(float) * 1_000.0

    nd_share_mean_pct = assumptions.get("nd_share_mean_pct")
    if nd_share_mean_pct is None:
        return ["CBO scenario missing `nd_share_mean_pct` in assumptions."]
    nd_share = float(nd_share_mean_pct) / 100.0

    us_map = series.set_index("year")["net_immigration_persons"].to_dict()
    cbo = projections.sort_values("year")
    expected_vals = [float(us_map[int(y)]) * nd_share for y in cbo["year"]]
    issues.extend(validate_series("CBO (Jan 2026)", cbo["year"], cbo["value"], expected_vals))

    return issues


def validate_pre_2020(
    projections: pd.DataFrame, migration: pd.DataFrame, assumptions: dict
) -> list[str]:
    """Validate Pre-2020 Trend scenario arithmetic."""
    issues = []
    pre_2020 = migration[migration["year"] < 2020]
    if pre_2020.empty:
        return ["Missing pre-2020 migration data for trend validation."]

    years = pre_2020["year"].to_numpy()
    values = pre_2020["nd_intl_migration"].to_numpy()
    X = years - years.min()
    slope, _, _, _, _ = stats.linregress(X, values)
    start_value = float(pre_2020.iloc[-1]["nd_intl_migration"])

    if not is_close(start_value, assumptions.get("start_value", start_value)):
        issues.append(
            f"Pre-2020 start value mismatch: data={start_value:.2f}, "
            f"assumption={assumptions.get('start_value'):.2f}"
        )

    if not is_close(slope, assumptions.get("trend_slope", slope)):
        issues.append(
            f"Pre-2020 slope mismatch: data={slope:.2f}, "
            f"assumption={assumptions.get('trend_slope'):.2f}"
        )

    pre_proj = projections.sort_values("year")
    expected_vals = [start_value + slope * (year - 2019) for year in pre_proj["year"]]
    issues.extend(
        validate_series(
            "Pre-2020 Trend", pre_proj["year"], pre_proj["value"], expected_vals
        )
    )

    return issues


def compute_cv(series: pd.Series) -> float:
    """Compute coefficient of variation."""
    mean = series.mean()
    return float(series.std(ddof=1) / mean) if mean else float("nan")


def main() -> int:
    """Run scenario validation checks."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    missing = [
        path
        for path in [
            SCENARIO_META_PATH,
            SCENARIO_PROJ_PATH,
            MIGRATION_PATH,
        ]
        if not path.exists()
    ]
    if missing:
        for path in missing:
            LOGGER.error("Missing required input: %s", path)
        return 1

    scenario_meta = load_json(SCENARIO_META_PATH)
    scenario_proj = pd.read_parquet(SCENARIO_PROJ_PATH)
    migration = pd.read_csv(MIGRATION_PATH)

    issues = []

    cbo_proj = scenario_proj[scenario_proj["scenario"] == "cbo_full"]
    cbo_assumptions = scenario_meta["results"]["scenarios"]["cbo_full"]["assumptions"]
    issues.extend(validate_cbo_full(cbo_proj, cbo_assumptions))

    pre_proj = scenario_proj[scenario_proj["scenario"] == "pre_2020_trend"]
    pre_assumptions = scenario_meta["results"]["scenarios"]["pre_2020_trend"][
        "assumptions"
    ]
    issues.extend(validate_pre_2020(pre_proj, migration, pre_assumptions))

    historical_cv = compute_cv(migration["nd_intl_migration"])
    LOGGER.info("Historical CV (2010--2024): %.3f", historical_cv)

    if COMBINED_PATH.exists():
        combined = pd.read_parquet(COMBINED_PATH)
        row_2045 = combined[combined["year"] == 2045]
        if not row_2045.empty:
            mc_mean = float(row_2045.iloc[0]["mc_mean"])
            mc_std = float(row_2045.iloc[0]["mc_std"])
            cv_2045 = mc_std / mc_mean if mc_mean else float("nan")
            LOGGER.info("Monte Carlo CV for 2045: %.3f", cv_2045)
        else:
            LOGGER.warning("No 2045 row found in combined forecasts.")
    else:
        LOGGER.warning("Combined forecasts parquet not found: %s", COMBINED_PATH)

    if issues:
        for issue in issues:
            LOGGER.error(issue)
        LOGGER.error("Scenario validation failed with %d issue(s).", len(issues))
        return 1

    LOGGER.info("Scenario validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
