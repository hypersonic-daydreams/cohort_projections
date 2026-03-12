#!/usr/bin/env python3
"""Walk-forward validation comparing SDC 2024 and 2026 projection methods.

Created: 2026-03-03
Author: Claude Code (automated)

Purpose
-------
Evaluate the relative accuracy of two population projection methodologies — the
North Dakota State Data Center (SDC) 2024 method and our enhanced 2026 method —
via walk-forward (expanding-window) cross-validation.  For each origin year (2005,
2010, 2015, 2020), load the base population, compute migration rates from data
available at that point in time, project forward, and compare projections to
observed population at future Census/PEP snapshot years.

Method
------
1. Load population snapshots for years 2000, 2005, 2010, 2015, 2020, 2024 using
   existing Census/PEP loaders (18 five-year age groups x 2 sexes x 53 counties).
2. Load pre-computed annualized residual migration rates for five periods:
   (2000,2005), (2005,2010), (2010,2015), (2015,2020), (2020,2024).
3. Load CDC ND 2020 life table 5-year survival rates and 2018-2022 blended
   annual ASFRs.
4. For each origin year, determine which migration periods are available (only
   periods ending at or before the origin) and which future snapshot years serve
   as validation targets.
5. SDC 2024 method: simple average across available periods, flat 60% Bakken
   dampening, constant survival/fertility/migration, 5-year projection steps.
6. 2026 method: period-specific Bakken dampening (boom-era only), male dampening,
   college-age smoothing, convergence schedule (recent/medium/long-term windows),
   0.5%/year mortality improvement, **annual projection steps** using the 1/5
   aging approximation for 5-year age groups.
7. Compute county-level (MAPE, RMSE, MPE) and state-level (APE, PE) metrics for
   each origin x method x validation year.
8. Write detailed CSVs and print comprehensive console report.

Key design decisions
--------------------
- **Walk-forward discipline**: Each origin year only uses migration data from
  periods that end at or before the origin, simulating real-time conditions.
- **Fair comparison**: Both methods use the same base population, same survival
  and fertility rates, and same 5-year age groups.
- **SDC stepping**: 5-year steps (faithful to the SDC method); values at
  non-step years are linearly interpolated.
- **2026 stepping**: Annual steps using the 1/5 aging approximation (faithful
  to the production engine's annual approach). Survival rates are converted from
  5-year to 1-year via S_1yr = S_5yr^(1/5). Fertility is already annualized.
  Migration rates are kept annualized. Mortality improvement is compounded
  annually.
- **SDC dampening**: Flat 60% factor applied post-averaging to Bakken county
  rates, matching the SDC 2024 approach.
- **2026 dampening**: Period-specific factors (0.50 for 2005-2010, 0.40 for
  2010-2015) applied pre-averaging to Bakken counties; 0.80 male dampening in
  boom periods; 50/50 college-age blending with statewide average.
- **Convergence schedule**: 2026 method ramps migration rates from recent toward
  long-term averages over the projection horizon.
- **Mortality improvement**: 2026 method reduces 1-year death probability by
  0.5%/year compound, applied annually.

Validation results
------------------
Run the script and check printed output and CSV files.  Expected: the 2026
method should show lower MAPE and more balanced MPE (less systematic over/under-
projection) compared to the SDC method, especially at longer horizons.

Inputs
------
- data/raw/nd_sdc_2024_projections/source_files/reference/Census 2000 County
  Age and Sex.xlsx — Census 2000 county age-sex population (also covers 2005).
- data/raw/nd_sdc_2024_projections/source_files/reference/cc-est2019-agesex-38
  (1).xlsx — PEP 2010-2019 county age-sex population.
- ~/workspace/shared-data/census/popest/parquet/2020-2024/county/
  cc-est2024-agesex-all.parquet — PEP 2020-2024 county age-sex population.
- data/processed/migration/residual_migration_rates.parquet — Annualized
  residual migration rates by county, age group, sex, period (5 periods).
- data/processed/sdc_2024/survival_rates_sdc_2024_full.csv — CDC ND 2020 life
  table 5-year survival rates by single-year age and sex.
- data/processed/sdc_2024/fertility_rates_5yr_summary_sdc_2024.csv — 2018-2022
  blended annual ASFRs by 5-year age group (7 groups, ages 15-49).

Output
------
Snapshot-based (original, 5-year step validation at Census/PEP years):
- data/analysis/walk_forward/county_detail.csv — One row per origin x method x
  validation_year x county with projected vs actual population.
- data/analysis/walk_forward/state_results.csv — One row per origin x method x
  validation_year with state-level error metrics.
- data/analysis/walk_forward/horizon_summary.csv — Error metrics aggregated by
  forecast horizon across all contributing origin years.
- data/analysis/walk_forward/method_comparison.csv — Side-by-side comparison
  of both methods for each origin x validation_year.

Annual-granularity (interpolated projections validated at every year):
- data/analysis/walk_forward/annual_state_results.csv — State-level error metrics
  for each origin x method x annual validation year.
- data/analysis/walk_forward/annual_county_detail.csv — County-level error metrics
  for each origin x method x annual validation year x county.
- data/analysis/walk_forward/annual_horizon_summary.csv — Error metrics aggregated
  by forecast horizon (1-19 years) across all contributing origin years.
- data/analysis/walk_forward/annual_method_comparison.csv — Side-by-side comparison
  of both methods at each annual horizon.
- data/analysis/walk_forward/projection_curves.csv — Full 50-year projection
  curves (state totals) for each origin x method, extending beyond validation.

Usage
-----
    python scripts/analysis/walk_forward_validation.py
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from cohort_projections.data.load.census_age_sex_population import (
    AGE_GROUP_LABELS,
    _ND_COUNTY_NAME_TO_FIPS,
    load_census_2000_county_age_sex,
    load_pep_2010_2019_county_age_sex,
    load_pep_2020_2024_county_age_sex,
)

# Reverse mapping: FIPS -> county name
FIPS_TO_COUNTY_NAME: dict[str, str] = {v: k for k, v in _ND_COUNTY_NAME_TO_FIPS.items()}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHARED_DATA = Path.home() / "workspace" / "shared-data"

# Data file paths
CENSUS_2000_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "nd_sdc_2024_projections"
    / "source_files"
    / "reference"
    / "Census 2000 County Age and Sex.xlsx"
)
PEP_2010_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "nd_sdc_2024_projections"
    / "source_files"
    / "reference"
    / "cc-est2019-agesex-38 (1).xlsx"
)
PEP_2024_PATH = (
    SHARED_DATA
    / "census"
    / "popest"
    / "parquet"
    / "2020-2024"
    / "county"
    / "cc-est2024-agesex-all.parquet"
)
MIGRATION_PATH = (
    PROJECT_ROOT / "data" / "processed" / "migration" / "residual_migration_rates.parquet"
)
SURVIVAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sdc_2024"
    / "survival_rates_sdc_2024_full.csv"
)
FERTILITY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sdc_2024"
    / "fertility_rates_5yr_summary_sdc_2024.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "data" / "analysis" / "walk_forward"

# 5-year age group labels (18 groups: 0-4 through 85+)
N_AGE_GROUPS = len(AGE_GROUP_LABELS)  # 18
AGE_BINS = list(range(0, 85, 5)) + [85]

# Origin years and their available migration periods
ORIGIN_YEARS = [2005, 2010, 2015, 2020]
ALL_PERIODS = [
    (2000, 2005),
    (2005, 2010),
    (2010, 2015),
    (2015, 2020),
    (2020, 2024),
]

# Validation targets: {origin_year: [(target_year, horizon_years), ...]}
VALIDATION_TARGETS: dict[int, list[tuple[int, int]]] = {
    2005: [(2010, 5), (2015, 10), (2020, 15), (2024, 19)],
    2010: [(2015, 5), (2020, 10), (2024, 14)],
    2015: [(2020, 5), (2024, 9)],
    2020: [(2024, 4)],
}

# Available migration periods for each origin year (periods ending <= origin)
AVAILABLE_PERIODS: dict[int, list[tuple[int, int]]] = {
    2005: [(2000, 2005)],
    2010: [(2000, 2005), (2005, 2010)],
    2015: [(2000, 2005), (2005, 2010), (2010, 2015)],
    2020: [(2000, 2005), (2005, 2010), (2010, 2015), (2015, 2020)],
}

# Mortality improvement rate (annual)
MORTALITY_IMPROVEMENT_RATE = 0.005

# Sex ratio at birth
MALE_BIRTH_FRACTION = 0.512
FEMALE_BIRTH_FRACTION = 0.488

# Projection horizon (max years forward)
MAX_PROJECTION_YEARS = 50
STEP = 5

# ---------------------------------------------------------------------------
# Per-Method Configuration (ADR-061 config refactor)
# ---------------------------------------------------------------------------
#
# Each projection method carries its own config dict so that config-level
# changes (county lists, dampening factors, convergence schedules) can be
# A/B tested between methods in a single walk-forward run.
#
# To create a new variant, copy a base config and override specific keys:
#     M2026R2_CONFIG = {**M2026R1_CONFIG, "college_fips": {"38017", "38035"}}

from typing import TypedDict


class MethodConfig(TypedDict, total=False):
    """Per-method configuration for walk-forward validation."""

    # Bakken oil-boom dampening
    bakken_fips: set[str]
    boom_period_dampening: dict[tuple[int, int], float]
    boom_male_dampening: float

    # College-age smoothing
    college_fips: set[str]
    college_age_groups: set[str]
    college_blend_factor: float

    # SDC flat dampening (post-averaging)
    sdc_bakken_dampening: float

    # Convergence schedule (5-year step boundaries)
    # recent_hold: steps at recent-medium blend (step 1)
    # medium_hold: steps at medium rate (steps 2..2+medium_hold-1)
    # transition_hold: steps at medium-longterm blend
    # after that: longterm
    convergence_recent_hold: int
    convergence_medium_hold: int
    convergence_transition_hold: int

    # Upstream data-processing parameters (EXP-C, EXP-D)
    gq_correction_fraction: float  # 1.0 = full GQ subtraction (default)
    rate_cap_general: float  # 0.08 = current general rate cap (default)


# Defaults for upstream parameters — used to detect overrides
_DEFAULT_GQ_CORRECTION_FRACTION: float = 1.0
_DEFAULT_RATE_CAP_GENERAL: float = 0.08


# Shared Bakken county set (used by all methods)
_BAKKEN_FIPS = {"38105", "38053", "38061", "38025", "38089"}

SDC_2024_CONFIG: MethodConfig = {
    "bakken_fips": _BAKKEN_FIPS,
    "sdc_bakken_dampening": 0.6,
}

M2026_CONFIG: MethodConfig = {
    # Bakken dampening
    "bakken_fips": _BAKKEN_FIPS,
    "boom_period_dampening": {(2005, 2010): 0.50, (2010, 2015): 0.40},
    "boom_male_dampening": 0.80,
    # College-age smoothing: original 4 counties
    "college_fips": {"38017", "38035", "38101", "38015"},
    "college_age_groups": {"15-19", "20-24"},
    "college_blend_factor": 0.5,
    # Convergence: 5-10-5 schedule (step 1 blend, steps 2-3 medium, step 4 transition, 5+ long)
    "convergence_recent_hold": 1,
    "convergence_medium_hold": 2,
    "convergence_transition_hold": 1,
    # Upstream data-processing parameters (defaults = no change from pre-computed rates)
    "gq_correction_fraction": _DEFAULT_GQ_CORRECTION_FRACTION,
    "rate_cap_general": _DEFAULT_RATE_CAP_GENERAL,
}

M2026R1_CONFIG: MethodConfig = {
    **M2026_CONFIG,
    # Expanded to 12 counties (ADR-061 Decision 4)
    "college_fips": {
        "38003", "38009", "38015", "38017", "38035",
        "38071", "38077", "38089", "38093", "38097", "38101", "38105",
    },
    # Extended college-age groups (ADR-061 Decision 1)
    "college_age_groups": {"15-19", "20-24", "25-29"},
    # Extended convergence: 5-15-5 schedule (ADR-061 Decision 3)
    "convergence_medium_hold": 3,
}

# Known state totals for sanity checking
KNOWN_STATE_TOTALS: dict[int, int] = {
    2000: 642_237,
    2005: 646_089,
    2010: 672_591,
    2015: 737_401,
    2020: 779_046,
    2024: 796_568,
}


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_population_snapshot(year: int) -> pd.DataFrame:
    """Load population snapshot for a given year.

    Supports any year from 2000 to 2024:
    - 2000-2009: Census 2000 county age-sex file (has POPESTIMATE columns)
    - 2010-2019: PEP cc-est2019-agesex file
    - 2020-2024: PEP cc-est2024-agesex-all file

    Returns DataFrame with columns: [county_fips, age_group, sex, population]
    """
    if 2000 <= year <= 2009:
        return load_census_2000_county_age_sex(CENSUS_2000_PATH, state_fips="38", year=year)
    elif 2010 <= year <= 2019:
        return load_pep_2010_2019_county_age_sex(PEP_2010_PATH, state_fips="38", year=year)
    elif 2020 <= year <= 2024:
        return load_pep_2020_2024_county_age_sex(PEP_2024_PATH, state_fips="38", year=year)
    else:
        raise ValueError(f"No loader available for year {year}")


def load_all_snapshots() -> dict[int, pd.DataFrame]:
    """Load all population snapshots needed for validation."""
    snapshot_years = {2000, 2005, 2010, 2015, 2020, 2024}
    snapshots: dict[int, pd.DataFrame] = {}
    for yr in sorted(snapshot_years):
        df = load_population_snapshot(yr)
        total = df["population"].sum()
        expected = KNOWN_STATE_TOTALS.get(yr, 0)
        print(f"  {yr}: {total:,.0f} (expected {expected:,})")
        snapshots[yr] = df
    return snapshots


def load_migration_rates_raw() -> pd.DataFrame:
    """Load raw migration rates (annualized) for all periods.

    Returns DataFrame with columns:
    [county_fips, age_group, sex, period_start, period_end, migration_rate]
    """
    df = pd.read_parquet(MIGRATION_PATH)
    return df[["county_fips", "age_group", "sex", "period_start", "period_end", "migration_rate"]]


def load_survival_rates() -> dict[tuple[str, str], float]:
    """Load 5-year survival rates keyed by (age_group, sex).

    Returns dict mapping (age_group_label, sex) -> survival_rate_5yr.
    """
    df = pd.read_csv(SURVIVAL_PATH)

    # Map single-year ages to 5-year group labels
    df["age_group"] = pd.cut(
        df["age"],
        bins=AGE_BINS + [200],
        labels=AGE_GROUP_LABELS,
        right=False,
        include_lowest=True,
    ).astype(str)

    # Take one rate per group (first age in each group)
    rates = (
        df.groupby(["age_group", "sex"], observed=True)["survival_rate_5yr"]
        .first()
        .to_dict()
    )
    return rates


def load_fertility_rates() -> dict[str, float]:
    """Load annual ASFRs keyed by age_group label (e.g., '15-19').

    Returns dict mapping age_group_label -> annual ASFR.
    """
    df = pd.read_csv(FERTILITY_PATH)
    rates: dict[str, float] = {}
    for _, row in df.iterrows():
        label = f"{int(row['age_start'])}-{int(row['age_end'])}"
        rates[label] = row["asfr_annual"]
    return rates


# ---------------------------------------------------------------------------
# GQ Correction Override — in-memory recomputation (EXP-C)
# ---------------------------------------------------------------------------

GQ_HISTORICAL_PATH = (
    PROJECT_ROOT / "data" / "processed" / "gq_county_age_sex_historical.parquet"
)


def recompute_migration_with_gq_override(
    snapshots: dict[int, pd.DataFrame],
    gq_correction_fraction: float,
) -> pd.DataFrame:
    """Recompute residual migration rates in-memory with an overridden GQ fraction.

    Loads GQ historical data, subtracts ``gq_correction_fraction`` of GQ from
    each snapshot year, then runs the residual migration computation on the
    adjusted household-only population.  This avoids mutating any on-disk
    ``data/processed/`` files.

    Args:
        snapshots: Population snapshots keyed by year (original, with full pop).
        gq_correction_fraction: Fraction of GQ to subtract (0.0 = none, 1.0 = full).

    Returns:
        DataFrame with the same schema as ``load_migration_rates_raw()``
        [county_fips, age_group, sex, period_start, period_end, migration_rate].
    """
    from cohort_projections.data.process.residual_migration import (
        compute_residual_migration_rates,
        subtract_gq_from_populations,
    )

    # Load GQ historical data
    gq_historical = pd.read_parquet(GQ_HISTORICAL_PATH)

    # Subtract overridden GQ fraction from snapshots
    adjusted_pops = subtract_gq_from_populations(
        snapshots, gq_historical, fraction=gq_correction_fraction
    )

    # Load survival rates for the residual computation
    survival_df = pd.read_csv(SURVIVAL_PATH)

    # Compute residual migration for each period
    all_period_rates: list[pd.DataFrame] = []
    for period_start, period_end in ALL_PERIODS:
        if period_start not in adjusted_pops or period_end not in adjusted_pops:
            continue
        period_rates = compute_residual_migration_rates(
            pop_start=adjusted_pops[period_start],
            pop_end=adjusted_pops[period_end],
            survival_rates=survival_df,
            period=(period_start, period_end),
        )
        all_period_rates.append(period_rates)

    combined = pd.concat(all_period_rates, ignore_index=True)
    return combined[
        ["county_fips", "age_group", "sex", "period_start", "period_end", "migration_rate"]
    ]


def maybe_recompute_mig_raw(
    mig_raw: pd.DataFrame,
    snapshots: dict[int, pd.DataFrame],
    config: MethodConfig,
) -> pd.DataFrame:
    """Conditionally recompute migration rates if GQ fraction differs from default.

    If the config's ``gq_correction_fraction`` equals the default (1.0) or is
    absent, returns the original ``mig_raw`` unchanged (same object).  Otherwise,
    triggers in-memory recomputation with the overridden fraction.

    Args:
        mig_raw: Pre-computed migration rates DataFrame.
        snapshots: Population snapshots keyed by year.
        config: Per-method config dict.

    Returns:
        Migration rates DataFrame (original or recomputed).
    """
    fraction = config.get("gq_correction_fraction", _DEFAULT_GQ_CORRECTION_FRACTION)
    if fraction == _DEFAULT_GQ_CORRECTION_FRACTION:
        return mig_raw
    return recompute_migration_with_gq_override(snapshots, fraction)


# ---------------------------------------------------------------------------
# SDC 2024 Method — Rate Preparation
# ---------------------------------------------------------------------------


def prepare_sdc_rates(
    mig_raw: pd.DataFrame,
    origin_year: int,
    config: MethodConfig | None = None,
) -> pd.DataFrame:
    """Prepare migration rates using SDC 2024 method.

    1. Filter to available periods for this origin year.
    2. Convert annualized rates to 5-year rates.
    3. Simple arithmetic average across all available periods.
    4. Apply flat Bakken dampening post-averaging.

    Returns DataFrame with columns: [county_fips, age_group, sex, migration_rate_5yr]
    """
    cfg = config or SDC_2024_CONFIG
    periods = AVAILABLE_PERIODS[origin_year]
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

    # Flat Bakken dampening (post-averaging)
    bakken_fips = cfg.get("bakken_fips", set())
    sdc_dampening = cfg.get("sdc_bakken_dampening", 0.6)
    bakken_mask = avg["county_fips"].isin(bakken_fips)
    avg.loc[bakken_mask, "migration_rate_5yr"] *= sdc_dampening

    return avg


# ---------------------------------------------------------------------------
# 2026 Method — Rate Preparation
# ---------------------------------------------------------------------------


def _compute_statewide_average_rate(
    period_rates: pd.DataFrame,
) -> pd.DataFrame:
    """Compute statewide average migration rate by age_group and sex.

    Args:
        period_rates: DataFrame with columns
            [county_fips, age_group, sex, rate_5yr]

    Returns:
        DataFrame with columns [age_group, sex, state_avg_rate]
    """
    return (
        period_rates.groupby(["age_group", "sex"], as_index=False)["rate_5yr"]
        .mean()
        .rename(columns={"rate_5yr": "state_avg_rate"})
    )


def _apply_2026_period_dampening(
    mig_raw: pd.DataFrame,
    periods: list[tuple[int, int]],
    config: MethodConfig,
    *,
    annualize: bool = False,
) -> pd.DataFrame:
    """Apply 2026 method period-specific dampening and convert to 5-year rates.

    1. Filter to requested periods.
    2. For boom-era periods, apply overall dampening factor to Bakken counties.
    3. For boom-era periods, apply male dampening factor to male rates in
       Bakken counties.
    4. College-age smoothing: blend with statewide average for college ages
       in college counties (all read from config).
    5. Convert annualized rates to 5-year rates (or keep annualized if
       ``annualize=True``).

    Args:
        mig_raw: Raw migration rates DataFrame.
        periods: List of (start, end) period tuples.
        config: Per-method config dict.
        annualize: If True, keep rates as annualized rates (column name
            ``rate_5yr`` is still used for interface consistency with
            downstream averaging, but the values are annualized).

    Returns DataFrame with columns:
    [county_fips, age_group, sex, period_start, period_end, rate_5yr]
    """
    bakken_fips = config.get("bakken_fips", set())
    boom_dampening = config.get("boom_period_dampening", {})
    boom_male = config.get("boom_male_dampening", 1.0)
    college_fips = config.get("college_fips", set())
    college_age_groups = config.get("college_age_groups", set())
    blend_factor = config.get("college_blend_factor", 0.5)

    mask = mig_raw.apply(
        lambda r: (r["period_start"], r["period_end"]) in periods, axis=1
    )
    df = mig_raw[mask].copy()

    # Apply boom-era dampening (pre-conversion to 5yr) on annualized rates
    for period, factor in boom_dampening.items():
        if period not in periods:
            continue
        period_mask = (df["period_start"] == period[0]) & (df["period_end"] == period[1])
        bakken_mask = df["county_fips"].isin(bakken_fips)

        # Overall dampening for Bakken counties in boom periods
        df.loc[period_mask & bakken_mask, "migration_rate"] *= factor

        # Additional male dampening in boom periods for Bakken counties
        male_mask = df["sex"] == "Male"
        df.loc[period_mask & bakken_mask & male_mask, "migration_rate"] *= boom_male

    # Convert annualized to 5-year rates (or keep annualized)
    if annualize:
        df["rate_5yr"] = df["migration_rate"]
    else:
        df["rate_5yr"] = (1 + df["migration_rate"]) ** 5 - 1

    # College-age smoothing: for each period, blend with statewide avg
    if college_fips and college_age_groups:
        result_parts = []
        for period in periods:
            p_mask = (df["period_start"] == period[0]) & (df["period_end"] == period[1])
            period_df = df[p_mask].copy()

            # Compute statewide averages for this period
            state_avg = _compute_statewide_average_rate(period_df)

            # Merge statewide avg onto period data
            period_df = period_df.merge(state_avg, on=["age_group", "sex"], how="left")

            # Apply college-age smoothing
            college_mask = (
                period_df["county_fips"].isin(college_fips)
                & period_df["age_group"].isin(college_age_groups)
            )
            period_df.loc[college_mask, "rate_5yr"] = (
                blend_factor * period_df.loc[college_mask, "rate_5yr"]
                + (1 - blend_factor) * period_df.loc[college_mask, "state_avg_rate"]
            )
            period_df = period_df.drop(columns=["state_avg_rate"])
            result_parts.append(period_df)

        return pd.concat(result_parts, ignore_index=True)

    return df


def prepare_2026_convergence_rates(
    mig_raw: pd.DataFrame,
    origin_year: int,
    config: MethodConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Prepare migration rate windows for 2026 convergence schedule.

    Returns dict with keys 'recent', 'medium', 'longterm', each a DataFrame
    with columns [county_fips, age_group, sex, migration_rate_5yr].

    Window definitions:
    - Origin 2005 (1 period): recent=medium=longterm = (2000,2005)
    - Origin 2010 (2 periods): recent=(2005,2010), medium=avg(2000-2010),
      longterm=avg(2000-2010)
    - Origin 2015 (3 periods): recent=(2010,2015), medium=avg(2005-2015),
      longterm=avg(2000-2015)
    - Origin 2020 (4 periods): recent=(2015,2020), medium=avg(2010-2020),
      longterm=avg(2000-2020)
    """
    cfg = config or M2026_CONFIG
    periods = AVAILABLE_PERIODS[origin_year]

    # Apply dampening/smoothing to all available periods
    dampened = _apply_2026_period_dampening(mig_raw, periods, cfg)

    def _avg_periods(
        df: pd.DataFrame, target_periods: list[tuple[int, int]]
    ) -> pd.DataFrame:
        mask = df.apply(
            lambda r: (r["period_start"], r["period_end"]) in target_periods, axis=1
        )
        subset = df[mask]
        return (
            subset.groupby(["county_fips", "age_group", "sex"], as_index=False)["rate_5yr"]
            .mean()
            .rename(columns={"rate_5yr": "migration_rate_5yr"})
        )

    if origin_year == 2005:
        # 1 period: no convergence effect
        avg = _avg_periods(dampened, periods)
        return {"recent": avg, "medium": avg.copy(), "longterm": avg.copy()}

    elif origin_year == 2010:
        recent_periods = [(2005, 2010)]
        medium_periods = [(2000, 2005), (2005, 2010)]
        longterm_periods = [(2000, 2005), (2005, 2010)]
        return {
            "recent": _avg_periods(dampened, recent_periods),
            "medium": _avg_periods(dampened, medium_periods),
            "longterm": _avg_periods(dampened, longterm_periods),
        }

    elif origin_year == 2015:
        recent_periods = [(2010, 2015)]
        medium_periods = [(2005, 2010), (2010, 2015)]
        longterm_periods = [(2000, 2005), (2005, 2010), (2010, 2015)]
        return {
            "recent": _avg_periods(dampened, recent_periods),
            "medium": _avg_periods(dampened, medium_periods),
            "longterm": _avg_periods(dampened, longterm_periods),
        }

    elif origin_year == 2020:
        recent_periods = [(2015, 2020)]
        medium_periods = [(2010, 2015), (2015, 2020)]
        longterm_periods = [(2000, 2005), (2005, 2010), (2010, 2015), (2015, 2020)]
        return {
            "recent": _avg_periods(dampened, recent_periods),
            "medium": _avg_periods(dampened, medium_periods),
            "longterm": _avg_periods(dampened, longterm_periods),
        }

    raise ValueError(f"Unknown origin year: {origin_year}")


def get_convergence_rate_for_step(
    step_number: int,
    windows: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Select the appropriate migration rate for a given projection step.

    Convergence schedule:
    - Step 1 (years 1-5): average of recent and medium
    - Steps 2-3 (years 6-15): medium rate
    - Step 4 (years 16-20): average of medium and long-term
    - Steps 5+ (years 21+): long-term rate

    Args:
        step_number: 1-based step number (step 1 = first 5-year projection).
        windows: dict with 'recent', 'medium', 'longterm' DataFrames.

    Returns:
        DataFrame with columns [county_fips, age_group, sex, migration_rate_5yr]
    """
    if step_number == 1:
        # Average of recent and medium
        merged = windows["recent"].merge(
            windows["medium"],
            on=["county_fips", "age_group", "sex"],
            suffixes=("_recent", "_medium"),
        )
        merged["migration_rate_5yr"] = (
            0.5 * merged["migration_rate_5yr_recent"]
            + 0.5 * merged["migration_rate_5yr_medium"]
        )
        return merged[["county_fips", "age_group", "sex", "migration_rate_5yr"]]

    elif step_number in (2, 3):
        return windows["medium"].copy()

    elif step_number == 4:
        # Average of medium and long-term
        merged = windows["medium"].merge(
            windows["longterm"],
            on=["county_fips", "age_group", "sex"],
            suffixes=("_medium", "_longterm"),
        )
        merged["migration_rate_5yr"] = (
            0.5 * merged["migration_rate_5yr_medium"]
            + 0.5 * merged["migration_rate_5yr_longterm"]
        )
        return merged[["county_fips", "age_group", "sex", "migration_rate_5yr"]]

    else:  # step >= 5
        return windows["longterm"].copy()


def prepare_2026_convergence_rates_annual(
    mig_raw: pd.DataFrame,
    origin_year: int,
    config: MethodConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Prepare ANNUALIZED migration rate windows for the 2026 annual engine.

    Identical to ``prepare_2026_convergence_rates`` except the underlying
    rates remain annualized (not compounded to 5-year equivalents).  This is
    the correct form for the annual stepping engine which applies migration
    once per year.

    Returns dict with keys 'recent', 'medium', 'longterm', each a DataFrame
    with columns [county_fips, age_group, sex, migration_rate_annual].
    """
    cfg = config or M2026_CONFIG
    periods = AVAILABLE_PERIODS[origin_year]

    # Apply dampening/smoothing but keep rates annualized
    dampened = _apply_2026_period_dampening(mig_raw, periods, cfg, annualize=True)

    def _avg_periods(
        df: pd.DataFrame, target_periods: list[tuple[int, int]]
    ) -> pd.DataFrame:
        mask = df.apply(
            lambda r: (r["period_start"], r["period_end"]) in target_periods, axis=1
        )
        subset = df[mask]
        return (
            subset.groupby(["county_fips", "age_group", "sex"], as_index=False)["rate_5yr"]
            .mean()
            .rename(columns={"rate_5yr": "migration_rate_annual"})
        )

    if origin_year == 2005:
        avg = _avg_periods(dampened, periods)
        return {"recent": avg, "medium": avg.copy(), "longterm": avg.copy()}

    elif origin_year == 2010:
        recent_periods = [(2005, 2010)]
        medium_periods = [(2000, 2005), (2005, 2010)]
        longterm_periods = [(2000, 2005), (2005, 2010)]
        return {
            "recent": _avg_periods(dampened, recent_periods),
            "medium": _avg_periods(dampened, medium_periods),
            "longterm": _avg_periods(dampened, longterm_periods),
        }

    elif origin_year == 2015:
        recent_periods = [(2010, 2015)]
        medium_periods = [(2005, 2010), (2010, 2015)]
        longterm_periods = [(2000, 2005), (2005, 2010), (2010, 2015)]
        return {
            "recent": _avg_periods(dampened, recent_periods),
            "medium": _avg_periods(dampened, medium_periods),
            "longterm": _avg_periods(dampened, longterm_periods),
        }

    elif origin_year == 2020:
        recent_periods = [(2015, 2020)]
        medium_periods = [(2010, 2015), (2015, 2020)]
        longterm_periods = [(2000, 2005), (2005, 2010), (2010, 2015), (2015, 2020)]
        return {
            "recent": _avg_periods(dampened, recent_periods),
            "medium": _avg_periods(dampened, medium_periods),
            "longterm": _avg_periods(dampened, longterm_periods),
        }

    raise ValueError(f"Unknown origin year: {origin_year}")


# ---------------------------------------------------------------------------
# Rate Cap — annual-rate equivalent of convergence_interpolation._apply_rate_cap
# ---------------------------------------------------------------------------

_DEFAULT_COLLEGE_CAP: float = 0.15
_DEFAULT_COLLEGE_AGES: frozenset[str] = frozenset({"15-19", "20-24"})


def _apply_annual_rate_cap(
    rate_series: pd.Series,
    age_groups: pd.Series,
    general_cap: float,
    college_cap: float = _DEFAULT_COLLEGE_CAP,
    college_ages: frozenset[str] | set[str] = _DEFAULT_COLLEGE_AGES,
) -> pd.Series:
    """Apply age-aware symmetric cap to annualized migration rates.

    Mirrors the logic of ``convergence_interpolation._apply_rate_cap`` but
    operates on annualized rates within the walk-forward validation context.

    College-age cells (default 15-19, 20-24) are capped at ``college_cap``
    (default 0.15). All other ages are capped at ``general_cap``.

    Args:
        rate_series: Series of annualized migration rates.
        age_groups: Series of age group labels aligned with *rate_series*.
        general_cap: Symmetric cap for non-college ages.
        college_cap: Symmetric cap for college ages (default 0.15).
        college_ages: Set of age group labels receiving the wider cap.

    Returns:
        Capped rate series (same index as input).
    """
    college_mask = age_groups.isin(college_ages)

    capped = rate_series.copy()
    capped = capped.clip(lower=-general_cap, upper=general_cap)
    capped[college_mask] = rate_series[college_mask].clip(
        lower=-college_cap, upper=college_cap
    )
    return capped


def get_convergence_rate_for_year(
    year_offset: int,
    windows: dict[str, pd.DataFrame],
    config: MethodConfig | None = None,
) -> pd.DataFrame:
    """Select the appropriate annualized migration rate for a given projection year.

    Uses the convergence schedule from config to map year offsets to rate windows.
    The schedule is defined by three parameters (all in 5-year steps):
    - ``convergence_recent_hold``: steps at recent-medium blend
    - ``convergence_medium_hold``: steps at medium rate
    - ``convergence_transition_hold``: steps at medium-longterm blend
    - After that: longterm rate

    Default (m2026): 1-2-1 = 5-10-5 years (recent blend / medium / transition / long)
    m2026r1:         1-3-1 = 5-15-5 years

    Args:
        year_offset: Years from origin (1-based; year 1 = first year forward).
        windows: dict with 'recent', 'medium', 'longterm' DataFrames, each
            having columns [county_fips, age_group, sex, migration_rate_annual].
        config: Per-method config dict. If None, uses M2026_CONFIG defaults.

    Returns:
        DataFrame with columns [county_fips, age_group, sex, migration_rate_annual]
    """
    cfg = config or M2026_CONFIG
    recent_hold = cfg.get("convergence_recent_hold", 1)
    medium_hold = cfg.get("convergence_medium_hold", 2)
    transition_hold = cfg.get("convergence_transition_hold", 1)

    # Map year offset to 5-year step number
    step_number = (year_offset - 1) // 5 + 1  # year 1-5 -> step 1, etc.

    # Determine phase based on step boundaries
    recent_end = recent_hold  # step <= recent_end: recent-medium blend
    medium_end = recent_end + medium_hold  # step <= medium_end: medium
    transition_end = medium_end + transition_hold  # step <= transition_end: transition

    if step_number <= recent_end:
        # Blend of recent and medium
        merged = windows["recent"].merge(
            windows["medium"],
            on=["county_fips", "age_group", "sex"],
            suffixes=("_recent", "_medium"),
        )
        merged["migration_rate_annual"] = (
            0.5 * merged["migration_rate_annual_recent"]
            + 0.5 * merged["migration_rate_annual_medium"]
        )
        result = merged[["county_fips", "age_group", "sex", "migration_rate_annual"]].copy()

    elif step_number <= medium_end:
        result = windows["medium"].copy()

    elif step_number <= transition_end:
        # Blend of medium and long-term
        merged = windows["medium"].merge(
            windows["longterm"],
            on=["county_fips", "age_group", "sex"],
            suffixes=("_medium", "_longterm"),
        )
        merged["migration_rate_annual"] = (
            0.5 * merged["migration_rate_annual_medium"]
            + 0.5 * merged["migration_rate_annual_longterm"]
        )
        result = merged[["county_fips", "age_group", "sex", "migration_rate_annual"]].copy()

    else:
        result = windows["longterm"].copy()

    # Apply rate cap if the config specifies a non-default value
    rate_cap = cfg.get("rate_cap_general", _DEFAULT_RATE_CAP_GENERAL)
    if rate_cap != _DEFAULT_RATE_CAP_GENERAL:
        result["migration_rate_annual"] = _apply_annual_rate_cap(
            result["migration_rate_annual"],
            result["age_group"],
            general_cap=rate_cap,
        )

    return result


def _build_mig_annual_lookup(
    mig_df: pd.DataFrame, county_fips: str
) -> dict[tuple[str, str], float]:
    """Build annualized migration rate lookup for a single county."""
    county_mig = mig_df[mig_df["county_fips"] == county_fips]
    lookup: dict[tuple[str, str], float] = {}
    for _, row in county_mig.iterrows():
        lookup[(row["age_group"], row["sex"])] = row["migration_rate_annual"]
    return lookup






# ---------------------------------------------------------------------------
# Projection Engines
# ---------------------------------------------------------------------------


def _build_mig_lookup(
    mig_df: pd.DataFrame, county_fips: str
) -> dict[tuple[str, str], float]:
    """Build migration rate lookup for a single county."""
    county_mig = mig_df[mig_df["county_fips"] == county_fips]
    lookup: dict[tuple[str, str], float] = {}
    for _, row in county_mig.iterrows():
        lookup[(row["age_group"], row["sex"])] = row["migration_rate_5yr"]
    return lookup


def _init_pop(
    base_pop: pd.DataFrame, county_fips: str
) -> dict[tuple[str, str], float]:
    """Initialize population dict for a county from snapshot data."""
    county_base = base_pop[base_pop["county_fips"] == county_fips]
    pop: dict[tuple[str, str], float] = {}
    for _, row in county_base.iterrows():
        pop[(row["age_group"], row["sex"])] = row["population"]

    # Ensure all age-sex combinations exist
    for ag in AGE_GROUP_LABELS:
        for sex in ["Male", "Female"]:
            if (ag, sex) not in pop:
                pop[(ag, sex)] = 0.0
    return pop


def project_sdc(
    base_pop: pd.DataFrame,
    survival: dict[tuple[str, str], float],
    fertility: dict[str, float],
    mig_rates: pd.DataFrame,
    county_fips: str,
    n_steps: int,
    origin_year: int,
) -> dict[int, float]:
    """Run SDC 2024 method projection for one county.

    Returns dict mapping year -> total population.
    """
    pop = _init_pop(base_pop, county_fips)
    mig_lookup = _build_mig_lookup(mig_rates, county_fips)

    results: dict[int, float] = {origin_year: sum(pop.values())}

    for step_idx in range(1, n_steps + 1):
        year = origin_year + step_idx * STEP
        new_pop: dict[tuple[str, str], float] = {}

        for sex in ["Male", "Female"]:
            # Survive ages 5-9 through 80-84
            for i in range(1, N_AGE_GROUPS - 1):
                prev_ag = AGE_GROUP_LABELS[i - 1]
                curr_ag = AGE_GROUP_LABELS[i]
                surv = survival.get((prev_ag, sex), 0.0)
                new_pop[(curr_ag, sex)] = pop[(prev_ag, sex)] * surv

            # Open-ended 85+
            surv_80_84 = survival.get(("80-84", sex), 0.0)
            surv_85p = survival.get(("85+", sex), 0.0)
            new_pop[("85+", sex)] = (
                pop[("80-84", sex)] * surv_80_84 + pop[("85+", sex)] * surv_85p
            )

            # Migrate ages 5-9 through 85+
            for i in range(1, N_AGE_GROUPS):
                ag = AGE_GROUP_LABELS[i]
                mig_rate = mig_lookup.get((ag, sex), 0.0)
                new_pop[(ag, sex)] *= 1 + mig_rate

        # Births
        total_births_5yr = 0.0
        for fert_ag, asfr in fertility.items():
            fem_pop = pop.get((fert_ag, "Female"), 0.0)
            total_births_5yr += fem_pop * asfr * 5

        male_births = total_births_5yr * MALE_BIRTH_FRACTION
        female_births = total_births_5yr * FEMALE_BIRTH_FRACTION

        # Birth survival
        new_pop[("0-4", "Male")] = male_births * survival.get(("0-4", "Male"), 0.0)
        new_pop[("0-4", "Female")] = female_births * survival.get(("0-4", "Female"), 0.0)

        # Migrate 0-4
        for sex in ["Male", "Female"]:
            mig_rate = mig_lookup.get(("0-4", sex), 0.0)
            new_pop[("0-4", sex)] *= 1 + mig_rate

        # Floor at 0
        for key in new_pop:
            new_pop[key] = max(0.0, new_pop[key])

        results[year] = sum(new_pop.values())
        pop = new_pop

    return results


def _get_improved_survival(
    base_survival: dict[tuple[str, str], float],
    step_number: int,
) -> dict[tuple[str, str], float]:
    """Compute mortality-improved survival rates for a given projection step.

    For step n, the midpoint year of the projection interval is step_n * 5 + 2.5
    years from the origin.

    q_5yr = 1 - survival_rate_5yr  (5-year death probability)
    q_improved = q_5yr * (1 - 0.005)^midpoint_year
    survival_improved = 1 - q_improved
    """
    midpoint_year = step_number * 5 - 2.5  # step 1 -> 2.5, step 2 -> 7.5, etc.
    improvement_factor = (1 - MORTALITY_IMPROVEMENT_RATE) ** midpoint_year

    improved: dict[tuple[str, str], float] = {}
    for key, surv in base_survival.items():
        q_5yr = 1 - surv
        q_improved = q_5yr * improvement_factor
        improved[key] = 1 - q_improved

    return improved


def project_2026(
    base_pop: pd.DataFrame,
    survival: dict[tuple[str, str], float],
    fertility: dict[str, float],
    convergence_windows: dict[str, pd.DataFrame],
    county_fips: str,
    n_steps: int,
    origin_year: int,
) -> dict[int, float]:
    """Run 2026 method projection for one county.

    Returns dict mapping year -> total population.
    """
    pop = _init_pop(base_pop, county_fips)

    results: dict[int, float] = {origin_year: sum(pop.values())}

    for step_idx in range(1, n_steps + 1):
        year = origin_year + step_idx * STEP

        # Get convergence-scheduled migration rates for this step
        step_mig = get_convergence_rate_for_step(step_idx, convergence_windows)
        mig_lookup = _build_mig_lookup(step_mig, county_fips)

        # Get mortality-improved survival rates for this step
        step_survival = _get_improved_survival(survival, step_idx)

        new_pop: dict[tuple[str, str], float] = {}

        for sex in ["Male", "Female"]:
            # Survive ages 5-9 through 80-84
            for i in range(1, N_AGE_GROUPS - 1):
                prev_ag = AGE_GROUP_LABELS[i - 1]
                curr_ag = AGE_GROUP_LABELS[i]
                surv = step_survival.get((prev_ag, sex), 0.0)
                new_pop[(curr_ag, sex)] = pop[(prev_ag, sex)] * surv

            # Open-ended 85+
            surv_80_84 = step_survival.get(("80-84", sex), 0.0)
            surv_85p = step_survival.get(("85+", sex), 0.0)
            new_pop[("85+", sex)] = (
                pop[("80-84", sex)] * surv_80_84 + pop[("85+", sex)] * surv_85p
            )

            # Migrate ages 5-9 through 85+
            for i in range(1, N_AGE_GROUPS):
                ag = AGE_GROUP_LABELS[i]
                mig_rate = mig_lookup.get((ag, sex), 0.0)
                new_pop[(ag, sex)] *= 1 + mig_rate

        # Births (use beginning-of-period female pop)
        total_births_5yr = 0.0
        for fert_ag, asfr in fertility.items():
            fem_pop = pop.get((fert_ag, "Female"), 0.0)
            total_births_5yr += fem_pop * asfr * 5

        male_births = total_births_5yr * MALE_BIRTH_FRACTION
        female_births = total_births_5yr * FEMALE_BIRTH_FRACTION

        # Birth survival (use improved rates)
        new_pop[("0-4", "Male")] = male_births * step_survival.get(("0-4", "Male"), 0.0)
        new_pop[("0-4", "Female")] = female_births * step_survival.get(
            ("0-4", "Female"), 0.0
        )

        # Migrate 0-4
        for sex in ["Male", "Female"]:
            mig_rate = mig_lookup.get(("0-4", sex), 0.0)
            new_pop[("0-4", sex)] *= 1 + mig_rate

        # Floor at 0
        for key in new_pop:
            new_pop[key] = max(0.0, new_pop[key])

        results[year] = sum(new_pop.values())
        pop = new_pop

    return results


def _get_improved_survival_annual(
    base_survival: dict[tuple[str, str], float],
    years_from_origin: int,
) -> dict[tuple[str, str], float]:
    """Compute mortality-improved 1-year survival rates for a given year offset.

    Converts 5-year base survival to 1-year via geometric interpolation:
        S_1yr = S_5yr^(1/5)

    Then applies annual mortality improvement:
        q_1yr = 1 - S_1yr
        q_improved = q_1yr * (1 - 0.005)^years_from_origin
        S_1yr_improved = 1 - q_improved

    Args:
        base_survival: 5-year survival rates keyed by (age_group, sex).
        years_from_origin: Number of years from the origin year (1-based).

    Returns:
        dict mapping (age_group, sex) -> improved 1-year survival rate.
    """
    improvement_factor = (1 - MORTALITY_IMPROVEMENT_RATE) ** years_from_origin

    improved: dict[tuple[str, str], float] = {}
    for key, surv_5yr in base_survival.items():
        # Convert 5-year survival to 1-year: S_1yr = S_5yr^(1/5)
        s_1yr = surv_5yr ** 0.2
        # Apply mortality improvement
        q_1yr = 1 - s_1yr
        q_improved = q_1yr * improvement_factor
        improved[key] = 1 - q_improved

    return improved


def _project_annual_core(
    base_pop: pd.DataFrame,
    survival: dict[tuple[str, str], float],
    fertility: dict[str, float],
    convergence_windows: dict[str, pd.DataFrame],
    county_fips: str,
    n_years: int,
    origin_year: int,
    config: MethodConfig | None = None,
) -> dict[int, float]:
    """Shared annual projection engine with config-driven convergence schedule.

    Args:
        base_pop: Base population snapshot DataFrame.
        survival: 5-year survival rates keyed by (age_group, sex).
        fertility: Annual ASFRs keyed by age_group label.
        convergence_windows: Dict with 'recent', 'medium', 'longterm' DataFrames,
            each with columns [county_fips, age_group, sex, migration_rate_annual].
        county_fips: FIPS code for the county.
        n_years: Number of years to project forward.
        origin_year: The origin year of the projection.
        config: Per-method config dict (passed to convergence getter).

    Returns:
        dict mapping year -> total county population.
    """
    pop = _init_pop(base_pop, county_fips)

    results: dict[int, float] = {origin_year: sum(pop.values())}

    for yr_offset in range(1, n_years + 1):
        year = origin_year + yr_offset

        # Get convergence-scheduled annualized migration rates for this year
        year_mig = get_convergence_rate_for_year(yr_offset, convergence_windows, config)
        mig_lookup = _build_mig_annual_lookup(year_mig, county_fips)

        # Get mortality-improved 1-year survival rates
        s1 = _get_improved_survival_annual(survival, yr_offset)

        new_pop: dict[tuple[str, str], float] = {}

        for sex in ["Male", "Female"]:
            # --- A. Aging + Survival using 1/5 approximation ---

            # Survive each cohort first (store survived values for aging calc)
            survived: dict[str, float] = {}
            for i in range(N_AGE_GROUPS):
                ag = AGE_GROUP_LABELS[i]
                survived[ag] = s1.get((ag, sex), 0.0) * pop.get((ag, sex), 0.0)

            # Age groups 5-9 through 80-84 (indices 1..16):
            #   pop(a, t+1) = (4/5)*survived(a) + (1/5)*survived(a-5)
            for i in range(1, N_AGE_GROUPS - 1):
                curr_ag = AGE_GROUP_LABELS[i]
                prev_ag = AGE_GROUP_LABELS[i - 1]
                new_pop[(curr_ag, sex)] = (
                    (4.0 / 5.0) * survived[curr_ag]
                    + (1.0 / 5.0) * survived[prev_ag]
                )

            # Terminal group 85+:
            #   pop(85+, t+1) = survived(85+) + (1/5)*survived(80-84)
            new_pop[("85+", sex)] = (
                survived["85+"]
                + (1.0 / 5.0) * survived["80-84"]
            )

            # 0-4 will be handled below after births

            # --- C. Migration for ages 5-9 through 85+ ---
            for i in range(1, N_AGE_GROUPS):
                ag = AGE_GROUP_LABELS[i]
                mig_rate = mig_lookup.get((ag, sex), 0.0)
                new_pop[(ag, sex)] += mig_rate * new_pop[(ag, sex)]

        # --- B. Births ---
        # Use beginning-of-period female population; fertility is already annual
        total_births = 0.0
        for fert_ag, asfr in fertility.items():
            fem_pop = pop.get((fert_ag, "Female"), 0.0)
            total_births += fem_pop * asfr

        male_births = total_births * MALE_BIRTH_FRACTION
        female_births = total_births * FEMALE_BIRTH_FRACTION

        # Birth survival: survive one year as infant using S_1yr(0-4)
        infant_surv_m = s1.get(("0-4", "Male"), 0.0)
        infant_surv_f = s1.get(("0-4", "Female"), 0.0)
        male_births_survived = male_births * infant_surv_m
        female_births_survived = female_births * infant_surv_f

        # 0-4 age group:
        #   pop(0-4, t+1) = (4/5)*survived(0-4) + births_survived
        for sex in ["Male", "Female"]:
            survived_04 = s1.get(("0-4", sex), 0.0) * pop.get(("0-4", sex), 0.0)
            births_surv = male_births_survived if sex == "Male" else female_births_survived
            new_pop[("0-4", sex)] = (4.0 / 5.0) * survived_04 + births_surv

        # Migrate 0-4
        for sex in ["Male", "Female"]:
            mig_rate = mig_lookup.get(("0-4", sex), 0.0)
            new_pop[("0-4", sex)] += mig_rate * new_pop[("0-4", sex)]

        # --- E. Floor at zero ---
        for key in new_pop:
            new_pop[key] = max(0.0, new_pop[key])

        results[year] = sum(new_pop.values())
        pop = new_pop

    return results


def project_2026_annual(
    base_pop: pd.DataFrame,
    survival: dict[tuple[str, str], float],
    fertility: dict[str, float],
    convergence_windows: dict[str, pd.DataFrame],
    county_fips: str,
    n_years: int,
    origin_year: int,
    config: MethodConfig | None = None,
) -> dict[int, float]:
    """Run 2026-family projection for one county with annual stepping.

    Uses the 1/5 aging approximation for 5-year age groups with annual steps:
    - Aging: (4/5) of survived cohort stays, (1/5) ages up from previous group
    - Births: annualized fertility (ASFR already annual)
    - Migration: annualized rates applied each year
    - Mortality improvement: compounded annually

    Args:
        base_pop: Base population snapshot DataFrame.
        survival: 5-year survival rates keyed by (age_group, sex).
        fertility: Annual ASFRs keyed by age_group label.
        convergence_windows: Dict with 'recent', 'medium', 'longterm' DataFrames,
            each with columns [county_fips, age_group, sex, migration_rate_annual].
        county_fips: FIPS code for the county.
        n_years: Number of years to project forward.
        origin_year: The origin year of the projection.
        config: Per-method config dict (controls convergence schedule).

    Returns:
        dict mapping year -> total county population.
    """
    return _project_annual_core(
        base_pop, survival, fertility, convergence_windows,
        county_fips, n_years, origin_year,
        config=config,
    )


# ---------------------------------------------------------------------------
# Method Dispatch & Helpers
# ---------------------------------------------------------------------------

# METHOD_DISPATCH — Method Versioning Registry (see ADR-061)
#
# This registry enables head-to-head comparison of projection methods within
# the walk-forward validation framework.  Each entry represents a fully
# self-contained projection variant so that accuracy, bias, and other metrics
# can be evaluated side-by-side across identical origin years and counties.
#
# Structure
# ---------
# Each key is a short method identifier (e.g. "sdc_2024", "m2026", "m2026r1").
# Each value is a dict with keys:
#
#   {
#       "config":    MethodConfig dict with all tunable parameters,
#       "prepare":   callable(mig_raw, origin_year, config) -> rates,
#       "project":   callable(base_pop, survival, fertility, rates,
#                             county_fips, n_periods, origin_year, config)
#                    -> dict[int, float],
#       "is_annual": bool,
#   }
#
# Adding a New Method Variant
# ---------------------------
# Config-only variants can reuse existing prepare/project functions — just
# copy a base config and override specific values:
#
#     "m2026r2": {
#         "config": {**M2026R1_CONFIG, "college_fips": {"38017", "38035"}},
#         "prepare": lambda mig, origin, cfg: prepare_2026_convergence_rates_annual(mig, origin, cfg),
#         "project": lambda base, surv, fert, rates, fips, n, origin, cfg: (
#             project_2026_annual(base, surv, fert, rates, fips, n, origin, cfg)
#         ),
#         "is_annual": True,
#     },
#
METHOD_DISPATCH: dict[str, dict[str, object]] = {
    "sdc_2024": {
        "config": SDC_2024_CONFIG,
        "prepare": lambda mig, origin, cfg: prepare_sdc_rates(mig, origin, cfg),
        "project": lambda base, surv, fert, rates, fips, n, origin, cfg: project_sdc(
            base, surv, fert, rates, fips, n, origin
        ),
        "is_annual": False,
    },
    "m2026": {
        "config": M2026_CONFIG,
        "prepare": lambda mig, origin, cfg: prepare_2026_convergence_rates_annual(mig, origin, cfg),
        "project": lambda base, surv, fert, rates, fips, n, origin, cfg: project_2026_annual(
            base, surv, fert, rates, fips, n, origin, cfg
        ),
        "is_annual": True,
    },
    "m2026r1": {
        "config": M2026R1_CONFIG,
        "prepare": lambda mig, origin, cfg: prepare_2026_convergence_rates_annual(mig, origin, cfg),
        "project": lambda base, surv, fert, rates, fips, n, origin, cfg: project_2026_annual(
            base, surv, fert, rates, fips, n, origin, cfg
        ),
        "is_annual": True,
    },
}


def _interpolate_sdc(
    sdc_proj: dict[int, float],
    origin_year: int,
    target_year: int,
) -> float:
    """Linearly interpolate between SDC 5-year step endpoints for a target year."""
    steps_from_origin = (target_year - origin_year) / STEP
    lower_step = int(np.floor(steps_from_origin))
    upper_step = int(np.ceil(steps_from_origin))
    lower_year = origin_year + lower_step * STEP
    upper_year = origin_year + upper_step * STEP

    if lower_step == upper_step or lower_year == upper_year:
        return sdc_proj.get(lower_year, 0.0)
    else:
        frac = (target_year - lower_year) / (upper_year - lower_year)
        pop_lower = sdc_proj.get(lower_year, 0.0)
        pop_upper = sdc_proj.get(upper_year, 0.0)
        return pop_lower + frac * (pop_upper - pop_lower)


def _output_path(base_name: str, label: str | None) -> Path:
    """Build output file path, optionally prefixed with a run label."""
    if label:
        return OUTPUT_DIR / f"{label}_{base_name}"
    return OUTPUT_DIR / base_name


# ---------------------------------------------------------------------------
# Validation Framework
# ---------------------------------------------------------------------------


def run_walk_forward_validation(
    snapshots: dict[int, pd.DataFrame],
    mig_raw: pd.DataFrame,
    survival: dict[tuple[str, str], float],
    fertility: dict[str, float],
    methods: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run walk-forward validation for all origin years and selected methods.

    Args:
        snapshots: Population snapshots keyed by year.
        mig_raw: Raw migration rates DataFrame.
        survival: 5-year survival rates keyed by (age_group, sex).
        fertility: Annual ASFRs keyed by age_group label.
        methods: List of method names to run (keys of METHOD_DISPATCH).
            If None, defaults to ``["sdc_2024", "m2026"]`` for backward
            compatibility.

    Returns:
        county_detail: DataFrame with per-county projected vs actual
        state_results: DataFrame with state-level results
    """
    if methods is None:
        methods = ["sdc_2024", "m2026"]

    county_records: list[dict] = []
    state_records: list[dict] = []

    counties = sorted(snapshots[2000]["county_fips"].unique())

    for origin_year in ORIGIN_YEARS:
        print(f"\n  Origin {origin_year}:")
        targets = VALIDATION_TARGETS[origin_year]
        max_target_year = max(t[0] for t in targets)
        n_steps = (max_target_year - origin_year + STEP - 1) // STEP  # ceil division
        # Ensure we project at least to cover all targets
        n_steps = max(n_steps, MAX_PROJECTION_YEARS // STEP)

        base_pop = snapshots[origin_year]

        # Determine maximum number of years needed for annual engines
        n_years_annual = MAX_PROJECTION_YEARS

        # Prepare rates for each method (with per-method GQ recomputation)
        method_rates: dict[str, object] = {}
        for method_name in methods:
            dispatch = METHOD_DISPATCH[method_name]
            cfg = dispatch["config"]  # type: ignore[index]
            method_mig = maybe_recompute_mig_raw(mig_raw, snapshots, cfg)
            method_rates[method_name] = dispatch["prepare"](method_mig, origin_year, cfg)  # type: ignore[operator]

        # Project all counties with each method
        for fips in counties:
            for method_name in methods:
                dispatch = METHOD_DISPATCH[method_name]
                cfg = dispatch["config"]  # type: ignore[index]
                rates = method_rates[method_name]

                if dispatch["is_annual"]:
                    proj = dispatch["project"](  # type: ignore[operator]
                        base_pop, survival, fertility, rates, fips, n_years_annual, origin_year, cfg
                    )
                else:
                    proj = dispatch["project"](  # type: ignore[operator]
                        base_pop, survival, fertility, rates, fips, n_steps, origin_year, cfg
                    )

                for target_year, horizon in targets:
                    # Get actual population for this county at target year
                    actual_df = snapshots[target_year]
                    actual_county = actual_df[actual_df["county_fips"] == fips]
                    actual_pop = actual_county["population"].sum() if len(actual_county) > 0 else 0.0

                    if dispatch["is_annual"]:
                        projected_pop = proj.get(target_year, 0.0)
                    else:
                        projected_pop = _interpolate_sdc(proj, origin_year, target_year)

                    error = projected_pop - actual_pop
                    pct_error = (error / actual_pop * 100) if actual_pop > 0 else 0.0

                    county_records.append(
                        {
                            "origin_year": origin_year,
                            "method": method_name,
                            "validation_year": target_year,
                            "horizon": horizon,
                            "county_fips": fips,
                            "projected": round(projected_pop, 1),
                            "actual": round(actual_pop, 1),
                            "error": round(error, 1),
                            "pct_error": round(pct_error, 4),
                        }
                    )

        # Compute state-level results from county detail
        for target_year, horizon in targets:
            for method_name in methods:
                matching = [
                    r
                    for r in county_records
                    if r["origin_year"] == origin_year
                    and r["method"] == method_name
                    and r["validation_year"] == target_year
                ]
                proj_state = sum(r["projected"] for r in matching)
                actual_state = sum(r["actual"] for r in matching)
                error = proj_state - actual_state
                pct_error = (error / actual_state * 100) if actual_state > 0 else 0.0

                state_records.append(
                    {
                        "origin_year": origin_year,
                        "method": method_name,
                        "validation_year": target_year,
                        "horizon": horizon,
                        "projected_state": round(proj_state, 0),
                        "actual_state": round(actual_state, 0),
                        "error": round(error, 0),
                        "pct_error": round(pct_error, 4),
                        "abs_pct_error": round(abs(pct_error), 4),
                    }
                )

        n_targets = len(targets)
        target_str = ", ".join(f"{t[0]}(h{t[1]})" for t in targets)
        print(f"    Periods: {len(AVAILABLE_PERIODS[origin_year])}, Targets: {n_targets} [{target_str}]")

    county_detail = pd.DataFrame(county_records)
    state_results = pd.DataFrame(state_records)

    return county_detail, state_results


# ---------------------------------------------------------------------------
# Metrics Computation
# ---------------------------------------------------------------------------


def compute_county_metrics(county_detail: pd.DataFrame) -> pd.DataFrame:
    """Compute county-level error metrics (MAPE, RMSE, MPE) for each
    origin x method x validation_year combination.

    Returns DataFrame with aggregated metrics.
    """
    records: list[dict] = []
    groups = county_detail.groupby(["origin_year", "method", "validation_year", "horizon"])

    for (origin, method, val_yr, horizon), group in groups:
        actual = group["actual"].values
        projected = group["projected"].values
        errors = projected - actual

        # Avoid division by zero
        valid_mask = actual > 0
        if valid_mask.sum() == 0:
            continue

        pct_errors = errors[valid_mask] / actual[valid_mask] * 100
        abs_pct_errors = np.abs(pct_errors)

        mape = abs_pct_errors.mean()
        rmse = np.sqrt((errors**2).mean())
        mpe = pct_errors.mean()

        records.append(
            {
                "origin_year": origin,
                "method": method,
                "validation_year": val_yr,
                "horizon": horizon,
                "county_mape": round(mape, 4),
                "county_rmse": round(rmse, 1),
                "county_mpe": round(mpe, 4),
            }
        )

    return pd.DataFrame(records)


def compute_horizon_summary(
    state_results: pd.DataFrame,
    county_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate metrics by horizon across origins.

    Returns DataFrame with one row per horizon x method.
    """
    records: list[dict] = []

    # Get unique horizons for each method (data-driven)
    for method in state_results["method"].unique():
        state_m = state_results[state_results["method"] == method]
        county_m = county_metrics[county_metrics["method"] == method]

        # Group by horizon
        all_horizons = sorted(state_m["horizon"].unique())
        for h in all_horizons:
            state_h = state_m[state_m["horizon"] == h]
            county_h = county_m[county_m["horizon"] == h]
            n_origins = len(state_h)

            records.append(
                {
                    "horizon": h,
                    "method": method,
                    "n_origins": n_origins,
                    "mean_state_ape": round(state_h["abs_pct_error"].mean(), 4),
                    "mean_county_mape": round(county_h["county_mape"].mean(), 4),
                    "mean_county_rmse": round(county_h["county_rmse"].mean(), 1),
                    "mean_state_mpe": round(state_h["pct_error"].mean(), 4),
                    "mean_county_mpe": round(county_h["county_mpe"].mean(), 4),
                }
            )

    return pd.DataFrame(records)


def compute_method_comparison(
    state_results: pd.DataFrame,
    county_metrics: pd.DataFrame,
    methods: list[str] | None = None,
) -> pd.DataFrame:
    """Create long-format method comparison.

    Returns DataFrame with one row per origin x validation_year x method,
    with columns [origin_year, validation_year, horizon, method,
    state_pct_error, state_abs_pct_error, county_mape, county_mpe].
    """
    if methods is None:
        methods = list(state_results["method"].unique())

    records: list[dict] = []

    for origin in ORIGIN_YEARS:
        for target_year, horizon in VALIDATION_TARGETS[origin]:
            for method in methods:
                s_rows = state_results[
                    (state_results["origin_year"] == origin)
                    & (state_results["method"] == method)
                    & (state_results["validation_year"] == target_year)
                ]
                c_rows = county_metrics[
                    (county_metrics["origin_year"] == origin)
                    & (county_metrics["method"] == method)
                    & (county_metrics["validation_year"] == target_year)
                ]

                if len(s_rows) == 0:
                    continue

                pe = s_rows["pct_error"].iloc[0]
                mape = c_rows["county_mape"].iloc[0] if len(c_rows) > 0 else np.nan
                mpe = c_rows["county_mpe"].iloc[0] if len(c_rows) > 0 else np.nan

                records.append(
                    {
                        "origin_year": origin,
                        "validation_year": target_year,
                        "horizon": horizon,
                        "method": method,
                        "state_pct_error": round(pe, 4),
                        "state_abs_pct_error": round(abs(pe), 4),
                        "county_mape": round(mape, 4),
                        "county_mpe": round(mpe, 4),
                    }
                )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Console Report
# ---------------------------------------------------------------------------


def print_report(
    state_results: pd.DataFrame,
    county_metrics: pd.DataFrame,
    horizon_summary: pd.DataFrame,
    method_comparison: pd.DataFrame,
) -> None:
    """Print comprehensive validation report to console."""

    all_methods = list(state_results["method"].unique())

    print("\n" + "=" * 80)
    print("WALK-FORWARD VALIDATION RESULTS")
    print("=" * 80)

    print(f"\n  Methods: {', '.join(all_methods)}")
    print(f"  Origins: {ORIGIN_YEARS}")

    # -----------------------------------------------------------------------
    # Table 1: State-level results by origin and method
    # -----------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("STATE-LEVEL RESULTS BY ORIGIN")
    print("-" * 80)
    print(
        f"  {'Origin':>6} {'Target':>6} {'Horizon':>7} {'Method':>10} "
        f"{'Projected':>11} {'Actual':>11} {'Error':>10} {'%Error':>8}"
    )
    print("  " + "-" * 74)

    for origin in ORIGIN_YEARS:
        for target_year, horizon in VALIDATION_TARGETS[origin]:
            for method in all_methods:
                row = state_results[
                    (state_results["origin_year"] == origin)
                    & (state_results["method"] == method)
                    & (state_results["validation_year"] == target_year)
                ]
                if len(row) == 0:
                    continue
                r = row.iloc[0]
                print(
                    f"  {origin:>6} {target_year:>6} {horizon:>5}yr "
                    f"{method:>10} "
                    f"{r['projected_state']:>11,.0f} {r['actual_state']:>11,.0f} "
                    f"{r['error']:>+10,.0f} {r['pct_error']:>+7.2f}%"
                )
            print()

    # -----------------------------------------------------------------------
    # Table 2: Horizon summary
    # -----------------------------------------------------------------------
    print("-" * 80)
    print("HORIZON SUMMARY (averaged across origins)")
    print("-" * 80)
    print(
        f"  {'Horizon':>7} {'Method':>10} {'N':>3} {'StateAPE':>9} "
        f"{'CtyMAPE':>8} {'CtyRMSE':>8} {'StateMPE':>9} {'CtyMPE':>8}"
    )
    print("  " + "-" * 68)

    for _, row in horizon_summary.sort_values(["horizon", "method"]).iterrows():
        print(
            f"  {row['horizon']:>5}yr {row['method']:>10} {row['n_origins']:>3} "
            f"{row['mean_state_ape']:>8.2f}% {row['mean_county_mape']:>7.2f}% "
            f"{row['mean_county_rmse']:>8,.0f} {row['mean_state_mpe']:>+8.2f}% "
            f"{row['mean_county_mpe']:>+7.2f}%"
        )

    # -----------------------------------------------------------------------
    # Section 3: Direction analysis
    # -----------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("DIRECTION ANALYSIS (positive = over-projection)")
    print("-" * 80)

    for method in all_methods:
        m_data = state_results[state_results["method"] == method]
        over = (m_data["pct_error"] > 0).sum()
        under = (m_data["pct_error"] < 0).sum()
        avg_pe = m_data["pct_error"].mean()
        print(f"  {method:>10}: Over={over}, Under={under}, Mean PE={avg_pe:+.2f}%")

    # -----------------------------------------------------------------------
    # Section 4: Method comparison (long format)
    # -----------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("METHOD COMPARISON BY ORIGIN x TARGET")
    print("-" * 80)
    print(
        f"  {'Origin':>6} {'Target':>6} {'Horizon':>7} {'Method':>10} "
        f"{'State%Err':>10} {'CtyMAPE':>8}"
    )
    print("  " + "-" * 56)

    for _, row in method_comparison.iterrows():
        print(
            f"  {row['origin_year']:>6} {row['validation_year']:>6} "
            f"{row['horizon']:>5}yr {row['method']:>10} "
            f"{row['state_pct_error']:>+9.2f}% {row['county_mape']:>7.2f}%"
        )

    # -----------------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("SUMMARY STATISTICS")
    print("-" * 80)

    for method in all_methods:
        m_state = state_results[state_results["method"] == method]
        m_county = county_metrics[county_metrics["method"] == method]
        print(f"\n  {method}:")
        print(f"    State APE — Mean: {m_state['abs_pct_error'].mean():.2f}%, "
              f"Max: {m_state['abs_pct_error'].max():.2f}%, "
              f"Min: {m_state['abs_pct_error'].min():.2f}%")
        print(f"    State PE  — Mean: {m_state['pct_error'].mean():+.2f}% "
              f"({'over-projects' if m_state['pct_error'].mean() > 0 else 'under-projects'} "
              f"on average)")
        print(f"    County MAPE — Mean: {m_county['county_mape'].mean():.2f}%, "
              f"Max: {m_county['county_mape'].max():.2f}%")

    print("\n" + "=" * 80)


# ---------------------------------------------------------------------------
# Annual Interpolation and Validation
# ---------------------------------------------------------------------------


def interpolate_county_annual(
    step_results: dict[int, float],
    origin_year: int,
    max_year: int | None = None,
) -> dict[int, float]:
    """Linearly interpolate county total population between 5-year step endpoints.

    Args:
        step_results: dict mapping step-endpoint year -> total population.
            E.g., {2005: 10000, 2010: 10500, 2015: 11000, ...}
        origin_year: The origin year of the projection.
        max_year: Optional cap on the maximum year to interpolate to.
            Defaults to the maximum step year.

    Returns:
        dict mapping every integer year from origin to max_year -> interpolated
        county total population.
    """
    step_years = sorted(step_results.keys())
    if max_year is None:
        max_year = step_years[-1]

    annual: dict[int, float] = {}
    for yr in range(origin_year, max_year + 1):
        if yr in step_results:
            annual[yr] = step_results[yr]
        else:
            # Find bracketing step years
            lower_yr = max(s for s in step_years if s <= yr)
            upper_yr = min(s for s in step_years if s >= yr)
            if lower_yr == upper_yr:
                annual[yr] = step_results[lower_yr]
            else:
                frac = (yr - lower_yr) / (upper_yr - lower_yr)
                annual[yr] = (
                    step_results[lower_yr]
                    + frac * (step_results[upper_yr] - step_results[lower_yr])
                )
    return annual


def load_annual_validation_actuals(
    origin_years: list[int],
    max_validation_year: int = 2024,
) -> dict[int, pd.DataFrame]:
    """Load actual annual PEP/Census population for all needed validation years.

    For each origin year, we need actuals for years origin+1 through
    max_validation_year.  This function loads each unique year once and caches.

    Returns:
        dict mapping year -> DataFrame with columns
        [county_fips, age_group, sex, population]
    """
    # Collect all unique years we need
    needed_years: set[int] = set()
    for oy in origin_years:
        for yr in range(oy + 1, max_validation_year + 1):
            needed_years.add(yr)

    actuals: dict[int, pd.DataFrame] = {}
    for yr in sorted(needed_years):
        try:
            df = load_population_snapshot(yr)
            actuals[yr] = df
        except (ValueError, FileNotFoundError) as e:
            print(f"  WARNING: Could not load year {yr}: {e}")
    return actuals


# ---------------------------------------------------------------------------
# Per-origin worker function (picklable — no lambdas)
# ---------------------------------------------------------------------------

# Local dispatch maps that avoid the lambda-based METHOD_DISPATCH for
# cross-process pickling.  Each method name is mapped to its concrete
# prepare/project functions and ``is_annual`` flag.

_PREPARE_DISPATCH: dict[str, object] = {
    "sdc_2024": prepare_sdc_rates,
    "m2026": prepare_2026_convergence_rates_annual,
    "m2026r1": prepare_2026_convergence_rates_annual,
}

_PROJECT_DISPATCH: dict[str, object] = {
    "sdc_2024": project_sdc,
    "m2026": project_2026_annual,
    "m2026r1": project_2026_annual,
}

_IS_ANNUAL: dict[str, bool] = {
    "sdc_2024": False,
    "m2026": True,
    "m2026r1": True,
}


def _run_single_origin_annual(
    *,
    origin_year: int,
    base_pop: pd.DataFrame,
    mig_raw: pd.DataFrame,
    survival: dict[tuple[str, str], float],
    fertility: dict[str, float],
    counties: list[str],
    methods: list[str],
    method_configs: dict[str, MethodConfig],
    actual_county_totals: dict[int, dict[str, float]],
    max_validation_year: int,
    method_mig_raw: dict[str, pd.DataFrame] | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Run walk-forward validation for a single origin year.

    This function is designed to be picklable for use with
    ``ProcessPoolExecutor``.  It avoids lambdas by using the module-level
    ``_PREPARE_DISPATCH`` / ``_PROJECT_DISPATCH`` / ``_IS_ANNUAL`` maps
    instead of ``METHOD_DISPATCH``.

    Args:
        origin_year: The origin year (e.g. 2005, 2010, 2015, 2020).
        base_pop: Population snapshot DataFrame for the origin year.
        mig_raw: Raw migration rates DataFrame.
        survival: 5-year survival rates keyed by (age_group, sex).
        fertility: Annual ASFRs keyed by age_group label.
        counties: Sorted list of county FIPS codes.
        methods: List of method identifiers to run.
        method_configs: Dict mapping method_id to its MethodConfig.
        actual_county_totals: Precomputed {year: {fips: total_pop}}.
        max_validation_year: Maximum year to validate against.
        method_mig_raw: Optional per-method migration rate overrides.
            Maps method_id -> migration DataFrame.  If absent or a method
            is not in this dict, ``mig_raw`` is used.

    Returns:
        Tuple of (county_records, state_records, curve_records).
    """
    n_steps = MAX_PROJECTION_YEARS // STEP
    n_years = MAX_PROJECTION_YEARS

    # Prepare rates for each method
    method_rates: dict[str, object] = {}
    for method_name in methods:
        cfg = method_configs[method_name]
        effective_mig = mig_raw
        if method_mig_raw and method_name in method_mig_raw:
            effective_mig = method_mig_raw[method_name]
        prepare_fn = _PREPARE_DISPATCH[method_name]
        method_rates[method_name] = prepare_fn(effective_mig, origin_year, cfg)  # type: ignore[operator]

    # Project all counties with each method
    method_county_annual: dict[str, dict[str, dict[int, float]]] = {
        m: {} for m in methods
    }

    for fips in counties:
        for method_name in methods:
            cfg = method_configs[method_name]
            rates = method_rates[method_name]
            is_annual = _IS_ANNUAL[method_name]
            project_fn = _PROJECT_DISPATCH[method_name]

            if is_annual:
                county_annual = project_fn(  # type: ignore[operator]
                    base_pop, survival, fertility, rates, fips, n_years, origin_year, cfg
                )
            else:
                step_proj = project_fn(  # type: ignore[operator]
                    base_pop, survival, fertility, rates, fips, n_steps, origin_year, cfg
                )
                county_annual = interpolate_county_annual(step_proj, origin_year)

            method_county_annual[method_name][fips] = county_annual

    # Build curve records
    curve_records: list[dict] = []
    for method_name in methods:
        county_annuals = method_county_annual[method_name]
        all_years = sorted(
            set().union(*(ca.keys() for ca in county_annuals.values()))
        )
        for yr in all_years:
            state_total = sum(
                ca.get(yr, 0.0) for ca in county_annuals.values()
            )
            curve_records.append(
                {
                    "origin_year": origin_year,
                    "method": method_name,
                    "year": yr,
                    "projected_state": round(state_total, 0),
                }
            )

    # Compute annual error metrics
    validation_years = [
        yr for yr in range(origin_year + 1, max_validation_year + 1)
        if yr in actual_county_totals
    ]

    county_records: list[dict] = []
    state_records: list[dict] = []

    for val_yr in validation_years:
        horizon = val_yr - origin_year

        for method_name in methods:
            county_annuals = method_county_annual[method_name]

            method_proj_state = 0.0
            method_actual_state = 0.0

            for fips in counties:
                projected = county_annuals[fips].get(val_yr, 0.0)
                actual = actual_county_totals[val_yr].get(fips, 0.0)
                error = projected - actual
                pct_error = (error / actual * 100) if actual > 0 else 0.0

                county_name = FIPS_TO_COUNTY_NAME.get(fips, fips)

                county_records.append(
                    {
                        "origin_year": origin_year,
                        "method": method_name,
                        "validation_year": val_yr,
                        "horizon": horizon,
                        "county_fips": fips,
                        "county_name": county_name,
                        "projected": round(projected, 1),
                        "actual": round(actual, 1),
                        "error": round(error, 1),
                        "pct_error": round(pct_error, 4),
                    }
                )

                method_proj_state += projected
                method_actual_state += actual

            state_error = method_proj_state - method_actual_state
            state_pct_error = (
                (state_error / method_actual_state * 100)
                if method_actual_state > 0
                else 0.0
            )
            state_records.append(
                {
                    "origin_year": origin_year,
                    "method": method_name,
                    "validation_year": val_yr,
                    "horizon": horizon,
                    "projected_state": round(method_proj_state, 0),
                    "actual_state": round(method_actual_state, 0),
                    "error": round(state_error, 0),
                    "pct_error": round(state_pct_error, 4),
                    "abs_pct_error": round(abs(state_pct_error), 4),
                }
            )

    return county_records, state_records, curve_records


def run_annual_validation(
    snapshots: dict[int, pd.DataFrame],
    mig_raw: pd.DataFrame,
    survival: dict[tuple[str, str], float],
    fertility: dict[str, float],
    max_validation_year: int = 2024,
    methods: list[str] | None = None,
    workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run annual-granularity walk-forward validation for all origins and methods.

    For each origin year and method:
    1. Non-annual methods: project in 5-year steps, linearly interpolate to annual.
    2. Annual methods: project with true annual stepping (no interpolation needed).
    3. Compare to actual annual PEP/Census data for origin+1 through
       max_validation_year.
    4. Also produce full 50-year projection curves.

    Args:
        snapshots: Population snapshots keyed by year.
        mig_raw: Raw migration rates DataFrame.
        survival: 5-year survival rates keyed by (age_group, sex).
        fertility: Annual ASFRs keyed by age_group label.
        max_validation_year: Maximum year to validate against.
        methods: List of method names to run (keys of METHOD_DISPATCH).
            If None, defaults to ``["sdc_2024", "m2026"]``.
        workers: Number of parallel workers.  ``1`` (default) runs
            sequentially on the main process.  ``0`` auto-detects
            ``min(len(ORIGIN_YEARS), cpu_count)``.  Values > 1 use a
            ``ProcessPoolExecutor`` with origin years as work units.

    Returns:
        annual_state_results: DataFrame with state-level annual error metrics
        annual_county_detail: DataFrame with county-level annual error metrics
        projection_curves: DataFrame with full 50-year projection curves
    """
    if methods is None:
        methods = ["sdc_2024", "m2026"]

    # Load all annual actuals
    print("\n  Loading annual validation actuals...")
    annual_actuals = load_annual_validation_actuals(ORIGIN_YEARS, max_validation_year)
    print(f"  Loaded actuals for {len(annual_actuals)} years "
          f"({min(annual_actuals.keys())}-{max(annual_actuals.keys())})")

    # Precompute county totals from annual actuals: {year: {fips: total_pop}}
    actual_county_totals: dict[int, dict[str, float]] = {}
    for yr, df in annual_actuals.items():
        county_tots = df.groupby("county_fips")["population"].sum().to_dict()
        actual_county_totals[yr] = county_tots

    counties = sorted(snapshots[2000]["county_fips"].unique())

    # Build method_configs from METHOD_DISPATCH
    method_configs: dict[str, MethodConfig] = {}
    for m in methods:
        method_configs[m] = METHOD_DISPATCH[m]["config"]  # type: ignore[index]

    # Pre-compute per-method migration rate overrides (GQ recomputation)
    method_mig_raw: dict[str, pd.DataFrame] = {}
    for method_name in methods:
        cfg = method_configs[method_name]
        recomputed = maybe_recompute_mig_raw(mig_raw, snapshots, cfg)
        if recomputed is not mig_raw:
            method_mig_raw[method_name] = recomputed

    # Resolve effective worker count
    n_origins = len(ORIGIN_YEARS)
    if workers == 0:
        effective_workers = min(n_origins, os.cpu_count() or 1)
    else:
        effective_workers = max(1, min(workers, n_origins))

    # Build shared keyword arguments for _run_single_origin_annual
    shared_kwargs: dict[str, object] = {
        "mig_raw": mig_raw,
        "survival": survival,
        "fertility": fertility,
        "counties": counties,
        "methods": methods,
        "method_configs": method_configs,
        "actual_county_totals": actual_county_totals,
        "max_validation_year": max_validation_year,
        "method_mig_raw": method_mig_raw if method_mig_raw else None,
    }

    # Collect results in deterministic ORIGIN_YEARS order
    annual_county_records: list[dict] = []
    annual_state_records: list[dict] = []
    curve_records: list[dict] = []

    if effective_workers <= 1:
        # Sequential execution — same code path as before
        for origin_year in ORIGIN_YEARS:
            print(f"\n  Origin {origin_year} (annual):")
            county_recs, state_recs, crv_recs = _run_single_origin_annual(
                origin_year=origin_year,
                base_pop=snapshots[origin_year],
                **shared_kwargs,  # type: ignore[arg-type]
            )
            annual_county_records.extend(county_recs)
            annual_state_records.extend(state_recs)
            curve_records.extend(crv_recs)
    else:
        # Parallel execution using ProcessPoolExecutor
        print(f"\n  Running {n_origins} origin years with {effective_workers} workers...")
        futures: dict[int, object] = {}  # origin_year -> Future

        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            for origin_year in ORIGIN_YEARS:
                fut = executor.submit(
                    _run_single_origin_annual,
                    origin_year=origin_year,
                    base_pop=snapshots[origin_year],
                    **shared_kwargs,  # type: ignore[arg-type]
                )
                futures[origin_year] = fut

        # Merge results in deterministic ORIGIN_YEARS order
        for origin_year in ORIGIN_YEARS:
            fut = futures[origin_year]
            try:
                county_recs, state_recs, crv_recs = fut.result()  # type: ignore[union-attr]
            except Exception as exc:
                # Graceful fallback: re-run this origin year sequentially
                print(f"  WARNING: Worker for origin {origin_year} failed "
                      f"({exc!r}), retrying sequentially...")
                county_recs, state_recs, crv_recs = _run_single_origin_annual(
                    origin_year=origin_year,
                    base_pop=snapshots[origin_year],
                    **shared_kwargs,  # type: ignore[arg-type]
                )
            annual_county_records.extend(county_recs)
            annual_state_records.extend(state_recs)
            curve_records.extend(crv_recs)

    annual_state_df = pd.DataFrame(annual_state_records)
    annual_county_df = pd.DataFrame(annual_county_records)
    projection_curves_df = pd.DataFrame(curve_records)

    return annual_state_df, annual_county_df, projection_curves_df


def compute_annual_horizon_summary(
    annual_state: pd.DataFrame,
    annual_county: pd.DataFrame,
) -> pd.DataFrame:
    """Compute horizon-level summary metrics from annual validation results.

    For each horizon × method, aggregate across all origins:
    - mean_state_ape: Mean absolute percentage error at state level
    - mean_county_mape: Mean county-level MAPE
    - mean_state_mpe: Mean signed percentage error at state level
    - mean_county_mpe: Mean signed county-level MPE

    Returns DataFrame with columns:
    [horizon, method, n_origins, mean_state_ape, mean_county_mape,
     mean_state_mpe, mean_county_mpe]
    """
    records: list[dict] = []

    for method in annual_state["method"].unique():
        state_m = annual_state[annual_state["method"] == method]
        county_m = annual_county[annual_county["method"] == method]

        all_horizons = sorted(state_m["horizon"].unique())
        for h in all_horizons:
            state_h = state_m[state_m["horizon"] == h]
            n_origins = len(state_h)

            # County MAPE for each origin at this horizon
            county_h = county_m[county_m["horizon"] == h]
            county_mapes = []
            county_mpes = []
            for _, grp in county_h.groupby("origin_year"):
                valid = grp[grp["actual"] > 0]
                if len(valid) > 0:
                    abs_pct = valid["pct_error"].abs().mean()
                    signed_pct = valid["pct_error"].mean()
                    county_mapes.append(abs_pct)
                    county_mpes.append(signed_pct)

            records.append(
                {
                    "horizon": h,
                    "method": method,
                    "n_origins": n_origins,
                    "mean_state_ape": round(state_h["abs_pct_error"].mean(), 4),
                    "mean_county_mape": round(
                        np.mean(county_mapes) if county_mapes else 0.0, 4
                    ),
                    "mean_state_mpe": round(state_h["pct_error"].mean(), 4),
                    "mean_county_mpe": round(
                        np.mean(county_mpes) if county_mpes else 0.0, 4
                    ),
                }
            )

    return pd.DataFrame(records)


def compute_annual_method_comparison(
    annual_state: pd.DataFrame,
    annual_county: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-horizon per-method summary from annual validation results.

    Long-format output: one row per horizon x method with aggregated metrics.

    Returns DataFrame with columns:
    [horizon, n_origins, method, mean_state_ape, mean_county_mape]
    """
    records: list[dict] = []

    all_horizons = sorted(annual_state["horizon"].unique())

    def _avg_county_mape(cdf: pd.DataFrame) -> float:
        mapes = []
        for _, grp in cdf.groupby("origin_year"):
            valid = grp[grp["actual"] > 0]
            if len(valid) > 0:
                mapes.append(valid["pct_error"].abs().mean())
        return float(np.mean(mapes)) if mapes else 0.0

    for h in all_horizons:
        for method in annual_state["method"].unique():
            state_h = annual_state[
                (annual_state["method"] == method) & (annual_state["horizon"] == h)
            ]
            county_h = annual_county[
                (annual_county["method"] == method) & (annual_county["horizon"] == h)
            ]

            if len(state_h) == 0:
                continue

            n_origins = len(state_h)
            ape = state_h["abs_pct_error"].mean()
            cmape = _avg_county_mape(county_h)

            records.append(
                {
                    "horizon": h,
                    "n_origins": n_origins,
                    "method": method,
                    "mean_state_ape": round(ape, 4),
                    "mean_county_mape": round(cmape, 4),
                }
            )

    return pd.DataFrame(records)


def print_annual_report(
    annual_state: pd.DataFrame,
    annual_horizon: pd.DataFrame,
    annual_comparison: pd.DataFrame,
) -> None:
    """Print summary of annual validation results."""
    all_methods = list(annual_state["method"].unique())
    n_methods = len(all_methods)

    print("\n" + "=" * 80)
    print("ANNUAL-GRANULARITY WALK-FORWARD VALIDATION")
    print("=" * 80)

    n_state = len(annual_state)
    per_method = n_state // n_methods if n_methods > 0 else 0
    print(f"\n  Total validation points: {per_method} per method "
          f"({n_methods} methods x {per_method} = {n_state} rows)")
    print(f"  Methods: {', '.join(all_methods)}")
    print(f"  Horizon range: 1 to {annual_state['horizon'].max()} years")

    # Error growth by horizon (long format)
    print("\n" + "-" * 80)
    print("ERROR GROWTH BY HORIZON (averaged across origins)")
    print("-" * 80)
    print(
        f"  {'Horizon':>7} {'N':>3} {'Method':>10} {'StateAPE':>9} {'CtyMAPE':>10}"
    )
    print("  " + "-" * 45)

    for _, row in annual_comparison.sort_values(["horizon", "method"]).iterrows():
        print(
            f"  {row['horizon']:>5}yr {row['n_origins']:>3} "
            f"{row['method']:>10} "
            f"{row['mean_state_ape']:>8.2f}% {row['mean_county_mape']:>9.2f}%"
        )

    # When do errors exceed 5%?
    print("\n" + "-" * 80)
    print("ERROR THRESHOLD ANALYSIS")
    print("-" * 80)

    for method in all_methods:
        method_comp = annual_comparison[annual_comparison["method"] == method]

        exceed_5_state = method_comp[method_comp["mean_state_ape"] > 5.0]
        exceed_5_county = method_comp[method_comp["mean_county_mape"] > 5.0]

        if len(exceed_5_state) > 0:
            first_h = exceed_5_state["horizon"].min()
            print(f"  {method} state APE exceeds 5%: first at horizon {first_h} years")
        else:
            print(f"  {method} state APE: never exceeds 5% within validation range")

        if len(exceed_5_county) > 0:
            first_h = exceed_5_county["horizon"].min()
            print(f"  {method} county MAPE exceeds 5%: first at horizon {first_h} years")
        else:
            print(f"  {method} county MAPE: never exceeds 5% within validation range")

    # Short vs long horizon comparison
    print("\n" + "-" * 80)
    print("SHORT vs LONG HORIZON PERFORMANCE")
    print("-" * 80)

    short = annual_comparison[annual_comparison["horizon"] <= 5]
    long_h = annual_comparison[annual_comparison["horizon"] > 10]

    for label, subset in [("Short horizon (1-5 yr)", short), ("Long horizon (>10 yr)", long_h)]:
        if len(subset) == 0:
            continue
        print(f"  {label}:")
        for method in all_methods:
            m_sub = subset[subset["method"] == method]
            if len(m_sub) > 0:
                avg_ape = m_sub["mean_state_ape"].mean()
                print(f"    {method}: {avg_ape:.2f}%")

    print("\n" + "=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run walk-forward validation and produce outputs."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Walk-forward validation of population projection methods"
    )
    parser.add_argument(
        "--run-label",
        type=str,
        default=None,
        help="Prefix for output CSV filenames (e.g., 'r1' -> 'r1_county_detail.csv')",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(METHOD_DISPATCH.keys()),
        help=f"Methods to run (default: all registered: {list(METHOD_DISPATCH.keys())})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of parallel workers for annual validation. "
            "0 = auto-detect min(len(ORIGIN_YEARS), cpu_count). "
            "1 = sequential (default)."
        ),
    )
    args = parser.parse_args()

    methods: list[str] = args.methods
    label: str | None = args.run_label
    n_workers: int = args.workers

    # Validate method names
    for m in methods:
        if m not in METHOD_DISPATCH:
            parser.error(f"Unknown method '{m}'. Available: {list(METHOD_DISPATCH.keys())}")

    print("=" * 80)
    print("Walk-Forward Validation")
    print("=" * 80)
    print(f"  Methods: {methods}")
    if label:
        print(f"  Run label: {label}")

    # 1. Load all population snapshots
    print("\nLoading population snapshots...")
    snapshots = load_all_snapshots()

    # 2. Load migration rates
    print("\nLoading migration rates...")
    mig_raw = load_migration_rates_raw()
    periods = sorted(
        mig_raw[["period_start", "period_end"]].drop_duplicates().values.tolist()
    )
    print(f"  Periods: {len(periods)} — {periods}")

    # 3. Load survival and fertility rates
    print("\nLoading survival and fertility rates...")
    survival = load_survival_rates()
    fertility = load_fertility_rates()
    print(f"  Survival rates: {len(survival)} (age_group, sex) pairs")
    print(f"  Fertility rates: {len(fertility)} age groups")

    # 4. Run validation
    print("\nRunning walk-forward validation...")
    county_detail, state_results = run_walk_forward_validation(
        snapshots, mig_raw, survival, fertility, methods=methods
    )

    # 5. Compute metrics
    print("\nComputing metrics...")
    county_metrics = compute_county_metrics(county_detail)
    horizon_summary = compute_horizon_summary(state_results, county_metrics)
    method_comparison = compute_method_comparison(state_results, county_metrics, methods=methods)

    # 6. Write output files
    print("\nWriting output files...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    county_detail_path = _output_path("county_detail.csv", label)
    # Ensure county_fips is zero-padded 5-char string in output
    county_detail_out = county_detail.copy()
    county_detail_out["county_fips"] = county_detail_out["county_fips"].astype(str).str.zfill(5)
    county_detail_out.to_csv(county_detail_path, index=False)
    print(f"  {county_detail_path.relative_to(PROJECT_ROOT)} ({len(county_detail_out)} rows)")

    state_results_path = _output_path("state_results.csv", label)
    state_results.to_csv(state_results_path, index=False)
    print(f"  {state_results_path.relative_to(PROJECT_ROOT)} ({len(state_results)} rows)")

    horizon_path = _output_path("horizon_summary.csv", label)
    horizon_summary.to_csv(horizon_path, index=False)
    print(f"  {horizon_path.relative_to(PROJECT_ROOT)} ({len(horizon_summary)} rows)")

    comparison_path = _output_path("method_comparison.csv", label)
    method_comparison.to_csv(comparison_path, index=False)
    print(f"  {comparison_path.relative_to(PROJECT_ROOT)} ({len(method_comparison)} rows)")

    # 7. Print console report
    print_report(state_results, county_metrics, horizon_summary, method_comparison)

    # -----------------------------------------------------------------------
    # 8. Annual-granularity validation
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Running annual-granularity validation...")
    print("=" * 80)

    annual_state, annual_county, projection_curves = run_annual_validation(
        snapshots, mig_raw, survival, fertility, methods=methods,
        workers=n_workers,
    )

    # 9. Compute annual summary metrics
    print("\nComputing annual summary metrics...")
    annual_horizon = compute_annual_horizon_summary(annual_state, annual_county)
    annual_comparison = compute_annual_method_comparison(annual_state, annual_county)

    # 10. Write annual output files (county_fips as zero-padded strings)
    print("\nWriting annual output files...")

    annual_state_path = _output_path("annual_state_results.csv", label)
    annual_state.to_csv(annual_state_path, index=False)
    print(f"  {annual_state_path.relative_to(PROJECT_ROOT)} ({len(annual_state)} rows)")

    annual_county_path = _output_path("annual_county_detail.csv", label)
    # Ensure county_fips is zero-padded 5-char string in output
    annual_county_out = annual_county.copy()
    annual_county_out["county_fips"] = annual_county_out["county_fips"].astype(str).str.zfill(5)
    annual_county_out.to_csv(annual_county_path, index=False)
    print(f"  {annual_county_path.relative_to(PROJECT_ROOT)} ({len(annual_county_out)} rows)")

    annual_horizon_path = _output_path("annual_horizon_summary.csv", label)
    annual_horizon.to_csv(annual_horizon_path, index=False)
    print(f"  {annual_horizon_path.relative_to(PROJECT_ROOT)} ({len(annual_horizon)} rows)")

    annual_comp_path = _output_path("annual_method_comparison.csv", label)
    annual_comparison.to_csv(annual_comp_path, index=False)
    print(f"  {annual_comp_path.relative_to(PROJECT_ROOT)} ({len(annual_comparison)} rows)")

    curves_path = _output_path("projection_curves.csv", label)
    projection_curves.to_csv(curves_path, index=False)
    print(f"  {curves_path.relative_to(PROJECT_ROOT)} ({len(projection_curves)} rows)")

    # 11. Print annual report
    print_annual_report(annual_state, annual_horizon, annual_comparison)

    print("\nDone.")


if __name__ == "__main__":
    main()
