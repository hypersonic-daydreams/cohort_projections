"""
Age-specific convergence interpolation for migration rates.

Implements Phase 2 of the Census Bureau method upgrade: computes time-varying
migration rates for each projection year (2025-2045) using a 5-10-5
convergence interpolation schedule.

Each county x age_group x sex cell converges independently from its recent
historical rate toward its long-term mean.

Convergence schedule (default 5-10-5):
    - Years 1-5:   linear interpolation from RECENT rate to MEDIUM rate
    - Years 6-15:  hold at MEDIUM rate
    - Years 16-20: linear interpolation from MEDIUM rate to LONG-TERM rate

Input: Phase 1 residual migration rates (per-period rates for 5 historical
periods).  Output: year-varying rates for the full 20-year projection horizon.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from cohort_projections.utils import get_logger_from_config, load_projection_config

logger = get_logger_from_config(__name__)


# ---------------------------------------------------------------------------
# Period-to-window mapping
# ---------------------------------------------------------------------------


def _map_config_window_to_periods(
    window_range: list[int],
    available_periods: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Map a config year range to the Phase 1 periods it covers.

    A period (period_start, period_end) is included if the window range
    overlaps with it, i.e. the period does not end before the window starts
    and does not start after the window ends.

    Args:
        window_range: Two-element list [start_year, end_year] from config.
        available_periods: List of (period_start, period_end) tuples from
            Phase 1 output.

    Returns:
        List of (period_start, period_end) tuples that fall within the
        window range.
    """
    win_start, win_end = window_range
    matched: list[tuple[int, int]] = []
    for ps, pe in available_periods:
        # Include the period if it overlaps with the window
        if pe >= win_start and ps <= win_end:
            matched.append((ps, pe))
    return sorted(matched)


# ---------------------------------------------------------------------------
# Window averaging
# ---------------------------------------------------------------------------


def compute_period_window_averages(
    all_period_rates: pd.DataFrame,
    recent_periods: list[tuple[int, int]],
    medium_periods: list[tuple[int, int]],
    longterm_periods: list[tuple[int, int]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute recent, medium, and long-term averaged migration rates.

    For each window, filters the Phase 1 period rates to the specified
    periods, then takes the simple arithmetic mean of ``migration_rate``
    per county x age_group x sex cell.

    Args:
        all_period_rates: Full period rates from Phase 1 (all 5 periods).
            Expected columns: [county_fips, age_group, sex, period_start,
            period_end, migration_rate].
        recent_periods: Which periods to include in "recent"
            (e.g., ``[(2020, 2024)]``).
        medium_periods: Which periods for "medium"
            (e.g., ``[(2010,2015),(2015,2020),(2020,2024)]``).
        longterm_periods: Which periods for "long-term" (e.g., all 5).

    Returns:
        Tuple of (recent_rates, medium_rates, longterm_rates) DataFrames.
        Each has columns [county_fips, age_group, sex, migration_rate].
    """
    group_cols = ["county_fips", "age_group", "sex"]

    def _avg_for_periods(df: pd.DataFrame, periods: list[tuple[int, int]]) -> pd.DataFrame:
        mask = pd.Series(False, index=df.index)
        for ps, pe in periods:
            mask = mask | ((df["period_start"] == ps) & (df["period_end"] == pe))
        filtered = df.loc[mask]
        if filtered.empty:
            logger.warning(f"No data found for periods {periods}")
            return pd.DataFrame(columns=group_cols + ["migration_rate"])
        averaged = filtered.groupby(group_cols, as_index=False).agg(
            migration_rate=("migration_rate", "mean")
        )
        return averaged

    recent_rates = _avg_for_periods(all_period_rates, recent_periods)
    medium_rates = _avg_for_periods(all_period_rates, medium_periods)
    longterm_rates = _avg_for_periods(all_period_rates, longterm_periods)

    logger.info(
        f"Window averages computed: recent={len(recent_rates)} cells "
        f"({len(recent_periods)} periods), medium={len(medium_rates)} cells "
        f"({len(medium_periods)} periods), longterm={len(longterm_rates)} cells "
        f"({len(longterm_periods)} periods)"
    )

    return recent_rates, medium_rates, longterm_rates


# ---------------------------------------------------------------------------
# Convergence interpolation
# ---------------------------------------------------------------------------


def _apply_rate_cap(
    rate: pd.Series,
    age_groups: pd.Series,
    rate_cap_config: dict[str, Any],
) -> tuple[pd.Series, int]:
    """Apply age-aware asymmetric migration rate cap.

    College-age cells (default 15-19, 20-24) are capped at a wider threshold
    to preserve legitimate university enrollment dynamics. All other age cells
    are capped at a tighter threshold to clip small-county statistical noise.

    Args:
        rate: Series of interpolated migration rates for a single year.
        age_groups: Series of age group labels aligned with *rate*.
        rate_cap_config: Dict with keys:
            - college_ages: list of age group labels for the wider cap
            - college_cap: float, symmetric cap for college ages (e.g. 0.15)
            - general_cap: float, symmetric cap for other ages (e.g. 0.08)

    Returns:
        Tuple of (capped_rate, n_clipped) where *n_clipped* is the number
        of cells that were modified by the cap.
    """
    college_ages = rate_cap_config.get("college_ages", ["15-19", "20-24"])
    college_cap = rate_cap_config.get("college_cap", 0.15)
    general_cap = rate_cap_config.get("general_cap", 0.08)

    college_mask = age_groups.isin(college_ages)

    # Apply general cap to all cells first, then overwrite college-age cells
    capped = rate.copy()
    capped = capped.clip(lower=-general_cap, upper=general_cap)
    capped[college_mask] = rate[college_mask].clip(lower=-college_cap, upper=college_cap)

    n_clipped = int((capped != rate).sum())
    return capped, n_clipped


def calculate_age_specific_convergence(
    recent_rates: pd.DataFrame,
    medium_rates: pd.DataFrame,
    longterm_rates: pd.DataFrame,
    projection_years: int = 20,
    convergence_schedule: dict[str, int] | None = None,
    rate_cap_config: dict[str, Any] | None = None,
) -> dict[int, pd.DataFrame]:
    """Calculate year-varying migration rates with age-specific convergence.

    Each county x age_group x sex cell converges independently through
    the 5-10-5 schedule:

    - Years 1 through *phase1*: linear from recent to medium
    - Years *phase1+1* through *phase1+phase2*: hold at medium
    - Years *phase1+phase2+1* through *total*: linear from medium to
      long-term

    After interpolation, an optional age-aware rate cap clips extreme
    values to guard against small-county statistical noise while
    preserving legitimate university enrollment dynamics.

    Args:
        recent_rates: DataFrame with [county_fips, age_group, sex,
            migration_rate] for the recent window.
        medium_rates: DataFrame with same schema for the medium window.
        longterm_rates: DataFrame with same schema for the long-term window.
        projection_years: Total projection horizon (default 20).
        convergence_schedule: Dict with keys:
            - recent_to_medium_years (default 5)
            - medium_hold_years (default 10)
            - medium_to_longterm_years (default 5)
        rate_cap_config: Optional dict with keys:
            - enabled: bool (default False)
            - college_ages: list of age group labels (default ["15-19", "20-24"])
            - college_cap: float (default 0.15)
            - general_cap: float (default 0.08)

    Returns:
        Dict mapping year_offset (1-based, 1 through *projection_years*)
        to DataFrame with columns [county_fips, age_group, sex,
        migration_rate].
    """
    if convergence_schedule is None:
        convergence_schedule = {
            "recent_to_medium_years": 5,
            "medium_hold_years": 10,
            "medium_to_longterm_years": 5,
        }

    phase1 = convergence_schedule["recent_to_medium_years"]
    phase2 = convergence_schedule["medium_hold_years"]
    phase3 = convergence_schedule["medium_to_longterm_years"]

    # Determine whether rate capping is enabled
    cap_enabled = (
        rate_cap_config is not None and rate_cap_config.get("enabled", False)
    )

    group_cols = ["county_fips", "age_group", "sex"]

    # Merge the three windows into one wide DataFrame
    merged = recent_rates[group_cols + ["migration_rate"]].rename(
        columns={"migration_rate": "recent"}
    )
    merged = merged.merge(
        medium_rates[group_cols + ["migration_rate"]].rename(columns={"migration_rate": "medium"}),
        on=group_cols,
        how="outer",
    )
    merged = merged.merge(
        longterm_rates[group_cols + ["migration_rate"]].rename(
            columns={"migration_rate": "longterm"}
        ),
        on=group_cols,
        how="outer",
    )
    merged = merged.fillna(0.0)

    results: dict[int, pd.DataFrame] = {}
    total_clipped = 0

    for year in range(1, projection_years + 1):
        if year <= phase1:
            # Linear interpolation: recent -> medium
            t = year / phase1
            rate = merged["recent"] * (1 - t) + merged["medium"] * t
        elif year <= phase1 + phase2:
            # Hold at medium
            rate = merged["medium"]
        else:
            # Linear interpolation: medium -> long-term
            years_into_phase3 = year - phase1 - phase2
            t = years_into_phase3 / phase3
            t = min(t, 1.0)
            rate = merged["medium"] * (1 - t) + merged["longterm"] * t

        # Apply age-aware rate cap (after interpolation, before storing)
        if cap_enabled:
            assert rate_cap_config is not None  # for type checker
            rate, n_clipped = _apply_rate_cap(
                rate, merged["age_group"], rate_cap_config
            )
            total_clipped += n_clipped

        year_df = merged[group_cols].copy()
        year_df["migration_rate"] = rate.values
        results[year] = year_df

    logger.info(
        f"Age-specific convergence computed for {len(merged)} cells "
        f"over {projection_years} years (phases: {phase1}+{phase2}+{phase3})"
    )
    if cap_enabled:
        logger.info(
            f"Rate cap applied: {total_clipped} cells clipped across "
            f"{projection_years} year offsets "
            f"({total_clipped / (len(merged) * projection_years) * 100:.1f}% of all cells)"
        )

    return results


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


def _compute_high_scenario_rate_increment(
    baseline_window: pd.DataFrame,
    project_root: Path,
) -> pd.DataFrame:
    """Compute per-county rate increment for the high scenario.

    Loads the BEBR baseline and high migration rate files, computes the
    per-county absolute difference (high - baseline), then converts that
    to a per-cell rate increment distributed uniformly across the 36
    age-group x sex cells in each county.

    The increment is expressed as a migration rate (proportion of
    cell-level population) and is added to the baseline window averages
    to produce high-scenario window averages.

    Args:
        baseline_window: Any window-average DataFrame used to identify
            the county x age_group x sex cell structure.
        project_root: Path to project root for locating data files.

    Returns:
        DataFrame with columns [county_fips, age_group, sex,
        rate_increment] containing the per-cell additive rate increment.
    """
    migration_dir = project_root / "data" / "processed" / "migration"
    baseline_pep = pd.read_parquet(migration_dir / "migration_rates_pep_baseline.parquet")
    high_pep = pd.read_parquet(migration_dir / "migration_rates_pep_high.parquet")

    # County-level absolute net migration totals
    bl_county = (
        baseline_pep.groupby("county_fips")["net_migration"]
        .sum()
        .reset_index()
        .rename(columns={"net_migration": "baseline_net"})
    )
    hi_county = (
        high_pep.groupby("county_fips")["net_migration"]
        .sum()
        .reset_index()
        .rename(columns={"net_migration": "high_net"})
    )
    diff_county = bl_county.merge(hi_county, on="county_fips")
    diff_county["abs_diff"] = diff_county["high_net"] - diff_county["baseline_net"]

    # Load population reference from residual rates (most recent period)
    residual_path = migration_dir / "residual_migration_rates.parquet"
    residual_df = pd.read_parquet(residual_path)
    max_period = residual_df["period_start"].max()
    recent_period = residual_df[residual_df["period_start"] == max_period]
    county_pop = (
        recent_period.groupby("county_fips")["pop_start"]
        .sum()
        .reset_index()
        .rename(columns={"pop_start": "county_population"})
    )

    # Merge diff with population to convert absolute diff to a rate
    diff_county = diff_county.merge(county_pop, on="county_fips", how="left")
    # Cells per county = 18 age groups x 2 sexes = 36
    cells_per_county = baseline_window.groupby("county_fips").size().iloc[0]
    diff_county["rate_increment_per_cell"] = diff_county["abs_diff"] / (
        diff_county["county_population"].clip(lower=1.0)
    ) / cells_per_county

    # Distribute the per-cell increment to all cells in the window
    group_cols = ["county_fips", "age_group", "sex"]
    cell_structure = baseline_window[group_cols].copy()
    increment_df = cell_structure.merge(
        diff_county[["county_fips", "rate_increment_per_cell"]],
        on="county_fips",
        how="left",
    )
    increment_df["rate_increment"] = increment_df["rate_increment_per_cell"].fillna(0.0)

    total_increment = diff_county["abs_diff"].sum()
    logger.info(
        f"High scenario rate increment computed: "
        f"+{total_increment:,.0f} additional migrants/year "
        f"across {len(diff_county)} counties, "
        f"distributed to {len(increment_df)} cells"
    )

    return increment_df[group_cols + ["rate_increment"]]


def _lift_window_averages(
    rates: pd.DataFrame,
    increment: pd.DataFrame,
) -> pd.DataFrame:
    """Add per-cell rate increment to a window-average DataFrame.

    Args:
        rates: Window-average DataFrame with columns
            [county_fips, age_group, sex, migration_rate].
        increment: DataFrame with columns
            [county_fips, age_group, sex, rate_increment].

    Returns:
        New DataFrame with lifted migration_rate values.
    """
    group_cols = ["county_fips", "age_group", "sex"]
    merged = rates.merge(increment, on=group_cols, how="left")
    merged["migration_rate"] = merged["migration_rate"] + merged["rate_increment"].fillna(0.0)
    return merged[group_cols + ["migration_rate"]]


def run_convergence_pipeline(
    config: dict[str, Any] | None = None,
    variant: str | None = None,
) -> dict[str, Any]:
    """Main pipeline orchestrator for convergence interpolation.

    Steps:
        1. Load Phase 1 period rates from parquet
        2. Determine which periods map to recent/medium/long-term
        3. Compute window averages
        3b. (If variant="high") Lift window averages by BEBR high-vs-baseline increment
        4. Apply convergence interpolation
        5. Save output parquet and metadata

    Args:
        config: Optional configuration dict.  If ``None``, loads from
            ``config/projection_config.yaml``.
        variant: Optional scenario variant.  When ``"high"``, the BEBR
            high-vs-baseline migration difference is added to all three
            window averages before convergence interpolation, producing
            scenario-specific convergence rates saved as
            ``convergence_rates_by_year_high.parquet``.  When ``None``
            (default), produces the baseline convergence rates.

    Returns:
        Dict with keys:
            - ``rates_by_year``: dict mapping year_offset to DataFrame
            - ``output_path``: Path to saved parquet
            - ``metadata_path``: Path to saved metadata JSON
            - ``total_rows``: total rows in output parquet
    """
    if config is None:
        config = load_projection_config()

    project_root = Path(__file__).parent.parent.parent.parent
    migration_config = config.get("rates", {}).get("migration", {})

    variant_label = f" (variant={variant})" if variant else ""
    logger.info(f"Starting convergence interpolation pipeline{variant_label}")

    # --- 1. Load Phase 1 period rates ---
    input_path = (
        project_root / "data" / "processed" / "migration" / "residual_migration_rates.parquet"
    )
    logger.info(f"Loading Phase 1 period rates from {input_path}")
    all_period_rates = pd.read_parquet(input_path)
    logger.info(
        f"Loaded {len(all_period_rates)} rows "
        f"({all_period_rates['county_fips'].nunique()} counties, "
        f"{all_period_rates['age_group'].nunique()} age groups, "
        f"{all_period_rates['sex'].nunique()} sexes)"
    )

    # --- 2. Determine period-to-window mapping ---
    interp_config = migration_config.get("interpolation", {})
    recent_range = interp_config.get("recent_period", [2022, 2024])
    medium_range = interp_config.get("medium_period", [2014, 2024])
    longterm_range = interp_config.get("longterm_period", [2000, 2024])
    convergence_schedule = interp_config.get(
        "convergence_schedule",
        {
            "recent_to_medium_years": 5,
            "medium_hold_years": 10,
            "medium_to_longterm_years": 5,
        },
    )
    projection_years = config.get("project", {}).get("projection_horizon", 20)

    # Extract available periods from data
    available_periods: list[tuple[int, int]] = sorted(
        all_period_rates[["period_start", "period_end"]]
        .drop_duplicates()
        .apply(lambda r: (int(r["period_start"]), int(r["period_end"])), axis=1)
        .tolist()
    )
    logger.info(f"Available periods: {available_periods}")

    recent_periods = _map_config_window_to_periods(recent_range, available_periods)
    medium_periods = _map_config_window_to_periods(medium_range, available_periods)
    longterm_periods = _map_config_window_to_periods(longterm_range, available_periods)

    logger.info(f"Recent periods: {recent_periods}")
    logger.info(f"Medium periods: {medium_periods}")
    logger.info(f"Long-term periods: {longterm_periods}")

    # --- 3. Compute window averages ---
    recent_rates, medium_rates, longterm_rates = compute_period_window_averages(
        all_period_rates, recent_periods, medium_periods, longterm_periods
    )

    # --- 3b. Lift window averages for high variant (ADR-046) ---
    if variant == "high":
        logger.info("Computing high-scenario rate increment from BEBR files")
        increment = _compute_high_scenario_rate_increment(recent_rates, project_root)
        recent_rates = _lift_window_averages(recent_rates, increment)
        medium_rates = _lift_window_averages(medium_rates, increment)
        longterm_rates = _lift_window_averages(longterm_rates, increment)
        logger.info("Window averages lifted for high scenario")

        # --- 3c. Apply migration floor for high variant (ADR-052) ---
        # For counties where BEBR-boosted rates are still net-negative at
        # the medium hold, lift all cells so the county mean reaches zero.
        high_growth_cfg = config.get("scenarios", {}).get("high_growth", {})
        floor_cfg = high_growth_cfg.get("migration_floor", {})
        floor_enabled = floor_cfg.get("enabled", False)
        floor_value = floor_cfg.get("floor_value", 0.0)

        if floor_enabled:
            n_floored = 0
            for window_rates in (recent_rates, medium_rates, longterm_rates):
                for fips in window_rates["county_fips"].unique():
                    mask = window_rates["county_fips"] == fips
                    county_mean = window_rates.loc[mask, "migration_rate"].mean()
                    if county_mean < floor_value:
                        lift = floor_value - county_mean
                        window_rates.loc[mask, "migration_rate"] += lift
                        n_floored += 1
            # n_floored counts across 3 windows, so divide by 3 for counties
            n_counties_floored = n_floored // 3
            if n_counties_floored > 0:
                logger.info(
                    f"Migration floor applied: {n_counties_floored} counties "
                    f"lifted to floor={floor_value:.3f} in high variant"
                )

    # --- 4. Apply convergence interpolation ---
    rate_cap_config = interp_config.get("rate_cap", None)
    rates_by_year = calculate_age_specific_convergence(
        recent_rates,
        medium_rates,
        longterm_rates,
        projection_years=projection_years,
        convergence_schedule=convergence_schedule,
        rate_cap_config=rate_cap_config,
    )

    # --- 5. Build output DataFrame and save ---
    output_dir = project_root / "data" / "processed" / "migration"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_frames = []
    for year_offset, year_df in sorted(rates_by_year.items()):
        frame = year_df.copy()
        frame["year_offset"] = year_offset
        output_frames.append(frame)

    output_df = pd.concat(output_frames, ignore_index=True)

    # Reorder columns to match spec
    output_df = output_df[["year_offset", "county_fips", "age_group", "sex", "migration_rate"]]

    # Variant-specific output file name
    suffix = f"_{variant}" if variant else ""
    output_path = output_dir / f"convergence_rates_by_year{suffix}.parquet"
    compression = config.get("output", {}).get("compression", "gzip")
    output_df.to_parquet(output_path, compression=compression, index=False)
    logger.info(f"Saved convergence rates to {output_path} ({len(output_df)} rows)")

    # --- 6. Save metadata ---
    metadata: dict[str, Any] = {
        "processing_date": datetime.now(UTC).isoformat(),
        "phase": "Phase 2 - Age-Specific Convergence Interpolation",
        "variant": variant,
        "input_file": str(input_path),
        "output_file": str(output_path),
        "rate_unit": "annual_rate",
        "total_rows": len(output_df),
        "projection_years": projection_years,
        "convergence_schedule": convergence_schedule,
        "rate_cap": rate_cap_config if rate_cap_config else {"enabled": False},
        "window_mapping": {
            "recent_config": recent_range,
            "recent_periods": [list(p) for p in recent_periods],
            "medium_config": medium_range,
            "medium_periods": [list(p) for p in medium_periods],
            "longterm_config": longterm_range,
            "longterm_periods": [list(p) for p in longterm_periods],
        },
        "dimensions": {
            "counties": int(output_df["county_fips"].nunique()),
            "age_groups": int(output_df["age_group"].nunique()),
            "sexes": int(output_df["sex"].nunique()),
            "years": int(output_df["year_offset"].nunique()),
            "cells_per_county_per_year": int(
                output_df["age_group"].nunique() * output_df["sex"].nunique()
            ),
        },
        "rate_summary": {
            "year_1_mean_rate": float(
                output_df.loc[output_df["year_offset"] == 1, "migration_rate"].mean()
            ),
            "year_5_mean_rate": float(
                output_df.loc[output_df["year_offset"] == 5, "migration_rate"].mean()
            ),
            "year_10_mean_rate": float(
                output_df.loc[output_df["year_offset"] == 10, "migration_rate"].mean()
            ),
            "year_20_mean_rate": float(
                output_df.loc[output_df["year_offset"] == 20, "migration_rate"].mean()
            ),
        },
    }

    metadata_path = output_dir / f"convergence_metadata{suffix}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")

    logger.info("=" * 70)
    logger.info(f"Convergence interpolation pipeline complete{variant_label}")
    logger.info(f"  Output rows: {len(output_df)}")
    logger.info(f"  Counties: {output_df['county_fips'].nunique()}")
    logger.info(f"  Years: 1-{projection_years}")
    logger.info(f"  Schedule: {convergence_schedule}")
    logger.info("=" * 70)

    return {
        "rates_by_year": rates_by_year,
        "output_path": output_path,
        "metadata_path": metadata_path,
        "total_rows": len(output_df),
    }
