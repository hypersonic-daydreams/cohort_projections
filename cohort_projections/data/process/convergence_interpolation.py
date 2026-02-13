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


def calculate_age_specific_convergence(
    recent_rates: pd.DataFrame,
    medium_rates: pd.DataFrame,
    longterm_rates: pd.DataFrame,
    projection_years: int = 20,
    convergence_schedule: dict[str, int] | None = None,
) -> dict[int, pd.DataFrame]:
    """Calculate year-varying migration rates with age-specific convergence.

    Each county x age_group x sex cell converges independently through
    the 5-10-5 schedule:

    - Years 1 through *phase1*: linear from recent to medium
    - Years *phase1+1* through *phase1+phase2*: hold at medium
    - Years *phase1+phase2+1* through *total*: linear from medium to
      long-term

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

        year_df = merged[group_cols].copy()
        year_df["migration_rate"] = rate.values
        results[year] = year_df

    logger.info(
        f"Age-specific convergence computed for {len(merged)} cells "
        f"over {projection_years} years (phases: {phase1}+{phase2}+{phase3})"
    )

    return results


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


def run_convergence_pipeline(
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Main pipeline orchestrator for convergence interpolation.

    Steps:
        1. Load Phase 1 period rates from parquet
        2. Determine which periods map to recent/medium/long-term
        3. Compute window averages
        4. Apply convergence interpolation
        5. Save output parquet and metadata

    Args:
        config: Optional configuration dict.  If ``None``, loads from
            ``config/projection_config.yaml``.

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

    # --- 4. Apply convergence interpolation ---
    rates_by_year = calculate_age_specific_convergence(
        recent_rates,
        medium_rates,
        longterm_rates,
        projection_years=projection_years,
        convergence_schedule=convergence_schedule,
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

    output_path = output_dir / "convergence_rates_by_year.parquet"
    compression = config.get("output", {}).get("compression", "gzip")
    output_df.to_parquet(output_path, compression=compression, index=False)
    logger.info(f"Saved convergence rates to {output_path} ({len(output_df)} rows)")

    # --- 6. Save metadata ---
    metadata = {
        "processing_date": datetime.now(UTC).isoformat(),
        "phase": "Phase 2 - Age-Specific Convergence Interpolation",
        "input_file": str(input_path),
        "output_file": str(output_path),
        "total_rows": len(output_df),
        "projection_years": projection_years,
        "convergence_schedule": convergence_schedule,
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

    metadata_path = output_dir / "convergence_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")

    logger.info("=" * 70)
    logger.info("Convergence interpolation pipeline complete")
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
