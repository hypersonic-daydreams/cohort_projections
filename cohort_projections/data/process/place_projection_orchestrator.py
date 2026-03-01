"""
Place projection orchestration for PP-003 Phase 2 (IMP-06).

This module applies county-constrained share-trending outputs to county
projections and materializes place-level outputs by confidence tier.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from cohort_projections.data.process.place_share_trending import (
    BALANCE_KEY,
    trend_all_places_in_county,
)
from cohort_projections.utils import get_logger_from_config

logger = get_logger_from_config(__name__)

TIER_HIGH = "HIGH"
TIER_MODERATE = "MODERATE"
TIER_LOWER = "LOWER"
PROJECTED_TIERS = {TIER_HIGH, TIER_MODERATE, TIER_LOWER}
SEX_ORDER = ["Male", "Female"]
ALLOWED_OUTLIER_FLAG_TYPES = {
    "SHARE_REVERSAL",
    "EXTREME_GROWTH",
    "NEAR_ZERO_SHARE",
    "SHARE_RESCALED",
    "POPULATION_DECLINE_TO_NEAR_ZERO",
}
TIER_ANNUAL_GROWTH_BANDS = {
    TIER_HIGH: 0.10,
    TIER_MODERATE: 0.15,
    TIER_LOWER: 0.25,
}
QA_FLOAT_TOLERANCE = 1e-9

HIGH_AGE_GROUPS = [
    "0-4",
    "5-9",
    "10-14",
    "15-19",
    "20-24",
    "25-29",
    "30-34",
    "35-39",
    "40-44",
    "45-49",
    "50-54",
    "55-59",
    "60-64",
    "65-69",
    "70-74",
    "75-79",
    "80-84",
    "85+",
]

MODERATE_AGE_GROUPS = ["0-17", "18-24", "25-44", "45-64", "65-84", "85+"]


def _project_root() -> Path:
    """Return repository root from module location."""
    return Path(__file__).resolve().parents[3]


def _normalize_fips(value: object, width: int) -> str:
    """Normalize FIPS-like values to zero-padded digit strings."""
    text = str(value).strip().removesuffix(".0")
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits.zfill(width)[-width:]


def _as_bool(value: object) -> bool:
    """Coerce mixed truthy/falsy values to bool."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _resolve_path(path_text: str, root: Path) -> Path:
    """Resolve relative config path against project root."""
    path = Path(path_text)
    return path if path.is_absolute() else root / path


def _five_year_age_group(age: int) -> str:
    """Map single-year age to standard 18-bin five-year age groups."""
    if age >= 85:
        return "85+"
    start = (age // 5) * 5
    end = start + 4
    return f"{start}-{end}"


def _broad_age_group(age: int) -> str:
    """Map single-year age to 6 broad age bins for MODERATE tier output."""
    if age <= 17:
        return "0-17"
    if age <= 24:
        return "18-24"
    if age <= 44:
        return "25-44"
    if age <= 64:
        return "45-64"
    if age <= 84:
        return "65-84"
    return "85+"


def _allocate_from_profile(
    place_total: float,
    profile_df: pd.DataFrame,
    groups: list[str],
) -> pd.DataFrame:
    """
    Allocate place total by an age-group/sex profile.

    Args:
        place_total: Place total population for one year.
        profile_df: DataFrame with columns ``age_group``, ``sex``, ``population``.
        groups: Ordered age-group labels expected in output.

    Returns:
        DataFrame with columns ``age_group``, ``sex``, ``population``.
    """
    if place_total < 0:
        raise ValueError("place_total must be non-negative.")

    merged = profile_df.groupby(["age_group", "sex"], as_index=False)["population"].sum()
    full_index = pd.MultiIndex.from_product([groups, SEX_ORDER], names=["age_group", "sex"])
    merged = merged.set_index(["age_group", "sex"]).reindex(full_index, fill_value=0.0).reset_index()

    county_total = float(merged["population"].sum())
    if county_total <= 0:
        if place_total == 0:
            merged["population"] = 0.0
            return merged
        raise ValueError("Cannot allocate positive place_total from zero county cohort total.")

    merged["allocation_share"] = merged["population"] / county_total
    merged["population"] = merged["allocation_share"] * float(place_total)
    merged = merged.drop(columns=["allocation_share"])

    # Keep exact add-up after floating-point arithmetic.
    diff = float(place_total) - float(merged["population"].sum())
    if not np.isclose(diff, 0.0):
        merged.loc[merged.index[-1], "population"] += diff

    return merged


def _resolve_variant_settings(
    variant_winner: Mapping[str, Any] | str,
    config: Mapping[str, Any],
) -> tuple[str, str]:
    """Resolve fitting/constraint methods from winner metadata or variant ID."""
    place_cfg = config.get("place_projections", {})
    model_cfg = place_cfg.get("model", {}) if isinstance(place_cfg, dict) else {}

    default_fitting = str(model_cfg.get("fitting_method", "ols")).lower()
    default_constraint = str(model_cfg.get("constraint_method", "proportional")).lower()

    if isinstance(variant_winner, str):
        variant_id = variant_winner.strip().upper().replace("_", "-")
        if variant_id.startswith("B-"):
            fitting = "wls"
        elif variant_id.startswith("A-"):
            fitting = "ols"
        else:
            fitting = default_fitting

        if variant_id.endswith("-II"):
            constraint = "cap_and_redistribute"
        elif variant_id.endswith("-I"):
            constraint = "proportional"
        else:
            constraint = default_constraint
        return fitting, constraint

    fitting = str(variant_winner.get("fitting_method", default_fitting)).lower()
    constraint = str(variant_winner.get("constraint_method", default_constraint)).lower()

    if fitting not in {"ols", "wls"}:
        raise ValueError(f"Unsupported fitting method: {fitting}")
    if constraint not in {"proportional", "cap_and_redistribute"}:
        raise ValueError(f"Unsupported constraint method: {constraint}")
    return fitting, constraint


def _projection_year_bounds(config: Mapping[str, Any]) -> tuple[int, int]:
    """Resolve base/end years from place block, then project defaults."""
    place_cfg = config.get("place_projections", {})
    out_cfg = place_cfg.get("output", {}) if isinstance(place_cfg, dict) else {}
    project_cfg = config.get("project", {})

    base_year = int(out_cfg.get("base_year", project_cfg.get("base_year", 2025)))
    default_end = base_year + int(project_cfg.get("projection_horizon", 30))
    end_year = int(out_cfg.get("end_year", default_end))
    return base_year, end_year


def _output_dir_for_scenario(config: Mapping[str, Any], scenario: str) -> Path:
    """Resolve scenario output directory from config."""
    root = _project_root()
    projection_output = (
        config.get("pipeline", {}).get("projection", {}).get("output_dir", "data/projections")
    )
    return _resolve_path(str(projection_output), root) / scenario / "place"


def _county_projection_dir(config: Mapping[str, Any], scenario: str) -> Path:
    """Resolve county projection input directory for a scenario."""
    root = _project_root()
    projection_output = (
        config.get("pipeline", {}).get("projection", {}).get("output_dir", "data/projections")
    )
    return _resolve_path(str(projection_output), root) / scenario / "county"


def _qa_dir_for_scenario(config: Mapping[str, Any], scenario: str) -> Path:
    """Resolve QA artifact output directory for a scenario."""
    return _output_dir_for_scenario(config, scenario) / "qa"


def _validate_crosswalk_uniqueness(crosswalk: pd.DataFrame) -> None:
    """Enforce one-row-per-place in projection crosswalk universe."""
    duplicates = crosswalk[crosswalk.duplicated(subset=["place_fips"], keep=False)]
    if duplicates.empty:
        return
    duplicate_ids = sorted(duplicates["place_fips"].astype(str).unique().tolist())
    raise ValueError(
        "Crosswalk projection universe must be unique on place_fips. "
        f"Duplicate place_fips: {duplicate_ids}"
    )


def _validate_projection_universe(expected_place_fips: set[str], produced_place_fips: set[str]) -> None:
    """Validate output universe matches the projected crosswalk exactly."""
    missing = sorted(expected_place_fips - produced_place_fips)
    extra = sorted(produced_place_fips - expected_place_fips)
    if not missing and not extra:
        return
    raise ValueError(
        "Monotonic FIPS constraint failed: output place universe does not match crosswalk. "
        f"Missing={missing[:10]}, Extra={extra[:10]}"
    )


def _enforce_county_hard_constraints(
    county_fips: str,
    county_share_projection: pd.DataFrame,
) -> None:
    """Enforce S06 hard constraints 1-4 at county-year granularity."""
    place_rows = county_share_projection[county_share_projection["row_type"] == "place"].copy()
    if place_rows.empty:
        raise ValueError(f"County {county_fips} has no place rows in projection output.")

    place_rows["projected_share"] = pd.to_numeric(place_rows["projected_share"], errors="coerce")
    place_rows["county_population"] = pd.to_numeric(place_rows["county_population"], errors="coerce")
    place_rows["projected_population"] = pd.to_numeric(
        place_rows["projected_population"], errors="coerce"
    )
    place_rows = place_rows.dropna(subset=["year", "projected_share", "county_population", "projected_population"])

    out_of_bounds = place_rows[
        (place_rows["projected_share"] < -QA_FLOAT_TOLERANCE)
        | (place_rows["projected_share"] > 1.0 + QA_FLOAT_TOLERANCE)
    ]
    if not out_of_bounds.empty:
        bad_row = out_of_bounds.iloc[0]
        raise ValueError(
            "Share bound hard constraint failed for county "
            f"{county_fips} in year {int(bad_row['year'])}: "
            f"share={float(bad_row['projected_share'])}"
        )

    share_sums = place_rows.groupby("year", as_index=False)["projected_share"].sum()
    overfull = share_sums[share_sums["projected_share"] > 1.0 + QA_FLOAT_TOLERANCE]
    if not overfull.empty:
        bad_row = overfull.iloc[0]
        raise ValueError(
            "County share-sum hard constraint failed for county "
            f"{county_fips} in year {int(bad_row['year'])}: "
            f"sum_place_shares={float(bad_row['projected_share'])}"
        )

    if (place_rows["projected_population"] < -QA_FLOAT_TOLERANCE).any():
        bad_row = place_rows[place_rows["projected_population"] < -QA_FLOAT_TOLERANCE].iloc[0]
        raise ValueError(
            "Non-negative population hard constraint failed for county "
            f"{county_fips}, place {bad_row['place_fips']}, year {int(bad_row['year'])}: "
            f"population={float(bad_row['projected_population'])}"
        )

    year_population = place_rows.groupby("year", as_index=False).agg(
        sum_place_pop=("projected_population", "sum"),
        county_population=("county_population", "first"),
    )
    inconsistent = year_population[
        year_population["sum_place_pop"] > year_population["county_population"] + QA_FLOAT_TOLERANCE
    ]
    if not inconsistent.empty:
        bad_row = inconsistent.iloc[0]
        raise ValueError(
            "Place-county consistency hard constraint failed for county "
            f"{county_fips} in year {int(bad_row['year'])}: "
            f"sum_place_pop={float(bad_row['sum_place_pop'])}, "
            f"county_total={float(bad_row['county_population'])}"
        )


def _load_state_total_series_from_counties(
    config: Mapping[str, Any],
    scenario: str,
    base_year: int,
    end_year: int,
) -> pd.Series | None:
    """Aggregate state totals from county projection parquet files for one scenario."""
    county_dir = _county_projection_dir(config, scenario)
    county_files = sorted(county_dir.glob(f"nd_county_*_projection_*_{scenario}.parquet"))
    if not county_files:
        return None

    totals_by_year: dict[int, float] = {}
    for county_path in county_files:
        county_df = pd.read_parquet(county_path, columns=["year", "population"])
        county_df["year"] = pd.to_numeric(county_df["year"], errors="coerce").astype("Int64")
        county_df["population"] = pd.to_numeric(county_df["population"], errors="coerce")
        county_df = county_df.dropna(subset=["year", "population"]).copy()
        county_df["year"] = county_df["year"].astype(int)
        county_df = county_df[(county_df["year"] >= base_year) & (county_df["year"] <= end_year)].copy()
        if county_df.empty:
            continue

        county_totals = county_df.groupby("year", as_index=False)["population"].sum()
        for _, row in county_totals.iterrows():
            year = int(row["year"])
            totals_by_year[year] = totals_by_year.get(year, 0.0) + float(row["population"])

    if not totals_by_year:
        return None

    series = pd.Series(totals_by_year, name=scenario, dtype=float).sort_index()
    series.index = series.index.astype(int)
    return series


def validate_state_scenario_ordering(
    config: Mapping[str, Any],
    base_year: int,
    end_year: int,
    required_scenarios: tuple[str, str, str] = ("restricted_growth", "baseline", "high_growth"),
    skip_if_missing: bool = True,
) -> bool:
    """
    Validate S06 hard constraint 6: restricted <= baseline <= high at state level.

    Returns ``True`` if evaluated successfully. Returns ``False`` only when
    ``skip_if_missing`` is enabled and required scenario county outputs are absent.
    """
    series_by_scenario: dict[str, pd.Series] = {}
    missing_scenarios: list[str] = []
    for scenario in required_scenarios:
        series = _load_state_total_series_from_counties(
            config=config,
            scenario=scenario,
            base_year=base_year,
            end_year=end_year,
        )
        if series is None:
            missing_scenarios.append(scenario)
            continue
        series_by_scenario[scenario] = series

    if missing_scenarios:
        if skip_if_missing:
            logger.warning(
                "Skipping state scenario ordering check because county projections are missing for: %s",
                ", ".join(missing_scenarios),
            )
            return False
        raise FileNotFoundError(
            "Cannot validate state scenario ordering; missing county outputs for: "
            f"{', '.join(missing_scenarios)}"
        )

    ordering = pd.concat(series_by_scenario.values(), axis=1)
    ordering.columns = list(series_by_scenario.keys())
    ordering = ordering.sort_index()
    if ordering.isna().any(axis=None):
        raise ValueError("State scenario ordering check failed due to missing year totals.")

    low, mid, high = required_scenarios
    violations = ordering[
        (ordering[low] > ordering[mid] + QA_FLOAT_TOLERANCE)
        | (ordering[mid] > ordering[high] + QA_FLOAT_TOLERANCE)
    ]
    if not violations.empty:
        bad_year = int(violations.index[0])
        bad_row = violations.iloc[0]
        raise ValueError(
            "Scenario ordering hard constraint failed at state level for year "
            f"{bad_year}: {low}={float(bad_row[low])}, "
            f"{mid}={float(bad_row[mid])}, {high}={float(bad_row[high])}"
        )

    return True


def _load_county_names(config: Mapping[str, Any]) -> dict[str, str]:
    """Load county FIPS -> county name map."""
    root = _project_root()
    county_ref = (
        config.get("geography", {})
        .get("reference_data", {})
        .get("counties_file", "data/raw/geographic/nd_counties.csv")
    )
    county_path = _resolve_path(str(county_ref), root)
    counties = pd.read_csv(county_path)
    counties["county_fips"] = counties["county_fips"].map(lambda v: _normalize_fips(v, 5))

    if "state_fips" in counties.columns:
        counties["state_fips"] = counties["state_fips"].map(lambda v: _normalize_fips(v, 2))
        counties = counties[counties["state_fips"] == "38"].copy()

    name_col = "county_name" if "county_name" in counties.columns else "name"
    return dict(zip(counties["county_fips"], counties[name_col].astype(str), strict=True))


def _load_crosswalk(config: Mapping[str, Any]) -> pd.DataFrame:
    """Load and normalize place-county crosswalk with confidence tiers."""
    root = _project_root()
    place_cfg = config.get("place_projections", {})
    crosswalk_path = place_cfg.get(
        "crosswalk_path",
        "data/processed/geographic/place_county_crosswalk_2020.csv",
    )
    path = _resolve_path(str(crosswalk_path), root)

    crosswalk = pd.read_csv(path)
    crosswalk["place_fips"] = crosswalk["place_fips"].map(lambda v: _normalize_fips(v, 7))
    crosswalk["county_fips"] = crosswalk["county_fips"].map(lambda v: _normalize_fips(v, 5))
    crosswalk["confidence_tier"] = crosswalk["confidence_tier"].astype(str).str.upper()
    crosswalk["historical_only"] = crosswalk["historical_only"].map(_as_bool)
    crosswalk["place_name"] = crosswalk["place_name"].astype(str)

    projected = crosswalk[
        (~crosswalk["historical_only"]) & (crosswalk["confidence_tier"].isin(PROJECTED_TIERS))
    ].copy()
    projected = projected.sort_values(["county_fips", "place_fips"]).reset_index(drop=True)
    if projected.empty:
        logger.warning("No projected places found in crosswalk (HIGH/MODERATE/LOWER + active).")
    return projected


def _load_share_history(config: Mapping[str, Any]) -> pd.DataFrame:
    """Load historical place shares used for trend fitting."""
    root = _project_root()
    place_cfg = config.get("place_projections", {})
    shares_path = place_cfg.get("historical_shares_path", "data/processed/place_shares_2000_2024.parquet")
    path = _resolve_path(str(shares_path), root)

    shares = pd.read_parquet(path)
    shares["county_fips"] = shares["county_fips"].map(lambda v: _normalize_fips(v, 5))
    shares["year"] = pd.to_numeric(shares["year"], errors="coerce").astype("Int64")
    shares = shares.dropna(subset=["year"]).copy()
    shares["year"] = shares["year"].astype(int)
    shares["share_raw"] = pd.to_numeric(shares["share_raw"], errors="coerce")
    shares["place_fips"] = shares["place_fips"].map(
        lambda v: _normalize_fips(v, 7) if pd.notna(v) else pd.NA
    )

    if "row_type" not in shares.columns:
        shares["row_type"] = "place"

    return shares


def _load_county_projection(
    county_fips: str,
    scenario: str,
    base_year: int,
    end_year: int,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Load one county projection parquet and filter to projection years."""
    county_dir = _county_projection_dir(config, scenario)
    expected = county_dir / f"nd_county_{county_fips}_projection_{base_year}_{end_year}_{scenario}.parquet"

    if expected.exists():
        path = expected
    else:
        candidates = sorted(county_dir.glob(f"nd_county_{county_fips}_projection_*_{scenario}.parquet"))
        if not candidates:
            raise FileNotFoundError(
                f"No county projection parquet found for county {county_fips} in {county_dir}"
            )
        path = candidates[0]

    county_df = pd.read_parquet(path)
    county_df["year"] = pd.to_numeric(county_df["year"], errors="coerce").astype("Int64")
    county_df["age"] = pd.to_numeric(county_df["age"], errors="coerce").astype("Int64")
    county_df["population"] = pd.to_numeric(county_df["population"], errors="coerce")
    county_df = county_df.dropna(subset=["year", "age", "sex", "population"]).copy()
    county_df["year"] = county_df["year"].astype(int)
    county_df["age"] = county_df["age"].astype(int)
    county_df = county_df[(county_df["year"] >= base_year) & (county_df["year"] <= end_year)].copy()
    return county_df


def _parquet_footer_metadata(metadata: Mapping[str, Any]) -> dict[bytes, bytes]:
    """Build required output footer key-value metadata."""
    projection = metadata.get("projection", {})
    geography = metadata.get("geography", {})
    share_model = metadata.get("share_model", {})

    footer = {
        "scenario": projection.get("scenario"),
        "geography_level": geography.get("level", "place"),
        "place_fips": geography.get("place_fips"),
        "county_fips": geography.get("county_fips"),
        "confidence_tier": geography.get("confidence_tier"),
        "projection_base_year": projection.get("base_year"),
        "projection_end_year": projection.get("end_year"),
        "model_method": projection.get("method", "share_of_county_trending"),
        "model_version": share_model.get("model_version", "1.0.0"),
        "crosswalk_vintage": share_model.get("crosswalk_vintage", "2020"),
        "processing_date": projection.get("processing_date"),
    }
    return {
        key.encode("utf-8"): str(value).encode("utf-8")
        for key, value in footer.items()
        if value is not None
    }


def _write_parquet_with_metadata(
    dataframe: pd.DataFrame,
    output_path: Path,
    compression: str,
    footer_metadata: Mapping[bytes, bytes],
) -> None:
    """Write parquet preserving Pandas metadata plus custom footer metadata."""
    table = pa.Table.from_pandas(dataframe, preserve_index=False)
    current = dict(table.schema.metadata or {})
    current.update(dict(footer_metadata))
    table = table.replace_schema_metadata(current)
    pq.write_table(table, output_path, compression=compression)


def _summary_row_from_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Convert per-place metadata to one-row summary schema."""
    geography = metadata["geography"]
    summary = metadata["summary_statistics"]
    share_model = metadata["share_model"]

    return {
        "place_fips": geography["place_fips"],
        "name": geography["name"],
        "county_fips": geography["county_fips"],
        "level": "place",
        "row_type": "place",
        "confidence_tier": geography["confidence_tier"],
        "base_population": summary["base_population"],
        "final_population": summary["final_population"],
        "absolute_growth": summary["absolute_growth"],
        "growth_rate": summary["growth_rate"],
        "base_share": share_model["base_share"],
        "final_share": share_model["final_share"],
        "processing_time": metadata["processing_time_seconds"],
    }


def _county_year_totals(county_df: pd.DataFrame) -> pd.DataFrame:
    """Return county totals by year for share scaling."""
    totals = county_df.groupby("year", as_index=False)["population"].sum()
    totals = totals.rename(columns={"population": "county_population"})
    return totals.sort_values("year").reset_index(drop=True)


def _share_history_for_county(
    share_history: pd.DataFrame,
    county_fips: str,
    place_fips_values: set[str],
) -> pd.DataFrame:
    """Prepare county historical share rows for projected places only."""
    county_rows = share_history[
        (share_history["county_fips"] == county_fips)
        & (share_history["row_type"].astype(str) == "place")
        & (share_history["place_fips"].isin(place_fips_values))
    ].copy()
    return county_rows[["county_fips", "year", "row_type", "place_fips", "share_raw"]]


def allocate_age_sex_detail(place_total: float, county_cohort_df: pd.DataFrame, tier: str) -> pd.DataFrame:
    """
    Allocate county cohort structure to a place total by confidence tier.

    Args:
        place_total: Place population for one projection year.
        county_cohort_df: County cohort rows for the same year.
        tier: ``HIGH``, ``MODERATE``, or ``LOWER``.

    Returns:
        Tier-specific DataFrame:
        - HIGH/MODERATE: columns ``age_group``, ``sex``, ``population``
        - LOWER: columns ``population`` (total only)
    """
    tier_upper = tier.upper()
    if tier_upper not in PROJECTED_TIERS:
        raise ValueError(f"Unsupported tier: {tier}")

    if tier_upper == TIER_LOWER:
        return pd.DataFrame({"population": [float(place_total)]})

    required = {"age", "sex", "population"}
    missing = required - set(county_cohort_df.columns)
    if missing:
        raise ValueError(f"county_cohort_df missing required columns: {sorted(missing)}")

    cohort = county_cohort_df.copy()
    cohort["age"] = pd.to_numeric(cohort["age"], errors="coerce").astype("Int64")
    cohort["population"] = pd.to_numeric(cohort["population"], errors="coerce")
    cohort = cohort.dropna(subset=["age", "sex", "population"]).copy()
    cohort["age"] = cohort["age"].astype(int)
    cohort = cohort[cohort["age"] >= 0].copy()

    collapsed = cohort.groupby(["age", "sex"], as_index=False)["population"].sum()
    if tier_upper == TIER_HIGH:
        collapsed["age_group"] = collapsed["age"].map(_five_year_age_group)
        allocated = _allocate_from_profile(
            place_total=float(place_total),
            profile_df=collapsed[["age_group", "sex", "population"]],
            groups=HIGH_AGE_GROUPS,
        )
    else:
        collapsed["age_group"] = collapsed["age"].map(_broad_age_group)
        allocated = _allocate_from_profile(
            place_total=float(place_total),
            profile_df=collapsed[["age_group", "sex", "population"]],
            groups=MODERATE_AGE_GROUPS,
        )

    return allocated[["age_group", "sex", "population"]]


def write_place_outputs(
    place_df: pd.DataFrame,
    metadata: dict[str, Any],
    scenario: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Write per-place parquet, metadata JSON, and summary CSV.

    Args:
        place_df: Place projection rows in tier-specific schema.
        metadata: Per-place metadata dictionary.
        scenario: Scenario key.
        config: Projection configuration.

    Returns:
        Dict containing output paths and summary row.
    """
    output_dir = _output_dir_for_scenario(config, scenario)
    output_dir.mkdir(parents=True, exist_ok=True)

    geography = metadata["geography"]
    projection = metadata["projection"]
    place_fips = geography["place_fips"]
    base_year = int(projection["base_year"])
    end_year = int(projection["end_year"])

    base_filename = f"nd_place_{place_fips}_projection_{base_year}_{end_year}_{scenario}"
    parquet_path = output_dir / f"{base_filename}.parquet"
    metadata_path = output_dir / f"{base_filename}_metadata.json"
    summary_path = output_dir / f"{base_filename}_summary.csv"

    compression = str(config.get("output", {}).get("compression", "gzip"))
    footer_metadata = _parquet_footer_metadata(metadata)
    _write_parquet_with_metadata(
        dataframe=place_df,
        output_path=parquet_path,
        compression=compression,
        footer_metadata=footer_metadata,
    )

    with open(metadata_path, "w", encoding="utf-8") as file_handle:
        json.dump(metadata, file_handle, indent=2)

    summary_row = _summary_row_from_metadata(metadata)
    pd.DataFrame([summary_row]).to_csv(summary_path, index=False)

    return {
        "parquet_path": parquet_path,
        "metadata_path": metadata_path,
        "summary_path": summary_path,
        "summary_row": summary_row,
    }


def write_run_level_metadata(
    all_places_metadata: list[dict[str, Any]],
    scenario: str,
    config: Mapping[str, Any],
) -> Path:
    """
    Write run-level places metadata (`places_metadata.json`).

    Args:
        all_places_metadata: List of per-place metadata dictionaries.
        scenario: Scenario key.
        config: Projection configuration.

    Returns:
        Path to written run-level metadata file.
    """
    output_dir = _output_dir_for_scenario(config, scenario)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "places_metadata.json"

    by_tier = dict.fromkeys([TIER_HIGH, TIER_MODERATE, TIER_LOWER], 0)
    for meta in all_places_metadata:
        tier = str(meta["geography"]["confidence_tier"]).upper()
        if tier in by_tier:
            by_tier[tier] += 1

    crosswalk_vintage = "2020"
    model_version = "1.0.0"
    if all_places_metadata:
        first_meta = all_places_metadata[0]
        crosswalk_vintage = str(first_meta["share_model"].get("crosswalk_vintage", "2020"))
        model_version = str(first_meta["share_model"].get("model_version", "1.0.0"))

    run_metadata = {
        "level": "place",
        "num_geographies": len(all_places_metadata),
        "successful": len(all_places_metadata),
        "failed": 0,
        "by_tier": by_tier,
        "total_processing_time_seconds": float(
            sum(float(meta.get("processing_time_seconds", 0.0)) for meta in all_places_metadata)
        ),
        "processing_date": datetime.now(UTC).isoformat(),
        "crosswalk_vintage": crosswalk_vintage,
        "model_version": model_version,
    }

    with open(metadata_path, "w", encoding="utf-8") as file_handle:
        json.dump(run_metadata, file_handle, indent=2)
    return metadata_path


def write_places_summary(
    all_places_summaries: list[dict[str, Any]],
    balance_rows: list[dict[str, Any]],
    scenario: str,
    config: Mapping[str, Any],
) -> Path:
    """
    Write aggregate `places_summary.csv` including balance-of-county rows.

    Args:
        all_places_summaries: Per-place summary rows.
        balance_rows: County balance summary rows.
        scenario: Scenario key.
        config: Projection configuration.

    Returns:
        Path to written summary file.
    """
    output_dir = _output_dir_for_scenario(config, scenario)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "places_summary.csv"

    columns = [
        "place_fips",
        "name",
        "county_fips",
        "level",
        "row_type",
        "confidence_tier",
        "base_population",
        "final_population",
        "absolute_growth",
        "growth_rate",
        "base_share",
        "final_share",
        "processing_time",
    ]

    summary_df = pd.DataFrame(all_places_summaries + balance_rows)
    if summary_df.empty:
        summary_df = pd.DataFrame(columns=columns)
    else:
        summary_df = summary_df[columns].sort_values(["county_fips", "row_type", "place_fips"]).reset_index(
            drop=True
        )

    summary_df.to_csv(summary_path, index=False)
    return summary_path


def _add_outlier_flag(
    outlier_flags: list[dict[str, Any]],
    place_fips: str,
    place_name: str,
    confidence_tier: str,
    flag_type: str,
    flag_detail: str,
    year: int | None = None,
) -> None:
    """Append one outlier flag row after validating the flag type."""
    if flag_type not in ALLOWED_OUTLIER_FLAG_TYPES:
        raise ValueError(f"Unsupported outlier flag_type: {flag_type}")
    outlier_flags.append(
        {
            "place_fips": place_fips,
            "name": place_name,
            "confidence_tier": confidence_tier,
            "flag_type": flag_type,
            "flag_detail": flag_detail,
            "year": year if year is not None else pd.NA,
        }
    )


def write_place_qa_artifacts(
    scenario: str,
    config: Mapping[str, Any],
    all_places_summaries: list[dict[str, Any]],
    share_sum_rows: list[dict[str, Any]],
    reconciliation_rows: list[dict[str, Any]],
    outlier_rows: list[dict[str, Any]],
    balance_rows: list[dict[str, Any]],
) -> dict[str, Path]:
    """
    Write S06 Section 5 QA artifacts for one scenario.

    Returns a mapping of artifact names to written file paths.
    """
    qa_dir = _qa_dir_for_scenario(config, scenario)
    qa_dir.mkdir(parents=True, exist_ok=True)

    tier_summary_path = qa_dir / "qa_tier_summary.csv"
    share_sum_path = qa_dir / "qa_share_sum_validation.csv"
    reconciliation_path = qa_dir / "qa_reconciliation_magnitude.csv"
    outlier_path = qa_dir / "qa_outlier_flags.csv"
    balance_path = qa_dir / "qa_balance_of_county.csv"

    tier_rows: list[dict[str, Any]] = []
    places_summary_df = pd.DataFrame(all_places_summaries)
    for tier in [TIER_HIGH, TIER_MODERATE, TIER_LOWER]:
        if places_summary_df.empty:
            tier_df = pd.DataFrame()
        else:
            tier_df = places_summary_df[places_summary_df["confidence_tier"] == tier].copy()

        if tier_df.empty:
            tier_rows.append(
                {
                    "confidence_tier": tier,
                    "place_count": 0,
                    "total_base_population": 0.0,
                    "total_final_population": 0.0,
                    "mean_growth_rate": 0.0,
                    "median_growth_rate": 0.0,
                    "min_growth_rate": 0.0,
                    "max_growth_rate": 0.0,
                }
            )
            continue

        growth_series = pd.to_numeric(tier_df["growth_rate"], errors="coerce").fillna(0.0)
        tier_rows.append(
            {
                "confidence_tier": tier,
                "place_count": int(len(tier_df)),
                "total_base_population": float(
                    pd.to_numeric(tier_df["base_population"], errors="coerce").fillna(0.0).sum()
                ),
                "total_final_population": float(
                    pd.to_numeric(tier_df["final_population"], errors="coerce").fillna(0.0).sum()
                ),
                "mean_growth_rate": float(growth_series.mean()),
                "median_growth_rate": float(growth_series.median()),
                "min_growth_rate": float(growth_series.min()),
                "max_growth_rate": float(growth_series.max()),
            }
        )

    tier_summary_df = pd.DataFrame(
        tier_rows,
        columns=[
            "confidence_tier",
            "place_count",
            "total_base_population",
            "total_final_population",
            "mean_growth_rate",
            "median_growth_rate",
            "min_growth_rate",
            "max_growth_rate",
        ],
    )
    tier_summary_df.to_csv(tier_summary_path, index=False)

    share_sum_columns = [
        "county_fips",
        "county_name",
        "year",
        "sum_place_shares",
        "balance_of_county_share",
        "constraint_satisfied",
        "rescaling_applied",
    ]
    share_sum_df = pd.DataFrame(share_sum_rows, columns=share_sum_columns)
    if not share_sum_df.empty:
        share_sum_df = share_sum_df.sort_values(["county_fips", "year"]).reset_index(drop=True)
    share_sum_df.to_csv(share_sum_path, index=False)

    reconciliation_columns = [
        "county_fips",
        "county_name",
        "year",
        "total_before_adjustment",
        "total_after_adjustment",
        "reconciliation_adjustment",
        "reconciliation_flag",
        "reconciliation_flag_threshold",
        "rescaling_applied",
    ]
    reconciliation_df = pd.DataFrame(reconciliation_rows, columns=reconciliation_columns)
    if not reconciliation_df.empty:
        reconciliation_df = reconciliation_df.sort_values(["county_fips", "year"]).reset_index(
            drop=True
        )
    reconciliation_df.to_csv(reconciliation_path, index=False)

    outlier_columns = ["place_fips", "name", "confidence_tier", "flag_type", "flag_detail", "year"]
    outlier_df = pd.DataFrame(outlier_rows, columns=outlier_columns)
    if outlier_df.empty:
        outlier_df = pd.DataFrame(columns=outlier_columns)
    else:
        invalid_types = set(outlier_df["flag_type"]) - ALLOWED_OUTLIER_FLAG_TYPES
        if invalid_types:
            raise ValueError(f"qa_outlier_flags contains unsupported flag types: {sorted(invalid_types)}")
        outlier_df = outlier_df.drop_duplicates().sort_values(
            ["place_fips", "flag_type", "year"], na_position="last"
        ).reset_index(drop=True)
    outlier_df.to_csv(outlier_path, index=False)

    balance_columns = [
        "county_fips",
        "county_name",
        "year",
        "county_total",
        "sum_of_places",
        "balance_of_county",
        "balance_share",
    ]
    balance_df = pd.DataFrame(balance_rows, columns=balance_columns)
    if not balance_df.empty:
        balance_df = balance_df.sort_values(["county_fips", "year"]).reset_index(drop=True)
    balance_df.to_csv(balance_path, index=False)

    return {
        "qa_dir": qa_dir,
        "qa_tier_summary": tier_summary_path,
        "qa_share_sum_validation": share_sum_path,
        "qa_reconciliation_magnitude": reconciliation_path,
        "qa_outlier_flags": outlier_path,
        "qa_balance_of_county": balance_path,
    }


def run_place_projections(
    scenario: str,
    config: Mapping[str, Any],
    variant_winner: Mapping[str, Any] | str,
) -> dict[str, Any]:
    """
    Run place projections for one scenario.

    Args:
        scenario: Scenario key (e.g., ``baseline``).
        config: Projection configuration dictionary.
        variant_winner: Backtest winner metadata or variant ID.

    Returns:
        Dictionary with output paths and run counts.
    """
    place_cfg = config.get("place_projections", {})
    if isinstance(place_cfg, dict) and not place_cfg.get("enabled", True):
        logger.info("Place projections disabled in config (`place_projections.enabled: false`).")
        return {"scenario": scenario, "enabled": False}

    fitting_method, constraint_method = _resolve_variant_settings(variant_winner, config)
    base_year, end_year = _projection_year_bounds(config)

    place_cfg_model = place_cfg.get("model", {}) if isinstance(place_cfg, dict) else {}
    epsilon = float(place_cfg_model.get("epsilon", 0.001))
    lambda_decay = float(place_cfg_model.get("lambda_decay", 0.9))
    reconciliation_threshold = float(place_cfg_model.get("reconciliation_flag_threshold", 0.05))
    history_start = int(place_cfg_model.get("history_start", 2000))
    history_end = int(place_cfg_model.get("history_end", 2024))

    crosswalk = _load_crosswalk(config)
    _validate_crosswalk_uniqueness(crosswalk)
    share_history = _load_share_history(config)
    county_names = _load_county_names(config)

    expected_place_fips = set(crosswalk["place_fips"].astype(str).tolist())
    produced_place_fips: set[str] = set()

    all_place_metadata: list[dict[str, Any]] = []
    all_place_summaries: list[dict[str, Any]] = []
    balance_rows: list[dict[str, Any]] = []
    qa_share_sum_rows: list[dict[str, Any]] = []
    qa_reconciliation_rows: list[dict[str, Any]] = []
    qa_balance_rows: list[dict[str, Any]] = []
    qa_outlier_rows: list[dict[str, Any]] = []

    for county_fips, county_places in crosswalk.groupby("county_fips"):
        county_df = _load_county_projection(
            county_fips=county_fips,
            scenario=scenario,
            base_year=base_year,
            end_year=end_year,
            config=config,
        )
        county_totals = _county_year_totals(county_df)
        projection_years = county_totals["year"].tolist()
        if not projection_years:
            raise ValueError(f"County {county_fips} has no projection years in {base_year}-{end_year}.")

        county_place_ids = set(county_places["place_fips"].tolist())
        history_county = _share_history_for_county(
            share_history=share_history,
            county_fips=county_fips,
            place_fips_values=county_place_ids,
        )
        if history_county.empty:
            raise ValueError(
                f"County {county_fips} has no historical place shares for projected universe."
            )

        trend_config = {
            "projection_years": projection_years,
            "epsilon": epsilon,
            "lambda_decay": lambda_decay,
            "fitting_method": fitting_method,
            "constraint_method": constraint_method,
            "reconciliation_flag_threshold": reconciliation_threshold,
        }
        county_share_projection = trend_all_places_in_county(
            place_share_history=history_county,
            county_pop_history=county_totals,
            config=trend_config,
        )

        _enforce_county_hard_constraints(county_fips=county_fips, county_share_projection=county_share_projection)

        place_projection_rows = county_share_projection[
            county_share_projection["row_type"] == "place"
        ].copy()
        place_projection_rows = place_projection_rows.sort_values(["place_fips", "year"]).reset_index(drop=True)

        projected_place_ids = set(place_projection_rows["place_fips"].astype(str).tolist())
        if projected_place_ids != county_place_ids:
            missing = sorted(county_place_ids - projected_place_ids)
            extra = sorted(projected_place_ids - county_place_ids)
            raise ValueError(
                "Monotonic FIPS hard constraint failed within county "
                f"{county_fips}. Missing={missing}, Extra={extra}"
            )

        county_name = county_names.get(county_fips, f"County {county_fips}")
        crosswalk_vintage = (
            str(county_places["source_vintage"].iloc[0]) if "source_vintage" in county_places else "2020"
        )

        raw_share_sum_by_year = (
            place_projection_rows.groupby("year", as_index=False)["projected_share_raw"].sum().set_index("year")
        )["projected_share_raw"].to_dict()
        share_sum_by_year = (
            place_projection_rows.groupby("year", as_index=False)["projected_share"].sum().set_index("year")
        )["projected_share"].to_dict()
        place_population_sum_by_year = (
            place_projection_rows.groupby("year", as_index=False)["projected_population"].sum().set_index("year")
        )["projected_population"].to_dict()
        county_population_by_year = dict(
            zip(county_totals["year"].astype(int), county_totals["county_population"], strict=True)
        )

        for year in projection_years:
            if year not in share_sum_by_year:
                raise ValueError(f"County {county_fips} missing projected shares for year {year}.")
            year_place_rows = place_projection_rows[place_projection_rows["year"] == year].copy()
            if year_place_rows.empty:
                raise ValueError(f"County {county_fips} has no place rows for year {year}.")

            sum_place_shares = float(share_sum_by_year[year])
            balance_share = 1.0 - sum_place_shares
            rescaling_applied = bool(
                (
                    pd.to_numeric(year_place_rows["projected_share"], errors="coerce")
                    - pd.to_numeric(year_place_rows["projected_share_raw"], errors="coerce")
                )
                .abs()
                .gt(QA_FLOAT_TOLERANCE)
                .any()
            )
            year_all_rows = county_share_projection[county_share_projection["year"] == year].copy()
            total_before_adjustment = float(
                pd.to_numeric(year_all_rows["projected_share_raw"], errors="coerce").sum()
            )
            total_after_adjustment = float(
                pd.to_numeric(year_all_rows["projected_share"], errors="coerce").sum()
            )

            reconciliation_adjustments = pd.to_numeric(
                year_all_rows["reconciliation_adjustment"], errors="coerce"
            ).dropna()
            if reconciliation_adjustments.empty:
                reconciliation_adjustment = abs(total_before_adjustment - 1.0)
            else:
                reconciliation_adjustment = float(reconciliation_adjustments.iloc[0])
                if not np.isclose(
                    reconciliation_adjustments.to_numpy(dtype=float),
                    reconciliation_adjustment,
                    atol=QA_FLOAT_TOLERANCE,
                ).all():
                    raise ValueError(
                        "Inconsistent reconciliation_adjustment values within county-year "
                        f"{county_fips}-{int(year)}."
                    )

            reconciliation_flags = year_all_rows["reconciliation_flag"].map(_as_bool).dropna()
            if reconciliation_flags.empty:
                reconciliation_flag = bool(
                    reconciliation_adjustment > reconciliation_threshold + QA_FLOAT_TOLERANCE
                )
            else:
                reconciliation_flag = bool(reconciliation_flags.iloc[0])
                if not (reconciliation_flags == reconciliation_flag).all():
                    raise ValueError(
                        "Inconsistent reconciliation_flag values within county-year "
                        f"{county_fips}-{int(year)}."
                    )

            qa_share_sum_rows.append(
                {
                    "county_fips": county_fips,
                    "county_name": county_name,
                    "year": int(year),
                    "sum_place_shares": sum_place_shares,
                    "balance_of_county_share": balance_share,
                    "constraint_satisfied": bool(sum_place_shares <= 1.0 + QA_FLOAT_TOLERANCE),
                    "rescaling_applied": rescaling_applied,
                }
            )
            qa_reconciliation_rows.append(
                {
                    "county_fips": county_fips,
                    "county_name": county_name,
                    "year": int(year),
                    "total_before_adjustment": total_before_adjustment,
                    "total_after_adjustment": total_after_adjustment,
                    "reconciliation_adjustment": reconciliation_adjustment,
                    "reconciliation_flag": reconciliation_flag,
                    "reconciliation_flag_threshold": reconciliation_threshold,
                    "rescaling_applied": rescaling_applied,
                }
            )

            county_total = float(county_population_by_year[int(year)])
            sum_of_places = float(place_population_sum_by_year.get(year, 0.0))
            balance_of_county = county_total - sum_of_places
            balance_share_pop = balance_of_county / county_total if county_total > 0 else 0.0
            qa_balance_rows.append(
                {
                    "county_fips": county_fips,
                    "county_name": county_name,
                    "year": int(year),
                    "county_total": county_total,
                    "sum_of_places": sum_of_places,
                    "balance_of_county": balance_of_county,
                    "balance_share": balance_share_pop,
                }
            )
            if balance_of_county < 0.0:
                logger.warning(
                    "Soft constraint flag: negative balance-of-county for %s (%s) year %s: %.6f",
                    county_name,
                    county_fips,
                    year,
                    balance_of_county,
                )

        share_sum_ok = bool(
            all(value <= 1.0 + QA_FLOAT_TOLERANCE for value in share_sum_by_year.values())
        )

        for _, place_row in county_places.iterrows():
            place_fips = str(place_row["place_fips"])
            place_name = str(place_row["place_name"])
            tier = str(place_row["confidence_tier"]).upper()

            place_share_rows = county_share_projection[
                (county_share_projection["row_type"] == "place")
                & (county_share_projection["place_fips"] == place_fips)
            ].copy()
            place_share_rows = place_share_rows.sort_values("year").reset_index(drop=True)
            if place_share_rows.empty:
                raise ValueError(
                    f"Monotonic FIPS hard constraint failed: missing projected rows for place {place_fips}."
                )
            produced_place_fips.add(place_fips)

            history_place_rows = history_county[history_county["place_fips"] == place_fips].copy()
            history_place_rows = history_place_rows.sort_values("year").reset_index(drop=True)
            if len(history_place_rows) >= 2 and len(place_share_rows) >= 2:
                historical_change = float(history_place_rows["share_raw"].iloc[-1] - history_place_rows["share_raw"].iloc[0])
                projected_change = float(
                    place_share_rows["projected_share"].iloc[-1] - place_share_rows["projected_share"].iloc[0]
                )

                if (
                    abs(historical_change) > QA_FLOAT_TOLERANCE
                    and abs(projected_change) > QA_FLOAT_TOLERANCE
                    and (historical_change * projected_change < 0.0)
                ):
                    _add_outlier_flag(
                        outlier_flags=qa_outlier_rows,
                        place_fips=place_fips,
                        place_name=place_name,
                        confidence_tier=tier,
                        flag_type="SHARE_REVERSAL",
                        flag_detail=(
                            "Historical and projected share directions differ "
                            f"(historical_change={historical_change:.6f}, "
                            f"projected_change={projected_change:.6f})."
                        ),
                    )

                if abs(projected_change) > 0.20:
                    _add_outlier_flag(
                        outlier_flags=qa_outlier_rows,
                        place_fips=place_fips,
                        place_name=place_name,
                        confidence_tier=tier,
                        flag_type="SHARE_REVERSAL",
                        flag_detail=(
                            "Share stability soft-constraint exceeded: "
                            f"{projected_change:.6f} change over horizon (>0.20)."
                        ),
                    )

            growth_band = TIER_ANNUAL_GROWTH_BANDS[tier]
            for idx, share_row in place_share_rows.iterrows():
                year = int(share_row["year"])
                projected_share = float(share_row["projected_share"])
                projected_share_raw = float(share_row["projected_share_raw"])
                projected_population = float(share_row["projected_population"])
                raw_sum_place_share = float(raw_share_sum_by_year.get(year, np.nan))

                if projected_share <= 0.001 + QA_FLOAT_TOLERANCE:
                    _add_outlier_flag(
                        outlier_flags=qa_outlier_rows,
                        place_fips=place_fips,
                        place_name=place_name,
                        confidence_tier=tier,
                        flag_type="NEAR_ZERO_SHARE",
                        flag_detail=f"Projected share {projected_share:.6f} is below 0.1% threshold.",
                        year=year,
                    )

                if (
                    np.isfinite(raw_sum_place_share)
                    and raw_sum_place_share > 1.0 + QA_FLOAT_TOLERANCE
                    and abs(projected_share - projected_share_raw) > QA_FLOAT_TOLERANCE
                ):
                    _add_outlier_flag(
                        outlier_flags=qa_outlier_rows,
                        place_fips=place_fips,
                        place_name=place_name,
                        confidence_tier=tier,
                        flag_type="SHARE_RESCALED",
                        flag_detail=(
                            "Projected share was adjusted during county reconciliation "
                            f"(raw_sum_place_shares={raw_sum_place_share:.6f})."
                        ),
                        year=year,
                    )

                if projected_population < 50.0:
                    _add_outlier_flag(
                        outlier_flags=qa_outlier_rows,
                        place_fips=place_fips,
                        place_name=place_name,
                        confidence_tier=tier,
                        flag_type="POPULATION_DECLINE_TO_NEAR_ZERO",
                        flag_detail=f"Projected population {projected_population:.3f} is below 50.",
                        year=year,
                    )

                if idx > 0:
                    previous_population = float(place_share_rows["projected_population"].iloc[idx - 1])
                    if previous_population <= 0.0:
                        annual_growth = np.inf if projected_population > 0.0 else 0.0
                    else:
                        annual_growth = (projected_population / previous_population) - 1.0

                    if abs(annual_growth) > growth_band:
                        _add_outlier_flag(
                            outlier_flags=qa_outlier_rows,
                            place_fips=place_fips,
                            place_name=place_name,
                            confidence_tier=tier,
                            flag_type="EXTREME_GROWTH",
                            flag_detail=(
                                f"Annual growth {annual_growth:.4f} exceeds tier band +/-{growth_band:.2f}."
                            ),
                            year=year,
                        )

            place_start_time = time.perf_counter()
            annual_blocks: list[pd.DataFrame] = []
            for _, year_row in place_share_rows.iterrows():
                year = int(year_row["year"])
                place_total = float(year_row["projected_population"])
                county_year = county_df[county_df["year"] == year]
                allocated = allocate_age_sex_detail(
                    place_total=place_total,
                    county_cohort_df=county_year,
                    tier=tier,
                )
                allocated.insert(0, "year", year)
                annual_blocks.append(allocated)

            place_df = pd.concat(annual_blocks, ignore_index=True)
            place_df = place_df.sort_values("year").reset_index(drop=True)
            if (place_df["population"] < -QA_FLOAT_TOLERANCE).any():
                raise ValueError(
                    f"Non-negative population hard constraint failed for place {place_fips}: "
                    "tier-detail allocation produced negative cohort values."
                )

            base_row = place_share_rows[place_share_rows["year"] == base_year]
            final_row = place_share_rows[place_share_rows["year"] == end_year]
            if base_row.empty:
                base_row = place_share_rows.iloc[[0]]
            if final_row.empty:
                final_row = place_share_rows.iloc[[-1]]

            base_population = float(base_row["projected_population"].iloc[0])
            final_population = float(final_row["projected_population"].iloc[0])
            absolute_growth = final_population - base_population
            growth_rate = absolute_growth / base_population if base_population > 0 else 0.0
            processing_time = round(time.perf_counter() - place_start_time, 4)

            share_within_bounds = bool(
                (
                    (place_share_rows["projected_share"] >= -QA_FLOAT_TOLERANCE)
                    & (place_share_rows["projected_share"] <= 1.0 + QA_FLOAT_TOLERANCE)
                ).all()
            )

            metadata = {
                "geography": {
                    "level": "place",
                    "place_fips": place_fips,
                    "name": place_name,
                    "county_fips": county_fips,
                    "confidence_tier": tier,
                    "base_population": base_population,
                },
                "projection": {
                    "base_year": base_year,
                    "end_year": end_year,
                    "scenario": scenario,
                    "method": "share_of_county_trending",
                    "processing_date": datetime.now(UTC).isoformat(),
                },
                "share_model": {
                    "trend_type": "logit_linear",
                    "fitting_method": fitting_method,
                    "constraint_method": constraint_method,
                    "base_share": float(base_row["projected_share"].iloc[0]),
                    "final_share": float(final_row["projected_share"].iloc[0]),
                    "share_change": float(final_row["projected_share"].iloc[0] - base_row["projected_share"].iloc[0]),
                    "historical_window": f"{history_start}-{history_end}",
                    "crosswalk_vintage": crosswalk_vintage,
                    "model_version": "1.0.0",
                },
                "summary_statistics": {
                    "base_population": base_population,
                    "final_population": final_population,
                    "absolute_growth": absolute_growth,
                    "growth_rate": growth_rate,
                    "years_projected": int(end_year - base_year),
                },
                "validation": {
                    "share_within_bounds": share_within_bounds,
                    "share_sum_check_passed": share_sum_ok,
                    "all_checks_passed": bool(
                        share_within_bounds
                        and share_sum_ok
                        and (place_df["population"] >= -QA_FLOAT_TOLERANCE).all()
                    ),
                },
                "processing_time_seconds": processing_time,
            }

            write_result = write_place_outputs(
                place_df=place_df,
                metadata=metadata,
                scenario=scenario,
                config=config,
            )
            all_place_metadata.append(metadata)
            all_place_summaries.append(write_result["summary_row"])

        county_balance = county_share_projection[county_share_projection["row_type"] == BALANCE_KEY].copy()
        county_balance = county_balance.sort_values("year").reset_index(drop=True)
        if not county_balance.empty:
            base_balance = county_balance[county_balance["year"] == base_year]
            final_balance = county_balance[county_balance["year"] == end_year]
            if base_balance.empty:
                base_balance = county_balance.iloc[[0]]
            if final_balance.empty:
                final_balance = county_balance.iloc[[-1]]

            base_pop = float(base_balance["projected_population"].iloc[0])
            final_pop = float(final_balance["projected_population"].iloc[0])
            growth = final_pop - base_pop
            growth_rate = growth / base_pop if base_pop > 0 else 0.0

            balance_rows.append(
                {
                    "place_fips": f"bal_{county_fips}",
                    "name": f"Balance of {county_name}",
                    "county_fips": county_fips,
                    "level": "place",
                    "row_type": "balance_of_county",
                    "confidence_tier": pd.NA,
                    "base_population": base_pop,
                    "final_population": final_pop,
                    "absolute_growth": growth,
                    "growth_rate": growth_rate,
                    "base_share": float(base_balance["projected_share"].iloc[0]),
                    "final_share": float(final_balance["projected_share"].iloc[0]),
                    "processing_time": 0.0,
                }
            )

    _validate_projection_universe(
        expected_place_fips=expected_place_fips,
        produced_place_fips=produced_place_fips,
    )

    summary_path = write_places_summary(
        all_places_summaries=all_place_summaries,
        balance_rows=balance_rows,
        scenario=scenario,
        config=config,
    )
    run_metadata_path = write_run_level_metadata(
        all_places_metadata=all_place_metadata,
        scenario=scenario,
        config=config,
    )
    qa_paths = write_place_qa_artifacts(
        scenario=scenario,
        config=config,
        all_places_summaries=all_place_summaries,
        share_sum_rows=qa_share_sum_rows,
        reconciliation_rows=qa_reconciliation_rows,
        outlier_rows=qa_outlier_rows,
        balance_rows=qa_balance_rows,
    )
    state_ordering_checked = validate_state_scenario_ordering(
        config=config,
        base_year=base_year,
        end_year=end_year,
        skip_if_missing=True,
    )

    logger.info(
        "Place projections complete for %s: %d places, %d county balances, %d QA outlier flags.",
        scenario,
        len(all_place_metadata),
        len(balance_rows),
        len(qa_outlier_rows),
    )
    return {
        "scenario": scenario,
        "enabled": True,
        "output_dir": _output_dir_for_scenario(config, scenario),
        "places_processed": len(all_place_metadata),
        "balance_rows": len(balance_rows),
        "summary_path": summary_path,
        "metadata_path": run_metadata_path,
        "qa_paths": qa_paths,
        "state_ordering_checked": state_ordering_checked,
    }


__all__ = [
    "allocate_age_sex_detail",
    "run_place_projections",
    "validate_state_scenario_ordering",
    "write_place_qa_artifacts",
    "write_place_outputs",
    "write_places_summary",
    "write_run_level_metadata",
]
