#!/usr/bin/env python3
"""
Build place-to-county crosswalk for ND place projections.

Created: 2026-02-28
ADR: 033 (PP3-S03 mapping strategy implementation)
Author: Codex / N. Haarstad

Purpose
-------
Create the authoritative place->county mapping artifact required by PP-003
Phase 1 city/place projections. The script applies deterministic assignment
rules from `docs/reviews/2026-02-28-place-county-mapping-strategy-note.md`:
single-county places map directly; multi-county places map to the county with
the largest area share and are flagged `multi_county_primary`.

Method
------
1. Load active ND place reference records (state FIPS 38, SUMLEV 162).
2. Load place-county overlaps from either:
   a) a relationship file with place/county keys and area components, or
   b) TIGER place + county shapefiles with intersection-based area shares.
3. Compute per-place county shares and select one primary county assignment.
4. Generate a supplemental detail table containing every county overlap for
   multi-county places.
5. Append dissolved historical-only places (Bantry, Churchs Ferry) so
   historical time-series joins remain complete through 2019.
6. Validate schema and invariants, then write both output CSV artifacts.

Key design decisions
--------------------
- Freeze to 2020-vintage assignment logic for Phase 1 consistency.
- Keep dissolved places in the crosswalk with `historical_only=True` so
  backtest/training joins can include pre-2020 records while projection output
  can exclude these rows downstream.
- Support both relationship-file and TIGER-overlay ingestion to avoid coupling
  to a single upstream source format.

Validation results
------------------
- Validation is executed at runtime; script logs:
  - active place coverage,
  - uniqueness on `place_fips`,
  - FIPS/null checks,
  - area share bounds.
- First-run production values are generated when this script is executed in the
  configured environment (expected active place count: 355).

Inputs
------
- `data/raw/geographic/nd_places.csv` (default via config):
  ND place reference used to define active place universe.
- Relationship file (optional, CLI `--relationship-file`):
  Census relationship-style table with place/county keys and area components.
- TIGER place/county shapefiles (optional, CLI `--place-shapefile`,
  `--county-shapefile`):
  Used when relationship-file input is not provided.

Outputs
-------
- `data/processed/geographic/place_county_crosswalk_2020.csv`
  Primary one-row-per-place assignment table.
- `data/processed/geographic/place_county_crosswalk_2020_multicounty_detail.csv`
  Supplemental overlap detail for multi-county places.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from cohort_projections.utils import load_projection_config

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ND_STATE_FIPS = "38"
DEFAULT_SOURCE_VINTAGE = "2020"

# Known dissolved places from PP3-S02/S03 readiness notes.
DISSOLVED_PLACE_OVERRIDES: dict[str, dict[str, str]] = {
    "3804740": {
        "place_name": "Bantry city",
        "county_fips": "38049",
    },
    "3814140": {
        "place_name": "Churchs Ferry city",
        "county_fips": "38071",
    },
}

VALID_ASSIGNMENT_TYPES = {"single_county", "multi_county_primary"}


def _normalize_fips(value: Any, width: int) -> str | None:
    """Normalize a FIPS-like value to zero-padded digits of fixed width."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.removesuffix(".0")
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return None
    return digits.zfill(width)[-width:]


def _read_delimited(path: Path) -> pd.DataFrame:
    """Read CSV/TSV-style delimited data with separator inference."""
    return pd.read_csv(path, dtype=str, sep=None, engine="python")


def _require_columns(df: pd.DataFrame, columns: list[str], context: str) -> None:
    """Raise a helpful error when required columns are missing."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"{context}: missing required columns: {missing_str}")


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return first matching column from candidate list (case-sensitive)."""
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def load_active_nd_places(places_csv_path: Path, state_fips: str = ND_STATE_FIPS) -> pd.DataFrame:
    """Load active ND places from the configured place reference CSV."""
    places = pd.read_csv(places_csv_path, dtype=str)
    _require_columns(places, ["STATE", "PLACE"], context=str(places_csv_path))

    places["STATE"] = places["STATE"].map(lambda v: _normalize_fips(v, 2))
    places["PLACE"] = places["PLACE"].map(lambda v: _normalize_fips(v, 5))
    places = places[places["STATE"] == state_fips].copy()

    if "SUMLEV" in places.columns:
        places = places[places["SUMLEV"] == "162"].copy()
    if "FUNCSTAT" in places.columns:
        places = places[places["FUNCSTAT"] == "A"].copy()

    places["place_fips"] = places["STATE"] + places["PLACE"]
    places["state_fips"] = places["STATE"]
    places["place_name"] = places.get("NAME", "").astype(str).str.strip()

    active = (
        places[["state_fips", "place_fips", "place_name"]]
        .drop_duplicates(subset=["place_fips"])
        .sort_values("place_fips")
        .reset_index(drop=True)
    )

    logger.info("Loaded %d active ND places from %s", len(active), places_csv_path)
    return active


def _compute_area_share_from_columns(df: pd.DataFrame) -> pd.Series:
    """
    Compute area share from available relationship-file columns.

    Supports two common patterns:
    1) pre-computed share/percent columns,
    2) part-area and place-total-area columns.
    """
    share_col = _pick_column(
        df,
        [
            "area_share",
            "AREA_SHARE",
            "AREASHARE",
            "PCT_AREA",
            "AREA_PCT",
            "AREAPCT",
        ],
    )
    if share_col:
        shares = pd.to_numeric(df[share_col], errors="coerce")
        if (shares > 1).any():
            shares = shares / 100.0
        return shares

    part_col = _pick_column(
        df,
        [
            "AREALAND_PART",
            "PART_AREALAND",
            "PART_AREA",
            "INT_AREA",
            "INTERSECTION_AREA",
        ],
    )
    total_col = _pick_column(
        df,
        [
            "AREALAND_PLACE",
            "PLACE_AREALAND",
            "PLACE_AREA",
            "TOTAL_PLACE_AREA",
            "TOTAL_AREA",
        ],
    )
    if part_col and total_col:
        part = pd.to_numeric(df[part_col], errors="coerce")
        total = pd.to_numeric(df[total_col], errors="coerce")
        return part / total

    raise ValueError(
        "Could not infer area share columns from relationship file. "
        "Provide one of AREA_SHARE/AREAPCT columns or part+total area columns."
    )


def load_overlaps_from_relationship_file(
    relationship_file: Path,
    state_fips: str = ND_STATE_FIPS,
) -> pd.DataFrame:
    """Load place-county overlap shares from a relationship file."""
    rel = _read_delimited(relationship_file)

    state_col = _pick_column(rel, ["STATEFP", "STATE", "STATEFP20"])
    place_col = _pick_column(rel, ["PLACEFP", "PLACE", "PLACEFP20"])
    county_col = _pick_column(rel, ["COUNTYFP", "COUNTY", "COUNTYFP20"])
    if not place_col or not county_col:
        raise ValueError(
            "Relationship file must include place and county code columns "
            "(e.g., PLACEFP/COUNTYFP)."
        )

    rel["state_fips"] = (
        rel[state_col].map(lambda v: _normalize_fips(v, 2)) if state_col else state_fips
    )
    rel["place_code"] = rel[place_col].map(lambda v: _normalize_fips(v, 5))
    rel["county_code"] = rel[county_col].map(lambda v: _normalize_fips(v, 3))
    rel["area_share"] = _compute_area_share_from_columns(rel)
    rel = rel[rel["state_fips"] == state_fips].copy()

    rel["place_fips"] = rel["state_fips"] + rel["place_code"]
    rel["county_fips"] = rel["state_fips"] + rel["county_code"]

    overlaps = (
        rel[["place_fips", "county_fips", "area_share"]]
        .dropna()
        .groupby(["place_fips", "county_fips"], as_index=False, sort=False)["area_share"]
        .sum()
    )
    overlaps["source_method"] = "census_relationship_file"
    logger.info(
        "Loaded %d place-county overlap rows from relationship file %s",
        len(overlaps),
        relationship_file,
    )
    return overlaps


def load_overlaps_from_tiger_shapefiles(
    place_shapefile: Path,
    county_shapefile: Path,
    state_fips: str = ND_STATE_FIPS,
) -> pd.DataFrame:
    """Load place-county overlaps by intersecting TIGER place/county geometries."""
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError(
            "geopandas is required for TIGER overlay mode. Install geo extras or "
            "use --relationship-file input."
        ) from exc

    places = gpd.read_file(place_shapefile)
    counties = gpd.read_file(county_shapefile)

    state_col_places = _pick_column(places, ["STATEFP", "STATEFP20"])
    state_col_counties = _pick_column(counties, ["STATEFP", "STATEFP20"])
    place_fips_col = _pick_column(places, ["GEOID", "GEOID20"])
    county_fips_col = _pick_column(counties, ["GEOID", "GEOID20"])
    place_code_col = _pick_column(places, ["PLACEFP", "PLACEFP20"])
    county_code_col = _pick_column(counties, ["COUNTYFP", "COUNTYFP20"])

    if not state_col_places or not state_col_counties:
        raise ValueError("TIGER shapefiles must include STATEFP columns.")
    if not place_fips_col and not place_code_col:
        raise ValueError("Place shapefile must include GEOID or PLACEFP columns.")
    if not county_fips_col and not county_code_col:
        raise ValueError("County shapefile must include GEOID or COUNTYFP columns.")

    places = places[places[state_col_places] == state_fips].copy()
    counties = counties[counties[state_col_counties] == state_fips].copy()

    if place_fips_col:
        places["place_fips"] = places[place_fips_col].map(lambda v: _normalize_fips(v, 7))
    else:
        places["place_fips"] = state_fips + places[place_code_col].map(
            lambda v: _normalize_fips(v, 5) or ""
        )

    if county_fips_col:
        counties["county_fips"] = counties[county_fips_col].map(lambda v: _normalize_fips(v, 5))
    else:
        counties["county_fips"] = state_fips + counties[county_code_col].map(
            lambda v: _normalize_fips(v, 3) or ""
        )

    places = places[["place_fips", "geometry"]].dropna(subset=["place_fips"]).copy()
    counties = counties[["county_fips", "geometry"]].dropna(subset=["county_fips"]).copy()

    # Use an equal-area projection for area-share calculations.
    places = places.to_crs(epsg=5070)
    counties = counties.to_crs(epsg=5070)

    places["place_area"] = places.geometry.area
    overlaps = gpd.overlay(
        places[["place_fips", "place_area", "geometry"]],
        counties[["county_fips", "geometry"]],
        how="intersection",
    )
    overlaps["intersection_area"] = overlaps.geometry.area
    overlaps["area_share"] = overlaps["intersection_area"] / overlaps["place_area"]

    overlap_df = (
        overlaps[["place_fips", "county_fips", "area_share"]]
        .dropna()
        .groupby(["place_fips", "county_fips"], as_index=False, sort=False)["area_share"]
        .sum()
    )
    overlap_df["source_method"] = "tiger_overlay"
    logger.info(
        "Computed %d place-county overlap rows from TIGER shapefiles",
        len(overlap_df),
    )
    return overlap_df


def _build_primary_and_detail_from_overlaps(
    overlaps: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert overlap rows to primary assignment table + multicounty detail."""
    overlap_required = ["place_fips", "county_fips", "area_share"]
    _require_columns(overlaps, overlap_required, context="overlaps")

    clean = overlaps[overlap_required].copy()
    clean["place_fips"] = clean["place_fips"].map(lambda v: _normalize_fips(v, 7))
    clean["county_fips"] = clean["county_fips"].map(lambda v: _normalize_fips(v, 5))
    clean["area_share"] = pd.to_numeric(clean["area_share"], errors="coerce")
    clean = clean.dropna(subset=["place_fips", "county_fips", "area_share"])
    clean = clean[clean["area_share"] > 0].copy()

    if clean.empty:
        raise ValueError("No usable overlap rows after cleaning.")

    # Resolve duplicates and stabilize ordering.
    clean = (
        clean.groupby(["place_fips", "county_fips"], as_index=False, sort=False)["area_share"]
        .sum()
        .sort_values(["place_fips", "area_share", "county_fips"], ascending=[True, False, True])
        .reset_index(drop=True)
    )
    # Normalize overlap shares within each place to absorb tiny geometry/precision drift.
    place_totals = clean.groupby("place_fips")["area_share"].transform("sum")
    clean["area_share"] = clean["area_share"] / place_totals
    clean["area_share"] = clean["area_share"].clip(lower=0.0, upper=1.0)

    county_counts = clean.groupby("place_fips")["county_fips"].nunique()
    clean["county_count"] = clean["place_fips"].map(county_counts)
    clean["county_rank"] = clean.groupby("place_fips")["area_share"].rank(
        method="first",
        ascending=False,
    )

    primary = clean[clean["county_rank"] == 1].copy()
    primary["assignment_type"] = primary["county_count"].map(
        lambda n: "single_county" if int(n) == 1 else "multi_county_primary"
    )

    detail = clean[clean["county_count"] > 1].copy()
    detail["assignment_type"] = "multi_county_primary"
    detail["is_primary"] = detail["county_rank"] == 1

    return primary, detail


def build_place_county_crosswalk(
    overlaps: pd.DataFrame,
    active_places: pd.DataFrame,
    source_vintage: str = DEFAULT_SOURCE_VINTAGE,
    source_method: str = "census_relationship_file",
    dissolved_place_overrides: dict[str, dict[str, str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build primary crosswalk and multi-county detail tables."""
    dissolved_place_overrides = dissolved_place_overrides or DISSOLVED_PLACE_OVERRIDES
    _require_columns(active_places, ["state_fips", "place_fips", "place_name"], context="active_places")

    primary, detail = _build_primary_and_detail_from_overlaps(overlaps)
    primary = primary.merge(active_places, on="place_fips", how="inner", validate="one_to_one")

    expected_active = set(active_places["place_fips"])
    actual_active = set(primary["place_fips"])
    missing = sorted(expected_active - actual_active)
    if missing:
        missing_preview = ", ".join(missing[:10])
        suffix = " ..." if len(missing) > 10 else ""
        raise ValueError(
            f"{len(missing)} active places are missing county assignments: {missing_preview}{suffix}"
        )

    primary["state_fips"] = primary["state_fips"].fillna(ND_STATE_FIPS)
    primary["historical_only"] = False
    primary["source_vintage"] = source_vintage
    primary["source_method"] = source_method

    detail = detail.merge(
        active_places[["place_fips", "place_name"]],
        on="place_fips",
        how="left",
        validate="many_to_one",
    )
    detail["state_fips"] = ND_STATE_FIPS
    detail["historical_only"] = False
    detail["source_vintage"] = source_vintage
    detail["source_method"] = source_method

    if dissolved_place_overrides:
        dissolved_rows = []
        for place_fips, payload in dissolved_place_overrides.items():
            dissolved_rows.append(
                {
                    "state_fips": ND_STATE_FIPS,
                    "place_fips": _normalize_fips(place_fips, 7),
                    "place_name": payload["place_name"],
                    "county_fips": _normalize_fips(payload["county_fips"], 5),
                    "assignment_type": "single_county",
                    "area_share": 1.0,
                    "historical_only": True,
                    "source_vintage": source_vintage,
                    "source_method": "manual_dissolved_override",
                }
            )
        dissolved_df = pd.DataFrame(dissolved_rows)
        primary = pd.concat([primary, dissolved_df], ignore_index=True)

    primary = primary[
        [
            "state_fips",
            "place_fips",
            "place_name",
            "county_fips",
            "assignment_type",
            "area_share",
            "historical_only",
            "source_vintage",
            "source_method",
        ]
    ].sort_values(["historical_only", "place_fips"], ascending=[True, True])

    detail = detail[
        [
            "state_fips",
            "place_fips",
            "place_name",
            "county_fips",
            "assignment_type",
            "area_share",
            "county_rank",
            "is_primary",
            "historical_only",
            "source_vintage",
            "source_method",
        ]
    ].sort_values(["place_fips", "county_rank", "county_fips"])

    return primary.reset_index(drop=True), detail.reset_index(drop=True)


def assign_confidence_tiers(
    population_2024: pd.DataFrame,
    high_threshold: int = 10_000,
    moderate_threshold: int = 2_500,
    lower_threshold: int = 500,
    tier_boundary_margin: float = 0.05,
) -> pd.DataFrame:
    """Assign HIGH/MODERATE/LOWER/EXCLUDED tiers from 2024 populations."""
    _require_columns(population_2024, ["place_fips", "population_2024"], context="population_2024")

    tiers = population_2024.copy()
    tiers["population_2024"] = pd.to_numeric(tiers["population_2024"], errors="coerce")
    tiers = tiers.dropna(subset=["place_fips", "population_2024"]).copy()

    def _tier(pop_value: float) -> str:
        if pop_value > high_threshold:
            return "HIGH"
        if pop_value >= moderate_threshold:
            return "MODERATE"
        if pop_value >= lower_threshold:
            return "LOWER"
        return "EXCLUDED"

    thresholds = [lower_threshold, moderate_threshold, high_threshold]

    def _is_boundary(pop_value: float) -> bool:
        return any(abs(pop_value - threshold) <= (threshold * tier_boundary_margin) for threshold in thresholds)

    tiers["confidence_tier"] = tiers["population_2024"].map(_tier)
    tiers["tier_boundary"] = tiers["population_2024"].map(_is_boundary)

    return tiers[["place_fips", "confidence_tier", "tier_boundary"]].drop_duplicates(
        subset=["place_fips"],
    )


def add_tiers_to_crosswalk(
    crosswalk: pd.DataFrame,
    tiers: pd.DataFrame,
) -> pd.DataFrame:
    """Attach confidence-tier fields to crosswalk."""
    _require_columns(crosswalk, ["place_fips"], context="crosswalk")
    _require_columns(tiers, ["place_fips", "confidence_tier", "tier_boundary"], context="tiers")

    enriched = crosswalk.merge(tiers, on="place_fips", how="left", validate="one_to_one")
    if "historical_only" in enriched.columns:
        historical_mask = enriched["historical_only"].fillna(False)
        enriched.loc[historical_mask, "confidence_tier"] = "EXCLUDED"
        enriched.loc[historical_mask, "tier_boundary"] = False
    return enriched


def validate_crosswalk(
    crosswalk: pd.DataFrame,
    expected_active_places: int | None = None,
) -> None:
    """Validate core PP3-S03 crosswalk invariants."""
    required = [
        "place_fips",
        "county_fips",
        "assignment_type",
        "area_share",
        "historical_only",
    ]
    _require_columns(crosswalk, required, context="crosswalk")

    non_historical = crosswalk[~crosswalk["historical_only"].fillna(False)].copy()
    if expected_active_places is not None and len(non_historical) != expected_active_places:
        raise ValueError(
            f"Expected {expected_active_places} active crosswalk rows, found {len(non_historical)}"
        )

    if crosswalk["place_fips"].duplicated().any():
        duplicates = crosswalk[crosswalk["place_fips"].duplicated(keep=False)]["place_fips"].unique()
        raise ValueError(f"Crosswalk has duplicate place_fips values: {sorted(duplicates)[:10]}")

    if crosswalk["place_fips"].isna().any() or crosswalk["county_fips"].isna().any():
        raise ValueError("Crosswalk contains null place_fips/county_fips values.")

    invalid_assignment = sorted(set(crosswalk["assignment_type"]) - VALID_ASSIGNMENT_TYPES)
    if invalid_assignment:
        raise ValueError(f"Invalid assignment_type values: {invalid_assignment}")

    shares = pd.to_numeric(crosswalk["area_share"], errors="coerce")
    if shares.isna().any():
        raise ValueError("Crosswalk contains non-numeric area_share values.")
    if ((shares <= 0) | (shares > 1.0)).any():
        bad = crosswalk.loc[(shares <= 0) | (shares > 1.0), ["place_fips", "area_share"]].head(5)
        raise ValueError(f"Crosswalk contains invalid area_share values:\n{bad}")


def _resolve_default_paths_from_config(config_path: Path) -> dict[str, Path]:
    """Resolve default input/output paths using project config."""
    config = load_projection_config(config_path=config_path)
    places_rel = config["geography"]["reference_data"]["places_file"]
    return {
        "places_file": PROJECT_ROOT / places_rel,
        "output_crosswalk": PROJECT_ROOT
        / "data"
        / "processed"
        / "geographic"
        / "place_county_crosswalk_2020.csv",
        "output_detail": PROJECT_ROOT
        / "data"
        / "processed"
        / "geographic"
        / "place_county_crosswalk_2020_multicounty_detail.csv",
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "projection_config.yaml",
        help="Path to projection config YAML.",
    )
    parser.add_argument(
        "--places-file",
        type=Path,
        help="Path to ND place reference CSV (defaults from config).",
    )
    parser.add_argument(
        "--relationship-file",
        type=Path,
        help="Relationship file with place/county area information.",
    )
    parser.add_argument(
        "--place-shapefile",
        type=Path,
        help="TIGER place shapefile path (used when relationship file not provided).",
    )
    parser.add_argument(
        "--county-shapefile",
        type=Path,
        help="TIGER county shapefile path (used when relationship file not provided).",
    )
    parser.add_argument(
        "--output-crosswalk",
        type=Path,
        help="Primary output crosswalk CSV path.",
    )
    parser.add_argument(
        "--output-detail",
        type=Path,
        help="Output path for multi-county detail CSV.",
    )
    parser.add_argument(
        "--source-vintage",
        default=DEFAULT_SOURCE_VINTAGE,
        help="Source vintage label written to output metadata columns.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute crosswalk build pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    defaults = _resolve_default_paths_from_config(args.config)

    places_file = args.places_file or defaults["places_file"]
    output_crosswalk = args.output_crosswalk or defaults["output_crosswalk"]
    output_detail = args.output_detail or defaults["output_detail"]

    active_places = load_active_nd_places(places_file)

    if args.relationship_file:
        overlaps = load_overlaps_from_relationship_file(args.relationship_file)
        source_method = "census_relationship_file"
    else:
        if not args.place_shapefile or not args.county_shapefile:
            raise ValueError(
                "Provide --relationship-file OR both --place-shapefile and --county-shapefile."
            )
        overlaps = load_overlaps_from_tiger_shapefiles(
            place_shapefile=args.place_shapefile,
            county_shapefile=args.county_shapefile,
        )
        source_method = "tiger_overlay"

    primary, detail = build_place_county_crosswalk(
        overlaps=overlaps,
        active_places=active_places,
        source_vintage=args.source_vintage,
        source_method=source_method,
    )
    validate_crosswalk(primary, expected_active_places=len(active_places))

    output_crosswalk.parent.mkdir(parents=True, exist_ok=True)
    output_detail.parent.mkdir(parents=True, exist_ok=True)
    primary.to_csv(output_crosswalk, index=False)
    detail.to_csv(output_detail, index=False)

    logger.info("Wrote primary crosswalk: %s (%d rows)", output_crosswalk, len(primary))
    logger.info("Wrote multicounty detail: %s (%d rows)", output_detail, len(detail))


if __name__ == "__main__":
    main()
