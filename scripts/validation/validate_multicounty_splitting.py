"""
Validate multi-county place splitting with real data (WS-B-08).

Purpose
-------
The multi-county place splitting module
(``cohort_projections/data/process/multicounty_allocation.py``) was
implemented in PP-005 but has only been tested with synthetic data.
This script validates the module against real projection outputs and
the actual TIGER-derived crosswalk weights.

Method
------
1. Load allocation weights from the real multicounty detail crosswalk.
2. For each of the 7 multicounty places, load the baseline place
   projection output (if available -- places below the 500-pop threshold
   are excluded from the projection pipeline).
3. For each year in the projection, split the place's population across
   constituent counties using ``split_multicounty_place()``.
4. Feed the split allocations into ``reaggregate_multicounty_place()``
   to recombine.
5. Verify the roundtrip invariant: reaggregated total == original total
   (within floating-point tolerance of 1e-9).
6. Report results including allocation weights, which places were
   validated, and pass/fail status.

Inputs
------
- ``data/processed/geographic/place_county_crosswalk_2020.csv``
- ``data/processed/geographic/place_county_crosswalk_2020_multicounty_detail.csv``
- ``data/projections/baseline/place/nd_place_{fips}_projection_2025_2055_baseline.parquet``
- ``config/projection_config.yaml``

Outputs
-------
- ``data/backtesting/multicounty_validation_results.txt``

Author: Claude / N. Haarstad
Created: 2026-03-01
Task: WS-B-08 (PP-005 deferred validation)
ADR: 058 (Multi-county place splitting)
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Resolve project root so imports work when running from any directory.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cohort_projections.data.process.multicounty_allocation import (  # noqa: E402
    identify_multicounty_places,
    load_allocation_weights,
    reaggregate_multicounty_place,
    split_multicounty_place,
)

# ---------------------------------------------------------------------------
# Paths (all relative to PROJECT_ROOT).
# ---------------------------------------------------------------------------
CROSSWALK_PATH = PROJECT_ROOT / "data" / "processed" / "geographic" / "place_county_crosswalk_2020.csv"
MULTICOUNTY_DETAIL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "geographic"
    / "place_county_crosswalk_2020_multicounty_detail.csv"
)
PLACE_PROJECTION_DIR = PROJECT_ROOT / "data" / "projections" / "baseline" / "place"
OUTPUT_PATH = PROJECT_ROOT / "data" / "backtesting" / "multicounty_validation_results.txt"

# Roundtrip tolerance (absolute).  The module uses remainder-assignment on the
# last county, so the invariant should hold to machine precision.
TOLERANCE = 1e-9


def _load_crosswalk() -> pd.DataFrame:
    """Load the primary crosswalk with string FIPS."""
    return pd.read_csv(CROSSWALK_PATH, dtype=str)


def _load_place_projection(place_fips: str) -> pd.DataFrame | None:
    """Load a single place projection parquet, or None if not found."""
    pattern = f"nd_place_{place_fips}_projection_2025_2055_baseline.parquet"
    path = PLACE_PROJECTION_DIR / pattern
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _place_name_lookup(crosswalk: pd.DataFrame) -> dict[str, str]:
    """Return {place_fips: place_name} from primary crosswalk."""
    return dict(zip(crosswalk["place_fips"], crosswalk["place_name"], strict=False))


def _county_name_from_detail(detail_path: Path) -> dict[str, str]:
    """Return {county_fips: county_fips} (we only have FIPS in detail)."""
    # The detail CSV does not carry county names, so just return FIPS.
    return {}


def run_validation() -> str:
    """Execute the full validation and return the report as a string."""
    buf = StringIO()

    def p(text: str = "") -> None:
        buf.write(text + "\n")

    timestamp = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    p("=" * 72)
    p("WS-B-08: Multi-County Place Splitting Validation")
    p(f"Timestamp: {timestamp}")
    p("=" * 72)
    p()

    # ------------------------------------------------------------------
    # 1. Load crosswalk and weights.
    # ------------------------------------------------------------------
    p("--- Step 1: Load crosswalk and allocation weights ---")
    crosswalk = _load_crosswalk()
    p(f"Primary crosswalk loaded: {len(crosswalk)} rows")

    multicounty_fips = identify_multicounty_places(crosswalk)
    p(f"Multi-county places identified: {len(multicounty_fips)}")
    for fips in multicounty_fips:
        name = _place_name_lookup(crosswalk).get(fips, "UNKNOWN")
        primary_county = crosswalk.loc[
            crosswalk["place_fips"] == fips, "county_fips"
        ].iloc[0]
        tier = crosswalk.loc[
            crosswalk["place_fips"] == fips, "confidence_tier"
        ].iloc[0]
        p(f"  {fips}: {name} (primary county: {primary_county}, tier: {tier})")
    p()

    weights = load_allocation_weights(CROSSWALK_PATH, MULTICOUNTY_DETAIL_PATH)
    p(f"Allocation weights loaded for {len(weights)} places.")
    p()

    # ------------------------------------------------------------------
    # 2. Report allocation weights.
    # ------------------------------------------------------------------
    p("--- Step 2: Allocation weights ---")
    for place_fips, county_weights in sorted(weights.items()):
        name = _place_name_lookup(crosswalk).get(place_fips, "UNKNOWN")
        p(f"  {place_fips} ({name}):")
        weight_total = 0.0
        for county_fips, w in sorted(county_weights.items(), key=lambda x: -x[1]):
            p(f"    {county_fips}: {w:.6f} ({w*100:.4f}%)")
            weight_total += w
        p(f"    SUM: {weight_total:.10f}")
        weights_sum_ok = abs(weight_total - 1.0) < 1e-9
        p(f"    Weights sum to 1.0: {'PASS' if weights_sum_ok else 'FAIL'}")
        p()

    # ------------------------------------------------------------------
    # 3. Load projections and validate roundtrip invariant.
    # ------------------------------------------------------------------
    p("--- Step 3: Roundtrip invariant validation (split -> reaggregate) ---")
    p()

    total_tests = 0
    total_passed = 0
    total_failed = 0
    places_with_projections = 0
    places_without_projections = 0
    place_results: list[dict] = []

    for place_fips in sorted(multicounty_fips):
        name = _place_name_lookup(crosswalk).get(place_fips, "UNKNOWN")
        tier = crosswalk.loc[
            crosswalk["place_fips"] == place_fips, "confidence_tier"
        ].iloc[0]

        proj_df = _load_place_projection(place_fips)
        if proj_df is None:
            p(f"  {place_fips} ({name}): NO PROJECTION FILE (tier={tier})")
            p("    Skipped -- place is below 500-pop threshold (EXCLUDED tier).")
            places_without_projections += 1
            p()
            continue

        places_with_projections += 1
        p(f"  {place_fips} ({name}): projection loaded ({len(proj_df)} years, tier={tier})")

        year_pass = 0
        year_fail = 0
        max_error = 0.0
        errors_by_year: list[tuple[int, float, float, float]] = []

        for _, row in proj_df.iterrows():
            year = int(row["year"])
            original_pop = float(row["population"])

            # --- Split ---
            allocations = split_multicounty_place(place_fips, original_pop, weights)

            # Verify split allocations sum to original.
            split_sum = sum(allocations.values())
            split_error = abs(split_sum - original_pop)
            total_tests += 1

            if split_error > TOLERANCE:
                year_fail += 1
                total_failed += 1
                errors_by_year.append((year, original_pop, split_sum, split_error))
                max_error = max(max_error, split_error)
                continue

            # --- Reaggregate ---
            # Build the county_projections dict in the format expected by
            # reaggregate_multicounty_place: each county needs a DataFrame
            # with year, place_fips, and projected_population columns.
            county_projections: dict[str, pd.DataFrame] = {}
            for county_fips, county_pop in allocations.items():
                county_projections[county_fips] = pd.DataFrame(
                    {
                        "year": [year],
                        "place_fips": [place_fips],
                        "projected_population": [county_pop],
                    }
                )

            reagg_df = reaggregate_multicounty_place(
                county_projections=county_projections,
                place_fips=place_fips,
                weights=weights,
            )
            reagg_pop = float(reagg_df["place_total"].iloc[0])
            roundtrip_error = abs(reagg_pop - original_pop)

            if roundtrip_error <= TOLERANCE:
                year_pass += 1
                total_passed += 1
            else:
                year_fail += 1
                total_failed += 1
                errors_by_year.append((year, original_pop, reagg_pop, roundtrip_error))
                max_error = max(max_error, roundtrip_error)

        status = "PASS" if year_fail == 0 else "FAIL"
        p(f"    Roundtrip invariant: {status} ({year_pass}/{year_pass + year_fail} years)")
        if max_error > 0:
            p(f"    Max roundtrip error: {max_error:.2e}")
        else:
            p("    Max roundtrip error: 0 (exact)")

        # Show allocation for base year.
        base_row = proj_df[proj_df["year"] == proj_df["year"].min()].iloc[0]
        base_pop = float(base_row["population"])
        base_alloc = split_multicounty_place(place_fips, base_pop, weights)
        p(f"    Base year ({int(base_row['year'])}) allocation:")
        for cfips, cpop in sorted(base_alloc.items(), key=lambda x: -x[1]):
            pct = cpop / base_pop * 100 if base_pop > 0 else 0
            p(f"      {cfips}: {cpop:>10.2f} ({pct:>6.2f}%)")
        p(f"      TOTAL: {sum(base_alloc.values()):>10.2f}")

        # Show allocation for final year.
        final_row = proj_df[proj_df["year"] == proj_df["year"].max()].iloc[0]
        final_pop = float(final_row["population"])
        final_alloc = split_multicounty_place(place_fips, final_pop, weights)
        p(f"    Final year ({int(final_row['year'])}) allocation:")
        for cfips, cpop in sorted(final_alloc.items(), key=lambda x: -x[1]):
            pct = cpop / final_pop * 100 if final_pop > 0 else 0
            p(f"      {cfips}: {cpop:>10.2f} ({pct:>6.2f}%)")
        p(f"      TOTAL: {sum(final_alloc.values()):>10.2f}")

        if errors_by_year:
            p("    FAILED YEARS:")
            for yr, orig, got, err in errors_by_year:
                p(f"      {yr}: original={orig:.6f}, got={got:.6f}, error={err:.2e}")

        place_results.append(
            {
                "place_fips": place_fips,
                "name": name,
                "tier": tier,
                "status": status,
                "years_tested": year_pass + year_fail,
                "years_passed": year_pass,
                "years_failed": year_fail,
                "max_error": max_error,
            }
        )
        p()

    # ------------------------------------------------------------------
    # 4. Additional validation: weight normalization.
    # ------------------------------------------------------------------
    p("--- Step 4: Weight normalization validation ---")
    weight_issues = 0
    for place_fips, county_weights in sorted(weights.items()):
        w_sum = sum(county_weights.values())
        if abs(w_sum - 1.0) > 1e-9:
            name = _place_name_lookup(crosswalk).get(place_fips, "UNKNOWN")
            p(f"  FAIL: {place_fips} ({name}) weights sum to {w_sum:.10f} (expected 1.0)")
            weight_issues += 1
    if weight_issues == 0:
        p(f"  All {len(weights)} places have properly normalized weights: PASS")
    p()

    # ------------------------------------------------------------------
    # 5. Additional validation: all weights are non-negative.
    # ------------------------------------------------------------------
    p("--- Step 5: Non-negative weight validation ---")
    neg_weight_issues = 0
    for place_fips, county_weights in sorted(weights.items()):
        for county_fips, w in county_weights.items():
            if w < 0:
                p(f"  FAIL: {place_fips} county {county_fips} has negative weight {w}")
                neg_weight_issues += 1
    if neg_weight_issues == 0:
        p("  All weights are non-negative: PASS")
    p()

    # ------------------------------------------------------------------
    # 6. Additional validation: split population is non-negative for all years.
    # ------------------------------------------------------------------
    p("--- Step 6: Non-negative allocation validation ---")
    neg_alloc_issues = 0
    for place_fips in sorted(multicounty_fips):
        proj_df = _load_place_projection(place_fips)
        if proj_df is None:
            continue
        for _, row in proj_df.iterrows():
            pop = float(row["population"])
            alloc = split_multicounty_place(place_fips, pop, weights)
            for cfips, cpop in alloc.items():
                if cpop < -TOLERANCE:
                    p(f"  FAIL: {place_fips} year {int(row['year'])} county {cfips}: {cpop:.6f}")
                    neg_alloc_issues += 1
    if neg_alloc_issues == 0:
        p("  All county allocations are non-negative: PASS")
    p()

    # ------------------------------------------------------------------
    # 7. Cross-check: weights match detail CSV values.
    # ------------------------------------------------------------------
    p("--- Step 7: Weights match detail CSV values ---")
    detail_df = pd.read_csv(MULTICOUNTY_DETAIL_PATH, dtype=str)
    detail_df["area_share"] = pd.to_numeric(detail_df["area_share"], errors="coerce")
    weight_mismatch = 0
    for _, drow in detail_df.iterrows():
        pfips = str(drow["place_fips"]).zfill(7)[-7:]
        cfips = str(drow["county_fips"]).zfill(5)[-5:]
        raw_share = float(drow["area_share"])

        if pfips not in weights:
            p(f"  WARN: {pfips} in detail CSV but not in loaded weights")
            continue

        # Compute expected normalized weight.
        place_detail = detail_df[
            detail_df["place_fips"].apply(lambda v: str(v).zfill(7)[-7:]) == pfips
        ]
        total_raw = place_detail["area_share"].sum()
        expected_weight = raw_share / total_raw

        if cfips in weights[pfips]:
            actual_weight = weights[pfips][cfips]
            if abs(actual_weight - expected_weight) > 1e-6:
                p(
                    f"  MISMATCH: {pfips}/{cfips} expected={expected_weight:.8f}, "
                    f"got={actual_weight:.8f}"
                )
                weight_mismatch += 1
        else:
            p(f"  MISSING: {pfips} missing county {cfips} in loaded weights")
            weight_mismatch += 1

    if weight_mismatch == 0:
        p("  All loaded weights match detail CSV after normalization: PASS")
    p()

    # ------------------------------------------------------------------
    # Summary.
    # ------------------------------------------------------------------
    p("=" * 72)
    p("SUMMARY")
    p("=" * 72)
    p(f"Multi-county places in crosswalk:       {len(multicounty_fips)}")
    p(f"Places with projection data:            {places_with_projections}")
    p(f"Places without projection data (excl.): {places_without_projections}")
    p(f"Total year-level roundtrip tests:       {total_tests}")
    p(f"Passed:                                 {total_passed}")
    p(f"Failed:                                 {total_failed}")
    p(f"Weight normalization issues:            {weight_issues}")
    p(f"Negative weight issues:                 {neg_weight_issues}")
    p(f"Negative allocation issues:             {neg_alloc_issues}")
    p(f"Weight/detail CSV mismatches:           {weight_mismatch}")
    p()

    all_pass = (
        total_failed == 0
        and weight_issues == 0
        and neg_weight_issues == 0
        and neg_alloc_issues == 0
        and weight_mismatch == 0
    )
    if all_pass:
        p("OVERALL RESULT: PASS")
        p()
        p(
            "The multi-county splitting module correctly splits and reaggregates\n"
            "population for all available multicounty place projections.\n"
            "The roundtrip invariant (split -> reaggregate == original) holds\n"
            "to machine precision for all tested years and places."
        )
    else:
        p("OVERALL RESULT: FAIL")
        p()
        p("See details above for specific failures.")

    p()
    p(f"Report generated: {timestamp}")
    p("Script: scripts/validation/validate_multicounty_splitting.py")
    p("Module: cohort_projections/data/process/multicounty_allocation.py")
    p("ADR: 058 (Multi-county place splitting)")
    p("Task: WS-B-08 (PP-005 deferred validation)")

    return buf.getvalue()


def main() -> None:
    """Run validation and write report to disk."""
    report = run_validation()

    # Print to stdout.
    print(report)

    # Write to file.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    print(f"\nReport written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
