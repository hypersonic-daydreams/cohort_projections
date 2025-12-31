#!/usr/bin/env python3
"""
Distance Analysis for Gravity Model
====================================

Calculates great-circle distances from origin countries to North Dakota
and re-estimates the gravity model with distance as a covariate.

This is an exploratory analysis for Task P3.19 of the Phase 3 revision plan.

Usage:
    python distance_analysis.py
"""

import json
import sys
from datetime import UTC, datetime
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.families import Poisson
from statsmodels.genmod.generalized_linear_model import GLM

# Project paths
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
DISTANCE_DIR = RESULTS_DIR / "distance_analysis"
DATA_DIR = (
    SCRIPT_DIR.parent.parent.parent / "data" / "processed" / "immigration" / "analysis"
)

# Ensure output directory exists
DISTANCE_DIR.mkdir(exist_ok=True)

# North Dakota geographic centroid (approximate)
# Source: US Census Bureau geographic centers
ND_CENTROID = {
    "lat": 47.5515,
    "lon": -100.0659,
    "name": "North Dakota",
}

# Country centroids (lat, lon)
# Sources: Natural Earth, CEPII GeoDist database, and standard geographic references
# These are approximate geographic centroids, not population-weighted
COUNTRY_CENTROIDS = {
    # Major sending countries to ND
    "Afghanistan": (33.9391, 67.7100),
    "Albania": (41.1533, 20.1683),
    "Armenia": (40.0691, 45.0382),
    "Australia": (-25.2744, 133.7751),
    "Austria": (47.5162, 14.5501),
    "Bangladesh": (23.6850, 90.3563),
    "Bosnia and Herzegovina": (43.9159, 17.6791),
    "Bulgaria": (42.7339, 25.4858),
    "Burma": (21.9162, 95.9560),  # Myanmar
    "Cambodia": (12.5657, 104.9910),
    "Cameroon": (7.3697, 12.3547),
    "Canada": (56.1304, -106.3468),
    "Cape Verde": (16.5388, -23.0418),
    "China": (35.8617, 104.1954),
    "Croatia": (45.1000, 15.2000),
    "Czechoslovakia (includes Czech Republic and Slovakia)": (
        49.8175,
        15.4730,
    ),  # Czech Republic centroid
    "Denmark": (56.2639, 9.5018),
    "Egypt": (26.8206, 30.8025),
    "Eritrea": (15.1794, 39.7823),
    "Ethiopia": (9.1450, 40.4897),
    "France": (46.2276, 2.2137),
    "Germany": (51.1657, 10.4515),
    "Ghana": (7.9465, -1.0232),
    "Greece": (39.0742, 21.8243),
    "Hungary": (47.1625, 19.5033),
    "India": (20.5937, 78.9629),
    "Indonesia": (-0.7893, 113.9213),
    "Iran": (32.4279, 53.6880),
    "Iraq": (33.2232, 43.6793),
    "Ireland": (53.1424, -7.6921),
    "Israel": (31.0461, 34.8516),
    "Italy": (41.8719, 12.5674),
    "Japan": (36.2048, 138.2529),
    "Jordan": (30.5852, 36.2384),
    "Kazakhstan": (48.0196, 66.9237),
    "Kenya": (-0.0236, 37.9062),
    "Korea": (35.9078, 127.7669),  # South Korea
    "Kuwait": (29.3117, 47.4818),
    "Laos": (19.8563, 102.4955),
    "Latvia": (56.8796, 24.6032),
    "Lebanon": (33.8547, 35.8623),
    "Liberia": (6.4281, -9.4295),
    "Lithuania": (55.1694, 23.8813),
    "Macedonia": (41.5124, 21.7453),
    "Malaysia": (4.2105, 101.9758),
    "Moldova": (47.4116, 28.3699),
    "Morocco": (31.7917, -7.0926),
    "Nepal": (28.3949, 84.1240),
    "Nigeria": (9.0820, 8.6753),
    "Norway": (60.4720, 8.4689),
    "Pakistan": (30.3753, 69.3451),
    "Philippines": (12.8797, 121.7740),
    "Poland": (51.9194, 19.1451),
    "Romania": (45.9432, 24.9668),
    "Russia": (61.5240, 105.3188),
    "Saudi Arabia": (23.8859, 45.0792),
    "Sierra Leone": (8.4606, -11.7799),
    "Singapore": (1.3521, 103.8198),
    "South Africa": (-30.5595, 22.9375),
    "Spain": (40.4637, -3.7492),
    "Sri Lanka": (7.8731, 80.7718),
    "Sudan": (12.8628, 30.2176),
    "Sweden": (60.1282, 18.6435),
    "Syria": (34.8021, 38.9968),
    "Thailand": (15.8700, 100.9925),
    "Turkey": (38.9637, 35.2433),
    "Ukraine": (48.3794, 31.1656),
    "United Kingdom (inc. Crown Dependencies)": (55.3781, -3.4360),
    "Uzbekistan": (41.3775, 64.5853),
    "Vietnam": (14.0583, 108.2772),
    "Yemen": (15.5527, 48.5164),
    "Yugoslavia": (44.0165, 21.0059),  # Serbia centroid as proxy
    # Regional aggregates (use representative centroids)
    "Caribbean": (18.1096, -77.2975),  # Jamaica as proxy
    "Central America": (14.6349, -90.5069),  # Guatemala as proxy
    "South America": (-14.2350, -51.9253),  # Brazil as proxy
    "Other Eastern Africa": (-6.3690, 34.8888),  # Tanzania as proxy
    "Other Eastern Asia": (36.2048, 138.2529),  # Japan as proxy
    "Other Eastern Europe": (52.2297, 21.0122),  # Poland as proxy
    "Other Middle Africa": (-4.0383, 21.7587),  # DRC as proxy
    "Other Northern Africa": (28.0339, 1.6596),  # Algeria as proxy
    "Other Northern America": (56.1304, -106.3468),  # Canada
    "Other South Central Asia": (28.3949, 84.1240),  # Nepal as proxy
    "Other South Eastern Asia": (15.8700, 100.9925),  # Thailand as proxy
    "Other Southern Africa": (-22.3285, 24.6849),  # Botswana as proxy
    "Other Southern Europe": (41.8719, 12.5674),  # Italy as proxy
    "Other Western Africa": (7.5400, -5.5471),  # Ivory Coast as proxy
    "Other Western Asia": (33.2232, 43.6793),  # Iraq as proxy
    "Other Australian and New Zealand Subregion": (-25.2744, 133.7751),  # Australia
    # Additional countries that may appear in the data
    "Mexico": (23.6345, -102.5528),
    "Bhutan": (27.5142, 90.4336),
    "Myanmar": (21.9162, 95.9560),
    "Somalia": (5.1521, 46.1996),
    "Democratic Republic of the Congo": (-4.0383, 21.7587),
    "Congo, Democratic Republic": (-4.0383, 21.7587),
    "Congo, Republic": (-0.2280, 15.8277),
    "Czech Republic": (49.8175, 15.4730),
    "South Korea": (35.9078, 127.7669),
    "Korea, South": (35.9078, 127.7669),
    "Burma (Myanmar)": (21.9162, 95.9560),
    "United Kingdom": (55.3781, -3.4360),
    "Russia (Russian Federation)": (61.5240, 105.3188),
    "China, People's Republic": (35.8617, 104.1954),
    "Serbia and Montenegro": (44.0165, 21.0059),
    # Additional European countries (added to improve coverage)
    "Belarus": (53.7098, 27.9534),
    "Belgium": (50.5039, 4.4699),
    "Netherlands": (52.1326, 5.2913),
    "Portugal": (39.3999, -8.2245),
    "Switzerland": (46.8182, 8.2275),
    "Other Northern Europe": (60.1282, 18.6435),  # Sweden as proxy
    "Other Western Europe": (50.5039, 4.4699),  # Belgium as proxy
}


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on Earth.

    Uses the Haversine formula:
    d = 2r * arcsin(sqrt(sin^2((lat2-lat1)/2) + cos(lat1)*cos(lat2)*sin^2((lon2-lon1)/2)))

    Args:
        lat1, lon1: Latitude and longitude of point 1 (degrees)
        lat2, lon2: Latitude and longitude of point 2 (degrees)

    Returns:
        Distance in kilometers
    """
    # Earth's radius in kilometers
    R = 6371.0

    # Convert to radians
    lat1_r = radians(lat1)
    lat2_r = radians(lat2)
    lon1_r = radians(lon1)
    lon2_r = radians(lon2)

    # Differences
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    # Haversine formula
    a = sin(dlat / 2) ** 2 + cos(lat1_r) * cos(lat2_r) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))

    return R * c


def calculate_distances_to_nd() -> pd.DataFrame:
    """
    Calculate distances from all country centroids to North Dakota.

    Returns:
        DataFrame with country, lat, lon, distance_km, log_distance
    """
    print("\n" + "=" * 70)
    print("CALCULATING DISTANCES TO NORTH DAKOTA")
    print("=" * 70)
    print(
        f"North Dakota centroid: ({ND_CENTROID['lat']:.4f}, {ND_CENTROID['lon']:.4f})"
    )

    distances = []

    for country, (lat, lon) in COUNTRY_CENTROIDS.items():
        dist_km = haversine_distance(lat, lon, ND_CENTROID["lat"], ND_CENTROID["lon"])
        distances.append(
            {
                "country": country,
                "latitude": lat,
                "longitude": lon,
                "distance_km": dist_km,
                "log_distance": np.log(dist_km),
            }
        )

    df = pd.DataFrame(distances)
    df = df.sort_values("distance_km")

    print(f"\nCalculated distances for {len(df)} countries")
    print("\nNearest 10 countries:")
    for _, row in df.head(10).iterrows():
        print(f"  {row['country']}: {row['distance_km']:.0f} km")

    print("\nFarthest 10 countries:")
    for _, row in df.tail(10).iterrows():
        print(f"  {row['country']}: {row['distance_km']:.0f} km")

    return df


def load_gravity_data() -> pd.DataFrame:
    """Load and prepare the gravity model data."""
    print("\n" + "=" * 70)
    print("LOADING GRAVITY MODEL DATA")
    print("=" * 70)

    # Load DHS LPR data
    dhs_lpr = pd.read_parquet(DATA_DIR / "dhs_lpr_by_state_country.parquet")
    print(f"DHS LPR data: {len(dhs_lpr):,} rows")

    # Load ACS foreign-born data
    acs_origin = pd.read_parquet(DATA_DIR / "acs_foreign_born_by_state_origin.parquet")
    print(f"ACS origin data: {len(acs_origin):,} rows")

    # Filter to country-level data
    dhs_countries = dhs_lpr[~dhs_lpr["is_region"]].copy()
    dhs_countries = dhs_countries[dhs_countries["region_country_of_birth"] != "Total"]

    # Get ACS 2023 country-level
    acs_2023 = acs_origin[
        (acs_origin["year"] == 2023) & (acs_origin["level"] == "country")
    ].copy()

    # Get state totals
    state_totals = acs_origin[
        (acs_origin["year"] == 2023) & (acs_origin["level"] == "total")
    ][["state_name", "foreign_born_pop"]].copy()
    state_totals.columns = ["state", "state_foreign_born_total"]

    # Country name harmonization
    country_mapping = {
        "China": "China, People's Republic",
        "Korea": "Korea, South",
        "Burma": "Burma (Myanmar)",
        "Czechoslovakia (includes Czech Republic and Slovakia)": "Czech Republic",
        "United Kingdom (inc. Crown Dependencies)": "United Kingdom",
        "Russia": "Russia (Russian Federation)",
        "Serbia": "Serbia and Montenegro",
        "Congo (Kinshasa)": "Congo, Democratic Republic",
        "Congo (Brazzaville)": "Congo, Republic",
    }
    reverse_mapping = {v: k for k, v in country_mapping.items()}

    dhs_countries["country_std"] = dhs_countries["region_country_of_birth"].replace(
        reverse_mapping
    )
    acs_2023["country_std"] = acs_2023["country"]

    # Build origin-state grid
    states = sorted(acs_2023["state_name"].unique())
    origins = sorted(acs_2023["country_std"].unique())
    od_grid = (
        pd.MultiIndex.from_product([states, origins], names=["state", "country_std"])
        .to_frame(index=False)
        .reset_index(drop=True)
    )

    # Merge flows
    dhs_flows = dhs_countries[
        ["state", "country_std", "lpr_count", "region_country_of_birth"]
    ].copy()
    gravity_df = od_grid.merge(dhs_flows, on=["state", "country_std"], how="left")
    gravity_df = gravity_df.rename(
        columns={
            "lpr_count": "flow",
            "region_country_of_birth": "origin_country",
        }
    )
    gravity_df["flow"] = gravity_df["flow"].fillna(0)
    gravity_df["origin_country"] = gravity_df["origin_country"].fillna(
        gravity_df["country_std"]
    )

    # Merge diaspora stock
    acs_2023 = acs_2023.rename(columns={"state_name": "state"})
    gravity_df = gravity_df.merge(
        acs_2023[["state", "country_std", "foreign_born_pop"]],
        on=["state", "country_std"],
        how="left",
    )
    gravity_df = gravity_df.rename(columns={"foreign_born_pop": "diaspora_stock"})
    gravity_df["diaspora_stock"] = gravity_df["diaspora_stock"].fillna(0)

    # Add state totals
    gravity_df = gravity_df.merge(state_totals, on="state", how="left")

    # Add national totals by origin
    national_by_origin = (
        acs_2023.groupby("country_std")["foreign_born_pop"].sum().reset_index()
    )
    national_by_origin.columns = ["country_std", "national_origin_total"]
    gravity_df = gravity_df.merge(national_by_origin, on="country_std", how="left")

    # Filter valid data
    gravity_df = gravity_df.dropna(subset=["diaspora_stock"])

    # Create log variables
    gravity_df["log_flow"] = np.log(gravity_df["flow"] + 1)
    gravity_df["log_diaspora"] = np.log(gravity_df["diaspora_stock"] + 1)
    gravity_df["log_state_total"] = np.log(gravity_df["state_foreign_born_total"] + 1)
    gravity_df["log_origin_total"] = np.log(gravity_df["national_origin_total"] + 1)

    print(f"\nGravity data prepared: {len(gravity_df):,} observations")
    print(f"  Unique states: {gravity_df['state'].nunique()}")
    print(f"  Unique origins: {gravity_df['country_std'].nunique()}")

    return gravity_df


def merge_distances(
    gravity_df: pd.DataFrame, distance_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge distance data with gravity data."""
    print("\n" + "=" * 70)
    print("MERGING DISTANCE DATA")
    print("=" * 70)

    # Create a mapping for country names to match
    distance_df = distance_df.copy()
    distance_df["country_match"] = distance_df["country"]

    # Merge on country_std
    merged = gravity_df.merge(
        distance_df[
            ["country", "latitude", "longitude", "distance_km", "log_distance"]
        ],
        left_on="country_std",
        right_on="country",
        how="left",
    )

    # Check for missing distances
    missing = merged[merged["distance_km"].isna()]["country_std"].unique()
    if len(missing) > 0:
        print(f"\nWARNING: {len(missing)} countries missing distance data:")
        for c in missing[:20]:  # Show first 20
            print(f"  - {c}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")

    # Keep only rows with distance data
    valid_merged = merged.dropna(subset=["distance_km"])

    print(f"\nMerged data: {len(valid_merged):,} observations")
    print(f"  Coverage: {len(valid_merged) / len(gravity_df) * 100:.1f}%")

    return valid_merged


def estimate_gravity_models(df: pd.DataFrame) -> dict:
    """
    Estimate gravity models with and without distance.

    Compares:
    1. Model without distance (baseline)
    2. Model with log(distance)
    """
    print("\n" + "=" * 70)
    print("ESTIMATING GRAVITY MODELS")
    print("=" * 70)

    # Clean data
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(
        subset=[
            "flow",
            "diaspora_stock",
            "state_foreign_born_total",
            "national_origin_total",
            "distance_km",
        ]
    )

    results = {}

    # Model 1: Without distance (replication of existing model)
    print("\n" + "-" * 70)
    print("MODEL 1: FULL GRAVITY (No Distance)")
    print("Flow ~ log(Diaspora) + log(Origin_Mass) + log(Dest_Mass)")
    print("-" * 70)

    y = df["flow"]
    X_no_dist = sm.add_constant(
        df[["log_diaspora", "log_origin_total", "log_state_total"]]
    )

    model_no_dist = GLM(y, X_no_dist, family=Poisson())
    result_no_dist = model_no_dist.fit()

    print(result_no_dist.summary())

    # Null model for pseudo R2
    null_model = GLM(y, np.ones(len(y)), family=Poisson())
    null_result = null_model.fit()
    pseudo_r2_no_dist = 1 - (result_no_dist.llf / null_result.llf)

    results["model_without_distance"] = {
        "specification": "Flow ~ log(Diaspora) + log(Origin_Mass) + log(Dest_Mass)",
        "n_observations": int(len(df)),
        "coefficients": {
            var: {
                "estimate": float(result_no_dist.params[var]),
                "std_error": float(result_no_dist.bse[var]),
                "z_statistic": float(result_no_dist.tvalues[var]),
                "p_value": float(result_no_dist.pvalues[var]),
            }
            for var in result_no_dist.params.index
        },
        "fit_statistics": {
            "log_likelihood": float(result_no_dist.llf),
            "aic": float(result_no_dist.aic),
            "bic": float(result_no_dist.bic),
            "pseudo_r2_mcfadden": float(pseudo_r2_no_dist),
            "deviance": float(result_no_dist.deviance),
        },
    }

    # Model 2: With distance
    print("\n" + "-" * 70)
    print("MODEL 2: FULL GRAVITY WITH DISTANCE")
    print("Flow ~ log(Diaspora) + log(Origin_Mass) + log(Dest_Mass) + log(Distance)")
    print("-" * 70)

    X_with_dist = sm.add_constant(
        df[["log_diaspora", "log_origin_total", "log_state_total", "log_distance"]]
    )

    model_with_dist = GLM(y, X_with_dist, family=Poisson())
    result_with_dist = model_with_dist.fit()

    print(result_with_dist.summary())

    pseudo_r2_with_dist = 1 - (result_with_dist.llf / null_result.llf)

    results["model_with_distance"] = {
        "specification": "Flow ~ log(Diaspora) + log(Origin_Mass) + log(Dest_Mass) + log(Distance)",
        "n_observations": int(len(df)),
        "coefficients": {
            var: {
                "estimate": float(result_with_dist.params[var]),
                "std_error": float(result_with_dist.bse[var]),
                "z_statistic": float(result_with_dist.tvalues[var]),
                "p_value": float(result_with_dist.pvalues[var]),
            }
            for var in result_with_dist.params.index
        },
        "fit_statistics": {
            "log_likelihood": float(result_with_dist.llf),
            "aic": float(result_with_dist.aic),
            "bic": float(result_with_dist.bic),
            "pseudo_r2_mcfadden": float(pseudo_r2_with_dist),
            "deviance": float(result_with_dist.deviance),
        },
    }

    # Model comparison
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    dist_coef = result_with_dist.params.get("log_distance", None)
    dist_se = result_with_dist.bse.get("log_distance", None)
    dist_p = result_with_dist.pvalues.get("log_distance", None)

    print(f"\n{'Metric':<30} {'Without Distance':>18} {'With Distance':>18}")
    print("-" * 70)
    print(f"{'AIC':<30} {result_no_dist.aic:>18.2f} {result_with_dist.aic:>18.2f}")
    print(f"{'BIC':<30} {result_no_dist.bic:>18.2f} {result_with_dist.bic:>18.2f}")
    print(f"{'Pseudo R2':<30} {pseudo_r2_no_dist:>18.4f} {pseudo_r2_with_dist:>18.4f}")
    print(
        f"{'Log-Likelihood':<30} {result_no_dist.llf:>18.2f} {result_with_dist.llf:>18.2f}"
    )

    if dist_coef is not None:
        print(f"\n{'Distance Coefficient':<30} {'N/A':>18} {dist_coef:>18.4f}")
        print(f"{'Distance SE':<30} {'N/A':>18} {dist_se:>18.4f}")
        print(f"{'Distance p-value':<30} {'N/A':>18} {dist_p:>18.6f}")

    # Interpretation
    print("\n" + "-" * 70)
    print("INTERPRETATION")
    print("-" * 70)

    if dist_coef is not None:
        if dist_p < 0.05:
            if dist_coef < 0:
                print(
                    "Distance has a significant NEGATIVE effect on immigration flows."
                )
                print(
                    f"A 1% increase in distance is associated with a {dist_coef:.4f}% decrease in flows."
                )
                print("This is consistent with standard gravity model predictions.")
            else:
                print(
                    "Distance has a significant POSITIVE effect on immigration flows."
                )
                print(
                    f"A 1% increase in distance is associated with a {dist_coef:.4f}% increase in flows."
                )
                print(
                    "This is UNEXPECTED - may indicate omitted variable bias or selection effects."
                )
        else:
            print(
                f"Distance does NOT have a statistically significant effect (p={dist_p:.4f})."
            )
            print("This may indicate:")
            print(
                "  - Distance matters less for specific migration streams (refugee vs. economic)"
            )
            print("  - Network effects dominate distance effects")
            print("  - Distance is confounded with other origin characteristics")

    # Network elasticity comparison
    diaspora_no_dist = result_no_dist.params.get("log_diaspora", None)
    diaspora_with_dist = result_with_dist.params.get("log_diaspora", None)

    if diaspora_no_dist is not None and diaspora_with_dist is not None:
        print("\nNetwork Elasticity Comparison:")
        print(f"  Without distance: {diaspora_no_dist:.4f}")
        print(f"  With distance: {diaspora_with_dist:.4f}")
        change = (diaspora_with_dist - diaspora_no_dist) / diaspora_no_dist * 100
        print(f"  Change: {change:+.1f}%")

    results["comparison"] = {
        "aic_improvement": float(result_no_dist.aic - result_with_dist.aic),
        "bic_improvement": float(result_no_dist.bic - result_with_dist.bic),
        "pseudo_r2_improvement": float(pseudo_r2_with_dist - pseudo_r2_no_dist),
        "distance_coefficient": float(dist_coef) if dist_coef is not None else None,
        "distance_p_value": float(dist_p) if dist_p is not None else None,
        "distance_significant_at_05": bool(dist_p < 0.05)
        if dist_p is not None
        else None,
        "network_elasticity_without_distance": float(diaspora_no_dist)
        if diaspora_no_dist is not None
        else None,
        "network_elasticity_with_distance": float(diaspora_with_dist)
        if diaspora_with_dist is not None
        else None,
    }

    return results


def create_summary_report(distance_df: pd.DataFrame, model_results: dict) -> str:
    """Create a markdown summary of the analysis."""

    comparison = model_results.get("comparison", {})
    model_with = model_results.get("model_with_distance", {})
    model_without = model_results.get("model_without_distance", {})

    dist_coef = comparison.get("distance_coefficient")
    dist_p = comparison.get("distance_p_value")

    report = f"""# Distance Analysis for North Dakota Immigration Gravity Model

**Generated:** {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}

## Overview

This exploratory analysis adds geographic distance as a covariate to the gravity model
of immigration flows to US states. Traditional gravity models predict that migration
flows decrease with distance (negative coefficient on log distance).

## Data

### North Dakota Centroid
- Latitude: {ND_CENTROID['lat']:.4f}
- Longitude: {ND_CENTROID['lon']:.4f}

### Distance Coverage
- Countries with centroid data: {len(distance_df)}
- Distance range: {distance_df['distance_km'].min():.0f} km to {distance_df['distance_km'].max():.0f} km

### Nearest Countries to North Dakota
| Country | Distance (km) |
|---------|--------------|
"""

    for _, row in distance_df.head(10).iterrows():
        report += f"| {row['country']} | {row['distance_km']:.0f} |\n"

    report += """
### Farthest Countries from North Dakota
| Country | Distance (km) |
|---------|--------------|
"""

    for _, row in distance_df.tail(10).iterrows():
        report += f"| {row['country']} | {row['distance_km']:.0f} |\n"

    report += f"""
## Model Results

### Model Comparison

| Metric | Without Distance | With Distance |
|--------|-----------------|---------------|
| N observations | {model_without.get('n_observations', 'N/A')} | {model_with.get('n_observations', 'N/A')} |
| AIC | {model_without.get('fit_statistics', {}).get('aic', 0):.2f} | {model_with.get('fit_statistics', {}).get('aic', 0):.2f} |
| BIC | {model_without.get('fit_statistics', {}).get('bic', 0):.2f} | {model_with.get('fit_statistics', {}).get('bic', 0):.2f} |
| Pseudo R-squared | {model_without.get('fit_statistics', {}).get('pseudo_r2_mcfadden', 0):.4f} | {model_with.get('fit_statistics', {}).get('pseudo_r2_mcfadden', 0):.4f} |

### Distance Coefficient

"""

    if dist_coef is not None:
        sig_str = "YES" if dist_p < 0.05 else "NO"
        report += f"""- **Coefficient:** {dist_coef:.4f}
- **Standard Error:** {model_with.get('coefficients', {}).get('log_distance', {}).get('std_error', 'N/A'):.4f}
- **p-value:** {dist_p:.6f}
- **Significant at 5%:** {sig_str}
"""
    else:
        report += "Distance coefficient not available.\n"

    # Format network elasticity values
    ne_without = comparison.get("network_elasticity_without_distance")
    ne_with = comparison.get("network_elasticity_with_distance")
    ne_without_str = f"{ne_without:.4f}" if ne_without is not None else "N/A"
    ne_with_str = f"{ne_with:.4f}" if ne_with is not None else "N/A"

    report += f"""
### Network Elasticity Comparison

| Model | Diaspora Elasticity |
|-------|-------------------|
| Without Distance | {ne_without_str} |
| With Distance | {ne_with_str} |

## Interpretation

"""

    if dist_coef is not None:
        if dist_p < 0.05:
            if dist_coef < 0:
                report += f"""Distance has a **statistically significant negative effect** on immigration flows to US states.

A 1% increase in distance from origin country is associated with a {abs(dist_coef):.2f}%
**decrease** in LPR admissions. This is consistent with traditional gravity model predictions
and the literature on migration costs.

**For North Dakota specifically:**
- Distant countries (e.g., Southeast Asia, Africa) face higher "migration friction"
- Nearby countries (Canada, Mexico) have geographic advantages
- However, this effect may be dominated by network effects and policy channels
"""
            else:
                report += f"""**UNEXPECTED:** Distance has a significant **positive** effect on immigration flows.

A 1% increase in distance is associated with a {dist_coef:.2f}% **increase** in flows.
This counterintuitive result may reflect:
1. Refugee resettlement patterns (distant conflict zones)
2. Selection effects (only highly motivated migrants from distant countries)
3. Omitted variables correlated with distance (e.g., colonial ties, language)
4. Policy factors (visa quotas, bilateral agreements)
"""
        else:
            report += f"""Distance **does not have a statistically significant effect** on immigration flows (p={dist_p:.4f}).

This suggests that for immigration to US states:
1. **Network effects dominate:** Existing diaspora communities are more important than geography
2. **Migration channels matter:** Refugee resettlement, family reunification, and employment-based
   immigration operate through mechanisms where distance is secondary
3. **US is a global destination:** Migration flows may be primarily driven by origin-country
   push factors and US-specific pull factors rather than geographic proximity

**Implications for North Dakota:**
- Geographic isolation may be less of a barrier than the network analysis suggests
- Policy interventions (refugee resettlement) can overcome distance barriers
- Building diaspora networks may be more effective than geographic marketing
"""

    report += """
## Technical Notes

### Distance Calculation
- Used Haversine (great-circle) formula
- Earth radius: 6,371 km
- Country centroids from Natural Earth / CEPII GeoDist database

### Model Specification
- Estimator: Poisson Pseudo-Maximum Likelihood (PPML)
- Standard errors: Model-based (not clustered in this exploratory analysis)
- Dependent variable: LPR admissions count by state-country pair

### Limitations
1. Centroids may not represent population-weighted origins
2. Distance may be endogenous to migration networks
3. Cross-sectional data cannot capture dynamic effects
4. Regional aggregates use proxy country centroids

---

*This analysis is exploratory and intended to inform whether distance should be included
in the main gravity model specification.*
"""

    return report


def main():
    """Main entry point."""
    print("=" * 70)
    print("DISTANCE ANALYSIS FOR GRAVITY MODEL")
    print(f"Started: {datetime.now(UTC).isoformat()}")
    print("=" * 70)

    try:
        # Calculate distances
        distance_df = calculate_distances_to_nd()

        # Save distance data
        distance_csv = DISTANCE_DIR / "country_distances.csv"
        distance_df.to_csv(distance_csv, index=False)
        print(f"\nDistance data saved: {distance_csv}")

        # Load gravity data
        gravity_df = load_gravity_data()

        # Merge distances
        merged_df = merge_distances(gravity_df, distance_df)

        # Estimate models
        model_results = estimate_gravity_models(merged_df)

        # Save model results
        results_json = DISTANCE_DIR / "gravity_with_distance_results.json"
        with open(results_json, "w") as f:
            json.dump(model_results, f, indent=2, default=str)
        print(f"\nModel results saved: {results_json}")

        # Create summary report
        report = create_summary_report(distance_df, model_results)
        report_md = DISTANCE_DIR / "distance_analysis_summary.md"
        with open(report_md, "w") as f:
            f.write(report)
        print(f"Summary report saved: {report_md}")

        # Final summary
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)

        print("\nOutput files:")
        print(f"  1. {distance_csv}")
        print(f"  2. {results_json}")
        print(f"  3. {report_md}")

        comparison = model_results.get("comparison", {})
        dist_coef = comparison.get("distance_coefficient")
        dist_p = comparison.get("distance_p_value")

        print("\nKey findings:")
        if dist_coef is not None:
            print(f"  - Distance coefficient: {dist_coef:.4f}")
            print(f"  - Distance p-value: {dist_p:.6f}")
            if dist_p < 0.05:
                print("  - Distance is SIGNIFICANT at 5% level")
            else:
                print("  - Distance is NOT significant at 5% level")

        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
